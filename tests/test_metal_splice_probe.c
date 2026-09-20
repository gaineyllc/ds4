/* Cross-context deviation probe for KV splicing (V4.1).
 *
 * Splicing a cached tool-definition block into a prompt it was not computed
 * in is exact for position -- RoPE is a rotation, so a block's K rows can be
 * re-rotated -- but not for context: the block's KV was produced while
 * attending to whatever preceded it.  This measures only that error, with
 * position held fixed:
 *
 *     A  = [system][block A ][block C][tail]
 *     A' = [system][block A'][block C][tail]
 *
 * A and A' are different blocks trimmed to the SAME token length, so block C
 * occupies identical positions in both runs and the only difference is what
 * it attended to.  Comparing logits at increasing depth into C shows how
 * large the deviation is and how fast it decays, i.e. how many tokens at a
 * splice seam would have to be recomputed.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/model.gguf make test-metal-splice-probe
 * Options: DS4_TEST_SSD_STREAMING=1, DS4_TEST_SSD_CACHE_GB=40,
 *          DS4_TEST_BLOCK_TOKENS (default 512).
 */
#include "ds4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *what) { fprintf(stderr, "FAIL: %s\n", what); exit(1); }

static char *read_file(const char *path, size_t cap) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    char *buf = malloc(cap + 1);
    size_t n = buf ? fread(buf, 1, cap, fp) : 0;
    fclose(fp);
    if (!buf) return NULL;
    buf[n] = '\0';
    return buf;
}

static float *logits_of(ds4_session *s, int vocab) {
    float *l = malloc((size_t)vocab * sizeof(float));
    if (!l || ds4_session_copy_logits(s, l, vocab) != vocab) fail("copy logits");
    return l;
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) { const float d = fabsf(a[i] - b[i]); if (!(d <= m)) m = d; }
    return m;
}

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* Tokenize `text` and keep exactly `want` tokens (fails when too short). */
static void tokens_exact(ds4_engine *e, const char *text, int want, ds4_tokens *out) {
    ds4_tokens all = {0};
    ds4_tokenize_text(e, text, &all);
    if (all.len < want) fail("source text too short for the requested block size");
    out->len = 0;
    for (int i = 0; i < want; i++) ds4_tokens_push(out, all.v[i]);
    ds4_tokens_free(&all);
}

static void concat(ds4_tokens *out, const ds4_tokens *a, const ds4_tokens *b) {
    for (int i = 0; i < a->len; i++) ds4_tokens_push(out, a->v[i]);
    for (int i = 0; i < b->len; i++) ds4_tokens_push(out, b->v[i]);
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) { fprintf(stderr, "SKIP: set DS4_TEST_MODEL\n"); return 0; }
    const char *cache_gb = getenv("DS4_TEST_SSD_CACHE_GB");
    const int block = getenv("DS4_TEST_BLOCK_TOKENS") ?
                      atoi(getenv("DS4_TEST_BLOCK_TOKENS")) : 512;
    if (block < 64) fail("DS4_TEST_BLOCK_TOKENS must be at least 64");

    /* Three distinct bodies of text stand in for a system prompt and two
     * interchangeable tool-definition blocks. */
    const char *sys_path = getenv("DS4_TEST_SYS_FILE");
    const char *a_path = getenv("DS4_TEST_A_FILE");
    const char *b_path = getenv("DS4_TEST_B_FILE");
    const char *c_path = getenv("DS4_TEST_C_FILE");
    const char *tail_path = getenv("DS4_TEST_TAIL_FILE");
    char *sys_text = read_file(sys_path ? sys_path : "README.md", 200000);
    char *a_text = read_file(a_path ? a_path : "ds4_server.c", 400000);
    char *b_text = read_file(b_path ? b_path : "ds4_engram.c", 400000);
    char *c_text = read_file(c_path ? c_path : "ds4_kvstore.c", 400000);
    char *tail_text = tail_path ? read_file(tail_path, 100000) : NULL;
    if (!sys_text || !a_text || !b_text || !c_text) fail("source files");
    if (tail_path && !tail_text) fail("tail file");

    ds4_engine_options opt = {
        .model_path = model, .backend = DS4_BACKEND_METAL, .n_threads = 1,
        .context_size = 16384, .power_percent = 100,
        .ssd_streaming = getenv("DS4_TEST_SSD_STREAMING") != NULL,
        .ssd_streaming_cache_bytes = cache_gb ?
            (uint64_t)strtoull(cache_gb, NULL, 10) << 30 : 0,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) fail("engine open");
    const int vocab = ds4_engine_vocab_size(engine);

    ds4_tokens sys = {0}, ba = {0}, bb = {0}, bc = {0};
    const int sys_tokens = getenv("DS4_TEST_SYS_TOKENS") ?
                           atoi(getenv("DS4_TEST_SYS_TOKENS")) : 1024;
    tokens_exact(engine, sys_text, sys_tokens, &sys);
    tokens_exact(engine, a_text, block, &ba);   /* block A  */
    tokens_exact(engine, b_text, block, &bb);   /* block A' -- same length */
    tokens_exact(engine, c_text, block, &bc);   /* block C  -- the spliced one */

    ds4_tokens pre_a = {0}, pre_b = {0};
    concat(&pre_a, &sys, &ba);
    concat(&pre_b, &sys, &bb);
    if (pre_a.len != pre_b.len) fail("prefixes differ in length");
    fprintf(stderr, "probe: prefix=%d tokens (system %d + block %d), spliced block=%d tokens\n",
            pre_a.len, sys.len, block, bc.len);

    ds4_session *sa = NULL, *sb = NULL;
    if (ds4_session_create(&sa, engine, 16384) || ds4_session_create(&sb, engine, 16384))
        fail("session create");

    /* Walk depth into block C, comparing the two contexts at identical
     * positions.  Depth 1 is the seam itself. */
    const int depths[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    char err[256];
    int agree = 0, total = 0;
    fprintf(stderr, "%8s  %12s  %12s  %s\n", "depth", "max|dlogit|", "top1 same", "top1 a / top1 b");
    for (size_t i = 0; i < sizeof(depths) / sizeof(depths[0]); i++) {
        const int d = depths[i];
        if (d > bc.len) break;
        ds4_tokens pa = {0}, pb = {0};
        for (int k = 0; k < pre_a.len; k++) ds4_tokens_push(&pa, pre_a.v[k]);
        for (int k = 0; k < pre_b.len; k++) ds4_tokens_push(&pb, pre_b.v[k]);
        for (int k = 0; k < d; k++) { ds4_tokens_push(&pa, bc.v[k]); ds4_tokens_push(&pb, bc.v[k]); }
        if (ds4_session_sync(sa, &pa, err, sizeof(err)) != 0) fail("sync a");
        if (ds4_session_sync(sb, &pb, err, sizeof(err)) != 0) fail("sync b");
        float *la = logits_of(sa, vocab), *lb = logits_of(sb, vocab);
        const int ta = argmax(la, vocab), tb = argmax(lb, vocab);
        total++; if (ta == tb) agree++;
        fprintf(stderr, "%8d  %12g  %12s  %d / %d\n",
                d, max_abs_diff(la, lb, vocab), ta == tb ? "yes" : "NO", ta, tb);
        free(la); free(lb);
        ds4_tokens_free(&pa); ds4_tokens_free(&pb);
    }
    fprintf(stderr, "probe done: top-1 agreement %d/%d\n", agree, total);

    /* The question that decides splicing: with the whole block in place and a
     * user turn appended, does the answer change?  Greedy, 12 tokens. */
    if (tail_text) {
        ds4_tokens tail = {0};
        ds4_tokenize_text(engine, tail_text, &tail);
        ds4_tokens pa = {0}, pb = {0};
        for (int k = 0; k < pre_a.len; k++) ds4_tokens_push(&pa, pre_a.v[k]);
        for (int k = 0; k < pre_b.len; k++) ds4_tokens_push(&pb, pre_b.v[k]);
        for (int k = 0; k < bc.len; k++) { ds4_tokens_push(&pa, bc.v[k]); ds4_tokens_push(&pb, bc.v[k]); }
        for (int k = 0; k < tail.len; k++) { ds4_tokens_push(&pa, tail.v[k]); ds4_tokens_push(&pb, tail.v[k]); }
        if (ds4_session_sync(sa, &pa, err, sizeof(err)) != 0) fail("sync a tail");
        if (ds4_session_sync(sb, &pb, err, sizeof(err)) != 0) fail("sync b tail");
        float *la = logits_of(sa, vocab), *lb = logits_of(sb, vocab);
        fprintf(stderr, "after tail (%d tokens): max|dlogit| %g  top1 %d / %d\n",
                tail.len, max_abs_diff(la, lb, vocab), argmax(la, vocab), argmax(lb, vocab));
        free(la); free(lb);
        int same = 0;
        for (int step = 0; step < 12; step++) {
            float *xa = logits_of(sa, vocab), *xb = logits_of(sb, vocab);
            const int ta = argmax(xa, vocab), tb = argmax(xb, vocab);
            free(xa); free(xb);
            fprintf(stderr, "  gen %2d: %6d %s %-6d\n", step, ta, ta == tb ? "==" : "!=", tb);
            if (ta == tb) same++;
            ds4_tokens_push(&pa, ta); ds4_tokens_push(&pb, tb);
            if (ds4_session_sync(sa, &pa, err, sizeof(err)) != 0) fail("gen a");
            if (ds4_session_sync(sb, &pb, err, sizeof(err)) != 0) fail("gen b");
        }
        fprintf(stderr, "greedy agreement after the blocks: %d/12\n", same);
        ds4_tokens_free(&tail); ds4_tokens_free(&pa); ds4_tokens_free(&pb);
    }
    ds4_session_free(sa); ds4_session_free(sb);
    ds4_tokens_free(&sys); ds4_tokens_free(&ba); ds4_tokens_free(&bb); ds4_tokens_free(&bc);
    ds4_tokens_free(&pre_a); ds4_tokens_free(&pre_b);
    ds4_engine_close(engine);
    free(sys_text); free(a_text); free(b_text); free(c_text); free(tail_text);
    return 0;
}
