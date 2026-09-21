/* Model-backed regression test: a V4.1 prompt read as back-to-back extensions
 * must succeed and match the same prompt read in fewer pieces.
 *
 * Found by continued-checkpoint segmentation (9bbe7f7): with 8,192-token
 * segments the second extension failed with "V4.1 layer-major prefill failed
 * at layer 39", from zero and after a disk restore alike, while 16,384-token
 * segments worked.
 *
 *   DS4_TEST_MODEL=... DS4_TEST_SSD_STREAMING=1 ./tests/test_metal_prefill_segments
 */
#include "ds4.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static char *read_file(const char *path, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *buf = malloc(cap + 1);
    size_t n = fread(buf, 1, cap, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

static void slice(ds4_tokens *out, const ds4_tokens *src, int to) {
    out->len = 0;
    for (int i = 0; i < to; i++) ds4_tokens_push(out, src->v[i]);
}

/* Read `all` up to each boundary in turn; returns the boundary that failed,
 * or 0 when every step succeeded. Fills logits at the last boundary. */
static int read_in_steps(ds4_engine *e, int ctx, const ds4_tokens *all,
                         const int *steps, int nsteps, float *logits, int vocab,
                         bool save_between) {
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, ctx)) { fprintf(stderr, "session create\n"); exit(1); }
    ds4_tokens p = {0};
    int failed = 0;
    for (int i = 0; i < nsteps && !failed; i++) {
        slice(&p, all, steps[i]);
        char err[256] = {0};
        if (ds4_session_sync(s, &p, err, sizeof(err)) != 0 ||
            ds4_session_pos(s) != steps[i]) {
            fprintf(stderr, "  step to %d failed: %s\n", steps[i], err);
            failed = steps[i];
        }
        /* What the server's KV disk store does at a continued boundary. */
        if (!failed && save_between && i + 1 < nsteps) {
            FILE *fp = tmpfile();
            char serr[256] = {0};
            if (!fp || ds4_session_save_payload(s, fp, serr, sizeof(serr)) != 0) {
                fprintf(stderr, "  save at %d failed: %s\n", steps[i], serr);
                failed = -2;
            } else {
                fprintf(stderr, "  saved payload at %d (%ld bytes)\n", steps[i], ftell(fp));
            }
            if (fp) fclose(fp);
        }
    }
    if (!failed && logits && ds4_session_copy_logits(s, logits, vocab) != vocab) failed = -1;
    free(p.v);
    ds4_session_free(s);
    return failed;
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model) { fprintf(stderr, "SKIP: set DS4_TEST_MODEL\n"); return 0; }
    const char *path = getenv("DS4_TEST_PROMPT_FILE");
    char *text = read_file(path ? path : "/tmp/p25k.txt", 200000);
    if (!text) { fprintf(stderr, "FAIL: prompt file\n"); return 1; }
    const int ctx = 32768;
    ds4_engine_options opt = {
        .model_path = model, .backend = DS4_BACKEND_METAL, .n_threads = 1,
        .context_size = ctx,
        .ssd_streaming = getenv("DS4_TEST_SSD_STREAMING") != NULL,
        .ssd_streaming_cache_bytes = getenv("DS4_TEST_SSD_CACHE_GB") ?
            (uint64_t)strtoull(getenv("DS4_TEST_SSD_CACHE_GB"), NULL, 10) << 30 : 0,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) { fprintf(stderr, "FAIL: engine open\n"); return 1; }
    const int vocab = ds4_engine_vocab_size(e);
    ds4_tokens all = {0};
    ds4_tokenize_text(e, text, &all);
    if (all.len < 16384) { fprintf(stderr, "FAIL: need 16384 tokens, have %d\n", all.len); return 1; }

    const char *only = getenv("DS4_TEST_CASE");
    struct { const char *name; int steps[4]; int n; } cases[] = {
        { "8192+8192",   { 8192, 16384 },        2 },
        { "16384 once",  { 16384 },              1 },
        { "4096+4096",   { 4096, 8192 },         2 },
        { "12288+4096",  { 12288, 16384 },       2 },
        { "8192+4096",   { 8192, 12288 },        2 },
    };
    float *ref = malloc((size_t)vocab * sizeof(float));
    float *got = malloc((size_t)vocab * sizeof(float));
    int bad = 0, ref_am = -1;
    bool have_ref = false;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        if (only && !strstr(only, cases[c].name)) continue;
        const bool save = getenv("DS4_TEST_SAVE_BETWEEN") != NULL;
        fprintf(stderr, "case %s%s\n", cases[c].name, save ? " (save between)" : "");
        const int f = read_in_steps(e, ctx, &all, cases[c].steps, cases[c].n, got, vocab, save);
        fprintf(stderr, "  -> %s\n", f ? "FAILED" : "ok");
        if (f) { bad++; continue; }
        /* Every case that ends at 16384 has read the same tokens, however
         * split, so they must agree on the next token. */
        if (cases[c].steps[cases[c].n - 1] != 16384) continue;
        int am = 0;
        for (int i = 1; i < vocab; i++) if (got[i] > got[am]) am = i;
        if (!have_ref) {
            memcpy(ref, got, (size_t)vocab * sizeof(float));
            ref_am = am; have_ref = true;
            fprintf(stderr, "  reference argmax %d (%.3f)\n", am, got[am]);
            continue;
        }
        float md = 0.0f;
        for (int i = 0; i < vocab; i++) {
            const float d = fabsf(got[i] - ref[i]);
            if (d > md) md = d;
        }
        fprintf(stderr, "  argmax %d vs %d, max |dlogit| %.3f\n", am, ref_am, md);
        if (am != ref_am) { fprintf(stderr, "  -> next token differs\n"); bad++; }
    }
    free(ref); free(got); free(all.v); free(text);
    ds4_engine_close(e);
    if (bad) { fprintf(stderr, "FAIL: %d case(s)\n", bad); return 1; }
    fprintf(stderr, "V4.1 segmented prefill PASS\n");
    return 0;
}
