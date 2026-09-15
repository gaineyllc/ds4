/* Model-backed check of ds4_session_rewind for V4.1: a session rewound to
 * an earlier position and re-synced must match a session that reached the
 * same tokens through a fresh prefix and the same replay, and stay close to
 * a single-sweep prefill.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/model.gguf make test-metal-rewind
 * Options: DS4_TEST_PROMPT_FILE (default README.md), DS4_TEST_SSD_STREAMING=1.
 */

#include "ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

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

static void slice(ds4_tokens *out, const ds4_tokens *src, int from, int to) {
    out->len = 0;
    for (int i = from; i < to; i++) ds4_tokens_push(out, src->v[i]);
}

static float *logits_of(ds4_session *s, int vocab) {
    float *l = malloc((size_t)vocab * sizeof(float));
    if (!l || ds4_session_copy_logits(s, l, vocab) != vocab) fail("copy logits");
    return l;
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (!(d <= m)) m = d;
    }
    return m;
}

static int argmax(const float *l, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (l[i] > l[best]) best = i;
    return best;
}

static void sync_or_fail(ds4_session *s, const ds4_tokens *t, const char *what) {
    char err[256];
    if (ds4_session_sync(s, t, err, sizeof(err)) != 0) {
        fprintf(stderr, "sync %s: %s\n", what, err);
        fail(what);
    }
}

/* Greedy-decode `steps` tokens through the sync API; returns the tokens. */
static void greedy(ds4_session *s, const ds4_tokens *prompt, int steps, int vocab, int *out) {
    ds4_tokens t = {0};
    slice(&t, prompt, 0, prompt->len);
    for (int i = 0; i < steps; i++) {
        float *l = logits_of(s, vocab);
        out[i] = argmax(l, vocab);
        free(l);
        ds4_tokens_push(&t, out[i]);
        sync_or_fail(s, &t, "greedy step");
    }
    ds4_tokens_free(&t);
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "SKIP: set DS4_TEST_MODEL\n");
        return 0;
    }
    const char *prompt_path = getenv("DS4_TEST_PROMPT_FILE");
    char *text = read_file(prompt_path && prompt_path[0] ? prompt_path : "README.md", 12000);
    if (!text) fail("prompt file");
    const int ctx = 4096;
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = ctx,
        .ssd_streaming = getenv("DS4_TEST_SSD_STREAMING") != NULL,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) fail("engine open");
    const int vocab = ds4_engine_vocab_size(engine);
    ds4_tokens all = {0};
    ds4_tokenize_text(engine, text, &all);
    int n = all.len < 1800 ? all.len : 1800;
    if (n < 700) fail("prompt too short (need 700 tokens)");
    ds4_tokens full = {0}, prefix = {0}, other = {0};
    slice(&full, &all, 0, n);
    const int deep = n - 300;       /* rewind past the 128-row window */
    slice(&prefix, &all, 0, deep);
    /* A prompt that shares the first deep tokens, then differs. */
    slice(&other, &all, 0, deep);
    for (int i = 0; i < 40; i++) ds4_tokens_push(&other, all.v[(deep + 7 * i + 3) % n]);

    ds4_session *fresh = NULL, *staged = NULL, *rewound = NULL;
    if (ds4_session_create(&fresh, engine, ctx) || ds4_session_create(&staged, engine, ctx) ||
        ds4_session_create(&rewound, engine, ctx)) fail("session create");

    /* Reference: one sweep over the full prompt, then a few greedy tokens. */
    sync_or_fail(fresh, &full, "fresh full");
    float *l_fresh = logits_of(fresh, vocab);
    int g_fresh[6];
    greedy(fresh, &full, 6, vocab, g_fresh);

    /* Staged: the prefix, then the tail as its own sweep (the extension path
     * every cache hit takes; its distance from the reference is the tail
     * path's own numerical noise). */
    sync_or_fail(staged, &prefix, "staged prefix");
    sync_or_fail(staged, &full, "staged tail");
    float *l_staged = logits_of(staged, vocab);
    int g_staged[6];
    greedy(staged, &full, 6, vocab, g_staged);

    /* Rewound: the same prefix and tail as `staged`, plus a few generated
     * tokens, rewound to the prefix, then the tail replayed: the prefix rows
     * were computed by the same sweep, so this must match `staged` exactly.
     * (A single sweep over the whole prompt computes the prefix rows in a
     * different batch shape and lands up to ~1 logit away -- the staged
     * distance above -- so the rewind is judged against `staged`.) */
    sync_or_fail(rewound, &prefix, "rewound prefix");
    sync_or_fail(rewound, &full, "rewound full");
    int g_tmp[6];
    greedy(rewound, &full, 6, vocab, g_tmp);
    if (ds4_session_pos(rewound) != n + 6) fail("rewound position before rewind");
    ds4_session_rewind(rewound, deep);
    if (ds4_session_pos(rewound) != deep) fail("rewind position");
    if (ds4_session_common_prefix(rewound, &full) != deep) fail("rewind kept no state");
    sync_or_fail(rewound, &full, "rewound tail");
    float *l_rewound = logits_of(rewound, vocab);
    int g_rewound[6];
    greedy(rewound, &full, 6, vocab, g_rewound);

    const float staged_vs_fresh = max_abs_diff(l_staged, l_fresh, vocab);
    const float rewound_vs_staged = max_abs_diff(l_rewound, l_staged, vocab);
    const float rewound_vs_fresh = max_abs_diff(l_rewound, l_fresh, vocab);
    fprintf(stderr, "logits: staged-vs-fresh %g  rewound-vs-staged %g  rewound-vs-fresh %g\n",
            staged_vs_fresh, rewound_vs_staged, rewound_vs_fresh);
    fprintf(stderr, "greedy: fresh %d %d %d %d %d %d | staged %d %d %d %d %d %d | rewound %d %d %d %d %d %d\n",
            g_fresh[0], g_fresh[1], g_fresh[2], g_fresh[3], g_fresh[4], g_fresh[5],
            g_staged[0], g_staged[1], g_staged[2], g_staged[3], g_staged[4], g_staged[5],
            g_rewound[0], g_rewound[1], g_rewound[2], g_rewound[3], g_rewound[4], g_rewound[5]);
    if (memcmp(g_rewound, g_staged, sizeof(g_staged)) != 0) fail("rewound greedy differs from staged");
    if (rewound_vs_staged > 1e-4f) fail("rewound logits differ from staged");
    if (argmax(l_rewound, vocab) != argmax(l_fresh, vocab)) fail("rewound argmax differs from fresh");

    /* Divergent prompt after a short rewind (inside the window), against a
     * fresh session that took the same prefix-then-tail route. */
    ds4_session *diverged = NULL;
    if (ds4_session_create(&diverged, engine, ctx)) fail("diverged session");
    sync_or_fail(diverged, &prefix, "diverged prefix");
    sync_or_fail(diverged, &other, "diverged tail");
    float *l_div = logits_of(diverged, vocab);
    ds4_session_rewind(rewound, deep);
    if (ds4_session_common_prefix(rewound, &other) != deep) fail("second rewind kept no state");
    sync_or_fail(rewound, &other, "rewound other");
    float *l_rw2 = logits_of(rewound, vocab);
    const float div_diff = max_abs_diff(l_rw2, l_div, vocab);
    fprintf(stderr, "diverged: rewound-vs-staged %g\n", div_diff);
    if (div_diff > 1e-4f) fail("diverged rewind logits differ");
    /* A rewind by only a few tokens, inside the window. */
    int g_a[4], g_b[4];
    greedy(rewound, &other, 4, vocab, g_a);
    ds4_session_rewind(rewound, other.len - 4);
    sync_or_fail(rewound, &other, "short rewind");
    greedy(rewound, &other, 4, vocab, g_b);
    if (memcmp(g_a, g_b, sizeof(g_a)) != 0) fail("short rewind changed greedy tokens");

    fprintf(stderr, "V4.1 rewind PASS n=%d deep=%d\n", n, deep);
    free(l_fresh); free(l_staged); free(l_rewound); free(l_div); free(l_rw2);
    ds4_session_free(fresh); ds4_session_free(staged); ds4_session_free(rewound); ds4_session_free(diverged);
    ds4_engine_close(engine);
    free(text);
    return 0;
}
