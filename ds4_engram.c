#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "ds4_engram.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#else
#include <pthread.h>
#endif

bool ds4_engram_layout_valid(const ds4_engram_layout *l) {
    if (!l || !l->token_map || !l->vocab_size ||
        !l->compressed_vocab_size || l->compressed_vocab_size > INT32_MAX ||
        l->pad_id >= l->compressed_vocab_size) return false;
    for (uint32_t i = 0; i < l->vocab_size; i++)
        if (l->token_map[i] >= l->compressed_vocab_size) return false;
    for (int layer = 0; layer < DS4_ENGRAM_LAYERS; layer++) {
        for (int i = 0; i < DS4_ENGRAM_NGRAM; i++) {
            uint64_t m = l->multipliers[layer][i];
            if (!(m & 1) || m > (uint64_t)INT64_MAX / l->compressed_vocab_size)
                return false;
        }
        uint64_t total = 0;
        for (int i = 0; i < DS4_ENGRAM_COLS; i++) {
            if (l->primes[layer][i] < 2) return false;
            total += l->primes[layer][i];
        }
        if (total != l->rows[layer]) return false;
    }
    return true;
}

void ds4_engram_history_reset(ds4_engram_history *h) {
    for (int i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) h->tail[i] = DS4_ENGRAM_DEAD;
}

bool ds4_engram_hash(const ds4_engram_layout *l, ds4_engram_history *h,
                     const int *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows) {
    if (!l || !h || !l->token_map || (count && (!tokens || !rows)) ||
        count > SIZE_MAX / (DS4_ENGRAM_LAYERS * DS4_ENGRAM_COLS * sizeof(*rows)))
        return false;
    for (int i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) {
        if (h->tail[i] < DS4_ENGRAM_DEAD ||
            (h->tail[i] >= 0 && (uint32_t)h->tail[i] >= l->compressed_vocab_size))
            return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= l->vocab_size) return false;
    }
    for (size_t i = 0; i < count; i++) {
        int32_t current = mask && !mask[i] ? DS4_ENGRAM_DEAD :
                          (int32_t)l->token_map[tokens[i]];
        uint32_t ids[DS4_ENGRAM_NGRAM];
        bool blocked = false;
        for (int j = 0; j < DS4_ENGRAM_NGRAM; j++) {
            int32_t id = j ? h->tail[j - 1] : current;
            blocked |= id == DS4_ENGRAM_DEAD;
            ids[j] = blocked ? l->pad_id : (uint32_t)id;
        }
        for (int layer = 0; layer < DS4_ENGRAM_LAYERS; layer++) {
            uint64_t hash = (uint64_t)ids[0] * l->multipliers[layer][0];
            uint32_t offset = 0;
            for (int j = 1; j < DS4_ENGRAM_NGRAM; j++) {
                hash ^= (uint64_t)ids[j] * l->multipliers[layer][j];
                for (int head = 0; head < DS4_ENGRAM_HEADS; head++) {
                    int col = (j - 1) * DS4_ENGRAM_HEADS + head;
                    uint32_t prime = l->primes[layer][col];
                    *rows++ = (uint32_t)(hash % prime) + offset;
                    offset += prime;
                }
            }
        }
        for (int j = DS4_ENGRAM_NGRAM - 2; j > 0; j--) h->tail[j] = h->tail[j - 1];
        h->tail[0] = current;
    }
    return true;
}

bool ds4_engram_table_open(ds4_engram_table *t, const char *path,
                           uint64_t offset, uint32_t rows) {
    if (!t) return false;
    *t = (ds4_engram_table){.fd = -1};
    uint64_t bytes = (uint64_t)rows * DS4_ENGRAM_ROW_BYTES;
    if (!path || !rows || offset > INT64_MAX || bytes > INT64_MAX - offset) {
        errno = EINVAL;
        return false;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) goto fail;
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || offset + bytes > (uint64_t)st.st_size) {
        errno = EINVAL;
        goto fail;
    }
#ifdef __APPLE__
    if (fcntl(fd, F_NOCACHE, 1) != 0 || fcntl(fd, F_RDAHEAD, 0) != 0) goto fail;
#endif
    *t = (ds4_engram_table){.fd = fd, .offset = offset, .rows = rows};
    return true;
fail: {
        int saved = errno;
        close(fd);
        errno = saved;
        return false;
    }
}

void ds4_engram_table_close(ds4_engram_table *t) {
    if (!t) return;
    if (t->fd >= 0) close(t->fd);
    *t = (ds4_engram_table){.fd = -1};
}

static float e4m3(uint8_t byte) {
    int exponent = (byte >> 3) & 15, mantissa = byte & 7;
    float value = exponent ? ldexpf((float)(8 + mantissa), exponent - 10) :
                             ldexpf((float)mantissa, -9);
    return byte & 128 ? -value : value;
}

typedef struct {
    uint32_t row, output;
} engram_request;

static int request_order(const void *a, const void *b) {
    const engram_request *x = a, *y = b;
    return (x->row > y->row) - (x->row < y->row);
}

/* ------------------------------------------------------------------------
 * I/O concurrency profile.
 *
 * Engram rows are 264 B and the table fd is F_NOCACHE, so every miss is a real
 * device round-trip and the reader is latency-bound, not CPU-bound: the threads
 * sit blocked in pread and burn nothing. That means useful width is set by how
 * many requests the NVMe queue will reward, NOT by core count -- on an M5 Max
 * (18 logical cores) measured random-264B throughput still climbs from 168k
 * rows/s at 16 threads to 239k at 96. Different Macs have different SSD
 * controllers and queue behaviour, so the width is measured on the actual
 * device at first use rather than hardcoded.
 *
 * Probe once per process, cache the answer keyed by device id so repeat runs
 * skip it. DS4_ENGRAM_READERS overrides and skips probing entirely.
 * ------------------------------------------------------------------------ */

enum {
    DS4_ENGRAM_MAX_READERS = 64,
    /*
     * Below this many rows, serial wins. A decode step asks for COLS=24 rows
     * per table: measured on an M5 Max, routing that through the pool costs
     * ~15% decode (13.8 -> 11.7 t/s) because waking the readers through one
     * mutex twice per token dwarfs the ~2.4 ms of reads it parallelises.
     * Prefill batches (up to 2048 x 24) amortise it and gain 1.3-2.4x.
     * Override with DS4_ENGRAM_MIN_PARALLEL.
     */
    DS4_ENGRAM_MIN_PARALLEL_DEFAULT = 256,
    DS4_ENGRAM_PROBE_READS = 192,    /* per width, per probe */
    DS4_ENGRAM_PAGE = 4096
};

static int g_engram_readers;
static pthread_mutex_t g_engram_init_mu = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    int fd;
    uint64_t span;
    int reads;
    unsigned seed;
} engram_probe_arg;

static void *engram_probe_worker(void *p) {
    engram_probe_arg *a = p;
    unsigned s = a->seed;
    uint8_t buf[DS4_ENGRAM_ROW_BYTES];
    for (int i = 0; i < a->reads; i++) {
        uint64_t r = ((uint64_t)rand_r(&s) << 31 | (uint64_t)rand_r(&s)) %
                     (a->span ? a->span : 1);
        if (pread(a->fd, buf, sizeof(buf),
                  (off_t)(r * DS4_ENGRAM_ROW_BYTES)) < 0 && errno != EINTR) {
            break;
        }
    }
    return NULL;
}

static double engram_probe_rate(int fd, uint64_t span, int width) {
    pthread_t th[DS4_ENGRAM_MAX_READERS];
    engram_probe_arg ag[DS4_ENGRAM_MAX_READERS];
    struct timespec t0, t1;
    int spawned = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < width; i++) {
        ag[i] = (engram_probe_arg){ fd, span, DS4_ENGRAM_PROBE_READS,
                                    (unsigned)(i * 2654435761u + width * 97u + 1u) };
        if (pthread_create(&th[i], NULL, engram_probe_worker, &ag[i]) != 0) break;
        spawned++;
    }
    for (int i = 0; i < spawned; i++) pthread_join(th[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (!spawned) return 0.0;
    double dt = (double)(t1.tv_sec - t0.tv_sec) +
                (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    if (dt <= 0.0) return 0.0;
    return (double)spawned * DS4_ENGRAM_PROBE_READS / dt;
}

static void engram_cache_path(char *buf, size_t n, dev_t dev) {
    const char *over = getenv("DS4_ENGRAM_IO_CACHE");
    if (over && over[0]) { snprintf(buf, n, "%s", over); return; }
    const char *home = getenv("HOME");
    snprintf(buf, n, "%s/.cache/ds4-engram-io-%llu",
             home && home[0] ? home : "/tmp", (unsigned long long)dev);
}

static int engram_cache_load(dev_t dev) {
    char path[PATH_MAX];
    engram_cache_path(path, sizeof(path), dev);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int w = 0;
    if (fscanf(f, "%d", &w) != 1) w = 0;
    fclose(f);
    return (w >= 1 && w <= DS4_ENGRAM_MAX_READERS) ? w : 0;
}

static void engram_cache_store(dev_t dev, int width) {
    char path[PATH_MAX];
    engram_cache_path(path, sizeof(path), dev);
    char dir[PATH_MAX];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) { *slash = '\0'; mkdir(dir, 0700); }
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "%d\n", width);
    fclose(f);
}

/*
 * Pick the smallest width within 5% of the best measured rate: past the knee
 * the extra threads buy noise, and a narrower pool leaves queue capacity for
 * the expert streamer sharing the same device.
 */
static int engram_calibrate(const ds4_engram_table *t) {
    static const int widths[] = { 8, 16, 24, 32, 48, 64 };
    const uint64_t span = t->rows ? t->rows : 1;
    double best = 0.0;
    double rate[sizeof(widths) / sizeof(*widths)];
    for (unsigned i = 0; i < sizeof(widths) / sizeof(*widths); i++) {
        rate[i] = engram_probe_rate(t->fd, span, widths[i]);
        if (rate[i] > best) best = rate[i];
    }
    if (best <= 0.0) return 16;
    int chosen = widths[sizeof(widths) / sizeof(*widths) - 1];
    for (unsigned i = 0; i < sizeof(widths) / sizeof(*widths); i++) {
        if (rate[i] >= best * 0.95) { chosen = widths[i]; break; }
    }
    fprintf(stderr,
            "ds4: engram I/O probe: %.0fk rows/s peak, using %d readers"
            " (8:%.0fk 16:%.0fk 24:%.0fk 32:%.0fk 48:%.0fk 64:%.0fk)\n",
            best / 1000.0, chosen,
            rate[0] / 1000.0, rate[1] / 1000.0, rate[2] / 1000.0,
            rate[3] / 1000.0, rate[4] / 1000.0, rate[5] / 1000.0);
    return chosen;
}

static int engram_readers(const ds4_engram_table *t) {
    pthread_mutex_lock(&g_engram_init_mu);
    if (g_engram_readers == 0) {
        int w = 0;
        const char *env = getenv("DS4_ENGRAM_READERS");
        if (env && env[0]) {
            long v = strtol(env, NULL, 10);
            if (v >= 1 && v <= DS4_ENGRAM_MAX_READERS) w = (int)v;
        }
        struct stat st;
        dev_t dev = (t && fstat(t->fd, &st) == 0) ? st.st_dev : 0;
        if (!w) w = engram_cache_load(dev);
        if (!w && t && t->fd >= 0) {
            w = engram_calibrate(t);
            engram_cache_store(dev, w);
        }
        g_engram_readers = w ? w : 16;
    }
    int r = g_engram_readers;
    pthread_mutex_unlock(&g_engram_init_mu);
    return r;
}

typedef struct {
    const ds4_engram_table *table;
    const engram_request *request;
    float *out;
    size_t count;
    uint32_t parts;
    int error[DS4_ENGRAM_MAX_READERS];
} engram_batch;

static void engram_decode_row(const uint8_t *raw, float *dst, int *err) {
    for (int j = 0; j < DS4_ENGRAM_DIM; j++) {
        uint8_t code = raw[j], scale = raw[DS4_ENGRAM_DIM + j / 32];
        if ((code & 127) == 127 || scale == 255) { *err = EDOM; return; }
        float value = ldexpf(e4m3(code), (int)scale - 127);
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
        memcpy(&value, &bits, sizeof(value));
        if (!isfinite(value)) { *err = EDOM; return; }
        dst[j] = value;
    }
}

/*
 * Sorted rows frequently land in the same 4 KiB page: a page holds 15.5 of
 * them, and F_NOCACHE makes the device read a full page regardless. Keep the
 * last window so a run of same-page rows costs one device read instead of one
 * per row. The window is two pages because a 264 B row can straddle a
 * boundary.
 */
typedef struct {
    uint8_t  data[2 * DS4_ENGRAM_PAGE];
    uint64_t base;
    size_t   len;
    bool     valid;
} engram_window;

static bool engram_window_fetch(int fd, engram_window *w,
                                uint64_t off, const uint8_t **out) {
    const uint64_t base = off & ~(uint64_t)(DS4_ENGRAM_PAGE - 1);
    const size_t need = (size_t)(off - base) + DS4_ENGRAM_ROW_BYTES;
    if (w->valid && base == w->base && w->len >= need) {
        *out = w->data + (off - base);
        return true;
    }
    const size_t want = need > DS4_ENGRAM_PAGE ? 2 * DS4_ENGRAM_PAGE
                                               : DS4_ENGRAM_PAGE;
    size_t done = 0;
    while (done < need) {
        ssize_t n = pread(fd, w->data + done, want - done, (off_t)(base + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { if (n == 0) errno = EIO; w->valid = false; return false; }
        done += (size_t)n;
        if (done >= want) break;
    }
    if (done < need) { errno = EIO; w->valid = false; return false; }
    w->base = base;
    w->len = done;
    w->valid = true;
    *out = w->data + (off - base);
    return true;
}

static void read_batch_part(void *context, size_t part) {
    engram_batch *batch = context;
    const engram_request *request = batch->request;
    const size_t begin = batch->count * part / batch->parts;
    const size_t end = batch->count * (part + 1) / batch->parts;
    const float *previous = NULL;
    engram_window window = { .valid = false };
    int err = 0;
    for (size_t i = begin; i < end; i++) {
        float *dst = batch->out + (size_t)request[i].output * DS4_ENGRAM_DIM;
        if (i > begin && request[i].row == request[i - 1].row) {
            memcpy(dst, previous, DS4_ENGRAM_DIM * sizeof(*dst));
            continue;
        }
        const uint64_t off = batch->table->offset +
                             (uint64_t)request[i].row * DS4_ENGRAM_ROW_BYTES;
        const uint8_t *raw = NULL;
        if (!engram_window_fetch(batch->table->fd, &window, off, &raw)) {
            batch->error[part] = errno ? errno : EIO;
            return;
        }
        engram_decode_row(raw, dst, &err);
        if (err) { batch->error[part] = err; return; }
        previous = dst;
    }
}

/* Persistent reader pool. Threads are created once and parked on a condvar:
 * a decode batch is only 24 rows, so per-batch thread creation (~30 us each)
 * would cost more than the reads it parallelises. Work is stolen part-wise so
 * a slow device read cannot stall a whole slice. */
static struct {
    pthread_t       th[DS4_ENGRAM_MAX_READERS];
    pthread_mutex_t mu;
    pthread_cond_t  start_cv, done_cv;
    uint32_t        nthreads, next_part, remaining;
    uint64_t        generation;
    engram_batch   *job;
    int             stopping, started;
} g_pool = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .start_cv = PTHREAD_COND_INITIALIZER,
    .done_cv = PTHREAD_COND_INITIALIZER,
};

static void *engram_pool_worker(void *arg) {
    (void)arg;
    uint64_t seen = 0;
    pthread_mutex_lock(&g_pool.mu);
    for (;;) {
        while (!g_pool.stopping && g_pool.generation == seen)
            pthread_cond_wait(&g_pool.start_cv, &g_pool.mu);
        if (g_pool.stopping) break;
        seen = g_pool.generation;
        engram_batch *job = g_pool.job;
        while (job && g_pool.next_part < job->parts) {
            const uint32_t part = g_pool.next_part++;
            pthread_mutex_unlock(&g_pool.mu);
            read_batch_part(job, part);
            pthread_mutex_lock(&g_pool.mu);
        }
        if (g_pool.remaining > 0 && --g_pool.remaining == 0)
            pthread_cond_signal(&g_pool.done_cv);
    }
    pthread_mutex_unlock(&g_pool.mu);
    return NULL;
}

static void engram_pool_shutdown(void) {
    pthread_mutex_lock(&g_pool.mu);
    if (!g_pool.started) { pthread_mutex_unlock(&g_pool.mu); return; }
    g_pool.stopping = 1;
    pthread_cond_broadcast(&g_pool.start_cv);
    const uint32_t n = g_pool.nthreads;
    pthread_mutex_unlock(&g_pool.mu);
    for (uint32_t i = 0; i < n; i++) pthread_join(g_pool.th[i], NULL);
    pthread_mutex_lock(&g_pool.mu);
    g_pool.started = 0;
    g_pool.nthreads = 0;
    g_pool.stopping = 0;
    pthread_mutex_unlock(&g_pool.mu);
}

static bool engram_pool_start(int width) {
    pthread_mutex_lock(&g_pool.mu);
    if (g_pool.started) { pthread_mutex_unlock(&g_pool.mu); return true; }
    g_pool.nthreads = 0;
    for (int i = 0; i < width; i++) {
        if (pthread_create(&g_pool.th[i], NULL, engram_pool_worker, NULL) != 0) break;
        g_pool.nthreads++;
    }
    g_pool.started = g_pool.nthreads > 0;
    const bool ok = g_pool.started;
    pthread_mutex_unlock(&g_pool.mu);
    if (ok) atexit(engram_pool_shutdown);
    return ok;
}

static void engram_pool_run(engram_batch *b) {
    pthread_mutex_lock(&g_pool.mu);
    g_pool.job = b;
    g_pool.next_part = 0;
    g_pool.remaining = g_pool.nthreads;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.start_cv);
    while (g_pool.remaining != 0)
        pthread_cond_wait(&g_pool.done_cv, &g_pool.mu);
    g_pool.job = NULL;
    pthread_mutex_unlock(&g_pool.mu);
}

static size_t engram_min_parallel(void) {
    static size_t cached;
    if (cached == 0) {
        size_t v = DS4_ENGRAM_MIN_PARALLEL_DEFAULT;
        const char *env = getenv("DS4_ENGRAM_MIN_PARALLEL");
        if (env && env[0]) {
            long n = strtol(env, NULL, 10);
            if (n >= 1) v = (size_t)n;
        }
        cached = v;
    }
    return cached;
}

/* Run `count` presorted requests, in parallel when it is worth it. */
static bool engram_run_requests(const ds4_engram_table *t,
                                engram_request *request,
                                size_t count,
                                float *out) {
    engram_batch batch = { .table = t, .request = request, .count = count,
                           .out = out, .parts = 1 };
    memset(batch.error, 0, sizeof(batch.error));
    int width = 1;
    if (count >= engram_min_parallel()) {
        width = engram_readers(t);
        if ((size_t)width > count) width = (int)count;
        if (width > 1 && engram_pool_start(engram_readers(t))) {
            /* Oversubscribe parts so a slow read cannot stall a whole slice,
             * but never past error[] -- part index is the error slot. */
            uint32_t parts = (uint32_t)width * 2u;
            if (parts > DS4_ENGRAM_MAX_READERS) parts = DS4_ENGRAM_MAX_READERS;
            if ((size_t)parts > count) parts = (uint32_t)count;
            batch.parts = parts ? parts : 1;
            engram_pool_run(&batch);
            for (uint32_t i = 0; i < batch.parts; i++) {
                if (batch.error[i]) { errno = batch.error[i]; return false; }
            }
            return true;
        }
    }
    batch.parts = 1;
    read_batch_part(&batch, 0);
    if (batch.error[0]) { errno = batch.error[0]; return false; }
    return true;
}

bool ds4_engram_read(const ds4_engram_table *t, const uint32_t *rows,
                     size_t count, float *out) {
    if (!t || t->fd < 0 || (count && (!rows || !out)) ||
        count > SIZE_MAX / (DS4_ENGRAM_DIM * sizeof(*out))) {
        errno = EINVAL;
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (rows[i] >= t->rows) { errno = EINVAL; return false; }
    }
    if (!count) return true;

    /* Decode asks for COLS rows per table per token. Sorting lets the window
     * coalesce same-page rows, and the pool hides the device latency that used
     * to be paid one strictly serial pread at a time. */
    engram_request stack_req[DS4_ENGRAM_COLS * 4];
    engram_request *request = stack_req;
    if (count > sizeof(stack_req) / sizeof(*stack_req)) {
        request = malloc(count * sizeof(*request));
        if (!request) return false;
    }
    for (size_t i = 0; i < count; i++)
        request[i] = (engram_request){ rows[i], (uint32_t)i };
    qsort(request, count, sizeof(*request), request_order);
    const bool ok = engram_run_requests(t, request, count, out);
    const int saved = errno;
    if (request != stack_req) free(request);
    errno = saved;
    return ok;
}

#ifndef __APPLE__
typedef struct {
    engram_batch *batch;
    size_t part;
} engram_reader;

static void *read_batch_thread(void *context) {
    engram_reader *reader = context;
    read_batch_part(reader->batch, reader->part);
    return NULL;
}
#endif

bool ds4_engram_read_batch(const ds4_engram_table *t, const uint32_t *rows,
                           size_t tokens, size_t stride, float *out) {
    if (!t || t->fd < 0 || (tokens && (!rows || !out || stride < DS4_ENGRAM_COLS)) ||
        tokens > SIZE_MAX / (DS4_ENGRAM_COLS * DS4_ENGRAM_DIM * sizeof(*out)) ||
        (tokens && tokens - 1 > (SIZE_MAX / sizeof(*rows) - DS4_ENGRAM_COLS) / stride)) {
        errno = EINVAL;
        return false;
    }
    for (size_t i = 0; i < tokens; i++) {
        for (size_t j = 0; j < DS4_ENGRAM_COLS; j++) {
            if (rows[i * stride + j] >= t->rows) { errno = EINVAL; return false; }
        }
    }
    if (!tokens) return true;
    enum { BATCH_TOKENS = 2048 };
    const size_t cap = tokens < BATCH_TOKENS ? tokens : BATCH_TOKENS;
    engram_request *request = malloc(cap * DS4_ENGRAM_COLS * sizeof(*request));
    if (!request) return false;
    bool ok = true;
    for (size_t start = 0; ok && start < tokens; start += cap) {
        const size_t n = tokens - start < cap ? tokens - start : cap;
        const size_t count = n * DS4_ENGRAM_COLS;
        for (size_t i = 0; i < count; i++) {
            request[i] = (engram_request){
                rows[(start + i / DS4_ENGRAM_COLS) * stride + i % DS4_ENGRAM_COLS],
                (uint32_t)i
            };
        }
        qsort(request, count, sizeof(*request), request_order);
        ok = engram_run_requests(t, request, count,
                                 out + start * DS4_ENGRAM_COLS * DS4_ENGRAM_DIM);
    }
    const int saved = errno;
    free(request);
    errno = saved;
    return ok;
}
