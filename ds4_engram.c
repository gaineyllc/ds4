#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "ds4_engram.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
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
    /* Cached reads by default.  The engram tables live inside the weights
     * file, which Metal also maps and wires pages of for the streaming expert
     * bank; an uncached read has to evict any cached page in its range first,
     * and twice in two days a reader wedged in pread on exactly this file and
     * stayed there (every later read of the file blocked behind it, the server
     * survived only a reboot).  No other file on the machine does both, and
     * nothing in the kernel log pointed at the disk.  Cached reads share the
     * page cache with those mappings instead of fighting them.
     * DS4_ENGRAM_NOCACHE=1 restores the uncached reads. */
    const char *nocache = getenv("DS4_ENGRAM_NOCACHE");
    if (nocache && nocache[0] == '1') {
        if (fcntl(fd, F_NOCACHE, 1) != 0) goto fail;
    }
    if (fcntl(fd, F_RDAHEAD, 0) != 0) goto fail;
#elif defined(POSIX_FADV_RANDOM)
    /* Hashed rows are scattered across the table. Read-ahead fetches unused
     * pages and competes with the model's limited memory headroom. */
    (void)posix_fadvise(fd, (off_t)offset, (off_t)bytes, POSIX_FADV_RANDOM);
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
    ds4_engram_table_unmap(t);
    if (t->fd >= 0) close(t->fd);
    *t = (ds4_engram_table){.fd = -1};
}

static float e4m3(uint8_t byte) {
    int exponent = (byte >> 3) & 15, mantissa = byte & 7;
    float value = exponent ? ldexpf((float)(8 + mantissa), exponent - 10) :
                             ldexpf((float)mantissa, -9);
    return byte & 128 ? -value : value;
}

/* All code/scale pairs fit in 128 KiB; this is a format lookup, not a cache
 * of model rows. Preserve the reference's F32 scaling and BF16 rounding. */
static uint16_t engram_bf16[256][256];
static pthread_once_t engram_bf16_once = PTHREAD_ONCE_INIT;

static void init_engram_bf16(void) {
    for (unsigned scale = 0; scale < 256; scale++) {
        for (unsigned code = 0; code < 256; code++) {
            uint16_t result = UINT16_MAX;
            if ((code & 127u) != 127u && scale != 255u) {
                float value = ldexpf(e4m3((uint8_t)code), (int)scale - 127);
                uint32_t bits;
                memcpy(&bits, &value, sizeof(bits));
                bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
                if ((bits & 0x7f800000u) != 0x7f800000u) result = (uint16_t)(bits >> 16);
            }
            engram_bf16[scale][code] = result;
        }
    }
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
 * Engram rows are 264 B and the table fd skips read-ahead, so a miss is a real
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
     * Decode asks for COLS=24 rows per table per token and those reads sit
     * on the critical path between one token's logits and the next token's
     * first layer. Serial, they cost 2.2 ms/token on an M5 Max; through the
     * pool 0.4-0.7 ms. That was a 15% loss (13.8 -> 11.7 t/s) when decode
     * still drained the GPU at every routed layer and the reads hid behind
     * those stalls; with the deferred expert path the token boundary is the
     * stall, and the pool is worth +12% (42.8 -> 47.9 t/s). Decode-sized
     * batches cap the width at DS4_ENGRAM_DECODE_READERS: the calibrated
     * width targets prefill throughput, and waking more readers than rows
     * pay for costs latency. Override with DS4_ENGRAM_MIN_PARALLEL.
     */
    DS4_ENGRAM_MIN_PARALLEL_DEFAULT = DS4_ENGRAM_COLS,
    DS4_ENGRAM_DECODE_READERS = 16,
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
    pthread_once(&engram_bf16_once, init_engram_bf16);
    for (int j = 0; j < DS4_ENGRAM_DIM; j++) {
        const uint16_t value = engram_bf16[raw[DS4_ENGRAM_DIM + j / 32]][raw[j]];
        if (value == UINT16_MAX) { *err = EDOM; return; }
        const uint32_t bits = (uint32_t)value << 16;
        memcpy(dst + j, &bits, sizeof(bits));
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

/* The pool runs one job at a time: callers on different threads (the V4.1
 * reader threads, a batch's background table read) queue here rather than
 * overwrite each other's job, generation and completion count, which left a
 * caller waiting for a completion that had already been consumed. */
static pthread_mutex_t g_pool_run_mu = PTHREAD_MUTEX_INITIALIZER;

static void engram_pool_run(engram_batch *b) {
    pthread_mutex_lock(&g_pool_run_mu);
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
    pthread_mutex_unlock(&g_pool_run_mu);
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
        if (count < 256 && width > DS4_ENGRAM_DECODE_READERS &&
            !getenv("DS4_ENGRAM_READERS")) width = DS4_ENGRAM_DECODE_READERS;
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

enum { TOKEN_READERS = 4 };

typedef struct {
    pthread_t thread;
    ds4_engram_reader *reader;
    unsigned index;
} engram_read_worker;

struct ds4_engram_reader {
    engram_read_worker workers[TOKEN_READERS];
    unsigned threads;
    pthread_mutex_t mutex;
    pthread_cond_t work, done;
    bool active, shutdown;
    uint64_t generation;
    unsigned completed[2];
    int errors[2];
    ds4_engram_table tables[2];
    uint32_t ids[2][DS4_ENGRAM_COLS];
    float *out;
};

static void *read_token_thread(void *context) {
    const engram_read_worker *worker = context;
    ds4_engram_reader *r = worker->reader;
    const unsigned index = worker->index;
    uint64_t seen = 0;
    pthread_mutex_lock(&r->mutex);
    for (;;) {
        while (r->generation == seen && !r->shutdown) pthread_cond_wait(&r->work, &r->mutex);
        if (r->shutdown) break;
        seen = r->generation;
        pthread_mutex_unlock(&r->mutex);
        for (unsigned table = 0; table < 2; table++) {
            const unsigned first = index * DS4_ENGRAM_COLS / TOKEN_READERS;
            const unsigned end = (index + 1u) * DS4_ENGRAM_COLS / TOKEN_READERS;
            const bool ok = ds4_engram_read(&r->tables[table], r->ids[table] + first,
                end - first, r->out + (table * DS4_ENGRAM_COLS + first) * DS4_ENGRAM_DIM);
            const int error = ok ? 0 : errno ? errno : EIO;
            pthread_mutex_lock(&r->mutex);
            if (!r->errors[table]) r->errors[table] = error;
            if (++r->completed[table] == TOKEN_READERS) {
                if (table == 1) r->active = false;
                pthread_cond_broadcast(&r->done);
            }
            pthread_mutex_unlock(&r->mutex);
        }
        pthread_mutex_lock(&r->mutex);
    }
    pthread_mutex_unlock(&r->mutex);
    return NULL;
}

ds4_engram_reader *ds4_engram_reader_create(void) {
    ds4_engram_reader *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    int error = pthread_mutex_init(&r->mutex, NULL);
    if (error) goto fail;
    error = pthread_cond_init(&r->work, NULL);
    if (error) goto mutex;
    error = pthread_cond_init(&r->done, NULL);
    if (error) goto work;
    for (unsigned i = 0; i < TOKEN_READERS; i++) {
        r->workers[i].reader = r;
        r->workers[i].index = i;
        error = pthread_create(&r->workers[i].thread, NULL, read_token_thread, &r->workers[i]);
        if (error) break;
        r->threads++;
    }
    if (!error) return r;
    pthread_mutex_lock(&r->mutex);
    r->shutdown = true;
    pthread_cond_broadcast(&r->work);
    pthread_mutex_unlock(&r->mutex);
    for (unsigned i = 0; i < r->threads; i++)
        if (pthread_join(r->workers[i].thread, NULL)) abort();
    pthread_cond_destroy(&r->done);
work:
    pthread_cond_destroy(&r->work);
mutex:
    pthread_mutex_destroy(&r->mutex);
fail:
    free(r);
    errno = error;
    return NULL;
}

bool ds4_engram_reader_start(ds4_engram_reader *r,
    const ds4_engram_table tables[2], const uint32_t ids[2][DS4_ENGRAM_COLS], float *out) {
    if (!r || !tables || !ids || !out) { errno = EINVAL; return false; }
    for (unsigned t = 0; t < 2; t++) {
        if (tables[t].fd < 0) { errno = EINVAL; return false; }
        for (unsigned col = 0; col < DS4_ENGRAM_COLS; col++)
            if (ids[t][col] >= tables[t].rows) { errno = EINVAL; return false; }
    }
    pthread_mutex_lock(&r->mutex);
    if (r->active || r->shutdown) {
        pthread_mutex_unlock(&r->mutex);
        errno = EBUSY;
        return false;
    }
    memcpy(r->tables, tables, sizeof(r->tables));
    memcpy(r->ids, ids, sizeof(r->ids));
    memset(r->errors, 0, sizeof(r->errors));
    r->out = out;
    memset(r->completed, 0, sizeof(r->completed));
    r->generation++;
    r->active = true;
    pthread_cond_broadcast(&r->work);
    pthread_mutex_unlock(&r->mutex);
    return true;
}

bool ds4_engram_reader_wait(ds4_engram_reader *r, unsigned table) {
    if (!r || table >= 2) { errno = EINVAL; return false; }
    pthread_mutex_lock(&r->mutex);
    if (!r->active && !r->completed[1]) {
        pthread_mutex_unlock(&r->mutex);
        errno = EINVAL;
        return false;
    }
    while (r->completed[table] < TOKEN_READERS) pthread_cond_wait(&r->done, &r->mutex);
    const int error = r->errors[table];
    pthread_mutex_unlock(&r->mutex);
    if (error) errno = error;
    return !error;
}

void ds4_engram_reader_free(ds4_engram_reader *r) {
    if (!r) return;
    pthread_mutex_lock(&r->mutex);
    r->shutdown = true;
    pthread_cond_broadcast(&r->work);
    pthread_mutex_unlock(&r->mutex);
    for (unsigned i = 0; i < r->threads; i++)
        if (pthread_join(r->workers[i].thread, NULL)) abort();
    pthread_cond_destroy(&r->done);
    pthread_cond_destroy(&r->work);
    pthread_mutex_destroy(&r->mutex);
    free(r);
}

/* ------------------------------------------------------------------------
 * Zero-copy Engram offload (spec 3.A / 3.B).
 *
 * This is additive to the pread path above, not a replacement: the same table
 * keeps its F_NOCACHE fd, and every existing reader stays valid while mapped.
 * F_NOCACHE governs read()/pread() on the descriptor only; pages faulted
 * through the mapping live in the unified buffer cache, which is what makes
 * them shareable between sessions and visible to a no-copy Metal buffer.
 *
 * The mapping is page-aligned below the table's GGUF offset because the table
 * does not begin on a page boundary; map_row0 carries that leading slack so
 * callers can address a row without repeating the alignment arithmetic.
 * ------------------------------------------------------------------------ */

bool ds4_engram_table_map(ds4_engram_table *t) {
    if (!t || t->fd < 0 || !t->rows) { errno = EINVAL; return false; }
    if (t->map_base) return true;

    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint64_t base = t->offset & ~(page - 1);
    const uint64_t slack = t->offset - base;
    const uint64_t bytes = (uint64_t)t->rows * DS4_ENGRAM_ROW_BYTES;
    if (slack > UINT64_MAX - bytes) { errno = EINVAL; return false; }
    uint64_t len = slack + bytes;
    if (len > UINT64_MAX - (page - 1)) { errno = EINVAL; return false; }
    len = (len + page - 1) & ~(page - 1);
    if (len > (uint64_t)SIZE_MAX) { errno = ENOMEM; return false; }

    /* MAP_SHARED, not MAP_PRIVATE: private pages are copy-on-write and would
     * stop the kernel sharing one physical copy across sessions. */
    void *p = mmap(NULL, (size_t)len, PROT_READ, MAP_SHARED, t->fd, (off_t)base);
    if (p == MAP_FAILED) return false;

    /* Without this the kernel runs its sequential read-ahead heuristic over a
     * 94 GiB table on every fault. Engram access is a hash scatter, so that
     * read-ahead is all waste and it competes with the expert streamer for the
     * same NVMe queue. Advisory: failure costs throughput, not correctness. */
    (void)madvise(p, (size_t)len, MADV_RANDOM);

    t->map_base = p;
    t->map_bytes = len;
    t->map_row0 = slack;
    return true;
}

void ds4_engram_table_unmap(ds4_engram_table *t) {
    if (!t || !t->map_base) return;
    munmap(t->map_base, (size_t)t->map_bytes);
    t->map_base = NULL;
    t->map_bytes = 0;
    t->map_row0 = 0;
}

bool ds4_engram_row_span(const ds4_engram_table *t, uint32_t row,
                         uint64_t *inner_offset, uint64_t *bytes) {
    if (!t || !t->map_base || row >= t->rows) { errno = EINVAL; return false; }
    if (inner_offset)
        *inner_offset = t->map_row0 + (uint64_t)row * DS4_ENGRAM_ROW_BYTES;
    if (bytes) *bytes = DS4_ENGRAM_ROW_BYTES;
    return true;
}

/* One advise per contiguous page run. A token's 24 rows are scattered by
 * construction, but a page holds 15.5 rows, so sorted rows still collapse into
 * far fewer runs than rows -- and one advise over a run is one queue entry
 * instead of 24. */
size_t ds4_engram_prefetch_rows(const ds4_engram_table *t,
                                const uint32_t *rows, size_t count) {
    if (!t || !t->map_base || !rows || !count) return 0;

    enum { PREFETCH_CHUNK = DS4_ENGRAM_COLS * 8 };
    uint32_t sorted[PREFETCH_CHUNK];
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    size_t advises = 0;

    for (size_t start = 0; start < count; start += PREFETCH_CHUNK) {
        size_t n = count - start;
        if (n > PREFETCH_CHUNK) n = PREFETCH_CHUNK;

        size_t kept = 0;
        for (size_t i = 0; i < n; i++) {
            const uint32_t row = rows[start + i];
            if (row < t->rows) sorted[kept++] = row;
        }
        if (!kept) continue;

        /* Insertion sort: kept is at most 192 and already near-sorted in the
         * common case, and this runs on the token path. */
        for (size_t i = 1; i < kept; i++) {
            const uint32_t v = sorted[i];
            size_t j = i;
            while (j && sorted[j - 1] > v) { sorted[j] = sorted[j - 1]; j--; }
            sorted[j] = v;
        }

        uint64_t run_begin = 0, run_end = 0;
        bool open = false;
        for (size_t i = 0; i < kept; i++) {
            if (i && sorted[i] == sorted[i - 1]) continue;
            const uint64_t off = t->map_row0 +
                                 (uint64_t)sorted[i] * DS4_ENGRAM_ROW_BYTES;
            uint64_t first = off & ~(page - 1);
            uint64_t last = (off + DS4_ENGRAM_ROW_BYTES + page - 1) & ~(page - 1);
            if (last > t->map_bytes) last = t->map_bytes;
            if (first >= last) continue;

            if (open && first <= run_end) {
                if (last > run_end) run_end = last;
                continue;
            }
            if (open) {
                posix_madvise((char *)t->map_base + run_begin,
                              (size_t)(run_end - run_begin), POSIX_MADV_WILLNEED);
                advises++;
            }
            run_begin = first;
            run_end = last;
            open = true;
        }
        if (open) {
            posix_madvise((char *)t->map_base + run_begin,
                          (size_t)(run_end - run_begin), POSIX_MADV_WILLNEED);
            advises++;
        }
    }
    return advises;
}
