/*
 * Can the expert read path go faster than 2 GB/s?
 *
 * Decode reads ~26 GB of routed experts at ~2.0 GB/s while prefill, on the same
 * device in the same run, sustains ~5.0 GB/s. Either 2 GB/s is what this access
 * pattern costs, or it is an artifact of how the reads are issued (one layer's
 * misses, advised and then immediately waited on -- queue depth ~1).
 *
 * This asks the device directly, outside ds4: random 3.16 MiB ranges (one
 * expert matrix) out of the Q2 GGUF, read five ways. Each mode uses its own
 * disjoint slice of the file so an earlier mode cannot warm a later one.
 *
 *   cc -O2 -o ioprobe ioprobe.c -pthread
 *   ./ioprobe <path> [n_ranges]
 */
#define _DARWIN_C_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>

#define RANGE_BYTES (3316 * 1024ULL)   /* ~ one expert matrix at Q2 */

static double now_s(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static int    g_fd;
static off_t *g_off;
static int    g_n;
static volatile uint64_t g_sink;

typedef struct { int lo, hi; } span;

static void *pread_worker(void *ud) {
    span *s = (span *)ud;
    void *buf = malloc(RANGE_BYTES);
    for (int i = s->lo; i < s->hi; i++) {
        size_t done = 0;
        while (done < RANGE_BYTES) {
            ssize_t n = pread(g_fd, (char *)buf + done, RANGE_BYTES - done,
                              g_off[i] + (off_t)done);
            if (n <= 0) break;
            done += (size_t)n;
        }
        g_sink += ((unsigned char *)buf)[0];
    }
    free(buf);
    return NULL;
}

/* One range split across `threads` workers -- the pattern a single-expert
 * install would use if each matrix were chunked, as opposed to the
 * one-range-per-thread pattern above. Different ask of the device: same total
 * bytes, but the concurrent requests are adjacent rather than scattered. */
typedef struct { int idx; int part; int nparts; } chunkarg;
static void *chunk_worker(void *ud) {
    chunkarg *c = (chunkarg *)ud;
    const uint64_t per = RANGE_BYTES / (uint64_t)c->nparts;
    const uint64_t off = per * (uint64_t)c->part;
    const uint64_t len = (c->part == c->nparts - 1)
                       ? RANGE_BYTES - off : per;
    void *buf = malloc(len);
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(g_fd, (char *)buf + done, len - done,
                          g_off[c->idx] + (off_t)(off + done));
        if (n <= 0) break;
        done += (size_t)n;
    }
    g_sink += ((unsigned char *)buf)[0];
    free(buf);
    return NULL;
}
static double run_pread_chunked(int lo, int hi, int nparts) {
    const double t0 = now_s();
    for (int i = lo; i < hi; i++) {
        pthread_t th[32]; chunkarg ca[32];
        for (int p = 0; p < nparts; p++) {
            ca[p].idx = i; ca[p].part = p; ca[p].nparts = nparts;
            pthread_create(&th[p], NULL, chunk_worker, &ca[p]);
        }
        for (int p = 0; p < nparts; p++) pthread_join(th[p], NULL);
    }
    return now_s() - t0;
}

static double run_pread(int lo, int hi, int threads) {
    const double t0 = now_s();
    pthread_t th[64];
    span sp[64];
    const int total = hi - lo, per = (total + threads - 1) / threads;
    for (int t = 0; t < threads; t++) {
        sp[t].lo = lo + t * per;
        sp[t].hi = sp[t].lo + per > hi ? hi : sp[t].lo + per;
        if (sp[t].lo > sp[t].hi) sp[t].lo = sp[t].hi;
        pthread_create(&th[t], NULL, pread_worker, &sp[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
    return now_s() - t0;
}

/* Advise a whole window of ranges before faulting any of them, so the device
 * sees `depth` requests in flight instead of one. This is what the no-copy
 * bank could do for a layer's misses without copying anything. */
static double run_fault_batched(const char *map, int lo, int hi, int depth) {
    const double t0 = now_s();
    const long pg = sysconf(_SC_PAGESIZE);
    for (int i = lo; i < hi; i += depth) {
        const int end = i + depth > hi ? hi : i + depth;
        for (int j = i; j < end; j++) {
            struct radvisory ra;
            ra.ra_offset = g_off[j];
            ra.ra_count  = (int)RANGE_BYTES;
            (void)fcntl(g_fd, F_RDADVISE, &ra);
        }
        for (int j = i; j < end; j++) {
            const char *p = map + g_off[j];
            for (uint64_t b = 0; b < RANGE_BYTES; b += (uint64_t)pg) g_sink += p[b];
        }
    }
    return now_s() - t0;
}

/* Fault the pages in, the way a no-copy MTLBuffer over the model map does. */
static double run_fault(const char *map, int lo, int hi, int advise) {
    const double t0 = now_s();
    const long pg = sysconf(_SC_PAGESIZE);
    for (int i = lo; i < hi; i++) {
        if (advise) {
            struct radvisory ra;
            ra.ra_offset = g_off[i];
            ra.ra_count  = (int)RANGE_BYTES;
            (void)fcntl(g_fd, F_RDADVISE, &ra);
        }
        const char *p = map + g_off[i];
        for (uint64_t b = 0; b < RANGE_BYTES; b += (uint64_t)pg) g_sink += p[b];
    }
    return now_s() - t0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: ioprobe <file> [n]\n"); return 1; }
    const int n_per_mode = argc > 2 ? atoi(argv[2]) : 150;

    g_fd = open(argv[1], O_RDONLY);
    if (g_fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(g_fd, &st)) { perror("fstat"); return 1; }

    const int modes = 11;
    g_n = n_per_mode * modes;
    g_off = malloc(sizeof(off_t) * (size_t)g_n);

    /* Disjoint slice per mode, random within it. */
    const uint64_t usable = (uint64_t)st.st_size - RANGE_BYTES;
    const uint64_t slice  = usable / (uint64_t)modes;
    srandom(argc > 3 ? (unsigned)atoi(argv[3]) : 12345u);
    for (int m = 0; m < modes; m++)
        for (int i = 0; i < n_per_mode; i++) {
            uint64_t r = ((uint64_t)random() << 31) ^ (uint64_t)random();
            g_off[m * n_per_mode + i] =
                (off_t)(m * slice + (r % slice)) & ~(off_t)0x3fff;
        }

    char *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, g_fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(map, (size_t)st.st_size, MADV_RANDOM);

    const double mb = (double)n_per_mode * RANGE_BYTES / (1024.0 * 1024.0);
    printf("%s\n%d ranges x %.2f MiB per mode\n\n",
           argv[1], n_per_mode, RANGE_BYTES / 1048576.0);
    printf("%-28s %8s %10s\n", "mode", "sec", "MB/s");

    struct { const char *name; int kind; int threads; } M[] = {
        { "mmap fault (no advise)",   0, 0  },
        { "F_RDADVISE + fault",       1, 0  },
        { "pread 1 thread",           2, 1  },
        { "pread 4 threads",          2, 4  },
        { "pread 16 threads",         2, 16 },
        { "advise x3 batched + fault", 3, 3  },
        { "advise x12 batched + fault",3, 12 },
        { "advise x48 batched + fault",3, 48 },
        { "1 range / 3 chunks",        4, 3  },
        { "1 range / 9 chunks",        4, 9  },
        { "3 ranges as 9 tasks",       2, 9  },
    };
    for (int m = 0; m < modes; m++) {
        const int lo = m * n_per_mode, hi = lo + n_per_mode;
        double d;
        if (M[m].kind == 2)      d = run_pread(lo, hi, M[m].threads);
        else if (M[m].kind == 3) d = run_fault_batched(map, lo, hi, M[m].threads);
        else if (M[m].kind == 4) d = run_pread_chunked(lo, hi, M[m].threads);
        else                     d = run_fault(map, lo, hi, M[m].kind);
        printf("%-28s %8.3f %10.1f\n", M[m].name, d, mb / d);
        fflush(stdout);
    }
    printf("\n(sink %llu)\n", (unsigned long long)g_sink);
    return 0;
}
