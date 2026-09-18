#ifndef DS4_ENGRAM_H
#define DS4_ENGRAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    DS4_ENGRAM_LAYERS = 2,
    DS4_ENGRAM_NGRAM = 4,
    DS4_ENGRAM_HEADS = 8,
    DS4_ENGRAM_COLS = 24,
    DS4_ENGRAM_DIM = 256,
    DS4_ENGRAM_ROW_BYTES = 264,
    DS4_ENGRAM_DEAD = -1
};

typedef struct {
    const uint32_t *token_map;
    uint32_t vocab_size, compressed_vocab_size, pad_id;
    uint32_t rows[DS4_ENGRAM_LAYERS];
    uint64_t multipliers[DS4_ENGRAM_LAYERS][DS4_ENGRAM_NGRAM];
    uint32_t primes[DS4_ENGRAM_LAYERS][DS4_ENGRAM_COLS];
} ds4_engram_layout;

/* Newest first; copying this value also snapshots the complete hash state. */
typedef struct {
    int32_t tail[DS4_ENGRAM_NGRAM - 1];
} ds4_engram_history;

bool ds4_engram_layout_valid(const ds4_engram_layout *layout);
void ds4_engram_history_reset(ds4_engram_history *history);
/* Validate the layout once at model load. Mask zero breaks all n-grams that
 * cross that position. Output is [token][layer][column], including masked rows;
 * the graph must suppress Engram at masked positions, as in the reference. */
bool ds4_engram_hash(const ds4_engram_layout *layout,
                     ds4_engram_history *history,
                     const int *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows);

typedef struct {
    int fd;
    uint64_t offset;
    uint32_t rows;
    /* Zero-copy mapping (spec 3.A). Null base means the table is pread-only. */
    void *map_base;          /* page-aligned mmap base, MAP_SHARED|PROT_READ */
    uint64_t map_bytes;      /* mapped length from map_base */
    uint64_t map_row0;       /* byte offset of row 0 within the mapping */
} ds4_engram_table;

/* A separate uncached file descriptor, never an mmap or Metal model view.
 * Each GGUF I8 row is 256 E4M3 bytes followed by 8 original E8M0 scales. */
bool ds4_engram_table_open(ds4_engram_table *table, const char *path,
                           uint64_t offset, uint32_t rows);
void ds4_engram_table_close(ds4_engram_table *table);
/* Output uses F32 storage for the reference's BF16-rounded values. No whole
 * table allocation; caller owns count * DIM floats. Failure invalidates output. */
bool ds4_engram_read(const ds4_engram_table *table, const uint32_t *rows,
                     size_t count, float *out);
/* Read COLS rows per token, restoring token order after deduplicated disk reads.
 * Input stride is in row IDs; output is packed [token][COLS][DIM]. Temporary
 * storage is bounded to 384 KiB, independent of the table and prefix size.
 * On macOS, large batches use bounded concurrent pread readers. */
bool ds4_engram_read_batch(const ds4_engram_table *table, const uint32_t *rows,
                           size_t tokens, size_t stride, float *out);

/* One reusable reader for a token's two tables. Tables/IDs are copied at
 * start; descriptors and output must remain valid until wait(reader, 1) or
 * free, even if wait(reader, 0) reports an error. No table data is cached.
 * One caller owns the reader. Creation failure permits synchronous fallback. */
typedef struct ds4_engram_reader ds4_engram_reader;
ds4_engram_reader *ds4_engram_reader_create(void);
bool ds4_engram_reader_start(ds4_engram_reader *reader,
    const ds4_engram_table tables[2], const uint32_t ids[2][DS4_ENGRAM_COLS], float *out);
bool ds4_engram_reader_wait(ds4_engram_reader *reader, unsigned table);
void ds4_engram_reader_free(ds4_engram_reader *reader);
/* ---- Zero-copy Engram offload (spec 3.A / 3.B) ----
 *
 * Maps the table's row region MAP_SHARED|PROT_READ and applies MADV_RANDOM so
 * the kernel does not run its sequential read-ahead over a 94 GiB table and
 * burn queue depth on rows no token asked for. The mapping is page-aligned
 * below the table's GGUF offset; map_row0 carries the leading slack so row r
 * lives at map_base + map_row0 + r * DS4_ENGRAM_ROW_BYTES.
 *
 * The mapping is additive: the pread path stays valid on the same table, and
 * a failed map leaves the table usable (map_base stays NULL). */
bool ds4_engram_table_map(ds4_engram_table *table);
void ds4_engram_table_unmap(ds4_engram_table *table);

/* Byte range of one row inside the mapping, for the Metal buffer's inner
 * offset. Returns false if the table is not mapped or the row is out of range. */
bool ds4_engram_row_span(const ds4_engram_table *table, uint32_t row,
                         uint64_t *inner_offset, uint64_t *bytes);

/* Stage rows into the unified page cache ahead of the GPU (spec 3.B.2).
 * posix_madvise(WILLNEED) over the pages covering each row; adjacent rows are
 * coalesced into one advise per contiguous page run. Advisory only: it never
 * blocks on I/O and never changes what a later read returns. Returns the
 * number of advise calls issued, or 0 when the table is not mapped. */
size_t ds4_engram_prefetch_rows(const ds4_engram_table *table,
                                const uint32_t *rows, size_t count);

#endif
