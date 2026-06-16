#ifndef DOLEN_Q_COMMON_H
#define DOLEN_Q_COMMON_H

#include <errno.h>
#include <limits.h>
#include <sys/types.h>

#include "ext/csafetensors.h"
#include "ext/json.h"

#include "dolen_common.h"

#ifndef DOLEN_QUANTIZE_CHUNK_BYTES
#define DOLEN_QUANTIZE_CHUNK_BYTES (16u * 1024u * 1024u)
#endif

typedef enum {
    ST_DTYPE_UNSUPPORTED = 0,
    ST_DTYPE_F16,
    ST_DTYPE_BF16,
    ST_DTYPE_F32
} st_dtype;

typedef struct {
    char *tensor_name;
    char *filename;
    st_dtype dtype;
    uint64_t data_offset;
    uint64_t data_nbytes;
    uint64_t num_elements;
    size_t processing_rank;
    int metadata_ready;
} weightmap_entry;

typedef struct {
    weightmap_entry *entries;
    size_t n_entries;
    char **unique_filenames;
    int n_unique_files;
    char *model_dir;
} safetensors_idx;

typedef struct {
    safetensors_idx index;
    FILE *source;
    const char *source_filename;
    size_t chunk_bytes;
} quantize_ctx;

void quantize_group(qtensor *qt, const float *weights, int rows, int cols);
int quantize_write_bytes(FILE *out, const void *data, size_t size, size_t count);

int load_safetensors_index(safetensors_idx *idx, const char *model_dir);
void free_safetensors_index(safetensors_idx *idx);

int quantize_ctx_open(quantize_ctx *ctx, const char *model_dir);
void quantize_ctx_close(quantize_ctx *ctx);

const weightmap_entry *quantize_find_tensor(const quantize_ctx *ctx, const char *name);
const weightmap_entry *quantize_find_last_tensor(
        const quantize_ctx *ctx, const char *const *names, size_t n_names);

int quantize_write_f32_tensor(quantize_ctx *ctx, FILE *out,
        const char *name, size_t expected_elements);
int quantize_write_f32_entry(quantize_ctx *ctx, FILE *out,
        const weightmap_entry *entry, size_t expected_elements);
int quantize_write_f32_or_zeros(quantize_ctx *ctx, FILE *out,
        const char *name, size_t expected_elements);
int quantize_write_f32_zeros(FILE *out, size_t elements);

int quantize_write_qtensor(quantize_ctx *ctx, FILE *out,
        const char *name, int rows, int cols);
int quantize_write_qtensor_or_empty(quantize_ctx *ctx, FILE *out,
        const char *name, int rows, int cols);
int quantize_write_qtensor_entry(quantize_ctx *ctx, FILE *out,
        const weightmap_entry *entry, int rows, int cols);
int quantize_write_empty_qtensor(FILE *out);

int quantize_write_scalar_or_default(quantize_ctx *ctx, FILE *out,
        const char *const *names, size_t n_names, float default_value);

#endif // DOLEN_Q_COMMON_H

