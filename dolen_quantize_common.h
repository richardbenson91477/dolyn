#ifndef DOLEN_Q_COMMON_H
#define DOLEN_Q_COMMON_H

#include "ext/csafetensors.h"
#include "ext/json.h"

#include "dolen_common_sampler.h"
#include "dolen_common_cmi.h"
#include "dolen_common_io.h"
#include "dolen_common_math.h"
#include "dolen_common_mem.h"
#include "dolen_common_qtensor.h"
#include "dolen_common_tokenizer.h"


#define DOLEN_QUANTIZE_CHUNK_BYTES (16u * 1024u * 1024u)


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


int quantize_write_bytes(FILE *out, const void *data, size_t size, size_t count);

int load_safetensors_index(safetensors_idx *idx, const char *model_dir);

void free_safetensors_index(safetensors_idx *idx);

int quantize_ctx_open(quantize_ctx *ctx, const char *model_dir);

void quantize_ctx_close(quantize_ctx *ctx);

const weightmap_entry *quantize_find_tensor(const quantize_ctx *ctx, const char *name);

const weightmap_entry *quantize_find_last_tensor(const quantize_ctx *ctx, const char *const *names, size_t n_names);

int quantize_write_tensor(quantize_ctx *ctx, FILE *out, const char *name,
        int rows, int cols, q_type_t type);

int quantize_write_tensor_or_empty(quantize_ctx *ctx, FILE *out, const char *name,
        int rows, int cols, q_type_t type);

int quantize_write_tensor_entry(quantize_ctx *ctx, FILE *out, const weightmap_entry *entry,
        int rows, int cols, q_type_t type);

int quantize_write_empty_tensor(FILE *out);

int quantize_write_scalar_or_default(quantize_ctx *ctx, FILE *out, const char *const *names, size_t n_names,
        float default_value);

q_type_t parse_q_type(const char *str);

#endif // DOLEN_Q_COMMON_H

