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
    char *_tensor_name_s;
    char *_file_name_s;
    st_dtype dtype;
    uint64_t data_offset;
    uint64_t data_nbytes;
    uint64_t num_elements;
    size_t processing_rank;
    int metadata_ready;
} weightmap_entry;

typedef struct {
    weightmap_entry *_wm_entries;
    size_t n_entries;
    char **__unique_filenames;
    int unique_files_n;
    char *_model_dir_s;
} safetensors_idx;

typedef struct {
    safetensors_idx index;
    FILE *_file;
    const char *_file_name_s;
    size_t chunk_bytes;
} quantize_ctx;


int quantize_write_bytes(FILE *_file, const void *_data, size_t size, size_t count);

int load_safetensors_index(safetensors_idx *_idx, const char *_model_dir_s);

void free_safetensors_index(safetensors_idx *idx);

int quantize_ctx_open(quantize_ctx *_q_ctx, const char *_model_dir);

void quantize_ctx_close(quantize_ctx *_q_ctx);

const weightmap_entry *quantize_find_tensor(const quantize_ctx *_q_ctx, const char *_name_s);

const weightmap_entry *quantize_find_last_tensor(const quantize_ctx *_q_ctx, const char *const *__names, size_t names_n);

int quantize_write_tensor(quantize_ctx *_qt_ctx, FILE *_file, const char *_name_s,
        int rows, int cols, q_type_t type);

int quantize_write_tensor_or_empty(quantize_ctx *_qt_ctx, FILE *_file, const char *_name_s,
        int rows, int cols, q_type_t type);

int quantize_write_tensor_entry(quantize_ctx *_qt_ctx, FILE *_file, const char *_name_s, const weightmap_entry *_wm_entry,
        int rows, int cols, q_type_t type);

int quantize_write_empty_tensor(FILE *_file);

int quantize_write_scalar_or_default(quantize_ctx *_qt_ctx, FILE *_file, const char *const *__names_s, size_t names_n,
        float default_value);

q_type_t parse_q_type(const char *_str_s);


#endif // DOLEN_Q_COMMON_H

