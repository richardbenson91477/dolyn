#ifndef DOLEN_Q_COMMON_H
#define DOLEN_Q_COMMON_H

#include "dolen_common.h"

#include "ext/csafetensors.h"
#include "ext/json.h"


typedef struct {
    char *tensor_name;
    char *filename;
} weightmap_entry;

typedef struct {
    weightmap_entry *entries;
    size_t n_entries;
    char **unique_filenames;
    int n_unique_files;
    char *model_dir;
} safetensors_idx;


void quantize_group(qtensor *qt, const float *weights, int rows, int cols);
int load_safetensors_index(safetensors_idx *idx, const char *model_dir);
void free_safetensors_index(safetensors_idx *idx);
float *extract_tensor_from_handle(void *st, const char *name, float *dest, size_t expected_size);
float *load_tensor_from_handle(void *st, const char *name, float *dest, size_t expected_size);
void load_and_quantize_from_handle(void *st, const char *name, qtensor *qt, int rows, int cols);


#endif // DOLEN_Q_COMMON_H

