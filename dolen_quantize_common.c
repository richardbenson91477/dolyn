#include "dolen_quantize_common.h"


void quantize_group(qtensor *qt, const float *weights, int rows, int cols) {
    qt->rows = rows;
    qt->cols = cols;
    int num_groups = (cols + GS - 1) / GS;

    qt->q = (int8_t *)a_calloc((size_t)rows * cols * sizeof(int8_t));
    qt->s = (float *)a_calloc((size_t)rows * num_groups * sizeof(float));

    for (int i = 0; i < rows; i++) {
        const float *row = weights + i * cols;
        float *row_s = qt->s + i * num_groups;
        int8_t *row_q = qt->q + i * cols;

        for (int g = 0; g < num_groups; g++) {
            int start = g * GS;
            int end = start + GS;
            if (end > cols) {
                end = cols;
            }

            float wmax = 0.0f;
            for (int j = start; j < end; j++) {
                float val = fabsf(row[j]);
                if (val > wmax) {
                    wmax = val;
                }
            }
            if (wmax < 1e-9f) {
                wmax = 1e-9f;
            }

            float scale = wmax / 127.0f;
            row_s[g] = scale;
            for (int j = start; j < end; j++) {
                float quant_value = row[j] / scale;
                row_q[j] = (int8_t)roundf(quant_value);
            }
        }
    }
}

int load_safetensors_index(safetensors_idx *idx, const char *model_dir) {
    char index_path[4096];
    snprintf(index_path, sizeof(index_path), "%s/model.safetensors.index.json", model_dir);

    FILE *f = fopen(index_path, "rb");
    if (! f) {
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json_str = (char *)a_calloc(size + 1);
    if (! json_str) {
        fclose(f);
        return -1;
    }
    if (fread(json_str, 1, size, f) != (size_t)size) {
        free(json_str);
        fclose(f);
        return -1;
    }
    json_str[size] = '\0';
    fclose(f);

    char error[256] = {0};
    JsonValue *root = json_parse(json_str, size, error, sizeof(error));
    free(json_str);
    if (! root) {
        return -1;
    }

    JsonValue *weight_map = json_object_get(root, "weight_map");
    if (! weight_map || weight_map->type != JSON_OBJECT) {
        json_free(root);
        return -1;
    }

    idx->n_entries = weight_map->data.object.count;
    idx->entries = (weightmap_entry *)a_calloc(idx->n_entries * sizeof(weightmap_entry));
    if (! idx->entries) {
        json_free(root);
        return -1;
    }

    char **temp_files = NULL;
    int n_temp_files = 0;

    for (size_t i = 0; i < idx->n_entries; i++) {
        JsonPair *pair = &weight_map->data.object.pairs[i];
        idx->entries[i].tensor_name = strdup(pair->key);
        idx->entries[i].filename = strdup(pair->value->data.string);

        int found = 0;
        for (int j = 0; j < n_temp_files; j++) {
            if (strcmp(temp_files[j], pair->value->data.string) == 0) {
                found = 1;
                break;
            }
        }
        if (! found) {
            n_temp_files++;
            temp_files = (char **)realloc(temp_files, n_temp_files * sizeof(char *));
            temp_files[n_temp_files - 1] = strdup(pair->value->data.string);
        }
    }

    idx->unique_filenames = temp_files;
    idx->n_unique_files = n_temp_files;
    idx->model_dir = strdup(model_dir);
    json_free(root);
    return 0;
}

void free_safetensors_index(safetensors_idx *idx) {
    for (size_t i = 0; i < idx->n_entries; i++) {
        free(idx->entries[i].tensor_name);
        free(idx->entries[i].filename);
    }
    free(idx->entries);
    for (int i = 0; i < idx->n_unique_files; i++) {
        free(idx->unique_filenames[i]);
    }
    free(idx->unique_filenames);
    free(idx->model_dir);
    memset(idx, 0, sizeof(safetensors_idx));
}

float *extract_tensor_from_handle(void *st_ptr, const char *name, float *dest, size_t expected_size) {
    csafetensors_t *st = (csafetensors_t *)st_ptr;
    const csafetensors_tensor_t *tensor = csafetensors_get_tensor(st, name);
    if (! tensor) {
        return NULL;
    }

    const uint8_t *data = csafetensors_get_tensor_data(st, tensor);
    if (! data) {
        return NULL;
    }

    size_t num_elements = csafetensors_shape_size(tensor);
    if (expected_size > 0 && (num_elements != expected_size)) {
        log_msg(stderr, "ERROR: Tensor %s size mismatch: got %zu, expected %zu\n", name, num_elements, expected_size);
    }

    float *output = dest;
    if (! dest) {
        output = (float *)a_calloc(num_elements * sizeof(float));
        if (! output) {
            return NULL;
        }
    }

    if (tensor->dtype == CSAFETENSORS_DTYPE_BFLOAT16) {
        const uint16_t *bf16_data = (const uint16_t *)data;
        for (size_t i = 0; i < num_elements; i++)
            output[i] = csafetensors_bf16_to_f32(bf16_data[i]);
    } else if (tensor->dtype == CSAFETENSORS_DTYPE_FLOAT16) {
        const uint16_t *f16_data = (const uint16_t *)data;
        for (size_t i = 0; i < num_elements; i++)
            output[i] = csafetensors_f16_to_f32(f16_data[i]);
    } else if (tensor->dtype == CSAFETENSORS_DTYPE_FLOAT32) {
        memcpy(output, data, num_elements * sizeof(float));
    } else {
        log_msg(stderr, "ERROR: Unsupported dtype for tensor %s\n", name);
        if (! dest) {
            free(output);
        }
        return NULL;
    }
    return output;
}

float *load_tensor_from_handle(void *st, const char *name, float *dest, size_t expected_size) {
    float *res = extract_tensor_from_handle(st, name, dest, expected_size);
    if ((! res) && dest) {
        log_msg(stderr, "ERROR: Failed to load tensor %s into provided buffer\n", name);
    }
    return res;
}

void load_and_quantize_from_handle(void *st, const char *name, qtensor *qt, int rows, int cols) {
    float *f = extract_tensor_from_handle(st, name, NULL, 0);
    if (! f) {
        log_msg(stderr, "ERROR: missing tensor %s in current shard\n", name);
        exit(EXIT_FAILURE);
    }
    quantize_group(qt, f, rows, cols);
    free(f);
}

