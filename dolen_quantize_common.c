#include "dolen_quantize_common.h"
#include <string.h>


static const quant_preset_t QUANT_PRESETS[] = {
        // Name     Embed       LM_Head     Attn        MLP
        {"Q4_K_M", Q_TYPE_Q8,  Q_TYPE_Q6,  Q_TYPE_Q4,  Q_TYPE_Q4},
        {"Q6_K",   Q_TYPE_Q8,  Q_TYPE_Q8,  Q_TYPE_Q6,  Q_TYPE_Q6},
        {"Q8_0",   Q_TYPE_Q8,  Q_TYPE_Q8,  Q_TYPE_Q8,  Q_TYPE_Q8},
        {"F16",    Q_TYPE_F16, Q_TYPE_F16, Q_TYPE_F16, Q_TYPE_F16},
        {"F32",    Q_TYPE_F32, Q_TYPE_F32, Q_TYPE_F32, Q_TYPE_F32}
};


const quant_preset_t *quantize_find_preset(const char *_name_s) {
    if (!_name_s) {
        return NULL;
    }
    for (size_t i = 0; i < sizeof(QUANT_PRESETS) / sizeof(QUANT_PRESETS[0]); i++) {
        if (strcmp(QUANT_PRESETS[i].name, _name_s) == 0) {
            return &QUANT_PRESETS[i];
        }
    }
    return NULL;
}

void quantize_print_presets(void) {
    log_msg(stdout, "Available presets:\n");
    for (size_t i = 0; i < sizeof(QUANT_PRESETS) / sizeof(QUANT_PRESETS[0]); i++) {
        log_msg(stdout, "  %s\n", QUANT_PRESETS[i].name);
    }
}

static inline void autofree_cleanup(void *p) {
    void **ptr = (void **)p;
    if (*ptr) {
        free(*ptr);
    }
}
#define autofree __attribute__((cleanup(autofree_cleanup)))

static inline void autojson_cleanup(void *p) {
    void **ptr = (void **)p;
    if (*ptr) {
        json_free(*ptr);
    }
}
#define autojson __attribute__((cleanup(autojson_cleanup)))

static int32_t checked_mul_size(size_t a, size_t b, size_t *_res) {
    if (a &&
        (b > (SIZE_MAX / a))) {
        return -1;
    }
    *_res = a * b;
    return 0;
}

static int32_t checked_add_u64(uint64_t a, uint64_t b, uint64_t *_res) {
    if (b > (UINT64_MAX - a)) {
        return -1;
    }
    *_res = a + b;
    return 0;
}

int32_t quantize_write_bytes(FILE *_file, const void *_data, size_t size, size_t count) {
    if (! count) {
        return 0;
    }
    if (fwrite(_data, size, count, _file) != count) {
        log_msg(stderr, "ERROR: Write failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static size_t dtype_size(st_dtype dtype) {
    switch (dtype) {
        case ST_DTYPE_F16:
        case ST_DTYPE_BF16:
            return sizeof(uint16_t);
        case ST_DTYPE_F32:
            return sizeof(float);
        default:
            return 0;
    }
}

static st_dtype parse_dtype(const char *_dtype_s) {
    if (! _dtype_s) {
        return ST_DTYPE_UNSUPPORTED;
    }
    else if (! strcmp(_dtype_s, "F16")) {
        return ST_DTYPE_F16;
    }
    else if (! strcmp(_dtype_s, "BF16")) {
        return ST_DTYPE_BF16;
    }
    else if (! strcmp(_dtype_s, "F32")) {
        return ST_DTYPE_F32;
    }
    else {
        return ST_DTYPE_UNSUPPORTED;
    }
}

static int32_t json_u64(JsonValue *_js_val, uint64_t *_res) {
    if ((! _js_val) ||
            (_js_val->type != JSON_NUMBER) ||
        _js_val->data.number < 0.0) {
        return -1;
    }
    double d = _js_val->data.number;
    uint64_t v = (uint64_t)d;
    if ((double)v != d) {
        return -1;
    }
    *_res = v;
    return 0;
}

static weightmap_entry *find_index_entry(safetensors_idx *_st_idx, const char *_file_name_s,
        const char *_tensor_name_s) {
    for (size_t i = 0; i < _st_idx->n_entries; i++) {
        weightmap_entry *_wm_entry = &_st_idx->_wm_entries[i];
        if ((! strcmp(_wm_entry->_file_name_s, _file_name_s)) &&
                (! strcmp(_wm_entry->_tensor_name_s, _tensor_name_s))) {
            return _wm_entry;
        }
    }
    return NULL;
}

static int32_t load_shard_metadata(safetensors_idx *_st_idx, const char *_file_name_s) {
    char _file_path_s[PATH_MAX];
    if (snprintf(_file_path_s, sizeof(_file_path_s), "%s/%s", _st_idx->_model_dir_s, _file_name_s) \
            >= (int32_t)sizeof(_file_path_s)) {
        log_msg(stderr, "ERROR: Safetensors path is too long\n");
        return -1;
    }
    FILE *_file = fopen(_file_path_s, "rb");
    if (! _file) {
        log_msg(stderr, "ERROR: Could not open %s\n", _file_path_s);
        return -1;
    }
    uint8_t length_bytes[8];
    if (fread(length_bytes, 1, sizeof(length_bytes), _file) != sizeof(length_bytes)) {
        log_msg(stderr, "ERROR: Could not read safetensors header length from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }
    uint64_t header_len_u64 = read_le64(length_bytes);
    if ((! header_len_u64) ||
            header_len_u64 > SIZE_MAX - 1) {
        log_msg(stderr, "ERROR: Invalid safetensors header length in %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }
    size_t header_len = (size_t)header_len_u64;
    autofree char *_header_s = (char *)a_calloc(header_len + 1);
    if ((! _header_s) ||
            (fread(_header_s, 1, header_len, _file) != header_len)) {
        log_msg(stderr, "ERROR: Could not read safetensors header from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }
    fclose(_file);
    _header_s[header_len] = '\0';
    char _error_s[256] = { 0 };
    autojson JsonValue *_js_root = json_parse(_header_s, header_len, _error_s, sizeof(_error_s));
    if ((! _js_root) ||
            _js_root->type != JSON_OBJECT) {
        log_msg(stderr, "ERROR: Invalid safetensors header in %s: %s\n", _file_path_s, _error_s);
        return -1;
    }
    uint64_t data_base;
    if (checked_add_u64(8, header_len_u64, &data_base)) {
        return -1;
    }
    for (size_t i = 0; i < _js_root->data.object.count; i++) {
        JsonPair *_js_pair = &_js_root->data.object.pairs[i];
        if (! strcmp(_js_pair->key, "__metadata__")) {
            continue;
        }
        weightmap_entry *_wm_entry = find_index_entry(_st_idx, _file_name_s, _js_pair->key);
        if (! _wm_entry) {
            continue;
        }
        JsonValue *_js_tensor = _js_pair->value;
        JsonValue *_js_dtype = json_object_get(_js_tensor, "dtype");
        JsonValue *_js_shape = json_object_get(_js_tensor, "shape");
        JsonValue *_js_offsets = json_object_get(_js_tensor, "data_offsets");
        if ((! _js_dtype) ||
                (_js_dtype->type != JSON_STRING) ||
                (! _js_shape) ||
                (_js_shape->type != JSON_ARRAY) ||
                (! _js_offsets) ||
                (_js_offsets->type != JSON_ARRAY) ||
                (_js_offsets->data.array.count != 2)) {
            log_msg(stderr, "ERROR: Invalid metadata for tensor %s\n", _js_pair->key);
            return -1;
        }
        uint64_t elements = 1;
        for (size_t d = 0; d < _js_shape->data.array.count; d++) {
            uint64_t dim;
            if (json_u64(json_array_get(_js_shape, d), &dim) ||
                    (dim &&
                    (elements > UINT64_MAX / dim))) {
                log_msg(stderr, "ERROR: Invalid shape for tensor %s\n", _js_pair->key);
                return -1;
            }
            elements *= dim;
        }
        uint64_t begin, end;
        if (json_u64(json_array_get(_js_offsets, 0), &begin) ||
                json_u64(json_array_get(_js_offsets, 1), &end) ||
                (end < begin)) {
            log_msg(stderr, "ERROR: Invalid data offsets for tensor %s\n", _js_pair->key);
            return -1;
        }
        _wm_entry->dtype = parse_dtype(_js_dtype->data.string);
        _wm_entry->num_elements = elements;
        _wm_entry->data_nbytes = end - begin;
        if (checked_add_u64(data_base, begin, &_wm_entry->data_offset)) {
            return -1;
        }
        _wm_entry->metadata_ready = 1;
        size_t item_size = dtype_size(_wm_entry->dtype);
        if (item_size &&
            ((elements > UINT64_MAX / item_size) ||
                (elements * item_size != _wm_entry->data_nbytes))) {
            log_msg(stderr, "ERROR: Byte size mismatch for tensor %s\n", _js_pair->key);
            return -1;
        }
    }
    return 0;
}

static void quantize_group_into_q8(int8_t *_q, float *_s, const float *_weights, int32_t rows, int32_t cols) {
    int32_t num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
    for (int32_t i = 0; i < rows; i++) {
        const float *_row = _weights + (size_t)i * cols;
        float *_row_s = _s + (size_t)i * num_groups;
        int8_t *_row_q = _q + (size_t)i * cols;
        for (int32_t g = 0; g < num_groups; g++) {
            int32_t start = g * GROUP_SIZE;
            int32_t end = start + GROUP_SIZE;
            if (end > cols) {
                end = cols;
            }
            float wmax = 0.0f;
            for (int32_t j = start; j < end; j++) {
                float val = fabsf(_row[j]);
                if (val > wmax) {
                    wmax = val;
                }
            }
            if (wmax < 1e-9f) {
                wmax = 1e-9f;
            }
            float scale = wmax / 127.0f;
            _row_s[g] = scale;
            for (int32_t j = start; j < end; j++) {
                float quant_value = _row[j] / scale;
                _row_q[j] = (int8_t)roundf(quant_value);
            }
        }
    }
}

static void quantize_group_into_q6(uint8_t *_q, float *_s, const float *_weights, int32_t rows, int32_t cols) {
    int32_t num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
    for (int32_t i = 0; i < rows; i++) {
        const float *_row = _weights + (size_t)i * cols;
        float *_row_s = _s + (size_t)i * num_groups;
        uint8_t *_row_q = _q + ((size_t)i * cols * 3) / 4;
        for (int32_t g = 0; g < num_groups; g++) {
            int32_t start = g * GROUP_SIZE;
            int32_t end = start + GROUP_SIZE;
            if (end > cols) {
                end = cols;
            }
            float wmax = 0.0f;
            for (int32_t j = start; j < end; j++) {
                float val = fabsf(_row[j]);
                if (val > wmax) {
                    wmax = val;
                }
            }
            if (wmax < 1e-9f) {
                wmax = 1e-9f;
            }
            // 6-bit signed range is [-32, 31]
            float scale = wmax / 31.0f;
            _row_s[g] = scale;
            float inv_scale = 1.0f / scale;
            for (int32_t j = start; j < end; j += 4) {
                float v0 = _row[j] * inv_scale;
                float v1 = (j + 1 < end) ? _row[j + 1] * inv_scale : 0.0f;
                float v2 = (j + 2 < end) ? _row[j + 2] * inv_scale : 0.0f;
                float v3 = (j + 3 < end) ? _row[j + 3] * inv_scale : 0.0f;
                int32_t q0 = (int32_t)roundf(v0);
                int32_t q1 = (int32_t)roundf(v1);
                int32_t q2 = (int32_t)roundf(v2);
                int32_t q3 = (int32_t)roundf(v3);
                q0 = q0 < -32 ? -32 : (q0 > 31 ? 31 : q0);
                q1 = q1 < -32 ? -32 : (q1 > 31 ? 31 : q1);
                q2 = q2 < -32 ? -32 : (q2 > 31 ? 31 : q2);
                q3 = q3 < -32 ? -32 : (q3 > 31 ? 31 : q3);
                uint8_t u0 = (uint8_t)(q0 & 0x3F);
                uint8_t u1 = (uint8_t)(q1 & 0x3F);
                uint8_t u2 = (uint8_t)(q2 & 0x3F);
                uint8_t u3 = (uint8_t)(q3 & 0x3F);
                // FIX: Use absolute index 'j', not relative '(j - start)'
                int32_t idx = j / 4 * 3;
                _row_q[idx] = (u0) | (u1 << 6);
                _row_q[idx + 1] = (u1 >> 2) | (u2 << 4);
                _row_q[idx + 2] = (u2 >> 4) | (u3 << 2);
            }
        }
    }
}

static void quantize_group_into_q4(uint8_t *_q, float *_s, const float *_weights, int32_t rows, int32_t cols) {
    int32_t num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
    for (int32_t i = 0; i < rows; i++) {
        const float *_row = _weights + (size_t)i * cols;
        float *_row_s = _s + (size_t)i * num_groups;
        uint8_t *_row_q = _q + ((size_t)i * cols) / 2;
        for (int32_t g = 0; g < num_groups; g++) {
            int32_t start = g * GROUP_SIZE;
            int32_t end = start + GROUP_SIZE;
            if (end > cols) {
                end = cols;
            }
            float wmax = 0.0f;
            for (int32_t j = start; j < end; j++) {
                float val = fabsf(_row[j]);
                if (val > wmax) {
                    wmax = val;
                }
            }
            if (wmax < 1e-9f) {
                wmax = 1e-9f;
            }
            float scale = wmax / 7.0f;
            _row_s[g] = scale;
            for (int32_t j = start; j < end; j += 2) {
                float v0 = _row[j] / scale;
                float v1 = (j + 1 < end) ? _row[j + 1] / scale : 0.0f;
                int32_t q0 = (int32_t)roundf(v0);
                int32_t q1 = (int32_t)roundf(v1);
                q0 = q0 < -7 ? -7 : (q0 > 7 ? 7 : q0);
                q1 = q1 < -7 ? -7 : (q1 > 7 ? 7 : q1);
                _row_q[j / 2] = ((uint8_t)(q1 & 0x0F) << 4) | (uint8_t)(q0 & 0x0F);
            }
        }
    }
}

static int32_t load_single_safetensors(safetensors_idx *_st_idx, const char *_model_dir_s) {
    char file_path[PATH_MAX];
    if (snprintf(file_path, sizeof(file_path), "%s/model.safetensors", _model_dir_s) >= (int32_t)sizeof(file_path)) {
        return -1;
    }

    FILE *_file = fopen(file_path, "rb");
    if (! _file) {
        log_msg(stderr, "ERROR: Could not open %s\n", file_path);
        return -1;
    }

    uint8_t length_bytes[8];
    if (fread(length_bytes, 1, sizeof(length_bytes), _file) != sizeof(length_bytes)) {
        log_msg(stderr, "ERROR: Could not read safetensors header length from %s\n", file_path);
        fclose(_file);
        return -1;
    }
    uint64_t header_len_u64 = read_le64(length_bytes);
    if ((! header_len_u64) ||
        (header_len_u64 > SIZE_MAX - 1)) {
        log_msg(stderr, "ERROR: Invalid safetensors header length in %s\n", file_path);
        fclose(_file);
        return -1;
    }
    size_t header_len = (size_t)header_len_u64;
    autofree char *_header_s = (char *)a_calloc(header_len + 1);
    if ((! _header_s) ||
        (fread(_header_s, 1, header_len, _file) != header_len)) {
        log_msg(stderr, "ERROR: Could not read safetensors header from %s\n", file_path);
        fclose(_file);
        return -1;
    }
    fclose(_file);
    _header_s[header_len] = '\0';
    char _error_s[256] = { 0 };
    autojson JsonValue *_js_root = json_parse(_header_s, header_len, _error_s, sizeof(_error_s));
    if ((! _js_root) ||
        (_js_root->type != JSON_OBJECT)) {
        log_msg(stderr, "ERROR: Invalid safetensors header in %s: %s\n", file_path, _error_s);
        return -1;
    }
    // Count valid tensors (ignoring __metadata__)
    size_t n_entries = 0;
    for (size_t i = 0; i < _js_root->data.object.count; i++) {
        JsonPair *_js_pair = &_js_root->data.object.pairs[i];
        if (! strcmp(_js_pair->key, "__metadata__")) {
            continue;
        }
        n_entries++;
    }
    _st_idx->n_entries = n_entries;
    _st_idx->_wm_entries = (weightmap_entry *)a_calloc(n_entries * sizeof(weightmap_entry));
    _st_idx->_model_dir_s = strdup(_model_dir_s);
    if ((! _st_idx->_wm_entries) ||
            (! _st_idx->_model_dir_s)) {
        free_safetensors_index(_st_idx);
        return -1;
    }
    _st_idx->unique_files_n = 1;
    _st_idx->__unique_filenames = (char **)a_calloc(1 * sizeof(char *));
    if (! _st_idx->__unique_filenames) {
        free_safetensors_index(_st_idx);
        return -1;
    }
    _st_idx->__unique_filenames[0] = strdup("model.safetensors");
    if (! _st_idx->__unique_filenames[0]) {
        free_safetensors_index(_st_idx);
        return -1;
    }
    size_t entry_idx = 0;
    for (size_t i = 0; i < _js_root->data.object.count; i++) {
        JsonPair *_js_pair = &_js_root->data.object.pairs[i];
        if (! strcmp(_js_pair->key, "__metadata__")) {
            continue;
        }
        _st_idx->_wm_entries[entry_idx]._tensor_name_s = strdup(_js_pair->key);
        _st_idx->_wm_entries[entry_idx]._file_name_s = strdup("model.safetensors");
        _st_idx->_wm_entries[entry_idx].processing_rank = entry_idx;
        if ((! _st_idx->_wm_entries[entry_idx]._tensor_name_s) ||
                (! _st_idx->_wm_entries[entry_idx]._file_name_s)) {
            free_safetensors_index(_st_idx);
            return -1;
        }
        entry_idx++;
    }
    // Reuse existing metadata parsing logic to populate dtypes, shapes, and offsets
    if (load_shard_metadata(_st_idx, "model.safetensors")) {
        free_safetensors_index(_st_idx);
        return -1;
    }
    return 0;
}

int32_t load_safetensors_index(safetensors_idx *_st_idx, const char *_model_dir_s) {
    memset(_st_idx, 0, sizeof(safetensors_idx));
    char index_path[PATH_MAX];
    if (snprintf(index_path, sizeof(index_path), "%s/model.safetensors.index.json", _model_dir_s)
        >= (int32_t)sizeof(index_path)) {
        return -1;
    }
    FILE *_file = fopen(index_path, "rb");
    if (! _file) {
        // Fallback: Check for single file model
        char single_path[PATH_MAX];
        if (snprintf(single_path, sizeof(single_path), "%s/model.safetensors", _model_dir_s)
            >= (int32_t)sizeof(single_path)) {
            return -1;
        }
        FILE *_single_file = fopen(single_path, "rb");
        if (_single_file) {
            fclose(_single_file);
            return load_single_safetensors(_st_idx, _model_dir_s);
        }
        return -1; // Neither index nor single file exists
    }
    if (fseeko(_file, 0, SEEK_END)) {
        fclose(_file);
        return -1;
    }
    off_t file_size = ftello(_file);
    if ((file_size < 0) ||
        fseeko(_file, 0, SEEK_SET)) {
        fclose(_file);
        return -1;
    }
    size_t size = (size_t)file_size;
    autofree char *_json_s = (char *)a_calloc(size + 1);
    if ((! _json_s) ||
        (fread(_json_s, 1, size, _file) != size)) {
        fclose(_file);
        return -1;
    }
    fclose(_file);
    _json_s[size] = '\0';
    char _error_s[256] = { 0 };
    autojson JsonValue *_js_root = json_parse(_json_s, size, _error_s, sizeof(_error_s));
    if (! _js_root) {
        return -1;
    }
    JsonValue *_js_weight_map = json_object_get(_js_root, "weight_map");
    if ((! _js_weight_map) ||
        (_js_weight_map->type != JSON_OBJECT)) {
        return -1;
    }
    _st_idx->n_entries = _js_weight_map->data.object.count;
    _st_idx->_wm_entries = (weightmap_entry *)a_calloc(_st_idx->n_entries * sizeof(weightmap_entry));
    _st_idx->_model_dir_s = strdup(_model_dir_s);
    if ((! _st_idx->_wm_entries) ||
        (! _st_idx->_model_dir_s)) {
        free_safetensors_index(_st_idx);
        return -1;
    }
    for (size_t i = 0; i < _st_idx->n_entries; i++) {
        JsonPair *_js_pair = &_js_weight_map->data.object.pairs[i];
        if ((! _js_pair->value) ||
            (_js_pair->value->type != JSON_STRING)) {
            free_safetensors_index(_st_idx);
            return -1;
        }
        _st_idx->_wm_entries[i]._tensor_name_s = strdup(_js_pair->key);
        _st_idx->_wm_entries[i]._file_name_s = strdup(_js_pair->value->data.string);
        if ((! _st_idx->_wm_entries[i]._tensor_name_s) ||
            (! _st_idx->_wm_entries[i]._file_name_s)) {
            free_safetensors_index(_st_idx);
            return -1;
        }
        int32_t found = 0;
        for (int32_t j = 0; j < _st_idx->unique_files_n; j++) {
            if (! strcmp(_st_idx->__unique_filenames[j], _js_pair->value->data.string)) {
                found = 1;
                break;
            }
        }
        if (! found) {
            char **files = (char **)realloc(_st_idx->__unique_filenames, (size_t)(_st_idx->unique_files_n + 1) *
            sizeof(char *));
            if (! files) {
                free_safetensors_index(_st_idx);
                return -1;
            }
            _st_idx->__unique_filenames = files;
            _st_idx->__unique_filenames[_st_idx->unique_files_n] = strdup(_js_pair->value->data.string);
            if (! _st_idx->__unique_filenames[_st_idx->unique_files_n]) {
                free_safetensors_index(_st_idx);
                return -1;
            }
            _st_idx->unique_files_n++;
        }
    }
    size_t rank = 0;
    for (int32_t file_i = 0; file_i < _st_idx->unique_files_n; file_i++) {
        for (size_t i = 0; i < _st_idx->n_entries; i++) {
            if (! strcmp(_st_idx->_wm_entries[i]._file_name_s, _st_idx->__unique_filenames[file_i])) {
                _st_idx->_wm_entries[i].processing_rank = rank++;
            }
        }
        if (load_shard_metadata(_st_idx, _st_idx->__unique_filenames[file_i])) {
            free_safetensors_index(_st_idx);
            return -1;
        }
    }
    return 0;
}

void free_safetensors_index(safetensors_idx *_st_idx) {
    if (! _st_idx) {
        return;
    }
    for (size_t i = 0; i < _st_idx->n_entries; i++) {
        free(_st_idx->_wm_entries[i]._tensor_name_s);
        free(_st_idx->_wm_entries[i]._file_name_s);
    }
    free(_st_idx->_wm_entries);
    for (int32_t i = 0; i < _st_idx->unique_files_n; i++) {
        free(_st_idx->__unique_filenames[i]);
    }
    free(_st_idx->__unique_filenames);
    free(_st_idx->_model_dir_s);
    memset(_st_idx, 0, sizeof(safetensors_idx));
}

int32_t quantize_ctx_open(quantize_ctx *_qt_ctx, const char *_model_dir_s) {
    memset(_qt_ctx, 0, sizeof(quantize_ctx));
    _qt_ctx->chunk_bytes = DOLEN_QUANTIZE_CHUNK_BYTES;
    return load_safetensors_index(&_qt_ctx->index, _model_dir_s);
}

void quantize_ctx_close(quantize_ctx *_qt_ctx) {
    if (! _qt_ctx) {
        return;
    }
    if (_qt_ctx->_file) {
        fclose(_qt_ctx->_file);
    }
    free_safetensors_index(&_qt_ctx->index);
    memset(_qt_ctx, 0, sizeof(quantize_ctx));
}

const weightmap_entry *quantize_find_tensor(const quantize_ctx *_qt_ctx, const char *_name_s) {
    for (size_t i = 0; i < _qt_ctx->index.n_entries; i++) {
        if (! strcmp(_qt_ctx->index._wm_entries[i]._tensor_name_s, _name_s)) {
            return &_qt_ctx->index._wm_entries[i];
        }
    }
    return NULL;
}

const weightmap_entry *quantize_find_last_tensor(const quantize_ctx *_qt_ctx,
        const char *const *__names_s, size_t n_names) {
    const weightmap_entry *_wm_best = NULL;
    for (size_t n = 0; n < n_names; n++) {
        const weightmap_entry *_wm_entry = quantize_find_tensor(_qt_ctx, __names_s[n]);
        if (_wm_entry &&
            ((! _wm_best) ||
            (_wm_entry->processing_rank > _wm_best->processing_rank))) {
            _wm_best = _wm_entry;
        }
    }
    return _wm_best;
}

static int32_t open_entry_source(quantize_ctx *_qt_ctx, const weightmap_entry *_wm_entry) {
    if ((! _wm_entry) ||
        (! _wm_entry->metadata_ready)) {
        return -1;
    }
    if (_qt_ctx->_file &&
        _qt_ctx->_file_name_s &&
        (! strcmp(_qt_ctx->_file_name_s, _wm_entry->_file_name_s))) {
        return 0;
    }
    if (_qt_ctx->_file) {
        fclose(_qt_ctx->_file);
        _qt_ctx->_file = NULL;
        _qt_ctx->_file_name_s = NULL;
    }
    char _file_path_s[PATH_MAX];
    if (snprintf(_file_path_s, sizeof(_file_path_s), "%s/%s",
        _qt_ctx->index._model_dir_s, _wm_entry->_file_name_s) >= (int32_t)sizeof(_file_path_s)) {
        return -1;
    }
    _qt_ctx->_file = fopen(_file_path_s, "rb");
    if ((! _qt_ctx->_file)) {
        log_msg(stderr, "ERROR: Could not open %s\n", _file_path_s);
        return -1;
    }
    _qt_ctx->_file_name_s = _wm_entry->_file_name_s;
    return 0;
}

static int32_t validate_entry(const weightmap_entry *_wm_entry, const char *_name_s, size_t expected_elements) {
    if (! _wm_entry) {
        log_msg(stderr, "ERROR: Missing tensor %s\n", _name_s ? _name_s : "(unknown)");
        return -1;
    }
    if (! _wm_entry->metadata_ready ||
        (! dtype_size(_wm_entry->dtype))) {
        log_msg(stderr, "ERROR: Unsupported or missing dtype for tensor %s\n", _wm_entry->_tensor_name_s);
        return -1;
    }
    if (_wm_entry->num_elements != expected_elements) {
        log_msg(stderr, "ERROR: Tensor %s size mismatch: got %llu, expected %zu\n", _wm_entry->_tensor_name_s,
            (uint64_t)_wm_entry->num_elements, expected_elements);
        return -1;
    }
    return 0;
}

static int32_t read_f32_range(quantize_ctx *_qt_ctx, const weightmap_entry *_wm_entry,
        uint64_t first_element, size_t elements, void *_raw, float *_f32) {
    size_t item_size = dtype_size(_wm_entry->dtype);
    uint64_t byte_offset;
    if (first_element > UINT64_MAX / item_size ||
        checked_add_u64(_wm_entry->data_offset, first_element * item_size, &byte_offset) ||
        seek_abs(_qt_ctx->_file, byte_offset)) {
        return -1;
    }
    if (fread(_raw, item_size, elements, _qt_ctx->_file) != elements) {
        log_msg(stderr, "ERROR: Failed reading tensor %s\n", _wm_entry->_tensor_name_s);
        return -1;
    }
    if (_wm_entry->dtype == ST_DTYPE_F32) {
        memcpy(_f32, _raw, elements * sizeof(float));
    }
    else if (_wm_entry->dtype == ST_DTYPE_F16) {
        const uint16_t *_src = (const uint16_t *)_raw;
        for (size_t i = 0; i < elements; i++) {
            _f32[i] = csafetensors_f16_to_f32(_src[i]);
        }
    }
    else if (_wm_entry->dtype == ST_DTYPE_BF16) {
        const uint16_t *_src = (const uint16_t *)_raw;
        for (size_t i = 0; i < elements; i++) {
            _f32[i] = csafetensors_bf16_to_f32(_src[i]);
        }
    }
    else {
        return -1;
    }
    return 0;
}

int32_t quantize_write_tensor_entry(quantize_ctx *_qt_ctx, FILE *_file, const char *_name_s,
        const weightmap_entry *_wm_entry, int32_t rows, int32_t cols, q_type_t q_type) {
    log_msg(stdout, "INFO: writing(2) \"%s\", %d rows, %d cols, %d q_type\n", _name_s, rows, cols, (int32_t)q_type);
    if (rows <= 0 ||
        cols <= 0) {
        log_msg(stderr, "ERROR: Invalid quantized tensor shape %d x %d\n", rows, cols);
        return -1;
    }
    size_t expected_elements;
    if (checked_mul_size((size_t)rows, (size_t)cols, &expected_elements) ||
        validate_entry(_wm_entry, _wm_entry ? _wm_entry->_tensor_name_s : NULL, expected_elements) ||
        open_entry_source(_qt_ctx, _wm_entry)) {
        return -1;
    }
    if (quantize_write_bytes(_file, &q_type, sizeof(q_type), 1) ||
        quantize_write_bytes(_file, &rows, sizeof(rows), 1) ||
        quantize_write_bytes(_file, &cols, sizeof(cols), 1)) {
        return -1;
    }
    if (q_type == Q_TYPE_F32) {
        size_t item_size = dtype_size(_wm_entry->dtype);
        size_t per_element = item_size + sizeof(float);
        size_t chunk_elements = _qt_ctx->chunk_bytes / per_element;
        if (! chunk_elements) {
            chunk_elements = 1;
        }
        if ((chunk_elements > expected_elements) &&
            (expected_elements > 0)) {
            chunk_elements = expected_elements;
        }
        size_t raw_bytes;
        if (checked_mul_size(chunk_elements, item_size, &raw_bytes)) {
            return -1;
        }
        autofree void *_raw = a_calloc(raw_bytes);
        autofree float *_f32 = (float *)a_calloc(chunk_elements * sizeof(float));
        if ((expected_elements > 0) &&
            ((! _raw) ||
            (! _f32))) {
            return -1;
        }
        for (size_t done = 0; done < expected_elements;) {
            size_t n = expected_elements - done;
            if (n > chunk_elements) {
                n = chunk_elements;
            }
            if (read_f32_range(_qt_ctx, _wm_entry, done, n, _raw, _f32) ||
                quantize_write_bytes(_file, _f32, sizeof(float), n)) {
                return -1;
            }
            done += n;
        }
        return 0;
    }
    else if (q_type == Q_TYPE_F16) {
        size_t item_size = dtype_size(_wm_entry->dtype);
        size_t per_element = item_size + sizeof(_Float16);
        size_t chunk_elements = _qt_ctx->chunk_bytes / per_element;
        if (! chunk_elements) {
            chunk_elements = 1;
        }
        if ((chunk_elements > expected_elements) &&
            (expected_elements > 0)) {
            chunk_elements = expected_elements;
        }
        size_t raw_bytes;
        if (checked_mul_size(chunk_elements, item_size, &raw_bytes)) {
            return -1;
        }
        autofree void *_raw = a_calloc(raw_bytes);
        autofree float *_f32 = (float *)a_calloc(chunk_elements * sizeof(float));
        autofree _Float16 *_f16 = (_Float16 *)a_calloc(chunk_elements * sizeof(_Float16));
        if ((expected_elements > 0) &&
            ((! _raw) ||
            (! _f32) ||
            (! _f16))) {
            return -1;
        }
        for (size_t done = 0; done < expected_elements;) {
            size_t n = expected_elements - done;
            if (n > chunk_elements) {
                n = chunk_elements;
            }
            if (read_f32_range(_qt_ctx, _wm_entry, done, n, _raw, _f32)) {
                return -1;
            }
            for (size_t i = 0; i < n; i++) {
                _f16[i] = (_Float16)_f32[i];
            }
            if (quantize_write_bytes(_file, _f16, sizeof(_Float16), n)) {
                return -1;
            }
            done += n;
        }
        return 0;
    }
    else if (q_type == Q_TYPE_Q8) {
        int32_t num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        uint64_t q_bytes = (uint64_t)rows * (uint64_t)cols;
        uint64_t s_bytes = (uint64_t)rows * (uint64_t)num_groups * sizeof(float);
        off_t current = ftello(_file);
        if (current < 0) {
            return -1;
        }
        uint64_t q_offset = (uint64_t)current;
        uint64_t s_offset, end_offset;
        if (checked_add_u64(q_offset, q_bytes, &s_offset) ||
            checked_add_u64(s_offset, s_bytes, &end_offset)) {
            return -1;
        }
        if (end_offset > 0) {
            if (seek_abs(_file, end_offset - 1) ||
                fputc(0, _file) == EOF) {
                return -1;
            }
        }
        size_t item_size = dtype_size(_wm_entry->dtype);
        size_t row_work_bytes =
            (size_t)cols * (item_size + sizeof(float) + sizeof(int8_t)) + (size_t)num_groups * sizeof(float);
        size_t rows_per_chunk = row_work_bytes ? _qt_ctx->chunk_bytes / row_work_bytes : 1;
        if (! rows_per_chunk) {
            rows_per_chunk = 1;
        }
        if (rows_per_chunk > (size_t)rows) {
            rows_per_chunk = (size_t)rows;
        }
        size_t max_elements;
        if (checked_mul_size(rows_per_chunk, (size_t)cols, &max_elements)) {
            return -1;
        }
        size_t raw_bytes;
        if (checked_mul_size(max_elements, item_size, &raw_bytes)) {
            return -1;
        }
        autofree void *_raw = a_calloc(raw_bytes);
        autofree float *_f32 = (float *)a_calloc(max_elements * sizeof(float));
        autofree int8_t *_q = (int8_t *)a_calloc(max_elements * sizeof(int8_t));
        autofree float *_s = (float *)a_calloc(rows_per_chunk * (size_t)num_groups * sizeof(float));
        if ((! _raw) ||
            (! _f32) ||
            (! _q) ||
            (! _s)) {
            return -1;
        }
        for (int32_t row_c = 0; row_c < rows;) {
            int32_t chunk_rows = rows - row_c;
            if ((size_t)chunk_rows > rows_per_chunk) {
                chunk_rows = (int32_t)rows_per_chunk;
            }
            size_t chunk_elements = (size_t)chunk_rows * cols;
            if (read_f32_range(_qt_ctx, _wm_entry, (uint64_t)row_c * cols, chunk_elements, _raw, _f32)) {
                return -1;
            }
            quantize_group_into_q8(_q, _s, _f32, chunk_rows, cols);
            if (seek_abs(_file, q_offset + (uint64_t)row_c * cols) ||
                quantize_write_bytes(_file, _q, sizeof(int8_t), chunk_elements) ||
                seek_abs(_file, s_offset + (uint64_t)row_c * num_groups * sizeof(float)) ||
                quantize_write_bytes(_file, _s, sizeof(float), (size_t)chunk_rows * num_groups)) {
                return -1;
            }
            row_c += chunk_rows;
        }
        return seek_abs(_file, end_offset);
    }
    else if (q_type == Q_TYPE_Q6) {
        int32_t num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        uint64_t q_bytes = ((uint64_t)rows * (uint64_t)cols * 3 + 3) / 4;
        uint64_t s_bytes = (uint64_t)rows * (uint64_t)num_groups * sizeof(float);
        off_t current = ftello(_file);
        if (current < 0) {
            return -1;
        }
        uint64_t q_offset = (uint64_t)current;
        uint64_t s_offset, end_offset;
        if (checked_add_u64(q_offset, q_bytes, &s_offset) ||
            checked_add_u64(s_offset, s_bytes, &end_offset)) {
            return -1;
        }
        if (end_offset > 0) {
            if (seek_abs(_file, end_offset - 1) ||
                fputc(0, _file) == EOF) {
                return -1;
            }
        }
        size_t item_size = dtype_size(_wm_entry->dtype);
        size_t row_work_bytes =
            (size_t)cols * (item_size + sizeof(float)) + ((size_t)cols * 3) / 4 + (size_t)num_groups * sizeof(float);
        size_t rows_per_chunk = row_work_bytes ? _qt_ctx->chunk_bytes / row_work_bytes : 1;
        if (! rows_per_chunk) {
            rows_per_chunk = 1;
        }
        if (rows_per_chunk > (size_t)rows) {
            rows_per_chunk = (size_t)rows;
        }
        size_t max_elements;
        if (checked_mul_size(rows_per_chunk, (size_t)cols, &max_elements)) {
            return -1;
        }
        size_t raw_bytes;
        if (checked_mul_size(max_elements, item_size, &raw_bytes)) {
            return -1;
        }
        autofree void *_raw = a_calloc(raw_bytes);
        autofree float *_f32 = (float *)a_calloc(max_elements * sizeof(float));
        autofree uint8_t *_q = (uint8_t *)a_calloc((max_elements * 3 + 3) / 4);
        autofree float *_s = (float *)a_calloc(rows_per_chunk * (size_t)num_groups * sizeof(float));
        if ((! _raw) ||
            (! _f32) ||
            (! _q) ||
            (! _s)) {
            return -1;
        }
        for (int32_t row_c = 0; row_c < rows;) {
            int32_t chunk_rows = rows - row_c;
            if ((size_t)chunk_rows > rows_per_chunk) {
                chunk_rows = (int32_t)rows_per_chunk;
            }
            size_t chunk_elements = (size_t)chunk_rows * cols;
            if (read_f32_range(_qt_ctx, _wm_entry, (uint64_t)row_c * cols, chunk_elements, _raw, _f32)) {
                return -1;
            }
            quantize_group_into_q6(_q, _s, _f32, chunk_rows, cols);
            if (seek_abs(_file, q_offset + ((uint64_t)row_c * cols * 3) / 4) ||
                quantize_write_bytes(_file, _q, 1, (chunk_elements * 3 + 3) / 4) ||
                seek_abs(_file, s_offset + (uint64_t)row_c * num_groups * sizeof(float)) ||
                quantize_write_bytes(_file, _s, sizeof(float), (size_t)chunk_rows * num_groups)) {
                return -1;
            }
            row_c += chunk_rows;
        }
        return seek_abs(_file, end_offset);
    }
    else if (q_type == Q_TYPE_Q4) {
        int32_t num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        uint64_t q_bytes = ((uint64_t)rows * (uint64_t)cols + 1) / 2;
        uint64_t s_bytes = (uint64_t)rows * (uint64_t)num_groups * sizeof(float);
        off_t current = ftello(_file);
        if (current < 0) {
            return -1;
        }
        uint64_t q_offset = (uint64_t)current;
        uint64_t s_offset, end_offset;
        if (checked_add_u64(q_offset, q_bytes, &s_offset) ||
            checked_add_u64(s_offset, s_bytes, &end_offset)) {
            return -1;
        }
        if (end_offset > 0) {
            if (seek_abs(_file, end_offset - 1) ||
                fputc(0, _file) == EOF) {
                return -1;
            }
        }
        size_t item_size = dtype_size(_wm_entry->dtype);
        size_t row_work_bytes =
            (size_t)cols * (item_size + sizeof(float)) + (size_t)cols / 2 + (size_t)num_groups * sizeof(float);
        size_t rows_per_chunk = row_work_bytes ? _qt_ctx->chunk_bytes / row_work_bytes : 1;
        if (! rows_per_chunk) {
            rows_per_chunk = 1;
        }
        if (rows_per_chunk > (size_t)rows) {
            rows_per_chunk = (size_t)rows;
        }
        size_t max_elements;
        if (checked_mul_size(rows_per_chunk, (size_t)cols, &max_elements)) {
            return -1;
        }
        size_t raw_bytes;
        if (checked_mul_size(max_elements, item_size, &raw_bytes)) {
            return -1;
        }
        autofree void *_raw = a_calloc(raw_bytes);
        autofree float *_f32 = (float *)a_calloc(max_elements * sizeof(float));
        autofree uint8_t *_q = (uint8_t *)a_calloc((max_elements + 1) / 2);
        autofree float *_s = (float *)a_calloc(rows_per_chunk * (size_t)num_groups * sizeof(float));
        if ((! _raw) ||
            (! _f32) ||
            (! _q) ||
            (! _s)) {
            return -1;
        }
        for (int32_t row_c = 0; row_c < rows;) {
            int32_t chunk_rows = rows - row_c;
            if ((size_t)chunk_rows > rows_per_chunk) {
                chunk_rows = (int32_t)rows_per_chunk;
            }
            size_t chunk_elements = (size_t)chunk_rows * cols;
            if (read_f32_range(_qt_ctx, _wm_entry, (uint64_t)row_c * cols, chunk_elements, _raw, _f32)) {
                return -1;
            }
            quantize_group_into_q4(_q, _s, _f32, chunk_rows, cols);
            if (seek_abs(_file, q_offset + ((uint64_t)row_c * cols) / 2) ||
                quantize_write_bytes(_file, _q, 1, (chunk_elements + 1) / 2) ||
                seek_abs(_file, s_offset + (uint64_t)row_c * num_groups * sizeof(float)) ||
                quantize_write_bytes(_file, _s, sizeof(float), (size_t)chunk_rows * num_groups)) {
                return -1;
            }
            row_c += chunk_rows;
        }
        return seek_abs(_file, end_offset);
    }
    return -1;
}

int32_t quantize_write_tensor(quantize_ctx *_qt_ctx, FILE *_file, const char *_name_s,
        int32_t rows, int32_t cols, q_type_t q_type) {
    log_msg(stdout, "INFO: writing \"%s\", %d rows, %d cols, %d q_type\n", _name_s, rows, cols, (int32_t)q_type);
    return quantize_write_tensor_entry(_qt_ctx, _file, _name_s, quantize_find_tensor(_qt_ctx, _name_s),
        rows, cols, q_type);
}

int32_t quantize_write_tensor_or_empty(quantize_ctx *_qt_ctx, FILE *_file, const char *_name_s,
        int32_t rows, int32_t cols, q_type_t q_type) {
    const weightmap_entry *_wm_entry = quantize_find_tensor(_qt_ctx, _name_s);
    if (! _wm_entry) {
        return quantize_write_empty_tensor(_file);
    }
    return quantize_write_tensor_entry(_qt_ctx, _file, _name_s, _wm_entry, rows, cols, q_type);
}

int32_t quantize_write_empty_tensor(FILE *_file) {
    q_type_t q_type = Q_TYPE_F32;
    int32_t zero = 0;
    return (quantize_write_bytes(_file, &q_type, sizeof(q_type_t), 1) ||
        quantize_write_bytes(_file, &zero, sizeof(zero), 1) ||
        quantize_write_bytes(_file, &zero, sizeof(zero), 1)) ? -1 : 0;
}

int32_t quantize_write_scalar_or_default(quantize_ctx *_qt_ctx, FILE *_file,
        const char *const *__names_s, size_t n_names, float default_value) {
    const weightmap_entry *_wm_entry = quantize_find_last_tensor(_qt_ctx, __names_s, n_names);
    if (! _wm_entry) {
        return quantize_write_bytes(_file, &default_value, sizeof(default_value), 1);
    }
    size_t expected_elements = 1;
    if (validate_entry(_wm_entry, _wm_entry ? _wm_entry->_tensor_name_s : NULL, expected_elements) ||
        open_entry_source(_qt_ctx, _wm_entry)) {
        return -1;
    }
    size_t item_size = dtype_size(_wm_entry->dtype);
    size_t per_element = item_size + sizeof(float);
    size_t chunk_elements = _qt_ctx->chunk_bytes / per_element;
    if (! chunk_elements) {
        chunk_elements = 1;
    }
    if ((chunk_elements > expected_elements) &&
        (expected_elements > 0)) {
        chunk_elements = expected_elements;
    }
    size_t raw_bytes;
    if (checked_mul_size(chunk_elements, item_size, &raw_bytes)) {
        return -1;
    }
    autofree void *_raw = a_calloc(raw_bytes);
    autofree float *_f32 = (float *)a_calloc(chunk_elements * sizeof(float));
    if ((expected_elements > 0) &&
        ((! _raw) ||
        (! _f32))) {
        return -1;
    }
    int32_t res = 0;
    if (read_f32_range(_qt_ctx, _wm_entry, 0, 1, _raw, _f32) ||
        quantize_write_bytes(_file, _f32, sizeof(float), 1)) {
        res = -1;
    }
    return res;
}

q_type_t parse_q_type(const char *_type_s) {
    if (! _type_s) {
        log_msg(stderr, "WARNING: Missing tensor type name. Defaulting to Q8.\n");
        return Q_TYPE_Q8;
    }
    else if ((! strcmp(_type_s, "F32")) ||
        (! strcmp(_type_s, "f32"))) {
        return Q_TYPE_F32;
    }
    else if ((! strcmp(_type_s, "F16")) ||
        (! strcmp(_type_s, "f16"))) {
        return Q_TYPE_F16;
    }
    else if ((! strcmp(_type_s, "Q8")) ||
        (! strcmp(_type_s, "q8"))) {
        return Q_TYPE_Q8;
    }
    else if ((! strcmp(_type_s, "Q6")) ||
        (! strcmp(_type_s, "q6"))) {
        return Q_TYPE_Q6;
    }
    else if ((! strcmp(_type_s, "Q4")) ||
        (! strcmp(_type_s, "q4"))) {
        return Q_TYPE_Q4;
    }

    log_msg(stderr, "WARNING: Unknown tensor type '%s'. Defaulting to Q8.\n", _type_s);
    return Q_TYPE_Q8;
}

