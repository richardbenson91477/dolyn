#include "dolen_quantize_common.h"

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a && (b > (SIZE_MAX / a))) {
        return -1;
    }
    *out = a * b;
    return 0;
}

static int checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (b > (UINT64_MAX - a)) {
        return -1;
    }
    *out = a + b;
    return 0;
}

int quantize_write_bytes(FILE *f, const void *data, size_t size, size_t count) {
    if (! count) {
        return 0;
    }
    if (fwrite(data, size, count, f) != count) {
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

static st_dtype parse_dtype(const char *dtype) {
    if (! dtype) {
        return ST_DTYPE_UNSUPPORTED;
    } else if (! strcmp(dtype, "F16")) {
        return ST_DTYPE_F16;
    } else if (! strcmp(dtype, "BF16")) {
        return ST_DTYPE_BF16;
    } else if (! strcmp(dtype, "F32")) {
        return ST_DTYPE_F32;
    } else {
        return ST_DTYPE_UNSUPPORTED;
    }
}

static int json_u64(JsonValue *value, uint64_t *out) {
    if ((! value) || (value->type != JSON_NUMBER) || value->data.number < 0.0) {
        return -1;
    }
    double d = value->data.number;
    uint64_t v = (uint64_t)d;
    if ((double)v != d)
        return -1;
    *out = v;
    return 0;
}

static weightmap_entry *find_index_entry(safetensors_idx *idx, const char *filename, const char *tensor_name) {
    for (size_t i = 0; i < idx->n_entries; i++) {
        weightmap_entry *entry = &idx->entries[i];
        if ((! strcmp(entry->filename, filename)) && (! strcmp(entry->tensor_name, tensor_name))) {
            return entry;
        }
    }
    return NULL;
}

static int load_shard_metadata(safetensors_idx *idx, const char *filename) {
    char filepath[PATH_MAX];
    if (snprintf(filepath, sizeof(filepath), "%s/%s", idx->model_dir, filename) >= (int)sizeof(filepath)) {
        log_msg(stderr, "ERROR: Safetensors path is too long\n");
        return -1;
    }

    FILE *f = fopen(filepath, "rb");
    if (! f) {
        log_msg(stderr, "ERROR: Could not open %s\n", filepath);
        return -1;
    }

    uint8_t length_bytes[8];
    if (fread(length_bytes, 1, sizeof(length_bytes), f) != sizeof(length_bytes)) {
        log_msg(stderr, "ERROR: Could not read safetensors header length from %s\n", filepath);
        fclose(f);
        return -1;
    }

    uint64_t header_len_u64 = read_le64(length_bytes);
    if ((! header_len_u64) || header_len_u64 > SIZE_MAX - 1) {
        log_msg(stderr, "ERROR: Invalid safetensors header length in %s\n", filepath);
        fclose(f);
        return -1;
    }
    size_t header_len = (size_t)header_len_u64;
    char *header = (char *)a_calloc(header_len + 1);
    if ((! header) || (fread(header, 1, header_len, f) != header_len)) {
        log_msg(stderr, "ERROR: Could not read safetensors header from %s\n", filepath);
        free(header);
        fclose(f);
        return -1;
    }
    fclose(f);
    header[header_len] = '\0';

    char error[256] = { 0 };
    JsonValue *root = json_parse(header, header_len, error, sizeof(error));
    free(header);
    if ((! root) || root->type != JSON_OBJECT) {
        log_msg(stderr, "ERROR: Invalid safetensors header in %s: %s\n", filepath, error);
        json_free(root);
        return -1;
    }

    uint64_t data_base;
    if (checked_add_u64(8, header_len_u64, &data_base)) {
        json_free(root);
        return -1;
    }

    for (size_t i = 0; i < root->data.object.count; i++) {
        JsonPair *pair = &root->data.object.pairs[i];
        if (! strcmp(pair->key, "__metadata__")) {
            continue;
        }

        weightmap_entry *entry = find_index_entry(idx, filename, pair->key);
        if (! entry) {
            continue;
        }

        JsonValue *tensor = pair->value;
        JsonValue *dtype_json = json_object_get(tensor, "dtype");
        JsonValue *shape = json_object_get(tensor, "shape");
        JsonValue *offsets = json_object_get(tensor, "data_offsets");
        if ((! dtype_json) || (dtype_json->type != JSON_STRING) || (! shape) || (shape->type != JSON_ARRAY) ||
                (! offsets) || (offsets->type != JSON_ARRAY) || (offsets->data.array.count != 2)) {
            log_msg(stderr, "ERROR: Invalid metadata for tensor %s\n", pair->key);
            json_free(root);
            return -1;
        }

        uint64_t elements = 1;
        for (size_t d = 0; d < shape->data.array.count; d++) {
            uint64_t dim;
            if (json_u64(json_array_get(shape, d), &dim) || (dim && (elements > UINT64_MAX / dim))) {
                log_msg(stderr, "ERROR: Invalid shape for tensor %s\n", pair->key);
                json_free(root);
                return -1;
            }
            elements *= dim;
        }

        uint64_t begin, end;
        if (json_u64(json_array_get(offsets, 0), &begin) || json_u64(json_array_get(offsets, 1), &end) || end < begin) {
            log_msg(stderr, "ERROR: Invalid data offsets for tensor %s\n", pair->key);
            json_free(root);
            return -1;
        }

        entry->dtype = parse_dtype(dtype_json->data.string);
        entry->num_elements = elements;
        entry->data_nbytes = end - begin;
        if (checked_add_u64(data_base, begin, &entry->data_offset)) {
            json_free(root);
            return -1;
        }
        entry->metadata_ready = 1;

        size_t item_size = dtype_size(entry->dtype);
        if (item_size && ((elements > UINT64_MAX / item_size) || (elements * item_size != entry->data_nbytes))) {
            log_msg(stderr, "ERROR: Byte size mismatch for tensor %s\n", pair->key);
            json_free(root);
            return -1;
        }
    }

    json_free(root);
    return 0;
}

static void quantize_group_into(int8_t *q, float *s, const float *weights, int rows, int cols) {
    int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
    for (int i = 0; i < rows; i++) {
        const float *row = weights + (size_t)i * cols;
        float *row_s = s + (size_t)i * num_groups;
        int8_t *row_q = q + (size_t)i * cols;

        for (int g = 0; g < num_groups; g++) {
            int start = g * GROUP_SIZE;
            int end = start + GROUP_SIZE;
            if (end > cols)
                end = cols;

            float wmax = 0.0f;
            for (int j = start; j < end; j++) {
                float val = fabsf(row[j]);
                if (val > wmax)
                    wmax = val;
            }
            if (wmax < 1e-9f)
                wmax = 1e-9f;

            float scale = wmax / 127.0f;
            row_s[g] = scale;
            for (int j = start; j < end; j++) {
                float quant_value = row[j] / scale;
                row_q[j] = (int8_t)roundf(quant_value);
            }
        }
    }
}

void quantize_group(qtensor *qt, const float *weights, int rows, int cols) {
    qt->rows = rows;
    qt->cols = cols;
    qt->type = Q_TYPE_Q8;
    int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;

    qt->data = a_calloc((size_t)rows * cols * sizeof(int8_t));
    qt->s = a_calloc((size_t)rows * num_groups * sizeof(float));
    if ((rows > 0 && cols > 0) && ((! qt->data) || (! qt->s))) {
        log_msg(stderr, "ERROR: Quantization allocation failed\n");
        exit(EXIT_FAILURE);
    }
    quantize_group_into((int8_t *)qt->data, qt->s, weights, rows, cols);
}

int load_safetensors_index(safetensors_idx *idx, const char *model_dir) {
    memset(idx, 0, sizeof(*idx));

    char index_path[PATH_MAX];
    if (snprintf(index_path, sizeof(index_path), "%s/model.safetensors.index.json", model_dir) >=
            (int)sizeof(index_path))
        return -1;

    FILE *f = fopen(index_path, "rb");
    if (! f) {
        return -1;
    }

    if (fseeko(f, 0, SEEK_END)) {
        fclose(f);
        return -1;
    }
    off_t file_size = ftello(f);
    if ((file_size < 0) || fseeko(f, 0, SEEK_SET)) {
        fclose(f);
        return -1;
    }

    size_t size = (size_t)file_size;
    char *json_str = (char *)a_calloc(size + 1);
    if ((! json_str) || fread(json_str, 1, size, f) != size) {
        free(json_str);
        fclose(f);
        return -1;
    }
    fclose(f);
    json_str[size] = '\0';

    char error[256] = { 0 };
    JsonValue *root = json_parse(json_str, size, error, sizeof(error));
    free(json_str);
    if (! root) {
        return -1;
    }

    JsonValue *weight_map = json_object_get(root, "weight_map");
    if ((! weight_map) || (weight_map->type != JSON_OBJECT)) {
        json_free(root);
        return -1;
    }

    idx->n_entries = weight_map->data.object.count;
    idx->entries = (weightmap_entry *)a_calloc(idx->n_entries * sizeof(weightmap_entry));
    idx->model_dir = strdup(model_dir);
    if ((! idx->entries) || (! idx->model_dir)) {
        json_free(root);
        free_safetensors_index(idx);
        return -1;
    }

    for (size_t i = 0; i < idx->n_entries; i++) {
        JsonPair *pair = &weight_map->data.object.pairs[i];
        if ((! pair->value) || (pair->value->type != JSON_STRING)) {
            json_free(root);
            free_safetensors_index(idx);
            return -1;
        }

        idx->entries[i].tensor_name = strdup(pair->key);
        idx->entries[i].filename = strdup(pair->value->data.string);
        if ((! idx->entries[i].tensor_name) || (! idx->entries[i].filename)) {
            json_free(root);
            free_safetensors_index(idx);
            return -1;
        }

        int found = 0;
        for (int j = 0; j < idx->n_unique_files; j++) {
            if (! strcmp(idx->unique_filenames[j], pair->value->data.string)) {
                found = 1;
                break;
            }
        }
        if (! found) {
            char **files = (char **)realloc(idx->unique_filenames, (size_t)(idx->n_unique_files + 1) * sizeof(char *));
            if (! files) {
                json_free(root);
                free_safetensors_index(idx);
                return -1;
            }
            idx->unique_filenames = files;
            idx->unique_filenames[idx->n_unique_files] = strdup(pair->value->data.string);
            if (! idx->unique_filenames[idx->n_unique_files]) {
                json_free(root);
                free_safetensors_index(idx);
                return -1;
            }
            idx->n_unique_files++;
        }
    }
    json_free(root);

    size_t rank = 0;
    for (int file_i = 0; file_i < idx->n_unique_files; file_i++) {
        for (size_t i = 0; i < idx->n_entries; i++) {
            if (! strcmp(idx->entries[i].filename, idx->unique_filenames[file_i])) {
                idx->entries[i].processing_rank = rank++;
            }
        }
        if (load_shard_metadata(idx, idx->unique_filenames[file_i])) {
            free_safetensors_index(idx);
            return -1;
        }
    }

    return 0;
}

void free_safetensors_index(safetensors_idx *idx) {
    if (! idx)
        return;

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
    memset(idx, 0, sizeof(*idx));
}

int quantize_ctx_open(quantize_ctx *ctx, const char *model_dir) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->chunk_bytes = DOLEN_QUANTIZE_CHUNK_BYTES;
    return load_safetensors_index(&ctx->index, model_dir);
}

void quantize_ctx_close(quantize_ctx *ctx) {
    if (! ctx)
        return;
    if (ctx->source)
        fclose(ctx->source);
    free_safetensors_index(&ctx->index);
    memset(ctx, 0, sizeof(*ctx));
}

const weightmap_entry *quantize_find_tensor(const quantize_ctx *ctx, const char *name) {
    for (size_t i = 0; i < ctx->index.n_entries; i++) {
        if (! strcmp(ctx->index.entries[i].tensor_name, name)) {
            return &ctx->index.entries[i];
        }
    }
    return NULL;
}

const weightmap_entry *quantize_find_last_tensor(const quantize_ctx *ctx, const char *const *names, size_t n_names) {
    const weightmap_entry *best = NULL;
    for (size_t n = 0; n < n_names; n++) {
        const weightmap_entry *entry = quantize_find_tensor(ctx, names[n]);
        if (entry && ((! best) || (entry->processing_rank > best->processing_rank))) {
            best = entry;
        }
    }
    return best;
}

static int open_entry_source(quantize_ctx *ctx, const weightmap_entry *entry) {
    if ((! entry) || (! entry->metadata_ready)) {
        return -1;
    }
    if (ctx->source && ctx->source_filename && (! strcmp(ctx->source_filename, entry->filename))) {
        return 0;
    }

    if (ctx->source) {
        fclose(ctx->source);
        ctx->source = NULL;
        ctx->source_filename = NULL;
    }

    char filepath[PATH_MAX];
    if (snprintf(filepath, sizeof(filepath), "%s/%s", ctx->index.model_dir, entry->filename) >= (int)sizeof(filepath))
        return -1;
    ctx->source = fopen(filepath, "rb");
    if ((! ctx->source)) {
        log_msg(stderr, "ERROR: Could not open %s\n", filepath);
        return -1;
    }
    ctx->source_filename = entry->filename;
    return 0;
}

static int validate_entry(const weightmap_entry *entry, const char *name, size_t expected_elements) {
    if (! entry) {
        log_msg(stderr, "ERROR: Missing tensor %s\n", name ? name : "(unknown)");
        return -1;
    }
    if (! entry->metadata_ready || (! dtype_size(entry->dtype))) {
        log_msg(stderr, "ERROR: Unsupported or missing dtype for tensor %s\n", entry->tensor_name);
        return -1;
    }
    if (entry->num_elements != expected_elements) {
        log_msg(stderr, "ERROR: Tensor %s size mismatch: got %llu, expected %zu\n", entry->tensor_name,
                (unsigned long long)entry->num_elements, expected_elements);
        return -1;
    }
    return 0;
}

static int read_f32_range(quantize_ctx *ctx, const weightmap_entry *entry, uint64_t first_element, size_t elements,
        void *raw, float *f32) {
    size_t item_size = dtype_size(entry->dtype);
    uint64_t byte_offset;
    if (first_element > UINT64_MAX / item_size ||
            checked_add_u64(entry->data_offset, first_element * item_size, &byte_offset) ||
            seek_abs(ctx->source, byte_offset)) {
        return -1;
    }

    if (fread(raw, item_size, elements, ctx->source) != elements) {
        log_msg(stderr, "ERROR: Failed reading tensor %s\n", entry->tensor_name);
        return -1;
    }

    if (entry->dtype == ST_DTYPE_F32) {
        memcpy(f32, raw, elements * sizeof(float));
    } else if (entry->dtype == ST_DTYPE_F16) {
        const uint16_t *src = (const uint16_t *)raw;
        for (size_t i = 0; i < elements; i++)
            f32[i] = csafetensors_f16_to_f32(src[i]);
    } else if (entry->dtype == ST_DTYPE_BF16) {
        const uint16_t *src = (const uint16_t *)raw;
        for (size_t i = 0; i < elements; i++)
            f32[i] = csafetensors_bf16_to_f32(src[i]);
    } else {
        return -1;
    }
    return 0;
}

int quantize_write_tensor_entry(
        quantize_ctx *ctx, FILE *out, const weightmap_entry *entry, int rows, int cols, q_type_t type) {
    if (rows <= 0 || cols <= 0) {
        log_msg(stderr, "ERROR: Invalid quantized tensor shape %d x %d\n", rows, cols);
        return -1;
    }

    size_t expected_elements;
    if (checked_mul_size((size_t)rows, (size_t)cols, &expected_elements) ||
            validate_entry(entry, entry ? entry->tensor_name : NULL, expected_elements) ||
            open_entry_source(ctx, entry)) {
        return -1;
    }

    if (quantize_write_bytes(out, &type, sizeof(type), 1) || quantize_write_bytes(out, &rows, sizeof(rows), 1) ||
            quantize_write_bytes(out, &cols, sizeof(cols), 1)) {
        return -1;
    }

    if (type == Q_TYPE_F32) {
        size_t item_size = dtype_size(entry->dtype);
        size_t per_element = item_size + sizeof(float);
        size_t chunk_elements = ctx->chunk_bytes / per_element;
        if (! chunk_elements) {
            chunk_elements = 1;
        }
        if ((chunk_elements > expected_elements) && (expected_elements > 0)) {
            chunk_elements = expected_elements;
        }

        size_t raw_bytes;
        if (checked_mul_size(chunk_elements, item_size, &raw_bytes))
            return -1;
        void *raw = a_calloc(raw_bytes);
        float *f32 = (float *)a_calloc(chunk_elements * sizeof(float));
        if ((expected_elements > 0) && ((! raw) || (! f32))) {
            free(raw);
            free(f32);
            return -1;
        }

        for (size_t done = 0; done < expected_elements;) {
            size_t n = expected_elements - done;
            if (n > chunk_elements)
                n = chunk_elements;
            if (read_f32_range(ctx, entry, done, n, raw, f32) || quantize_write_bytes(out, f32, sizeof(float), n)) {
                free(raw);
                free(f32);
                return -1;
            }
            done += n;
        }
        free(raw);
        free(f32);
        return 0;
    } else if (type == Q_TYPE_F16) {
        size_t item_size = dtype_size(entry->dtype);
        size_t per_element = item_size + sizeof(_Float16);
        size_t chunk_elements = ctx->chunk_bytes / per_element;
        if (! chunk_elements) {
            chunk_elements = 1;
        }
        if ((chunk_elements > expected_elements) && (expected_elements > 0)) {
            chunk_elements = expected_elements;
        }

        size_t raw_bytes;
        if (checked_mul_size(chunk_elements, item_size, &raw_bytes))
            return -1;
        void *raw = a_calloc(raw_bytes);
        float *f32 = (float *)a_calloc(chunk_elements * sizeof(float));
        _Float16 *f16 = (_Float16 *)a_calloc(chunk_elements * sizeof(_Float16));
        if ((expected_elements > 0) && ((! raw) || (! f32) || (! f16))) {
            free(raw);
            free(f32);
            free(f16);
            return -1;
        }

        for (size_t done = 0; done < expected_elements;) {
            size_t n = expected_elements - done;
            if (n > chunk_elements)
                n = chunk_elements;
            if (read_f32_range(ctx, entry, done, n, raw, f32)) {
                free(raw);
                free(f32);
                free(f16);
                return -1;
            }
            for (size_t i = 0; i < n; i++)
                f16[i] = (_Float16)f32[i];
            if (quantize_write_bytes(out, f16, sizeof(_Float16), n)) {
                free(raw);
                free(f32);
                free(f16);
                return -1;
            }
            done += n;
        }
        free(raw);
        free(f32);
        free(f16);
        return 0;
    } else if (type == Q_TYPE_Q8) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        uint64_t q_bytes = (uint64_t)rows * (uint64_t)cols;
        uint64_t s_bytes = (uint64_t)rows * (uint64_t)num_groups * sizeof(float);

        off_t current = ftello(out);
        if (current < 0)
            return -1;
        uint64_t q_offset = (uint64_t)current;
        uint64_t s_offset, end_offset;
        if (checked_add_u64(q_offset, q_bytes, &s_offset) || checked_add_u64(s_offset, s_bytes, &end_offset))
            return -1;

        if (end_offset > 0) {
            if (seek_abs(out, end_offset - 1) || fputc(0, out) == EOF)
                return -1;
        }

        size_t item_size = dtype_size(entry->dtype);
        size_t row_work_bytes =
                (size_t)cols * (item_size + sizeof(float) + sizeof(int8_t)) + (size_t)num_groups * sizeof(float);
        size_t rows_per_chunk = row_work_bytes ? ctx->chunk_bytes / row_work_bytes : 1;
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

        void *raw = a_calloc(raw_bytes);
        float *f32 = (float *)a_calloc(max_elements * sizeof(float));
        int8_t *q = (int8_t *)a_calloc(max_elements * sizeof(int8_t));
        float *s = (float *)a_calloc(rows_per_chunk * (size_t)num_groups * sizeof(float));
        if ((! raw) || (! f32) || (! q) || (! s)) {
            free(raw);
            free(f32);
            free(q);
            free(s);
            return -1;
        }

        for (int row = 0; row < rows;) {
            int chunk_rows = rows - row;
            if ((size_t)chunk_rows > rows_per_chunk)
                chunk_rows = (int)rows_per_chunk;
            size_t chunk_elements = (size_t)chunk_rows * cols;

            if (read_f32_range(ctx, entry, (uint64_t)row * cols, chunk_elements, raw, f32)) {
                free(raw);
                free(f32);
                free(q);
                free(s);
                return -1;
            }
            quantize_group_into(q, s, f32, chunk_rows, cols);

            if (seek_abs(out, q_offset + (uint64_t)row * cols) ||
                    quantize_write_bytes(out, q, sizeof(int8_t), chunk_elements) ||
                    seek_abs(out, s_offset + (uint64_t)row * num_groups * sizeof(float)) ||
                    quantize_write_bytes(out, s, sizeof(float), (size_t)chunk_rows * num_groups)) {
                free(raw);
                free(f32);
                free(q);
                free(s);
                return -1;
            }
            row += chunk_rows;
        }
        free(raw);
        free(f32);
        free(q);
        free(s);
        return seek_abs(out, end_offset);
    }
    return -1;
}

int quantize_write_tensor(quantize_ctx *ctx, FILE *out, const char *name, int rows, int cols, q_type_t type) {
    return quantize_write_tensor_entry(ctx, out, quantize_find_tensor(ctx, name), rows, cols, type);
}

int quantize_write_tensor_or_empty(quantize_ctx *ctx, FILE *out, const char *name, int rows, int cols, q_type_t type) {
    const weightmap_entry *entry = quantize_find_tensor(ctx, name);
    if (! entry) {
        return quantize_write_empty_tensor(out);
    }
    return quantize_write_tensor_entry(ctx, out, entry, rows, cols, type);
}

int quantize_write_empty_tensor(FILE *out) {
    q_type_t type = Q_TYPE_F32;
    int zero = 0;
    return (quantize_write_bytes(out, &type, sizeof(type), 1) || quantize_write_bytes(out, &zero, sizeof(zero), 1) ||
                   quantize_write_bytes(out, &zero, sizeof(zero), 1))
                   ? -1
                   : 0;
}

int quantize_write_scalar_or_default(
        quantize_ctx *ctx, FILE *out, const char *const *names, size_t n_names, float default_value) {
    const weightmap_entry *entry = quantize_find_last_tensor(ctx, names, n_names);
    if (! entry) {
        return quantize_write_bytes(out, &default_value, sizeof(default_value), 1);
    }

    size_t expected_elements = 1;
    if (validate_entry(entry, entry ? entry->tensor_name : NULL, expected_elements) || open_entry_source(ctx, entry))
        return -1;

    size_t item_size = dtype_size(entry->dtype);
    size_t per_element = item_size + sizeof(float);
    size_t chunk_elements = ctx->chunk_bytes / per_element;
    if (! chunk_elements) {
        chunk_elements = 1;
    }
    if ((chunk_elements > expected_elements) && (expected_elements > 0)) {
        chunk_elements = expected_elements;
    }

    size_t raw_bytes;
    if (checked_mul_size(chunk_elements, item_size, &raw_bytes)) {
        return -1;
    }
    void *raw = a_calloc(raw_bytes);
    float *f32 = (float *)a_calloc(chunk_elements * sizeof(float));
    if ((expected_elements > 0) && ((! raw) || (! f32))) {
        free(raw);
        free(f32);
        return -1;
    }

    int res = 0;
    if (read_f32_range(ctx, entry, 0, 1, raw, f32) || quantize_write_bytes(out, f32, sizeof(float), 1)) {
        res = -1;
    }

    free(raw);
    free(f32);
    return res;
}
