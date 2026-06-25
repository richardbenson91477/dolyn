#include "dolen_common_qtensor.h"
#include "dolen_common_io.h"
#include "dolen_common_mem.h"


void dequantize_row(float *output, const qtensor *qt, int row_idx) {
    if ((row_idx >= qt->rows) ||
            (row_idx < 0)) {
        log_msg(stderr, "ERROR: Row index %d out of bounds (max %d)\n", row_idx, qt->rows);
        exit(EXIT_FAILURE);
    }
    int cols = qt->cols;

    if (qt->type == Q_TYPE_F32) {
        const float *row = (const float *)qt->data + (size_t)row_idx * cols;
        memcpy(output, row, cols * sizeof(float));
    }
    else if (qt->type == Q_TYPE_F16) {
        const _Float16 *row = (const _Float16 *)qt->data + (size_t)row_idx * cols;
        for (int j = 0; j < cols; j++) {
            output[j] = (float)row[j];
        }
    }
    else if (qt->type == Q_TYPE_Q8) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        const float *row_s = qt->s + row_idx * num_groups;
        const int8_t *row_q = (const int8_t *)qt->data + row_idx * cols;

        for (int g = 0; g < num_groups; g++) {
            int start = g * GROUP_SIZE;
            int end = start + GROUP_SIZE;
            if (end > cols) {
                end = cols;
            }
            float scale = row_s[g];
            for (int j = start; j < end; j++) {
                output[j] = (float)row_q[j] * scale;
            }
        }
    }
    else if (qt->type == Q_TYPE_Q6) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        const float *row_s = qt->s + row_idx * num_groups;
        const uint8_t *row_q = (const uint8_t *)qt->data + ((size_t)row_idx * cols * 3) / 4;

        for (int g = 0; g < num_groups; g++) {
            int start = g * GROUP_SIZE;
            int end = start + GROUP_SIZE;
            if (end > cols) {
                end = cols;
            }
            float scale = row_s[g];
            for (int j = start; j < end; j += 4) {
                int idx = j / 4 * 3; 
                uint8_t b0 = row_q[idx];
                uint8_t b1 = row_q[idx + 1];
                uint8_t b2 = row_q[idx + 2];

                uint8_t u0 = b0 & 0x3F;
                uint8_t u1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2);
                uint8_t u2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4);
                uint8_t u3 = (b2 >> 2) & 0x3F;

                output[j] = (float)((int32_t)((u0 ^ 0x20) - 0x20)) * scale;
                if (j + 1 < end) {
                    output[j + 1] = (float)((int32_t)((u1 ^ 0x20) - 0x20)) * scale;
                }
                if (j + 2 < end) {
                    output[j + 2] = (float)((int32_t)((u2 ^ 0x20) - 0x20)) * scale;
                }
                if (j + 3 < end) {
                    output[j + 3] = (float)((int32_t)((u3 ^ 0x20) - 0x20)) * scale;
                }
            }
        }
    }
    else if (qt->type == Q_TYPE_Q4) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        const float *row_s = qt->s + row_idx * num_groups;
        const uint8_t *row_q = (const uint8_t *)qt->data + ((size_t)row_idx * cols) / 2;

        for (int g = 0; g < num_groups; g++) {
            int start = g * GROUP_SIZE;
            int end = start + GROUP_SIZE;
            if (end > cols) {
                end = cols;
            }
            float scale = row_s[g];
            for (int j = start; j < end; j += 2) {
                int8_t p = (int8_t)row_q[j / 2];
                // FIX: Cast to int8_t BEFORE the right shift to force 8-bit sign extension
                int8_t w0 = ((int8_t)(p << 4)) >> 4; 
                output[j] = (float)w0 * scale;
                if (j + 1 < end) {
                    int8_t w1 = (int8_t)(p >> 4);
                    output[j + 1] = (float)w1 * scale;
                }
            }
        }
    }
}

void matmul_qt(float *restrict output, const float *restrict input, const qtensor *restrict qt) {
    int cols = qt->cols;
    int rows = qt->rows;

    if (qt->type == Q_TYPE_F32) {
        const float *w_data = (const float *)qt->data;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            const float *w_row = w_data + (size_t)i * cols;
#pragma omp simd reduction(+ : sum)
            for (int j = 0; j < cols; j++) {
                sum += input[j] * w_row[j];
            }
            output[i] = sum;
        }
    }
    else if (qt->type == Q_TYPE_F16) {
        const _Float16 *w_data = (const _Float16 *)qt->data;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            const _Float16 *w_row = w_data + (size_t)i * cols;
#pragma omp simd reduction(+ : sum)
            for (int j = 0; j < cols; j++) {
                sum += input[j] * (float)w_row[j];
            }
            output[i] = sum;
        }
    }
    else if (qt->type == Q_TYPE_Q8) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        int full_groups = cols / GROUP_SIZE;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            const int8_t *row_q = (const int8_t *)qt->data + (size_t)i * cols;
            const float *row_s = qt->s + (size_t)i * num_groups;

            for (int g = 0; g < full_groups; g++) {
                float group_sum = 0.0f;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : group_sum)
                for (int j = 0; j < GROUP_SIZE; j++) {
                    group_sum += input[offset + j] * (float)row_q[offset + j];
                }
                sum += group_sum * row_s[g];
            }

            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < cols) {
                float group_sum = 0.0f;
#pragma omp simd reduction(+ : group_sum)
                for (int j = rem_start; j < cols; j++) {
                    group_sum += input[j] * (float)row_q[j];
                }
                sum += group_sum * row_s[full_groups];
            }
            output[i] = sum;
        }
    }
    else if (qt->type == Q_TYPE_Q6) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        int full_groups = cols / GROUP_SIZE;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            const uint8_t *row_q = (const uint8_t *)qt->data + ((size_t)i * cols * 3) / 4;
            const float *row_s = qt->s + (size_t)i * num_groups;

            for (int g = 0; g < full_groups; g++) {
                float group_sum = 0.0f;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : group_sum)
                for (int j = 0; j < GROUP_SIZE; j += 4) {
                    int idx = (offset + j) / 4 * 3;
                    uint8_t b0 = row_q[idx];
                    uint8_t b1 = row_q[idx + 1];
                    uint8_t b2 = row_q[idx + 2];

                    uint8_t u0 = b0 & 0x3F;
                    uint8_t u1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2);
                    uint8_t u2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4);
                    uint8_t u3 = (b2 >> 2) & 0x3F;

                    int32_t w0 = (int32_t)((u0 ^ 0x20) - 0x20);
                    int32_t w1 = (int32_t)((u1 ^ 0x20) - 0x20);
                    int32_t w2 = (int32_t)((u2 ^ 0x20) - 0x20);
                    int32_t w3 = (int32_t)((u3 ^ 0x20) - 0x20);

                    group_sum += input[offset + j]     * (float)w0;
                    group_sum += input[offset + j + 1] * (float)w1;
                    group_sum += input[offset + j + 2] * (float)w2;
                    group_sum += input[offset + j + 3] * (float)w3;
                }
                sum += group_sum * row_s[g];
            }

            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < cols) {
                float group_sum = 0.0f;
                for (int j = rem_start; j < cols; j += 4) {
                    int idx = j / 4 * 3;
                    uint8_t b0 = row_q[idx];
                    uint8_t b1 = row_q[idx + 1];
                    uint8_t b2 = row_q[idx + 2];

                    uint8_t u0 = b0 & 0x3F;
                    uint8_t u1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2);
                    uint8_t u2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4);
                    uint8_t u3 = (b2 >> 2) & 0x3F;

                    int32_t w0 = (int32_t)((u0 ^ 0x20) - 0x20);
                    int32_t w1 = (int32_t)((u1 ^ 0x20) - 0x20);
                    int32_t w2 = (int32_t)((u2 ^ 0x20) - 0x20);
                    int32_t w3 = (int32_t)((u3 ^ 0x20) - 0x20);

                    group_sum += input[j] * (float)w0;
                    if (j + 1 < cols) {
                        group_sum += input[j + 1] * (float)w1;
                    }
                    if (j + 2 < cols) {
                        group_sum += input[j + 2] * (float)w2;
                    }
                    if (j + 3 < cols) {
                        group_sum += input[j + 3] * (float)w3;
                    }
                }
                sum += group_sum * row_s[full_groups];
            }
            output[i] = sum;
        }
    }
    else if (qt->type == Q_TYPE_Q4) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        int full_groups = cols / GROUP_SIZE;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            const uint8_t *row_q = (const uint8_t *)qt->data + ((size_t)i * cols) / 2;
            const float *row_s = qt->s + (size_t)i * num_groups;

            for (int g = 0; g < full_groups; g++) {
                float group_sum = 0.0f;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : group_sum)
                for (int j = 0; j < GROUP_SIZE; j += 2) {
                    int8_t p = (int8_t)row_q[(offset + j) / 2];
                    // FIX: Cast to int8_t BEFORE the right shift
                    int8_t w0 = ((int8_t)(p << 4)) >> 4; 
                    int8_t w1 = (int8_t)(p >> 4);
                    group_sum += input[offset + j] * (float)w0;
                    group_sum += input[offset + j + 1] * (float)w1;
                }
                sum += group_sum * row_s[g];
            }

            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < cols) {
                float group_sum = 0.0f;
#pragma omp simd reduction(+ : group_sum)
                for (int j = rem_start; j < cols; j += 2) {
                    int8_t p = (int8_t)row_q[j / 2];
                    // FIX: Cast to int8_t BEFORE the right shift
                    int8_t w0 = ((int8_t)(p << 4)) >> 4; 
                    group_sum += input[j] * (float)w0;
                    if (j + 1 < cols) {
                        int8_t w1 = (int8_t)(p >> 4);
                        group_sum += input[j + 1] * (float)w1;
                    }
                }
                sum += group_sum * row_s[full_groups];
            }
            output[i] = sum;
        }
    }
}

void quantize_vec(qtensor *xq, const float *x, int n) {
    int num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    xq->rows = 1;
    xq->cols = n;
    xq->type = Q_TYPE_Q8;

    int8_t *q_data = (int8_t *)xq->data;

#pragma omp parallel for schedule(static)
    for (int g = 0; g < num_groups; g++) {
        int start = g * GROUP_SIZE;
        int end = start + GROUP_SIZE < n ? start + GROUP_SIZE : n;
        float wmax = 0.0f;
        for (int i = start; i < end; i++) {
            float v = x[i] < 0.0f ? -x[i] : x[i];
            if (v > wmax) {
                wmax = v;
            }
        }
        float scale = wmax < 1e-9f ? 1e-9f : wmax / 127.0f;
        xq->s[g] = scale;
        float inv_scale = 1.0f / scale;

        for (int i = start; i < end; i++) {
            int32_t q = (int32_t)roundf(x[i] * inv_scale);
            q_data[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
        }
    }
}

void matmul_qq(float *restrict output, const qtensor *restrict x, const qtensor *restrict w) {
    int n = x->cols;
    int d = w->rows;

    const int8_t *x_q = (const int8_t *)x->data;
    const float *x_s = x->s;
    int n_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    int full_groups = n / GROUP_SIZE;

    if (w->type == Q_TYPE_F32) {
        const float *w_data = (const float *)w->data;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            const float *w_row = w_data + (size_t)i * n;
            for (int g = 0; g < full_groups; g++) {
                float group_sum = 0.0f;
                const int offset = g * GROUP_SIZE;
                float scale = x_s[g];
#pragma omp simd reduction(+ : group_sum)
                for (int k = 0; k < GROUP_SIZE; k++) {
                    group_sum += (float)x_q[offset + k] * w_row[offset + k];
                }
                val += group_sum * scale;
            }
            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < n) {
                float group_sum = 0.0f;
                float scale = x_s[full_groups];
#pragma omp simd reduction(+ : group_sum)
                for (int k = rem_start; k < n; k++) {
                    group_sum += (float)x_q[k] * w_row[k];
                }
                val += group_sum * scale;
            }
            output[i] = val;
        }
    }
    else if (w->type == Q_TYPE_F16) {
        const _Float16 *w_data = (const _Float16 *)w->data;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            const _Float16 *w_row = w_data + (size_t)i * n;
            for (int g = 0; g < full_groups; g++) {
                float group_sum = 0.0f;
                const int offset = g * GROUP_SIZE;
                float scale = x_s[g];
#pragma omp simd reduction(+ : group_sum)
                for (int k = 0; k < GROUP_SIZE; k++) {
                    group_sum += (float)x_q[offset + k] * (float)w_row[offset + k];
                }
                val += group_sum * scale;
            }
            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < n) {
                float group_sum = 0.0f;
                float scale = x_s[full_groups];
#pragma omp simd reduction(+ : group_sum)
                for (int k = rem_start; k < n; k++) {
                    group_sum += (float)x_q[k] * (float)w_row[k];
                }
                val += group_sum * scale;
            }
            output[i] = val; 
        }
    }
    else if (w->type == Q_TYPE_Q8) {
        const int8_t *w_q = (const int8_t *)w->data;
        const float *w_s = w->s;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            const int8_t *w_row = w_q + (size_t)i * n;
            const float *w_row_s = w_s + (size_t)i * n_groups;

            for (int g = 0; g < full_groups; g++) {
                int32_t acc = 0;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : acc)
                for (int k = 0; k < GROUP_SIZE; k++) {
                    acc += (int32_t)x_q[offset + k] * (int32_t)w_row[offset + k];
                }
                val += (float)acc * w_row_s[g] * x_s[g];
            }
            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < n) {
                int32_t acc = 0;
#pragma omp simd reduction(+ : acc)
                for (int k = rem_start; k < n; k++) {
                    acc += (int32_t)x_q[k] * (int32_t)w_row[k];
                }
                val += (float)acc * w_row_s[full_groups] * x_s[full_groups];
            }
            output[i] = val;
        }
    }
    else if (w->type == Q_TYPE_Q6) {
        const uint8_t *w_q = (const uint8_t *)w->data;
        const float *w_s = w->s;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            const uint8_t *w_row = w_q + ((size_t)i * n * 3) / 4;
            const float *w_row_s = w_s + (size_t)i * n_groups;

            for (int g = 0; g < full_groups; g++) {
                int32_t acc = 0;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : acc)
                for (int k = 0; k < GROUP_SIZE; k += 4) {
                    int idx = (offset + k) / 4 * 3;
                    uint8_t b0 = w_row[idx];
                    uint8_t b1 = w_row[idx + 1];
                    uint8_t b2 = w_row[idx + 2];

                    uint8_t u0 = b0 & 0x3F;
                    uint8_t u1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2);
                    uint8_t u2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4);
                    uint8_t u3 = (b2 >> 2) & 0x3F;

                    int32_t w0 = (int32_t)((u0 ^ 0x20) - 0x20);
                    int32_t w1 = (int32_t)((u1 ^ 0x20) - 0x20);
                    int32_t w2 = (int32_t)((u2 ^ 0x20) - 0x20);
                    int32_t w3 = (int32_t)((u3 ^ 0x20) - 0x20);

                    acc += (int32_t)x_q[offset + k]     * w0;
                    acc += (int32_t)x_q[offset + k + 1] * w1;
                    acc += (int32_t)x_q[offset + k + 2] * w2;
                    acc += (int32_t)x_q[offset + k + 3] * w3;
                }
                val += (float)acc * w_row_s[g] * x_s[g];
            }
            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < n) {
                int32_t acc = 0;
                for (int k = rem_start; k < n; k += 4) {
                    int idx = k / 4 * 3;
                    uint8_t b0 = w_row[idx];
                    uint8_t b1 = w_row[idx + 1];
                    uint8_t b2 = w_row[idx + 2];

                    uint8_t u0 = b0 & 0x3F;
                    uint8_t u1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2);
                    uint8_t u2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4);
                    uint8_t u3 = (b2 >> 2) & 0x3F;

                    int32_t w0 = (int32_t)((u0 ^ 0x20) - 0x20);
                    int32_t w1 = (int32_t)((u1 ^ 0x20) - 0x20);
                    int32_t w2 = (int32_t)((u2 ^ 0x20) - 0x20);
                    int32_t w3 = (int32_t)((u3 ^ 0x20) - 0x20);

                    acc += (int32_t)x_q[k] * w0;
                    if (k + 1 < n) {
                        acc += (int32_t)x_q[k + 1] * w1;
                    }
                    if (k + 2 < n) {
                        acc += (int32_t)x_q[k + 2] * w2;
                    }
                    if (k + 3 < n) {
                        acc += (int32_t)x_q[k + 3] * w3;
                    }
                }
                val += (float)acc * w_row_s[full_groups] * x_s[full_groups];
            }
            output[i] = val;
        }
    }
    else if (w->type == Q_TYPE_Q4) {
        const uint8_t *w_q = (const uint8_t *)w->data;
        const float *w_s = w->s;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            const uint8_t *w_row = w_q + ((size_t)i * n) / 2;
            const float *w_row_s = w_s + (size_t)i * n_groups;

            for (int g = 0; g < full_groups; g++) {
                int32_t acc = 0;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : acc)
                for (int k = 0; k < GROUP_SIZE; k += 2) {
                    int8_t p = (int8_t)w_row[(offset + k) / 2];
                    // FIX: Cast to int8_t BEFORE the right shift
                    int8_t w0 = ((int8_t)(p << 4)) >> 4; 
                    int8_t w1 = (int8_t)(p >> 4);
                    acc += (int32_t)x_q[offset + k] * w0;
                    acc += (int32_t)x_q[offset + k + 1] * w1;
                }
                val += (float)acc * w_row_s[g] * x_s[g];
            }
            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < n) {
                int32_t acc = 0;
#pragma omp simd reduction(+ : acc)
                for (int k = rem_start; k < n; k += 2) {
                    int8_t p = (int8_t)w_row[k / 2];
                    // FIX: Cast to int8_t BEFORE the right shift
                    int8_t w0 = ((int8_t)(p << 4)) >> 4; 
                    acc += (int32_t)x_q[k] * w0;
                    if (k + 1 < n) {
                        int8_t w1 = (int8_t)(p >> 4);
                        acc += (int32_t)x_q[k + 1] * w1;
                    }
                }
                val += (float)acc * w_row_s[full_groups] * x_s[full_groups];
            }
            output[i] = val;
        }
    }
}

void free_qt(qtensor *qt) {
    if (! qt) {
        return;
    }

    free(qt->data);
    qt->data = NULL;

    free(qt->s);
    qt->s = NULL;

    qt->rows = 0;
    qt->cols = 0;
}

void free_qt_array(qtensor *arr, int n) {
    if (! arr) {
        return;
    }
    for (int i = 0; i < n; i++) {
        free_qt(&arr[i]);
    }
    free(arr);
}

void read_qt(FILE *f, qtensor *qt) {
    fread(&qt->type, sizeof(q_type_t), 1, f);
    fread(&qt->rows, sizeof(int), 1, f);
    fread(&qt->cols, sizeof(int), 1, f);

    if ((qt->rows <= 0) ||
            (qt->cols <= 0)) {
        qt->data = NULL;
        qt->s = NULL;
        return;
    }

    int num_groups = (qt->cols + GROUP_SIZE - 1) / GROUP_SIZE;
    size_t elements = (size_t)qt->rows * qt->cols;

    if (qt->type == Q_TYPE_F32) {
        qt->data = a_calloc(elements * sizeof(float));
        qt->s = NULL;
        fread(qt->data, sizeof(float), elements, f);
    }
    else if (qt->type == Q_TYPE_F16) {
        qt->data = a_calloc(elements * sizeof(_Float16));
        qt->s = NULL;
        fread(qt->data, sizeof(_Float16), elements, f);
    }
    else if (qt->type == Q_TYPE_Q8) {
        qt->data = a_calloc(elements * sizeof(int8_t));
        qt->s = a_calloc((size_t)qt->rows * num_groups * sizeof(float));
        fread(qt->data, sizeof(int8_t), elements, f);
        fread(qt->s, sizeof(float), (size_t)qt->rows * num_groups, f);
    }
    else if (qt->type == Q_TYPE_Q6) {
        size_t data_bytes = (size_t)(((uint64_t)elements * 3 + 3) / 4);
        qt->data = a_calloc(data_bytes);
        qt->s = a_calloc((size_t)qt->rows * num_groups * sizeof(float));
        fread(qt->data, 1, data_bytes, f);
        fread(qt->s, sizeof(float), (size_t)qt->rows * num_groups, f);
    }
    else if (qt->type == Q_TYPE_Q4) {
        size_t data_bytes = (elements + 1) / 2;
        qt->data = a_calloc(data_bytes);
        qt->s = a_calloc((size_t)qt->rows * num_groups * sizeof(float));
        fread(qt->data, 1, data_bytes, f);
        fread(qt->s, sizeof(float), (size_t)qt->rows * num_groups, f);
    }
}

