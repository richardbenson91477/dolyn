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
    } else if (qt->type == Q_TYPE_F16) {
        const _Float16 *row = (const _Float16 *)qt->data + (size_t)row_idx * cols;
        for (int j = 0; j < cols; j++) {
            output[j] = (float)row[j];
        }
    } else if (qt->type == Q_TYPE_Q8) {
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
    } else if (qt->type == Q_TYPE_F16) {
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
    } else if (qt->type == Q_TYPE_Q8) {
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
}

void quantize_vec(qtensor *xq, const float *x, int n) {
    int num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    xq->rows = 1;
    xq->cols = n;
    xq->type = Q_TYPE_Q8;

#pragma omp parallel for schedule(static) if (num_groups > 32)
    for (int g = 0; g < num_groups; g++) {
        int start = g * GROUP_SIZE;
        int end = start + GROUP_SIZE < n ? start + GROUP_SIZE : n;
        float wmax = 0.0f;
        for (int i = start; i < end; i++) {
            float v = fabsf(x[i]);
            if (v > wmax) {
                wmax = v;
            }
        }
        float scale = wmax < 1e-9f ? 1e-9f : wmax / 127.0f;
        xq->s[g] = scale;
        float inv_scale = 1.0f / scale;
        int8_t *q_data = (int8_t *)xq->data;
        for (int i = start; i < end; i++) {
            q_data[i] = (int8_t)roundf(x[i] * inv_scale);
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
    } else if (w->type == Q_TYPE_F16) {
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
    } else if (w->type == Q_TYPE_Q8) {
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
    } else if (qt->type == Q_TYPE_F16) {
        qt->data = a_calloc(elements * sizeof(_Float16));
        qt->s = NULL;
        fread(qt->data, sizeof(_Float16), elements, f);
    } else if (qt->type == Q_TYPE_Q8) {
        qt->data = a_calloc(elements * sizeof(int8_t));
        qt->s = a_calloc((size_t)qt->rows * num_groups * sizeof(float));
        fread(qt->data, sizeof(int8_t), elements, f);
        fread(qt->s, sizeof(float), (size_t)qt->rows * num_groups, f);
    }
}

void write_qt(FILE *f, qtensor *qt) {
    fwrite(&qt->type, sizeof(q_type_t), 1, f);
    fwrite(&qt->rows, sizeof(int), 1, f);
    fwrite(&qt->cols, sizeof(int), 1, f);

    if ((qt->rows <= 0) ||
            (qt->cols <= 0)) {
        return;
    }

    int num_groups = (qt->cols + GROUP_SIZE - 1) / GROUP_SIZE;
    size_t elements = (size_t)qt->rows * qt->cols;

    if (qt->type == Q_TYPE_F32) {
        fwrite(qt->data, sizeof(float), elements, f);
    } else if (qt->type == Q_TYPE_F16) {
        fwrite(qt->data, sizeof(_Float16), elements, f);
    } else if (qt->type == Q_TYPE_Q8) {
        fwrite(qt->data, sizeof(int8_t), elements, f);
        fwrite(qt->s, sizeof(float), (size_t)qt->rows * num_groups, f);
    }
}

