#include "dolen_common_qtensor.h"
#include "dolen_common_io.h"
#include "dolen_common_mem.h"


void dequantize_row(float *_output, const qtensor *_qt, int row_idx) {
    if ((row_idx >= _qt->rows) ||
            (row_idx < 0)) {
        log_msg(stderr, "ERROR: Row index %d out of bounds (max %d)\n", row_idx, _qt->rows);
        exit(EXIT_FAILURE);
    }
    int cols = _qt->cols;

    if (_qt->type == Q_TYPE_F32) {
        const float *_row = (const float *)_qt->_data + (size_t)row_idx * cols;
        memcpy(_output, _row, cols * sizeof(float));
    }
    else if (_qt->type == Q_TYPE_F16) {
        const _Float16 *_row = (const _Float16 *)_qt->_data + (size_t)row_idx * cols;
        for (int j = 0; j < cols; j++) {
            _output[j] = (float)_row[j];
        }
    }
    else if (_qt->type == Q_TYPE_Q8) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        const float *_row_s = _qt->_scales + row_idx * num_groups;
        const int8_t *_row_q = (const int8_t *)_qt->_data + row_idx * cols;

        for (int g = 0; g < num_groups; g++) {
            int start = g * GROUP_SIZE;
            int end = start + GROUP_SIZE;
            if (end > cols) {
                end = cols;
            }
            float scale = _row_s[g];
            for (int j = start; j < end; j++) {
                _output[j] = (float)_row_q[j] * scale;
            }
        }
    }
    else if (_qt->type == Q_TYPE_Q6) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        const float *_row_s = _qt->_scales + row_idx * num_groups;
        const uint8_t *_row_q = (const uint8_t *)_qt->_data + ((size_t)row_idx * cols * 3) / 4;

        for (int g = 0; g < num_groups; g++) {
            int start = g * GROUP_SIZE;
            int end = start + GROUP_SIZE;
            if (end > cols) {
                end = cols;
            }
            float scale = _row_s[g];
            for (int j = start; j < end; j += 4) {
                int idx = j / 4 * 3; 
                uint8_t b0 = _row_q[idx];
                uint8_t b1 = _row_q[idx + 1];
                uint8_t b2 = _row_q[idx + 2];

                uint8_t u0 = b0 & 0x3F;
                uint8_t u1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2);
                uint8_t u2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4);
                uint8_t u3 = (b2 >> 2) & 0x3F;

                _output[j] = (float)((int32_t)((u0 ^ 0x20) - 0x20)) * scale;
                if (j + 1 < end) {
                    _output[j + 1] = (float)((int32_t)((u1 ^ 0x20) - 0x20)) * scale;
                }
                if (j + 2 < end) {
                    _output[j + 2] = (float)((int32_t)((u2 ^ 0x20) - 0x20)) * scale;
                }
                if (j + 3 < end) {
                    _output[j + 3] = (float)((int32_t)((u3 ^ 0x20) - 0x20)) * scale;
                }
            }
        }
    }
    else if (_qt->type == Q_TYPE_Q4) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        const float *_row_s = _qt->_scales + row_idx * num_groups;
        const uint8_t *_row_q = (const uint8_t *)_qt->_data + ((size_t)row_idx * cols) / 2;

        for (int g = 0; g < num_groups; g++) {
            int start = g * GROUP_SIZE;
            int end = start + GROUP_SIZE;
            if (end > cols) {
                end = cols;
            }
            float scale = _row_s[g];
            for (int j = start; j < end; j += 2) {
                int8_t p = (int8_t)_row_q[j / 2];
                // FIX: Cast to int8_t BEFORE the right shift to force 8-bit sign extension
                int8_t w0 = ((int8_t)(p << 4)) >> 4; 
                _output[j] = (float)w0 * scale;
                if (j + 1 < end) {
                    int8_t w1 = (int8_t)(p >> 4);
                    _output[j + 1] = (float)w1 * scale;
                }
            }
        }
    }
}

void matmul_qt(float *restrict _output, const float *restrict _input, const qtensor *restrict _qt) {
    int cols = _qt->cols;
    int rows = _qt->rows;

    if (_qt->type == Q_TYPE_F32) {
        const float *_w_data = (const float *)_qt->_data;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            const float *_w_row = _w_data + (size_t)i * cols;
#pragma omp simd reduction(+ : sum)
            for (int j = 0; j < cols; j++) {
                sum += _input[j] * _w_row[j];
            }
            _output[i] = sum;
        }
    }
    else if (_qt->type == Q_TYPE_F16) {
        const _Float16 *_w_data = (const _Float16 *)_qt->_data;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            const _Float16 *_w_row = _w_data + (size_t)i * cols;
#pragma omp simd reduction(+ : sum)
            for (int j = 0; j < cols; j++) {
                sum += _input[j] * (float)_w_row[j];
            }
            _output[i] = sum;
        }
    }
    else if (_qt->type == Q_TYPE_Q8) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        int full_groups = cols / GROUP_SIZE;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            const int8_t *_row_q = (const int8_t *)_qt->_data + (size_t)i * cols;
            const float *_row_s = _qt->_scales + (size_t)i * num_groups;

            for (int g = 0; g < full_groups; g++) {
                float group_sum = 0.0f;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : group_sum)
                for (int j = 0; j < GROUP_SIZE; j++) {
                    group_sum += _input[offset + j] * (float)_row_q[offset + j];
                }
                sum += group_sum * _row_s[g];
            }

            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < cols) {
                float group_sum = 0.0f;
#pragma omp simd reduction(+ : group_sum)
                for (int j = rem_start; j < cols; j++) {
                    group_sum += _input[j] * (float)_row_q[j];
                }
                sum += group_sum * _row_s[full_groups];
            }
            _output[i] = sum;
        }
    }
    else if (_qt->type == Q_TYPE_Q6) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        int full_groups = cols / GROUP_SIZE;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            const uint8_t *_row_q = (const uint8_t *)_qt->_data + ((size_t)i * cols * 3) / 4;
            const float *_row_s = _qt->_scales + (size_t)i * num_groups;

            for (int g = 0; g < full_groups; g++) {
                float group_sum = 0.0f;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : group_sum)
                for (int j = 0; j < GROUP_SIZE; j += 4) {
                    int idx = (offset + j) / 4 * 3;
                    uint8_t b0 = _row_q[idx];
                    uint8_t b1 = _row_q[idx + 1];
                    uint8_t b2 = _row_q[idx + 2];

                    uint8_t u0 = b0 & 0x3F;
                    uint8_t u1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2);
                    uint8_t u2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4);
                    uint8_t u3 = (b2 >> 2) & 0x3F;

                    int32_t w0 = (int32_t)((u0 ^ 0x20) - 0x20);
                    int32_t w1 = (int32_t)((u1 ^ 0x20) - 0x20);
                    int32_t w2 = (int32_t)((u2 ^ 0x20) - 0x20);
                    int32_t w3 = (int32_t)((u3 ^ 0x20) - 0x20);

                    group_sum += _input[offset + j]     * (float)w0;
                    group_sum += _input[offset + j + 1] * (float)w1;
                    group_sum += _input[offset + j + 2] * (float)w2;
                    group_sum += _input[offset + j + 3] * (float)w3;
                }
                sum += group_sum * _row_s[g];
            }

            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < cols) {
                float group_sum = 0.0f;
                for (int j = rem_start; j < cols; j += 4) {
                    int idx = j / 4 * 3;
                    uint8_t b0 = _row_q[idx];
                    uint8_t b1 = _row_q[idx + 1];
                    uint8_t b2 = _row_q[idx + 2];

                    uint8_t u0 = b0 & 0x3F;
                    uint8_t u1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2);
                    uint8_t u2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4);
                    uint8_t u3 = (b2 >> 2) & 0x3F;

                    int32_t w0 = (int32_t)((u0 ^ 0x20) - 0x20);
                    int32_t w1 = (int32_t)((u1 ^ 0x20) - 0x20);
                    int32_t w2 = (int32_t)((u2 ^ 0x20) - 0x20);
                    int32_t w3 = (int32_t)((u3 ^ 0x20) - 0x20);

                    group_sum += _input[j] * (float)w0;
                    if (j + 1 < cols) {
                        group_sum += _input[j + 1] * (float)w1;
                    }
                    if (j + 2 < cols) {
                        group_sum += _input[j + 2] * (float)w2;
                    }
                    if (j + 3 < cols) {
                        group_sum += _input[j + 3] * (float)w3;
                    }
                }
                sum += group_sum * _row_s[full_groups];
            }
            _output[i] = sum;
        }
    }
    else if (_qt->type == Q_TYPE_Q4) {
        int num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE;
        int full_groups = cols / GROUP_SIZE;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            const uint8_t *_row_q = (const uint8_t *)_qt->_data + ((size_t)i * cols) / 2;
            const float *_row_s = _qt->_scales + (size_t)i * num_groups;

            for (int g = 0; g < full_groups; g++) {
                float group_sum = 0.0f;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : group_sum)
                for (int j = 0; j < GROUP_SIZE; j += 2) {
                    int8_t p = (int8_t)_row_q[(offset + j) / 2];
                    // FIX: Cast to int8_t BEFORE the right shift
                    int8_t w0 = ((int8_t)(p << 4)) >> 4; 
                    int8_t w1 = (int8_t)(p >> 4);
                    group_sum += _input[offset + j] * (float)w0;
                    group_sum += _input[offset + j + 1] * (float)w1;
                }
                sum += group_sum * _row_s[g];
            }

            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < cols) {
                float group_sum = 0.0f;
#pragma omp simd reduction(+ : group_sum)
                for (int j = rem_start; j < cols; j += 2) {
                    int8_t p = (int8_t)_row_q[j / 2];
                    // FIX: Cast to int8_t BEFORE the right shift
                    int8_t w0 = ((int8_t)(p << 4)) >> 4; 
                    group_sum += _input[j] * (float)w0;
                    if (j + 1 < cols) {
                        int8_t w1 = (int8_t)(p >> 4);
                        group_sum += _input[j + 1] * (float)w1;
                    }
                }
                sum += group_sum * _row_s[full_groups];
            }
            _output[i] = sum;
        }
    }
}

void quantize_vec(qtensor *_xq, const float *_x, int n) {
    int num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    _xq->rows = 1;
    _xq->cols = n;
    _xq->type = Q_TYPE_Q8;

    int8_t *_q_data = (int8_t *)_xq->_data;

#pragma omp parallel for schedule(static)
    for (int g = 0; g < num_groups; g++) {
        int start = g * GROUP_SIZE;
        int end = start + GROUP_SIZE < n ? start + GROUP_SIZE : n;
        float wmax = 0.0f;
        for (int i = start; i < end; i++) {
            float v = _x[i] < 0.0f ? -_x[i] : _x[i];
            if (v > wmax) {
                wmax = v;
            }
        }
        float scale = wmax < 1e-9f ? 1e-9f : wmax / 127.0f;
        _xq->_scales[g] = scale;
        float inv_scale = 1.0f / scale;

        for (int i = start; i < end; i++) {
            int32_t q = (int32_t)roundf(_x[i] * inv_scale);
            _q_data[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
        }
    }
}

void matmul_qq(float *restrict _output, const qtensor *restrict _x, const qtensor *restrict _w) {
    int n = _x->cols;
    int d = _w->rows;

    const int8_t *_x_q = (const int8_t *)_x->_data;
    const float *_x_s = _x->_scales;
    int n_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    int full_groups = n / GROUP_SIZE;

    if (_w->type == Q_TYPE_F32) {
        const float *_w_data = (const float *)_w->_data;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            const float *_w_row = _w_data + (size_t)i * n;
            for (int g = 0; g < full_groups; g++) {
                float group_sum = 0.0f;
                const int offset = g * GROUP_SIZE;
                float scale = _x_s[g];
#pragma omp simd reduction(+ : group_sum)
                for (int k = 0; k < GROUP_SIZE; k++) {
                    group_sum += (float)_x_q[offset + k] * _w_row[offset + k];
                }
                val += group_sum * scale;
            }
            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < n) {
                float group_sum = 0.0f;
                float scale = _x_s[full_groups];
#pragma omp simd reduction(+ : group_sum)
                for (int k = rem_start; k < n; k++) {
                    group_sum += (float)_x_q[k] * _w_row[k];
                }
                val += group_sum * scale;
            }
            _output[i] = val;
        }
    }
    else if (_w->type == Q_TYPE_F16) {
        const _Float16 *_w_data = (const _Float16 *)_w->_data;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            const _Float16 *_w_row = _w_data + (size_t)i * n;
            for (int g = 0; g < full_groups; g++) {
                float group_sum = 0.0f;
                const int offset = g * GROUP_SIZE;
                float scale = _x_s[g];
#pragma omp simd reduction(+ : group_sum)
                for (int k = 0; k < GROUP_SIZE; k++) {
                    group_sum += (float)_x_q[offset + k] * (float)_w_row[offset + k];
                }
                val += group_sum * scale;
            }
            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < n) {
                float group_sum = 0.0f;
                float scale = _x_s[full_groups];
#pragma omp simd reduction(+ : group_sum)
                for (int k = rem_start; k < n; k++) {
                    group_sum += (float)_x_q[k] * (float)_w_row[k];
                }
                val += group_sum * scale;
            }
            _output[i] = val; 
        }
    }
    else if (_w->type == Q_TYPE_Q8) {
        const int8_t *_w_q = (const int8_t *)_w->_data;
        const float *_w_s = _w->_scales;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            const int8_t *_w_row = _w_q + (size_t)i * n;
            const float *_w_row_s = _w_s + (size_t)i * n_groups;

            for (int g = 0; g < full_groups; g++) {
                int32_t acc = 0;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : acc)
                for (int k = 0; k < GROUP_SIZE; k++) {
                    acc += (int32_t)_x_q[offset + k] * (int32_t)_w_row[offset + k];
                }
                val += (float)acc * _w_row_s[g] * _x_s[g];
            }
            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < n) {
                int32_t acc = 0;
#pragma omp simd reduction(+ : acc)
                for (int k = rem_start; k < n; k++) {
                    acc += (int32_t)_x_q[k] * (int32_t)_w_row[k];
                }
                val += (float)acc * _w_row_s[full_groups] * _x_s[full_groups];
            }
            _output[i] = val;
        }
    }
    else if (_w->type == Q_TYPE_Q6) {
        const uint8_t *_w_q = (const uint8_t *)_w->_data;
        const float *_w_s = _w->_scales;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            const uint8_t *_w_row = _w_q + ((size_t)i * n * 3) / 4;
            const float *_w_row_s = _w_s + (size_t)i * n_groups;

            for (int g = 0; g < full_groups; g++) {
                int32_t acc = 0;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : acc)
                for (int k = 0; k < GROUP_SIZE; k += 4) {
                    int idx = (offset + k) / 4 * 3;
                    uint8_t b0 = _w_row[idx];
                    uint8_t b1 = _w_row[idx + 1];
                    uint8_t b2 = _w_row[idx + 2];

                    uint8_t u0 = b0 & 0x3F;
                    uint8_t u1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2);
                    uint8_t u2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4);
                    uint8_t u3 = (b2 >> 2) & 0x3F;

                    int32_t w0 = (int32_t)((u0 ^ 0x20) - 0x20);
                    int32_t w1 = (int32_t)((u1 ^ 0x20) - 0x20);
                    int32_t w2 = (int32_t)((u2 ^ 0x20) - 0x20);
                    int32_t w3 = (int32_t)((u3 ^ 0x20) - 0x20);

                    acc += (int32_t)_x_q[offset + k]     * w0;
                    acc += (int32_t)_x_q[offset + k + 1] * w1;
                    acc += (int32_t)_x_q[offset + k + 2] * w2;
                    acc += (int32_t)_x_q[offset + k + 3] * w3;
                }
                val += (float)acc * _w_row_s[g] * _x_s[g];
            }
            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < n) {
                int32_t acc = 0;
                for (int k = rem_start; k < n; k += 4) {
                    int idx = k / 4 * 3;
                    uint8_t b0 = _w_row[idx];
                    uint8_t b1 = _w_row[idx + 1];
                    uint8_t b2 = _w_row[idx + 2];

                    uint8_t u0 = b0 & 0x3F;
                    uint8_t u1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2);
                    uint8_t u2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4);
                    uint8_t u3 = (b2 >> 2) & 0x3F;

                    int32_t w0 = (int32_t)((u0 ^ 0x20) - 0x20);
                    int32_t w1 = (int32_t)((u1 ^ 0x20) - 0x20);
                    int32_t w2 = (int32_t)((u2 ^ 0x20) - 0x20);
                    int32_t w3 = (int32_t)((u3 ^ 0x20) - 0x20);

                    acc += (int32_t)_x_q[k] * w0;
                    if (k + 1 < n) {
                        acc += (int32_t)_x_q[k + 1] * w1;
                    }
                    if (k + 2 < n) {
                        acc += (int32_t)_x_q[k + 2] * w2;
                    }
                    if (k + 3 < n) {
                        acc += (int32_t)_x_q[k + 3] * w3;
                    }
                }
                val += (float)acc * _w_row_s[full_groups] * _x_s[full_groups];
            }
            _output[i] = val;
        }
    }
    else if (_w->type == Q_TYPE_Q4) {
        const uint8_t *_w_q = (const uint8_t *)_w->_data;
        const float *_w_s = _w->_scales;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            const uint8_t *_w_row = _w_q + ((size_t)i * n) / 2;
            const float *_w_row_s = _w_s + (size_t)i * n_groups;

            for (int g = 0; g < full_groups; g++) {
                int32_t acc = 0;
                const int offset = g * GROUP_SIZE;
#pragma omp simd reduction(+ : acc)
                for (int k = 0; k < GROUP_SIZE; k += 2) {
                    int8_t p = (int8_t)_w_row[(offset + k) / 2];
                    // FIX: Cast to int8_t BEFORE the right shift
                    int8_t w0 = ((int8_t)(p << 4)) >> 4; 
                    int8_t w1 = (int8_t)(p >> 4);
                    acc += (int32_t)_x_q[offset + k] * w0;
                    acc += (int32_t)_x_q[offset + k + 1] * w1;
                }
                val += (float)acc * _w_row_s[g] * _x_s[g];
            }
            int rem_start = full_groups * GROUP_SIZE;
            if (rem_start < n) {
                int32_t acc = 0;
#pragma omp simd reduction(+ : acc)
                for (int k = rem_start; k < n; k += 2) {
                    int8_t p = (int8_t)_w_row[k / 2];
                    // FIX: Cast to int8_t BEFORE the right shift
                    int8_t w0 = ((int8_t)(p << 4)) >> 4; 
                    acc += (int32_t)_x_q[k] * w0;
                    if (k + 1 < n) {
                        int8_t w1 = (int8_t)(p >> 4);
                        acc += (int32_t)_x_q[k + 1] * w1;
                    }
                }
                val += (float)acc * _w_row_s[full_groups] * _x_s[full_groups];
            }
            _output[i] = val;
        }
    }
}

void free_qt(qtensor *_qt) {
    if (! _qt) {
        return;
    }

    free(_qt->_data);
    _qt->_data = NULL;

    free(_qt->_scales);
    _qt->_scales = NULL;

    _qt->rows = 0;
    _qt->cols = 0;
}

void free_qt_array(qtensor *_qt_arr, int arr_n) {
    if (! arr_n) {
        return;
    }
    for (int i = 0; i < arr_n; i++) {
        free_qt(&_qt_arr[i]);
    }
    free(_qt_arr);
}

void read_qt(FILE *_file, qtensor *_qt) {
    fread(&_qt->type, sizeof(q_type_t), 1, _file);
    fread(&_qt->rows, sizeof(int), 1, _file);
    fread(&_qt->cols, sizeof(int), 1, _file);

    if ((_qt->rows <= 0) ||
            (_qt->cols <= 0)) {
        _qt->_data = NULL;
        _qt->_scales = NULL;
        return;
    }

    int num_groups = (_qt->cols + GROUP_SIZE - 1) / GROUP_SIZE;
    size_t elements = (size_t)_qt->rows * _qt->cols;

    if (_qt->type == Q_TYPE_F32) {
        _qt->_data = a_calloc(elements * sizeof(float));
        _qt->_scales = NULL;
        fread(_qt->_data, sizeof(float), elements, _file);
    }
    else if (_qt->type == Q_TYPE_F16) {
        _qt->_data = a_calloc(elements * sizeof(_Float16));
        _qt->_scales = NULL;
        fread(_qt->_data, sizeof(_Float16), elements, _file);
    }
    else if (_qt->type == Q_TYPE_Q8) {
        _qt->_data = a_calloc(elements * sizeof(int8_t));
        _qt->_scales = a_calloc((size_t)_qt->rows * num_groups * sizeof(float));
        fread(_qt->_data, sizeof(int8_t), elements, _file);
        fread(_qt->_scales, sizeof(float), (size_t)_qt->rows * num_groups, _file);
    }
    else if (_qt->type == Q_TYPE_Q6) {
        size_t data_bytes = (size_t)(((uint64_t)elements * 3 + 3) / 4);
        _qt->_data = a_calloc(data_bytes);
        _qt->_scales = a_calloc((size_t)_qt->rows * num_groups * sizeof(float));
        fread(_qt->_data, 1, data_bytes, _file);
        fread(_qt->_scales, sizeof(float), (size_t)_qt->rows * num_groups, _file);
    }
    else if (_qt->type == Q_TYPE_Q4) {
        size_t data_bytes = (elements + 1) / 2;
        _qt->_data = a_calloc(data_bytes);
        _qt->_scales = a_calloc((size_t)_qt->rows * num_groups * sizeof(float));
        fread(_qt->_data, 1, data_bytes, _file);
        fread(_qt->_scales, sizeof(float), (size_t)_qt->rows * num_groups, _file);
    }
}

