#include "dolen_common_math.h"

void rmsnorm(float *_o, float *_x, float *_weight, int size, float eps) {
    float ss = 0.0f;

#pragma omp simd reduction(+ : ss)
    for (int j = 0; j < size; j++) {
        ss += _x[j] * _x[j];
    }
    ss = 1.0f / sqrtf((ss / size) + eps);

#pragma omp simd
    for (int j = 0; j < size; j++) {
        _o[j] = _weight[j] * (ss * _x[j]);
    }
}

void rmsnorm_gemma(float *_o, float *_x, float *_weight, int size, float eps) {
    float ss = 0.0f;

#pragma omp simd reduction(+ : ss)
    for (int j = 0; j < size; j++) {
        ss += _x[j] * _x[j];
    }
    ss = 1.0f / sqrtf(ss / size + eps);

#pragma omp simd
    for (int j = 0; j < size; j++) {
        _o[j] = (1.0f + _weight[j]) * (ss * _x[j]);
    }
}

void rmsnorm_gated(float *_o, float *_x, float *_gate, float *_weight, int heads_n, int d_v, float eps) {
#pragma omp parallel for
    for (int h = 0; h < heads_n; h++) {
        float *_x_h = _x + h * d_v;
        float *_gate_h = _gate + h * d_v;
        float *_o_h = _o + h * d_v;

        float ss = 0.0f;
#pragma omp simd reduction(+ : ss)
        for (int j = 0; j < d_v; j++) {
            ss += _x_h[j] * _x_h[j];
        }

        ss /= d_v;
        ss += eps;
        ss = 1.0f / sqrtf(ss);

#pragma omp simd
        for (int j = 0; j < d_v; j++) {
            float x_norm = ss * _x_h[j];
            _o_h[j] = _weight[j] * x_norm * silu(_gate_h[j]);
        }
    }
}

void rmsnorm_g4(float *_o, float *_x, float *_weight, int size, float eps, bool with_scale_) {
    float ss = 0.0f;

#pragma omp simd reduction(+ : ss)
    for (int j = 0; j < size; j++) {
        ss += _x[j] * _x[j];
    }
    ss = 1.0f / sqrtf(ss / size + eps);

    if (with_scale_ &&
            _weight) {
#pragma omp simd
        for (int j = 0; j < size; j++) {
            _o[j] = _x[j] * ss * _weight[j];
        }
    }
    else {
#pragma omp simd
        for (int j = 0; j < size; j++) {
            _o[j] = _x[j] * ss;
        }
    }
}

float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

void softmax(float *_x, int size) {
    float max_val = _x[0];
#pragma omp simd reduction(max : max_val)
    for (int i = 1; i < size; i++) {
        max_val = fmaxf(max_val, _x[i]);
    }

    float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
    for (int i = 0; i < size; i++) {
        _x[i] = expf(_x[i] - max_val);
        sum += _x[i];
    }

    if (sum < 1e-10f) {
        float uniform = 1.0f / size;
        for (int i = 0; i < size; i++) {
            _x[i] = uniform;
        }
        return;
    }

    float inv_sum = 1.0f / sum;

#pragma omp simd
    for (int i = 0; i < size; i++) {
        _x[i] *= inv_sum;
    }
}

float silu(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float softplus(float x) {
    return logf(1.0f + expf(x));
}

void l2norm(float *_x, int size) {
    float ss = 0.0f;
#pragma omp simd reduction(+ : ss)
    for (int i = 0; i < size; i++) {
        ss += _x[i] * _x[i];
    }
    ss = 1.0f / sqrtf(ss + 1e-6f);
#pragma omp simd
    for (int i = 0; i < size; i++) {
        _x[i] *= ss;
    }
}

float matmul_scalar(float *_x, float *_w, int n) {
    float val = 0.0f;

#pragma omp simd reduction(+ : val)
    for (int i = 0; i < n; i++) {
        val += _w[i] * _x[i];
    }
    return val;
}

