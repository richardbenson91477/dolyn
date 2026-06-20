#include "dolen_common_math.h"

void rmsnorm(float *o, float *x, float *weight, int size, float eps) {
    float ss = 0.0f;

#pragma omp simd reduction(+ : ss)
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss = 1.0f / sqrtf((ss / size) + eps);

#pragma omp simd
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void rmsnorm_gemma(float *o, float *x, float *weight, int size, float eps) {
    float ss = 0.0f;

#pragma omp simd reduction(+ : ss)
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss = 1.0f / sqrtf(ss / size + eps);

#pragma omp simd
    for (int j = 0; j < size; j++) {
        o[j] = (1.0f + weight[j]) * (ss * x[j]);
    }
}

void rmsnorm_gated(float *o, float *x, float *gate, float *weight, int n_heads, int d_v, float eps) {
#pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        float *x_h = x + h * d_v;
        float *gate_h = gate + h * d_v;
        float *o_h = o + h * d_v;

        float ss = 0.0f;
#pragma omp simd reduction(+ : ss)
        for (int j = 0; j < d_v; j++) {
            ss += x_h[j] * x_h[j];
        }

        ss /= d_v;
        ss += eps;
        ss = 1.0f / sqrtf(ss);

#pragma omp simd
        for (int j = 0; j < d_v; j++) {
            float x_norm = ss * x_h[j];
            o_h[j] = weight[j] * x_norm * silu(gate_h[j]);
        }
    }
}

void rmsnorm_g4u(float *o, float *x, float *weight, int size, float eps, int with_scale) {
    float ss = 0.0f;

#pragma omp simd reduction(+ : ss)
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss = 1.0f / sqrtf(ss / size + eps);

    if (with_scale &&
            weight) {
#pragma omp simd
        for (int j = 0; j < size; j++) {
            o[j] = x[j] * ss * weight[j];
        }
    } else {
#pragma omp simd
        for (int j = 0; j < size; j++) {
            o[j] = x[j] * ss;
        }
    }
}

float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

void softmax(float *x, int size) {
    float max_val = x[0];
#pragma omp simd reduction(max : max_val)
    for (int i = 1; i < size; i++) {
        max_val = fmaxf(max_val, x[i]);
    }

    float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    if (sum < 1e-10f) {
        float uniform = 1.0f / size;
        for (int i = 0; i < size; i++) {
            x[i] = uniform;
        }
        return;
    }

    float inv_sum = 1.0f / sum;

#pragma omp simd
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
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

void l2norm(float *x, int size) {
    float ss = 0.0f;
#pragma omp simd reduction(+ : ss)
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss + 1e-6f);
#pragma omp simd
    for (int i = 0; i < size; i++) {
        x[i] *= ss;
    }
}

float matmul_scalar(float *x, float *w, int n) {
    float val = 0.0f;

#pragma omp simd reduction(+ : val)
    for (int i = 0; i < n; i++) {
        val += w[i] * x[i];
    }
    return val;
}

