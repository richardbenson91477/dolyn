// Math Utilities
#ifndef DOLEN_COMMON_MATH_H
#define DOLEN_COMMON_MATH_H

#include <math.h>
#include <stdbool.h>


void rmsnorm(float *_o, float *_x, float *_weight, int size, float eps);

void rmsnorm_gemma(float *_o, float *_x, float *_weight, int size, float eps);

void rmsnorm_gated(float *_o, float *_x, float *_gate, float *_weight, int heads_n, int d_v, float eps);

void rmsnorm_g4(float *_o, float *_x, float *_weight, int size, float eps, bool with_scale_);

float gelu(float x);

void softmax(float *_x, int size);

float silu(float x);

float sigmoid(float x);

float softplus(float x);

void l2norm(float *_x, int size);

float matmul_scalar(float *_x, float *_w, int n);


#endif // DOLEN_COMMON_MATH_H

