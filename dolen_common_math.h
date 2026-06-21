// Math Utilities
#ifndef DOLEN_COMMON_MATH_H
#define DOLEN_COMMON_MATH_H

#include <math.h>


void rmsnorm(float *o, float *x, float *weight, int size, float eps);

void rmsnorm_gemma(float *o, float *x, float *weight, int size, float eps);

void rmsnorm_gated(float *o, float *x, float *gate, float *weight, int n_heads, int d_v, float eps);

void rmsnorm_g4u(float *o, float *x, float *weight, int size, float eps, int with_scale);

float gelu(float x);

void softmax(float *x, int size);

float silu(float x);

float sigmoid(float x);

float softplus(float x);

void l2norm(float *x, int size);

float matmul_scalar(float *x, float *w, int n);


#endif // DOLEN_COMMON_MATH_H

