// Sampler
#ifndef DOLEN_COMMON_SAMPLER_H
#define DOLEN_COMMON_SAMPLER_H

#include <stdint.h>


typedef struct {
    float prob;
    int32_t index;
} prob_index;

typedef struct {
    int32_t vocab_size;
    prob_index *_prob_index;
    float temp;
    int32_t top_k;
    float top_p;
    uint64_t rng_state;
} sampler;


int32_t sample_argmax(float *_probs, int32_t n);

uint32_t random_u32(uint64_t *_state);

float random_f32(uint64_t *_state);

int32_t sample_mult(float *_probs, int32_t n, float coin);

int32_t compare_prob(const void *_a, const void *_b);

int32_t sample_top(float *_probs, int32_t n, int32_t top_k, float top_p, prob_index *_prob_index, float coin);

int32_t sample(sampler *_sampler, float *_logits);

void build_sampler(sampler *_sampler, int32_t vocab_size, float temp, int32_t top_k, float top_p, uint64_t rng_seed);

void free_sampler(sampler *_sampler);


#endif // DOLEN_COMMON_SAMPLER_H

