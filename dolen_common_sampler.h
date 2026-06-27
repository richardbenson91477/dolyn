// Sampler
#ifndef DOLEN_COMMON_SAMPLER_H
#define DOLEN_COMMON_SAMPLER_H


typedef struct {
    float prob;
    int index;
} prob_index;

typedef struct {
    int vocab_size;
    prob_index *_prob_index;
    float temp;
    int top_k;
    float top_p;
    unsigned long long rng_state;
} sampler;


int sample_argmax(float *_probs, int n);

unsigned int random_u32(unsigned long long *_state);

float random_f32(unsigned long long *_state);

int sample_mult(float *_probs, int n, float coin);

int compare_prob(const void *_a, const void *_b);

int sample_top(float *_probs, int n, int top_k, float top_p, prob_index *_prob_index, float coin);

int sample(sampler *_sampler, float *_logits);

void build_sampler(sampler *_sampler, int vocab_size, float temp, int top_k, float top_p,
        unsigned long long rng_seed);

void free_sampler(sampler *_sampler);


#endif // DOLEN_COMMON_SAMPLER_H

