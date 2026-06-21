// Sampler
#ifndef DOLEN_COMMON_SAMPLER_H
#define DOLEN_COMMON_SAMPLER_H


typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    int topk;
    float topp;
    unsigned long long rng_state;
} Sampler;


int sample_argmax(float *probs, int n);

unsigned int random_u32(unsigned long long *state);

float random_f32(unsigned long long *state);

int sample_mult(float *probs, int n, float coin);

int compare_prob(const void *a, const void *b);

int sample_top(float *probs, int n, int topk, float topp, ProbIndex *probindex, float coin);

int sample(Sampler *sampler, float *logits);

void build_sampler(Sampler *sampler, int vocab_size, float temperature, int topk, float topp, unsigned long long rng_seed);

void free_sampler(Sampler *sampler);


#endif // DOLEN_COMMON_SAMPLER_H

