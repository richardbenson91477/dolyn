#include "dolen_common_sampler.h"
#include "dolen_common_math.h"
#include "dolen_common_mem.h"


int sample_argmax(float *probs, int n) {
    int max_i = 0;
    float max_p = probs[0];

    for (int i = 1; i < n; i++) {
        if (probs[i] > max_p) {
            max_i = i;
            max_p = probs[i];
        }
    }

    return max_i;
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float *probs, int n, float coin) {
    float cdf = 0.0f;

    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare_prob(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *)a;
    ProbIndex *b_ = (ProbIndex *)b;

    if (a_->prob > b_->prob) {
        return -1;
    }

    if (a_->prob < b_->prob) {
        return 1;
    }

    return 0;
}

int sample_top(float *probs, int n, int topk, float topp, ProbIndex *probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);

    for (int i = 0; i < n; i++) {
        if (probs[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probs[i];
            n0++;
        }
    }

    if (n0 == 0) {
        return sample_argmax(probs, n);
    }

    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);

    if (topk > 0 && n0 > topk) {
        n0 = topk;
    }

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }

    return probindex[last_idx].index;
}

int sample(Sampler *sampler, float *logits) {
    int next;

    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
#pragma omp parallel for if (sampler->vocab_size > 4096)
        for (int q = 0; q < sampler->vocab_size; q++) {
            logits[q] /= sampler->temperature;
        }

        softmax(logits, sampler->vocab_size);

        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_top(logits, sampler->vocab_size, sampler->topk, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

void build_sampler(
        Sampler *sampler, int vocab_size, float temperature, int topk, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topk = topk;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = a_calloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}

