#include "dolen_common_sampler.h"
#include "dolen_common_math.h"
#include "dolen_common_mem.h"


int sample_argmax(float *_probs, int n) {
    int max_i = 0;
    float max_p = _probs[0];

    for (int i = 1; i < n; i++) {
        if (_probs[i] > max_p) {
            max_i = i;
            max_p = _probs[i];
        }
    }

    return max_i;
}

unsigned int random_u32(unsigned long long *_state) {
    *_state ^= *_state >> 12;
    *_state ^= *_state << 25;
    *_state ^= *_state >> 27;
    return (*_state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *_state) {
    return (random_u32(_state) >> 8) / 16777216.0f;
}

int sample_mult(float *_probs, int n, float coin) {
    float cdf = 0.0f;

    for (int i = 0; i < n; i++) {
        cdf += _probs[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare_prob(const void *_a, const void *_b) {
    prob_index *_prob_index_a = (prob_index *)_a;
    prob_index *_prob_index_b = (prob_index *)_b;

    if (_prob_index_a->prob > _prob_index_b->prob) {
        return -1;
    }

    if (_prob_index_a->prob < _prob_index_b->prob) {
        return 1;
    }

    return 0;
}

int sample_top(float *_probs, int n, int top_k, float top_p, prob_index *_prob_index, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - top_p) / (n - 1);

    for (int i = 0; i < n; i++) {
        if (_probs[i] >= cutoff) {
            _prob_index[n0].index = i;
            _prob_index[n0].prob = _probs[i];
            n0++;
        }
    }

    if (! n0) {
        return sample_argmax(_probs, n);
    }

    qsort(_prob_index, n0, sizeof(prob_index), compare_prob);

    if ((top_k > 0) &&
            (n0 > top_k)) {
        n0 = top_k;
    }

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += _prob_index[i].prob;
        if (cumulative_prob > top_p) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += _prob_index[i].prob;
        if (r < cdf) {
            return _prob_index[i].index;
        }
    }

    return _prob_index[last_idx].index;
}

int sample(sampler *_sampler, float *_logits) {
    int next;

    if (_sampler->temp == 0.0f) {
        next = sample_argmax(_logits, _sampler->vocab_size);
    }
    else {
#pragma omp parallel for
        for (int q = 0; q < _sampler->vocab_size; q++) {
            _logits[q] /= _sampler->temp;
        }

        softmax(_logits, _sampler->vocab_size);

        float coin = random_f32(&_sampler->rng_state);
        if (_sampler->top_p <= 0 ||
                _sampler->top_p >= 1) {
            next = sample_mult(_logits, _sampler->vocab_size, coin);
        }
        else {
            next = sample_top(_logits, _sampler->vocab_size, _sampler->top_k, _sampler->top_p, _sampler->_prob_index, coin);
        }
    }
    return next;
}

void build_sampler(sampler *_sampler, int vocab_size, float temp, int top_k, float top_p,
        unsigned long long rng_seed) {
    _sampler->vocab_size = vocab_size;
    _sampler->temp = temp;
    _sampler->top_k = top_k;
    _sampler->top_p = top_p;
    _sampler->rng_state = rng_seed;
    _sampler->_prob_index = a_calloc(_sampler->vocab_size * sizeof(prob_index));
}

void free_sampler(sampler *_sampler) {
    free(_sampler->_prob_index);
}

