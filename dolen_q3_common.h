#ifndef DOLEN_Q3_COMMON_H
#define DOLEN_Q3_COMMON_H


#include "dolen_common_cmi.h"
#include "dolen_common_io.h"
#include "dolen_common_math.h"
#include "dolen_common_mem.h"
#include "dolen_common_qtensor.h"


typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    int head_dim;
    int shared_classifier;
    float rope_theta;
    float rope_scaling_factor;
    float rms_norm_eps;
} config_q3;

typedef struct {
    qtensor embed_tokens_weight;
    qtensor *rms_att_weight; 
    qtensor *rms_ffn_weight;
    qtensor *wq;
    qtensor *wk;
    qtensor *wv;
    qtensor *wo;
    qtensor *q_norm;
    qtensor *k_norm;
    qtensor *w1;
    qtensor *w2;
    qtensor *w3;
    qtensor rms_final_weight;
    qtensor wcls;
} weights_q3;

typedef struct {
    float *x;
    float *xb;
    float *hb;
    float *hb2;
    qtensor xq;
    qtensor hq;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
    float *cos_cache;
    float *sin_cache;
    int allocated;
} state_q3;

typedef struct {
    config_q3 config;
    weights_q3 weights;
    state_q3 state;
    Tokenizer tokenizer;
} Q3;


void alloc_state_q3(state_q3 *s, config_q3 *p);

void free_state_q3(state_q3 *s);

void free_q3(Q3 *t);


#endif // DOLEN_Q3_COMMON_H

