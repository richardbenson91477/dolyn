#ifndef DOLEN_IG4_1_COMMON_H
#define DOLEN_IG4_1_COMMON_H


#include "dolen_common_cmi.h"
#include "dolen_common_io.h"
#include "dolen_common_math.h"
#include "dolen_common_mem.h"
#include "dolen_common_qtensor.h"


typedef struct {
    int dim;
    int n_heads;
    int n_kv_heads;
    int n_layer;
    int n_mlp;
    int vocab_size;
    int seq_len;
    float rope_theta;
    float rms_norm_eps;
    int tie_word_embeddings;
    int d_head;

    float embedding_multiplier;
    float attention_multiplier;
    float residual_multiplier;
    float logits_scaling;
} config_ig4_1;

typedef struct {
    qtensor token_embedding_table;
    qtensor *rms_att_weight;
    qtensor *wq;
    qtensor *wk;
    qtensor *wv;
    qtensor *wo;
    qtensor *rms_ffn_weight;
    qtensor *w1;
    qtensor *w2;
    qtensor *w3;
    qtensor rms_final_weight;
    qtensor wcls;
} weights_ig4_1;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
    qtensor xq;
    qtensor hq;
    float *cos_cache;
    float *sin_cache;
    int allocated;
} state_ig4_1;

typedef struct {
    config_ig4_1 config;
    weights_ig4_1 weights;
    state_ig4_1 state;
} IG4_1;


void alloc_state_ig4_1(state_ig4_1 *s, config_ig4_1 *p);

void free_state_ig4_1(state_ig4_1 *s);

void free_ig4_1(IG4_1 *model_ig4_1);


#endif // DOLEN_IG4_1_COMMON_H

