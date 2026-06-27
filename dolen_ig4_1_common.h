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
    qtensor embed_tokens_weight;
    qtensor *_rms_att_weight;
    qtensor *_wq;
    qtensor *_wk;
    qtensor *_wv;
    qtensor *_wo;
    qtensor *_rms_ffn_weight;
    qtensor *_w1;
    qtensor *_w2;
    qtensor *_w3;
    qtensor rms_final_weight;
    qtensor wcls;
} weights_ig4_1;

typedef struct {
    float *_x;
    float *_xb;
    float *_xb2;
    float *_hb;
    float *_hb2;
    float *_q;
    float *_k;
    float *_v;
    float *_att;
    float *_logits;
    float *_key_cache;
    float *_value_cache;
    qtensor xq;
    qtensor hq;
    float *_cos_cache;
    float *_sin_cache;
    int allocated;
} state_ig4_1;

typedef struct {
    config_ig4_1 config;
    weights_ig4_1 weights;
    state_ig4_1 state;
    tokenizer tokenizer;
} IG4_1;


void alloc_state_ig4_1(state_ig4_1 *_state, config_ig4_1 *_config);

void free_state_ig4_1(state_ig4_1 *_state);

void free_ig4_1(IG4_1 *_model_ig4_1);


#endif // DOLEN_IG4_1_COMMON_H

