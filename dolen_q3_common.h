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
    int bos_token_id;
    int eos_token_id;
} config_q3;

typedef struct {
    qtensor embed_tokens_weight;
    qtensor *_rms_att_weight;
    qtensor *_rms_ffn_weight;
    qtensor *_wq;
    qtensor *_wk;
    qtensor *_wv;
    qtensor *_wo;
    qtensor *_q_norm;
    qtensor *_k_norm;
    qtensor *_w1;
    qtensor *_w2;
    qtensor *_w3;
    qtensor rms_final_weight;
    qtensor wcls;
} weights_q3;

typedef struct {
    float *_x;
    float *_xb;
    float *_hb;
    float *_hb2;
    qtensor xq;
    qtensor hq;
    float *_q;
    float *_k;
    float *_v;
    float *_att;
    float *_logits;
    float *_key_cache;
    float *_value_cache;
    float *_cos_cache;
    float *_sin_cache;
    int allocated;
} state_q3;

typedef struct {
    config_q3 config;
    weights_q3 weights;
    state_q3 state;
    tokenizer tokenizer1;
} Q3;


void alloc_state_q3(state_q3 *_state, config_q3 *_config);

void free_state_q3(state_q3 *_state);

void free_q3(Q3 *_model);


#endif // DOLEN_Q3_COMMON_H
