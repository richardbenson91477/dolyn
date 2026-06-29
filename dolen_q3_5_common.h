#ifndef DOLEN_Q3_5_COMMON_H
#define DOLEN_Q3_5_COMMON_H

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
    float rope_partial_rotary_factor;
    float rms_norm_eps;
    int tie_word_embeddings;
    int d_head;
    int n_linear_k_heads;
    int n_linear_v_heads;
    int d_linear_k;
    int d_linear_v;  
    int linear_conv_kernel;
    int n_full_attn_layers;
    int n_linear_attn_layers;
    int bos_token_id;
    int eos_token_id;
} config_q3_5;

typedef struct {
    qtensor embed_tokens_weight;
    qtensor *_rms_att_weight;
    qtensor *_wq;
    qtensor *_wk;
    qtensor *_wv;
    qtensor *_wo;
    qtensor *_q_norm;
    qtensor *_k_norm;
    qtensor *_in_proj_qkv;
    qtensor *_in_proj_z;
    qtensor *_in_proj_b;
    qtensor *_in_proj_a;
    qtensor *_conv1d_weight;
    qtensor *_dt_bias;
    qtensor *_A_log;
    qtensor *_linear_norm;
    qtensor *_out_proj;
    qtensor *_rms_ffn_weight;
    qtensor *_w1;
    qtensor *_w2;
    qtensor *_w3;
    qtensor rms_final_weight;
    qtensor wcls;
} weights_q3_5;

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
    float *_gate;
    float *_key_cache;
    float *_value_cache;
    float *_qkv;
    float *_z;
    float *_beta;
    float *_g;
    float *_linear_out;
    float *_conv_state;
    float *_S;
    float *_delta_S;
    qtensor xq;
    qtensor hq;
    float *_cos_cache;
    float *_sin_cache;
    int allocated;
} state_q3_5;

typedef struct {
    config_q3_5 config;
    weights_q3_5 weights;
    state_q3_5 state;
    tokenizer tokenizer1;
    int *_layer_types;
    int *_attn_layer_indices;
    int *_deltanet_layer_indices;
} Q3_5;


void alloc_state_q3_5(state_q3_5 *_state, config_q3_5 *_config);

void free_state_q3_5(state_q3_5 *_state);

void free_q3_5(Q3_5 *_model);


#endif // DOLEN_Q3_5_COMMON_H

