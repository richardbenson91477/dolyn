#ifndef dolen_Q3_5_COMMON_H
#define DOLEN_Q3_5_COMMON_H

#include "dolen_common.h"

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
} config_q3_5;

typedef struct {
    qtensor token_embedding_table;
    float *rms_att_weight;
    qtensor *wq;
    qtensor *wk;
    qtensor *wv;
    qtensor *wo;
    float *q_norm;
    float *k_norm;
    qtensor *in_proj_qkv;
    qtensor *in_proj_z;
    float *in_proj_b;
    float *in_proj_a;
    float *conv1d_weight;
    float *dt_bias;
    float *A_log;
    float *linear_norm;
    qtensor *out_proj;
    float *rms_ffn_weight;
    qtensor *w1;
    qtensor *w2;
    qtensor *w3;
    float *rms_final_weight;
    qtensor wcls;
} weights_q3_5;

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
    float *gate;
    float *key_cache;
    float *value_cache;
    float *qkv;
    float *z;
    float *beta;
    float *g;
    float *linear_out;
    float *conv_state;
    float *S;
    float *delta_S;
    qtensor xq;
    qtensor hq;
    float *cos_cache;
    float *sin_cache;
    int allocated;
} state_q3_5;

typedef struct {
    config_q3_5 config;
    weights_q3_5 weights;
    state_q3_5 state;
    int *layer_types;
    int *attn_layer_indices;
    int *deltanet_layer_indices;
} Q3_5;

void alloc_state_q3_5(state_q3_5 *s, config_q3_5 *p);

void free_state_q3_5(state_q3_5 *s);

void free_q3_5(Q3_5 *model_q3_5);


#endif //DOLEN_Q3_5_COMMON_H

