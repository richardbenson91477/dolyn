#ifndef DOLEN_G4U_COMMON_H
#define DOLEN_G4U_COMMON_H

#include "dolen_common.h"

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int n_global_kv_heads;
    int vocab_size;
    int seq_len;
    int head_dim;
    int global_head_dim;
    int sliding_window;
    int tie_word_embeddings;
    float rope_theta_full;
    float rope_theta_sliding;
    float rope_partial_factor;
    float rms_norm_eps;
    float final_logit_softcapping;
    int attention_k_eq_v;
} config_gemma4u;

typedef struct {
    qtensor embed_tokens;
    float *rms_input_layernorm;
    float *rms_post_attn_layernorm;
    float *rms_pre_ffn_layernorm;
    float *rms_post_ffn_layernorm;
    float *rms_q_norm;
    float *rms_k_norm;
    float *rms_final_norm;
    int *norm_offsets;
    qtensor *q_proj;
    qtensor *k_proj;
    qtensor *v_proj;
    qtensor *o_proj;
    qtensor *gate_proj;
    qtensor *up_proj;
    qtensor *down_proj;
} weights_gemma4u;

typedef struct {
    float *x;
    float *xb;
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
    float *cos_cache_full;
    float *sin_cache_full;
    float *cos_cache_sliding;
    float *sin_cache_sliding;
    int allocated;
} state_gemma4u;

typedef struct {
    config_gemma4u config;
    weights_gemma4u weights;
    state_gemma4u state;
    int *layer_types;
} Gemma4Unified;

void alloc_state_gemma4u(state_gemma4u *s, config_gemma4u *p);
void free_state_gemma4u(state_gemma4u *s);
void free_gemma4u(Gemma4Unified *model);
int load_quantized_gemma4u(const char *filepath, Gemma4Unified *model, int seq_n_max);
float *forward_gemma4u(Gemma4Unified *model, int token, int pos);

#endif // DOLEN_G4U_COMMON_H
