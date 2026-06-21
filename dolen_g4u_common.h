#ifndef DOLEN_G4U_COMMON_H
#define DOLEN_G4U_COMMON_H


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
    int original_max_seq_len;
    int use_rope_freqs;
} config_g4u;

typedef struct {
    qtensor embed_tokens_weight;
    qtensor *rms_input_layernorm;
    qtensor *rms_post_attn_layernorm;
    qtensor *rms_pre_ffn_layernorm;
    qtensor *rms_post_ffn_layernorm;
    qtensor *rms_q_norm;
    qtensor *rms_k_norm;
    qtensor rms_final_norm;
    qtensor *q_proj;
    qtensor *k_proj;
    qtensor *v_proj;
    qtensor *o_proj;
    qtensor *gate_proj;
    qtensor *up_proj;
    qtensor *down_proj;
    float *layer_scalars;
    qtensor rope_freqs_full;
} weights_g4u;

typedef struct {
    float *x;
    float *xb;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *k_raw;
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
} state_g4u;

typedef struct {
    config_g4u config;
    weights_g4u weights;
    state_g4u state;
    int *layer_types;
} G4U;


void alloc_state_g4u(state_g4u *s, config_g4u *p, weights_g4u *w);

void free_state_g4u(state_g4u *s);

void free_g4u(G4U *model);

int load_quantized_g4u(const char *filepath, G4U *model, int seq_n_max);

float *forward_g4u(G4U *model, int token, int pos);


#endif // DOLEN_G4U_COMMON_H

