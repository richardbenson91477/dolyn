#ifndef DOLEN_G4_COMMON_H
#define DOLEN_G4_COMMON_H


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
} config_g4;

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
} weights_g4;

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
    float **key_cache;   // one pointer per layer
    float **value_cache; // one pointer per layer
    qtensor xq;
    qtensor hq; 
    float *cos_cache_full;
    float *sin_cache_full;
    float *cos_cache_sliding;
    float *sin_cache_sliding;
    int allocated;
    int n_layers;        // needed for freeing per-layer caches
} state_g4;

typedef struct {
    config_g4 config;
    weights_g4 weights;
    state_g4 state;
    int *layer_types;
} G4;


void alloc_state_g4(state_g4 *s, config_g4 *p, weights_g4 *w, const int *layer_types);

void free_state_g4(state_g4 *s);

void free_g4(G4 *model);

int load_quantized_g4(const char *filepath, G4 *model, int seq_n_max);

float *forward_g4(G4 *model, int token, int pos);


#endif // DOLEN_G4_COMMON_H
