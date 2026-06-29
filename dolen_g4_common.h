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
    int bos_token_id;
    int eos_token_id;
} config_g4;

typedef struct {
    qtensor embed_tokens_weight;
    qtensor *_rms_input_layernorm;
    qtensor *_rms_post_attn_layernorm;
    qtensor *_rms_pre_ffn_layernorm;
    qtensor *_rms_post_ffn_layernorm;
    qtensor *_rms_q_norm;
    qtensor *_rms_k_norm;
    qtensor rms_final_norm;
    qtensor *_q_proj;
    qtensor *_k_proj;
    qtensor *_v_proj;
    qtensor *_o_proj;
    qtensor *_gate_proj;
    qtensor *_up_proj;
    qtensor *_down_proj;
    float *_layer_scalars;
    qtensor rope_freqs_full;
} weights_g4;

typedef struct {
    float *_x;
    float *_xb;
    float *_hb;
    float *_hb2;
    float *_q;
    float *_k;
    float *_k_raw;
    float *_v;
    float *_att;
    float *_logits;
    float **__key_cache;
    float **__value_cache;
    qtensor xq;
    qtensor hq;
    float *_cos_cache_full;
    float *_sin_cache_full;
    float *_cos_cache_sliding;
    float *_sin_cache_sliding;
    int allocated;
    int n_layers;
} state_g4;

typedef struct {
    config_g4 config;
    weights_g4 weights;
    state_g4 state;
    tokenizer tokenizer;
    int *_layer_types;
} G4;

void alloc_state_g4(state_g4 *_state, config_g4 *_config, weights_g4 *_weights, const int *_layer_types);
void free_state_g4(state_g4 *_state);
void free_g4(G4 *_model);
int load_quantized_g4(const char *_path_s, G4 *_model, int seq_n_max);
float *forward_g4(G4 *_model, int token, int pos);

#endif // DOLEN_G4_COMMON_H