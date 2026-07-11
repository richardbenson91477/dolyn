#ifndef DOLEN_G4_COMMON_H
#define DOLEN_G4_COMMON_H

#include "dolen_common_cmi.h"
#include "dolen_common_io.h"
#include "dolen_common_math.h"
#include "dolen_common_mem.h"
#include "dolen_common_qtensor.h"


typedef struct {
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t n_global_kv_heads;
    int32_t vocab_size;
    int32_t seq_n;
    int32_t head_dim;
    int32_t global_head_dim;
    int32_t sliding_window;
    int32_t tie_word_embeddings;
    float rope_theta_full;
    float rope_theta_sliding;
    float rope_partial_factor;
    float rms_norm_eps;
    float final_logit_softcapping;
    int32_t attention_k_eq_v;
    int32_t original_max_pos_embeds;
    int32_t use_rope_freqs;
    int32_t bos_token_id;
    int32_t eos_token_id;
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
    int32_t seq_n;
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
    _Float16 **__key_cache;
    _Float16 **__value_cache;
    qtensor xq;
    qtensor hq;
    float *_cos_cache_full;
    float *_sin_cache_full;
    float *_cos_cache_sliding;
    float *_sin_cache_sliding;
    int32_t n_layers;
} state_g4;

typedef struct {
    config_g4 config;
    weights_g4 weights;
    state_g4 state;
    tokenizer tokenizer;
    int32_t *_layer_types;
} G4;


bool alloc_state_g4(G4 *_g4, int32_t seq_n);

void free_state_g4(state_g4 *_state);

void free_g4(G4 *_model);

bool load_quantized_g4(const char *_path_s, G4 *_model);

float *forward_g4(G4 *_model, int32_t token, int32_t pos);


#endif // DOLEN_G4_COMMON_H

