#ifndef DOLEN_G4_COMMON_H
#define DOLEN_G4_COMMON_H

#include "dolen_common_cmi.h"
#include "dolen_common_io.h"
#include "dolen_common_math.h"
#include "dolen_common_mem.h"
#include "dolen_common_qtensor.h"

typedef struct {
    /* --- base fields (same layout/meaning as config_g4) --- */
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
    float   rope_theta_full;
    float   rope_theta_sliding;
    float   rope_partial_factor;
    float   rms_norm_eps;
    float   final_logit_softcapping;
    int32_t attention_k_eq_v;
    int32_t original_max_pos_embeds;
    int32_t use_rope_freqs;
    int32_t bos_token_id;
    int32_t eos_token_id;
    /* --- G4E additions --- */
    int32_t hidden_size_per_layer_input;   /* PLE per-layer dim  (0 = disabled) */
    int32_t vocab_size_per_layer_input;    /* PLE vocab size     */
    int32_t num_kv_shared_layers;          /* KV-sharing count   (0 = disabled) */
} config_g4;

typedef struct {
    /* base weights (identical to weights_g4) */
    qtensor  embed_tokens_weight;
    qtensor *_rms_input_layernorm;
    qtensor *_rms_post_attn_layernorm;
    qtensor *_rms_pre_ffn_layernorm;
    qtensor *_rms_post_ffn_layernorm;
    qtensor *_rms_q_norm;
    qtensor *_rms_k_norm;          /* empty for shared layers */
    qtensor  rms_final_norm;
    qtensor *_q_proj;
    qtensor *_k_proj;              /* empty for shared layers */
    qtensor *_v_proj;              /* empty for shared layers / alt-attn */
    qtensor *_o_proj;
    qtensor *_gate_proj;
    qtensor *_up_proj;
    qtensor *_down_proj;
    float   *_layer_scalars;
    qtensor  rope_freqs_full;

    /* PLE weights (valid when hidden_size_per_layer_input > 0) */
    qtensor  embed_tokens_per_layer;         /* [vocab_per_layer, n_layers*ple_dim] */
    qtensor  per_layer_model_projection;     /* [n_layers*ple_dim, dim]             */
    qtensor  per_layer_projection_norm;      /* F32 [ple_dim] \u2014 shared across layers */
    qtensor *_ple_gate;                      /* per-layer [ple_dim, dim]            */
    qtensor *_ple_projection;                /* per-layer [dim, ple_dim]            */
    qtensor *_ple_post_norm;                 /* per-layer F32 [dim]                 */
} weights_g4;

typedef struct {
    int32_t  seq_n;
    float   *_x;
    float   *_xb;
    float   *_hb;
    float   *_hb2;
    float   *_q;
    float   *_k;
    float   *_k_raw;
    float   *_v;
    float   *_att;
    float   *_logits;
    _Float16 **__key_cache;
    _Float16 **__value_cache;
    qtensor  xq;
    qtensor  hq;
    float   *_cos_cache_full;
    float   *_sin_cache_full;
    float   *_cos_cache_sliding;
    float   *_sin_cache_sliding;
    int32_t  n_layers;

    /* PLE scratch */
    float   *_ple_combined;     /* [n_layers * ple_dim] final PLE input  */
    float   *_ple_proj_raw;     /* [n_layers * ple_dim] raw projection   */
    float   *_ple_gate_out;     /* [ple_dim]                             */
    float   *_ple_proj_out;     /* [dim]                                 */

    /* Shared KV caches (one per layer-type) */
    _Float16 *_shared_key_full;
    _Float16 *_shared_value_full;
    _Float16 *_shared_key_sliding;
    _Float16 *_shared_value_sliding;
} state_g4;

typedef struct {
    config_g4  config;
    weights_g4 weights;
    state_g4   state;
    tokenizer   tokenizer;
    int32_t    *_layer_types;

    /* pre-computed KV-sharing metadata */
    int32_t _first_kv_shared_layer_idx;  /* n_layers when no sharing */
    int32_t _store_full_layer_idx;       /* -1 when no sharing       */
    int32_t _store_sliding_layer_idx;    /* -1 when no sharing       */
} G4;


bool   alloc_state_g4(G4 *_model, int32_t seq_n);
void   free_state_g4(state_g4 *_state);
void   free_g4(G4 *_model);
bool   load_quantized_g4(const char *_path_s, G4 *_model);
float *forward_g4(G4 *_model, int32_t token, int32_t pos);

#endif /* DOLEN_G4_COMMON_H */

