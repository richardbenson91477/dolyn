#ifndef DOLEN_MS_COMMON_H
#define DOLEN_MS_COMMON_H

#include "dolen_common_cmi.h"
#include "dolen_common_io.h"
#include "dolen_common_math.h"
#include "dolen_common_mem.h"
#include "dolen_common_qtensor.h"


typedef struct {
    int32_t dim;                    /* hidden_size */
    int32_t hidden_dim;             /* intermediate_size */
    int32_t n_layers;               /* num_hidden_layers */
    int32_t n_heads;                /* num_attention_heads */
    int32_t n_kv_heads;             /* num_key_value_heads */
    int32_t vocab_size;
    int32_t seq_len;                /* max_position_embeddings */
    int32_t head_dim;
    int32_t shared_classifier;      /* tie_word_embeddings */
    float rope_theta;
    float rms_norm_eps;
    int32_t bos_token_id;
    int32_t eos_token_id;
    int32_t sliding_window;         /* Mistral sliding window limit */
} config_ms;

typedef struct {
    qtensor embed_tokens_weight;
    qtensor *_rms_att_weight;   /* input_layernorm per layer */
    qtensor *_rms_ffn_weight;   /* post_attention_layernorm per layer */
    qtensor *_wq;
    qtensor *_wk;
    qtensor *_wv;
    qtensor *_wo;
    qtensor *_w1;               /* gate_proj */
    qtensor *_w2;               /* down_proj */
    qtensor *_w3;               /* up_proj */
    /* NO QKV Biases for Mistral */
    qtensor rms_final_weight;   /* model.norm */
    qtensor wcls;               /* lm_head (if not tied) */
} weights_ms;

typedef struct {
    float *_x;                  /* current hidden state */
    float *_xb;                 /* buffer for intermediate results */
    float *_hb;                 /* buffer for mlp gate activations */
    float *_hb2;                /* buffer for mlp up projections */
    qtensor xq;                 /* quantized intermediate tensor */
    qtensor hq;                 /* quantized mlp hidden tensor */
    float *_q;                  /* query buffer (all heads) */
    float *_k;                  /* key buffer (kv heads) */
    float *_v;                  /* value buffer (kv heads) */
    float *_att;                /* attention scores */
    float *_logits;
    float *_key_cache;
    float *_value_cache;
    float *_cos_cache;          /* precomputed cos */
    float *_sin_cache;          /* precomputed sin */
    int32_t allocated;
} state_ms;

typedef struct {
    config_ms config;
    weights_ms weights;
    state_ms state;
    tokenizer tokenizer1;
} MS;


void alloc_state_ms(state_ms *_state, config_ms *_config);

void free_state_ms(state_ms *_state);

void free_ms(MS *_model);


#endif /* DOLEN_MS_COMMON_H */
