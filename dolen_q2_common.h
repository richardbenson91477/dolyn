#ifndef DOLEN_Q2_COMMON_H
#define DOLEN_Q2_COMMON_H

#include "dolen_common_cmi.h"
#include "dolen_common_io.h"
#include "dolen_common_math.h"
#include "dolen_common_mem.h"
#include "dolen_common_qtensor.h"

typedef struct {
    int dim;                    /* hidden_size */
    int hidden_dim;             /* intermediate_size */
    int n_layers;               /* num_hidden_layers */
    int n_heads;                /* num_attention_heads */
    int n_kv_heads;             /* num_key_value_heads */
    int vocab_size;
    int seq_len;                /* max_position_embeddings */
    int head_dim;
    int shared_classifier;      /* tie_word_embeddings */
    float rope_theta;
    float rope_scaling_factor;  /* default 1.0, may be overridden */
    float rms_norm_eps;
    int bos_token_id;
    int eos_token_id;
} config_q2;

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
    
    /* Qwen2 Attention Biases */
    qtensor *_q_bias;
    qtensor *_k_bias;
    qtensor *_v_bias;

    qtensor rms_final_weight;   /* model.norm */
    qtensor wcls;               /* lm_head (if not tied) */
} weights_q2;

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
    int allocated;
} state_q2;

typedef struct {
    config_q2 config;
    weights_q2 weights;
    state_q2 state;
    tokenizer tokenizer1;
} Q2;

void alloc_state_q2(state_q2 *_state, config_q2 *_config);
void free_state_q2(state_q2 *_state);
void free_q2(Q2 *_model);

#endif /* DOLEN_Q2_COMMON_H */