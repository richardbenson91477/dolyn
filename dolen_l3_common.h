#ifndef DOLEN_L3_COMMON_H
#define DOLEN_L3_COMMON_H

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
    int32_t vocab_size;
    int32_t seq_len;
    int32_t head_dim;
    float rope_theta;
    float rms_norm_eps;
    int32_t tie_word_embeddings;
    int32_t bos_token_id;
    int32_t eos_token_id;
} config_l3;

typedef struct {
    qtensor embed_tokens_weight;
    qtensor *_rms_att_weight;
    qtensor *_wq;
    qtensor *_wk;
    qtensor *_wv;
    qtensor *_wo;
    qtensor *_rms_ffn_weight;
    qtensor *_w1;
    qtensor *_w2;
    qtensor *_w3;
    qtensor rms_final_weight;
    qtensor wcls;
} weights_l3;

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
    float **__key_cache;
    float **__value_cache;
    qtensor xq;
    qtensor hq;
    float *_cos_cache;
    float *_sin_cache;
    int32_t allocated;
    int32_t n_layers;
} state_l3;

typedef struct {
    config_l3 config;
    weights_l3 weights;
    state_l3 state;
    tokenizer tokenizer1;
} L3;


void alloc_state_l3(state_l3 *_state, config_l3 *_config);

void free_state_l3(state_l3 *_state);

void free_l3(L3 *_model);

int32_t load_quantized_l3(const char *_path_s, L3 *_model, int32_t seq_n_max);

float *forward_l3(L3 *_model, int32_t token, int32_t pos);


#endif // DOLEN_L3_COMMON_H

