#ifndef DOLEN_L3_COMMON_H
#define DOLEN_L3_COMMON_H

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
    int vocab_size;
    int seq_len;
    int head_dim;
    float rope_theta;
    float rms_norm_eps;
    int tie_word_embeddings;
} config_l3;

typedef struct {
    qtensor embed_tokens_weight;
    qtensor *rms_att_weight;
    qtensor *wq;
    qtensor *wk;
    qtensor *wv;
    qtensor *wo;
    qtensor *rms_ffn_weight;
    qtensor *w1;
    qtensor *w2;
    qtensor *w3;
    qtensor rms_final_weight;
    qtensor wcls;
} weights_l3;

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
    float **key_cache;
    float **value_cache;
    qtensor xq;
    qtensor hq;
    float *cos_cache;
    float *sin_cache;
    int allocated;
    int n_layers;
} state_l3;

typedef struct {
    config_l3 config;
    weights_l3 weights;
    state_l3 state;
    Tokenizer tokenizer;
} L3;

void alloc_state_l3(state_l3 *s, config_l3 *p);
void free_state_l3(state_l3 *s);
void free_l3(L3 *model);
int load_quantized_l3(const char *filepath, L3 *model, int seq_n_max);
float *forward_l3(L3 *model, int token, int pos);

#endif // DOLEN_L3_COMMON_H