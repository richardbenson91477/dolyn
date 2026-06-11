#ifndef DOLEN_Q3_COMMON_H
#define DOLEN_Q3_COMMON_H

#include "dolen_common.h"

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    int head_dim;
    int shared_classifier;
    float rope_theta;
    float rope_scaling_factor;
    float rms_norm_eps;
} config_qwen3;

typedef struct {
    qtensor token_embedding_table;
    float *rms_att_weight;
    float *rms_ffn_weight;
    qtensor *wq;
    qtensor *wk;
    qtensor *wv;
    qtensor *wo;
    float *q_norm_weights;
    float *k_norm_weights;
    qtensor *w1;
    qtensor *w2;
    qtensor *w3;
    float *rms_final_weight;
    qtensor wcls;
} weights_qwen3;

typedef struct {
    float *x;
    float *xb;
    float *hb;
    float *hb2;
    qtensor xq;
    qtensor hq;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
    float *cos_cache;
    float *sin_cache;
    int allocated;
} state_qwen3;

typedef struct {
    config_qwen3 config;
    weights_qwen3 weights;
    state_qwen3 state;
} Qwen3;

static token_map special_tokens_qwen3[] = {
    {"<|endoftext|\x3e", 151643},
    {"<|im_start|\x3e", 151644},
    {"<|im_end|\x3e", 151645},
    {"<|object_ref_start|\x3e", 151646},
    {"<|object_ref_end|\x3e", 151647},
    {"<|box_start|\x3e", 151648},
    {"<|box_end|\x3e", 151649},
    {"<|quad_start|\x3e", 151650},
    {"<|quad_end|\x3e", 151651},
    {"<|vision_start|\x3e", 151652},
    {"<|vision_end|\x3e", 151653},
    {"<|vision_pad|\x3e", 151654},
    {"<|image_pad|\x3e", 151655},
    {"<|video_pad|\x3e", 151656},
    {NULL, 0}
};


void alloc_state_qwen3(state_qwen3 *s, config_qwen3 *p);
void free_state_qwen3(state_qwen3 *s);
void free_qwen3(Qwen3 *t);


#endif //DOLEN_Q3_COMMON_H

