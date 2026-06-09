#ifndef dolen3_5_COMMON_H
#define DOLEN3_5_COMMON_H

#include "dolen_common.h"

typedef struct {
    int dim;
    int n_heads;
    int n_kv_heads;
    int n_layer;
    int n_mlp;
    int vocab_size;
    int seq_len;
    float rope_theta;
    float rms_norm_eps;
    int tie_word_embeddings;
    int d_head;
    int n_linear_k_heads;
    int n_linear_v_heads;
    int d_linear_k;
    int d_linear_v;
    int linear_conv_kernel;
    int n_full_attn_layers;
    int n_linear_attn_layers;
} config_qwen3_5;

typedef struct {
    qtensor token_embedding_table;
    float *rms_att_weight;
    qtensor *wq;
    qtensor *wk;
    qtensor *wv;
    qtensor *wo;
    float *q_norm;
    float *k_norm;
    qtensor *in_proj_qkv;
    qtensor *in_proj_z;
    float *in_proj_b;
    float *in_proj_a;
    float *conv1d_weight;
    float *dt_bias;
    float *A_log;
    float *linear_norm;
    qtensor *out_proj;
    float *rms_ffn_weight;
    qtensor *w1;
    qtensor *w2;
    qtensor *w3;
    float *rms_final_weight;
    qtensor wcls;
} weights_qwen3_5;

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
    float *gate;
    float *key_cache;
    float *value_cache;
    float *qkv;
    float *z;
    float *beta;
    float *g;
    float *linear_out;
    float *conv_state;
    float *S;
    float *delta_S;
    qtensor xq;
    qtensor hq;
    float *cos_cache;
    float *sin_cache;
    int allocated;
} state_qwen3_5;

typedef struct {
    config_qwen3_5 config;
    weights_qwen3_5 weights;
    state_qwen3_5 state;
    int *layer_types;
    int *attn_layer_indices;
    int *deltanet_layer_indices;
} Qwen3_5;

static token_map special_tokens_qwen3_5[] = {
    {"<|endoftext|\x3e", 248044},
    {"<|im_start|\x3e", 248045},
    {"<|im_end|\x3e", 248046},
    {"<|object_ref_start|\x3e", 248047},
    {"<|object_ref_end|\x3e", 248048},
    {"<|box_start|\x3e", 248049},
    {"<|box_end|\x3e", 248050},
    {"<|quad_start|\x3e", 248051},
    {"<|quad_end|\x3e", 248052},
    {"<|vision_start|\x3e", 248053},
    {"<|vision_end|\x3e", 248054},
    {"<|vision_pad|\x3e", 248055},
    {"<|image_pad|\x3e", 248056},
    {"<|video_pad|\x3e", 248057},
    {NULL, 0}
};


void alloc_state_qwen3_5(state_qwen3_5 *s, config_qwen3_5 *p);
void free_state_qwen3_5(state_qwen3_5 *s);
void free_qwen3_5(Qwen3_5 *model_qwen3_5);


#endif //DOLEN3_5_COMMON_H

