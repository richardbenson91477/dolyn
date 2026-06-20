#include "dolen_g4u_common.h"

void alloc_state_g4u(state_g4u *s, config_g4u *p, weights_g4u *w) {
    (void)w;
    int max_head_dim = p->head_dim > p->global_head_dim ? p->head_dim : p->global_head_dim;
    int max_kv_heads = p->n_kv_heads > p->n_global_kv_heads ? p->n_kv_heads : p->n_global_kv_heads;
    int max_kv_dim = max_kv_heads * max_head_dim;
    int attn_out_dim = p->n_heads * max_head_dim;

    int cache_stride_full = p->global_head_dim / 2;
    int cache_stride_sliding = p->head_dim / 2;

    int max_act_dim = p->dim;
    if (attn_out_dim > max_act_dim) max_act_dim = attn_out_dim;
    if (p->hidden_dim > max_act_dim) max_act_dim = p->hidden_dim;

    s->x = a_calloc((size_t)p->dim * sizeof(float));
    s->xb = a_calloc((size_t)max_act_dim * sizeof(float));
    s->hb = a_calloc((size_t)p->hidden_dim * sizeof(float));
    s->hb2 = a_calloc((size_t)p->hidden_dim * sizeof(float));
    s->q = a_calloc((size_t)attn_out_dim * sizeof(float));
    s->k = a_calloc((size_t)max_kv_dim * sizeof(float));
    s->k_raw = a_calloc((size_t)max_kv_dim * sizeof(float));
    s->v = a_calloc((size_t)max_kv_dim * sizeof(float));
    s->att = a_calloc((size_t)p->n_heads * p->seq_len * sizeof(float));
    s->logits = a_calloc((size_t)p->vocab_size * sizeof(float));

    s->key_cache = a_calloc((size_t)p->n_layers * p->seq_len * max_kv_dim * sizeof(float));
    s->value_cache = a_calloc((size_t)p->n_layers * p->seq_len * max_kv_dim * sizeof(float));

    int num_groups_xq = (max_act_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    s->xq.data = a_calloc((size_t)max_act_dim * sizeof(int8_t));
    s->xq.s = a_calloc((size_t)num_groups_xq * sizeof(float));
    s->xq.type = Q_TYPE_Q8;
    s->xq.rows = 1;
    s->xq.cols = max_act_dim;

    int num_groups_hq = (p->hidden_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    s->hq.data = a_calloc((size_t)p->hidden_dim * sizeof(int8_t));
    s->hq.s = a_calloc((size_t)num_groups_hq * sizeof(float));
    s->hq.type = Q_TYPE_Q8;
    s->hq.rows = 1;
    s->hq.cols = p->hidden_dim;

    int rotary_dim_full = (int)(p->rope_partial_factor * p->global_head_dim);

    if (cache_stride_full > 0) {
        s->cos_cache_full = a_calloc((size_t)p->seq_len * cache_stride_full * sizeof(float));
        s->sin_cache_full = a_calloc((size_t)p->seq_len * cache_stride_full * sizeof(float));

        for (int pos = 0; pos < p->seq_len; pos++) {
            for (int i = 0; i < cache_stride_full; i++) {
                float cos_val, sin_val;
                if (i < rotary_dim_full / 2) {
                    float freq = 1.0f / powf(p->rope_theta_full, (float)(2 * i) / p->global_head_dim);
                    float val = (float)pos * freq;
                    cos_val = cosf(val);
                    sin_val = sinf(val);
                } else {
                    cos_val = 1.0f;
                    sin_val = 0.0f;
                }
                s->cos_cache_full[pos * cache_stride_full + i] = cos_val;
                s->sin_cache_full[pos * cache_stride_full + i] = sin_val;
            }
        }
    }

    if (cache_stride_sliding > 0) {
        s->cos_cache_sliding = a_calloc((size_t)p->seq_len * cache_stride_sliding * sizeof(float));
        s->sin_cache_sliding = a_calloc((size_t)p->seq_len * cache_stride_sliding * sizeof(float));
        for (int pos = 0; pos < p->seq_len; pos++) {
            for (int i = 0; i < cache_stride_sliding; i++) {
                float freq = 1.0f / powf(p->rope_theta_sliding, (float)(2 * i) / p->head_dim);
                float val = (float)pos * freq;
                s->cos_cache_sliding[pos * cache_stride_sliding + i] = cosf(val);
                s->sin_cache_sliding[pos * cache_stride_sliding + i] = sinf(val);
            }
        }
    } else {
        s->cos_cache_sliding = NULL;
        s->sin_cache_sliding = NULL;
    }

    if (!s->x || !s->xb || !s->hb || !s->hb2 || !s->q || !s->k || !s->k_raw || !s->v || !s->att ||
            !s->logits || !s->key_cache || !s->value_cache || !s->xq.data || !s->xq.s || !s->hq.data || !s->hq.s) {
        log_msg(stderr, "ERROR: Alloc failed!\n");
        exit(EXIT_FAILURE);
    }
    if (p->seq_len > 1 &&
            (!s->cos_cache_full || !s->sin_cache_full || !s->cos_cache_sliding || !s->sin_cache_sliding)) {
        log_msg(stderr, "ERROR: Alloc failed for RoPE cache!\n");
        exit(EXIT_FAILURE);
    }
    s->allocated = 1;
}

void free_state_g4u(state_g4u *s) {
    if (!s->allocated) return;

    free(s->x);
    free(s->xb);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->k_raw);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
    free(s->xq.data);
    free(s->xq.s);
    free(s->hq.data);
    free(s->hq.s);
    free(s->cos_cache_full);
    free(s->sin_cache_full);
    free(s->cos_cache_sliding);
    free(s->sin_cache_sliding);

    s->allocated = 0;
}

void free_g4u(G4U *model) {
    if (!model) return;

    config_g4u *p = &model->config;
    weights_g4u *w = &model->weights;

    free(model->layer_types);
    
    free_qt(&w->embed_tokens);
    free_qt(&w->rms_final_norm);
    free_qt(&w->rope_freqs_full);

    free_qt_array(w->rms_input_layernorm, p->n_layers);
    free_qt_array(w->rms_post_attn_layernorm, p->n_layers);
    free_qt_array(w->rms_pre_ffn_layernorm, p->n_layers);
    free_qt_array(w->rms_post_ffn_layernorm, p->n_layers);
    free_qt_array(w->rms_q_norm, p->n_layers);
    free_qt_array(w->rms_k_norm, p->n_layers);

    free_qt_array(w->q_proj, p->n_layers);
    free_qt_array(w->k_proj, p->n_layers);
    free_qt_array(w->v_proj, p->n_layers);
    free_qt_array(w->o_proj, p->n_layers);
    free_qt_array(w->gate_proj, p->n_layers);
    free_qt_array(w->up_proj, p->n_layers);
    free_qt_array(w->down_proj, p->n_layers);

    free(w->layer_scalars);

    if (model->state.allocated == 1) {
        free_state_g4u(&model->state);
    }

    memset(model, 0, sizeof(G4U));
}