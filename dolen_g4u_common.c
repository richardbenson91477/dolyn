#include "dolen_g4u_common.h"

void alloc_state_gemma4u(state_gemma4u *s, config_gemma4u *p) {
    int max_head_dim = p->head_dim > p->global_head_dim ? p->head_dim : p->global_head_dim;

    int kv_dim_sliding = p->n_kv_heads * p->head_dim;
    int kv_dim_full = p->n_global_kv_heads * p->global_head_dim;
    int max_kv_dim = kv_dim_sliding > kv_dim_full ? kv_dim_sliding : kv_dim_full;

    int attn_out_dim = p->n_heads * max_head_dim;

    int rotary_full = (int)((float)p->global_head_dim * p->rope_partial_factor);
    int rotary_sliding = p->head_dim;

    int half_rotary_full = rotary_full / 2;
    int half_rotary_sliding = rotary_sliding / 2;

    int max_act_dim = p->dim;
    if (attn_out_dim > max_act_dim) max_act_dim = attn_out_dim;
    if (p->hidden_dim > max_act_dim) max_act_dim = p->hidden_dim;

    s->x = a_calloc((size_t)p->dim * sizeof(float));
    s->xb = a_calloc((size_t)max_act_dim * sizeof(float));
    s->hb = a_calloc((size_t)p->hidden_dim * sizeof(float));
    s->hb2 = a_calloc((size_t)p->hidden_dim * sizeof(float));
    s->q = a_calloc((size_t)attn_out_dim * sizeof(float));
    s->k = a_calloc((size_t)max_kv_dim * sizeof(float));
    s->v = a_calloc((size_t)max_kv_dim * sizeof(float));
    s->att = a_calloc((size_t)p->n_heads * p->seq_len * sizeof(float));
    s->logits = a_calloc((size_t)p->vocab_size * sizeof(float));

    s->key_cache = a_calloc((size_t)p->n_layers * p->seq_len * max_kv_dim * sizeof(float));
    s->value_cache = a_calloc((size_t)p->n_layers * p->seq_len * max_kv_dim * sizeof(float));

    int num_groups_xq = (max_act_dim + GS - 1) / GS;
    s->xq.q = a_calloc((size_t)max_act_dim * sizeof(int8_t));
    s->xq.s = a_calloc((size_t)num_groups_xq * sizeof(float));
    s->xq.rows = 1;
    s->xq.cols = max_act_dim;

    int num_groups_hq = (p->hidden_dim + GS - 1) / GS;
    s->hq.q = a_calloc((size_t)p->hidden_dim * sizeof(int8_t));
    s->hq.s = a_calloc((size_t)num_groups_hq * sizeof(float));
    s->hq.rows = 1;
    s->hq.cols = p->hidden_dim;

    if (half_rotary_full > 0) {
        s->cos_cache_full = a_calloc((size_t)p->seq_len * half_rotary_full * sizeof(float));
        s->sin_cache_full = a_calloc((size_t)p->seq_len * half_rotary_full * sizeof(float));
        for (int pos = 0; pos < p->seq_len; pos++) {
            for (int i = 0; i < half_rotary_full; i++) {
                float freq = 1.0f / powf(p->rope_theta_full, (float)(2 * i) / rotary_full);
                float val = pos * freq;
                s->cos_cache_full[pos * half_rotary_full + i] = cosf(val);
                s->sin_cache_full[pos * half_rotary_full + i] = sinf(val);
            }
        }
    } else { s->cos_cache_full = NULL; s->sin_cache_full = NULL; }

    if (half_rotary_sliding > 0) {
        s->cos_cache_sliding = a_calloc((size_t)p->seq_len * half_rotary_sliding * sizeof(float));
        s->sin_cache_sliding = a_calloc((size_t)p->seq_len * half_rotary_sliding * sizeof(float));
        for (int pos = 0; pos < p->seq_len; pos++) {
            for (int i = 0; i < half_rotary_sliding; i++) {
                float freq = 1.0f / powf(p->rope_theta_sliding, (float)(2 * i) / rotary_sliding);
                float val = pos * freq;
                s->cos_cache_sliding[pos * half_rotary_sliding + i] = cosf(val);
                s->sin_cache_sliding[pos * half_rotary_sliding + i] = sinf(val);
            }
        }
    } else { s->cos_cache_sliding = NULL; s->sin_cache_sliding = NULL; }

    if (!s->x || !s->xb || !s->hb || !s->hb2 || !s->q || !s->k || !s->v ||
        !s->att || !s->logits || !s->key_cache || !s->value_cache ||
        !s->xq.q || !s->xq.s || !s->hq.q || !s->hq.s) {
        log_msg(stderr, "ERROR: Alloc failed!\n"); exit(EXIT_FAILURE);
    }

    if (p->seq_len > 1 && (!s->cos_cache_full || !s->sin_cache_full || !s->cos_cache_sliding || !s->sin_cache_sliding)) {
        log_msg(stderr, "ERROR: Alloc failed for RoPE cache!\n"); exit(EXIT_FAILURE);
    }

    s->allocated = 1;
}

void free_state_gemma4u(state_gemma4u *s) {
    if (!s->allocated) return;
    free(s->x); free(s->xb); free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
    free(s->xq.q); free(s->xq.s);
    free(s->hq.q); free(s->hq.s);
    free(s->cos_cache_full); free(s->sin_cache_full);
    free(s->cos_cache_sliding); free(s->sin_cache_sliding);
    s->allocated = 0;
}

void free_gemma4u(Gemma4Unified *model) {
    weights_gemma4u *w = &model->weights;
    int n_layer = model->config.n_layers;
    free_qt(&w->embed_tokens);
    free(w->rms_input_layernorm); free(w->rms_post_attn_layernorm);
    free(w->rms_pre_ffn_layernorm); free(w->rms_post_ffn_layernorm);
    free(w->rms_q_norm); free(w->rms_k_norm); free(w->rms_final_norm);
    free(w->norm_offsets);
    free_qt_array(w->q_proj, n_layer);
    free_qt_array(w->k_proj, n_layer);
    free_qt_array(w->v_proj, n_layer);
    free_qt_array(w->o_proj, n_layer);
    free_qt_array(w->gate_proj, n_layer);
    free_qt_array(w->up_proj, n_layer);
    free_qt_array(w->down_proj, n_layer);
    if (model->state.allocated) free_state_gemma4u(&model->state);
    free(model->layer_types);
}