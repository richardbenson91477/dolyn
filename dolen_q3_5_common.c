#include "dolen_q3_5_common.h"

void alloc_state_q3_5(state_q3_5 *s, config_q3_5 *p) {
    int dim = p->dim;
    int head_size = p->d_head > 0 ? p->d_head : dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int hidden_dim = p->n_mlp;
    int q_dim = p->n_heads * head_size * 2;
    int attn_dim = p->n_heads * head_size;
    size_t n_kv_layers = (size_t)p->n_full_attn_layers;
    size_t n_linear_layers = (size_t)p->n_linear_attn_layers;
    int value_dim = p->n_linear_v_heads * p->d_linear_v;

    int max_act_dim = dim;
    if (q_dim > max_act_dim) {
        max_act_dim = q_dim;
    }
    if (attn_dim > max_act_dim) {
        max_act_dim = attn_dim;
    }
    if (hidden_dim > max_act_dim) {
        max_act_dim = hidden_dim;
    }
    if (value_dim > max_act_dim) {
        max_act_dim = value_dim;
    }

    s->x = a_calloc((size_t)dim * sizeof(float));
    s->xb = a_calloc((size_t)max_act_dim * sizeof(float));
    s->xb2 = a_calloc((size_t)dim * sizeof(float));
    s->hb = a_calloc((size_t)hidden_dim * sizeof(float));
    s->hb2 = a_calloc((size_t)hidden_dim * sizeof(float));
    s->q = a_calloc((size_t)q_dim * sizeof(float));
    s->k = a_calloc((size_t)kv_dim * sizeof(float));
    s->v = a_calloc((size_t)kv_dim * sizeof(float));
    s->att = a_calloc((size_t)p->n_heads * p->seq_len * sizeof(float));
    s->logits = a_calloc((size_t)p->vocab_size * sizeof(float));
    s->gate = a_calloc((size_t)p->n_heads * head_size * sizeof(float));

    int num_groups = (max_act_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    s->xq.q = (int8_t *)a_calloc((size_t)max_act_dim * sizeof(int8_t));
    s->xq.s = (float *)a_calloc((size_t)num_groups * sizeof(float));
    s->xq.rows = 1;
    s->xq.cols = max_act_dim;

    s->hq.q = (int8_t *)a_calloc((size_t)max_act_dim * sizeof(int8_t));
    s->hq.s = (float *)a_calloc((size_t)num_groups * sizeof(float));
    s->hq.rows = 1;
    s->hq.cols = max_act_dim;

    if (n_kv_layers > 0) {
        s->key_cache = a_calloc(n_kv_layers * p->seq_len * kv_dim * sizeof(float));
        s->value_cache = a_calloc(n_kv_layers * p->seq_len * kv_dim * sizeof(float));
    }

    if (n_linear_layers > 0) {
        int key_dim = p->n_linear_k_heads * p->d_linear_k;
        int conv_dim = key_dim * 2 + value_dim;

        s->qkv = a_calloc((size_t)conv_dim * sizeof(float));
        s->z = a_calloc((size_t)value_dim * sizeof(float));
        s->beta = a_calloc((size_t)p->n_linear_v_heads * sizeof(float));
        s->g = a_calloc((size_t)p->n_linear_v_heads * sizeof(float));
        s->linear_out = a_calloc((size_t)value_dim * sizeof(float));
        s->conv_state = a_calloc(n_linear_layers * conv_dim * p->linear_conv_kernel * sizeof(float));
        s->S = a_calloc(n_linear_layers * p->n_linear_v_heads * p->d_linear_k * p->d_linear_v * sizeof(float));
        s->delta_S = a_calloc((size_t)p->n_linear_v_heads * p->d_linear_v * sizeof(float));
    }

    int rotary_partial = (int)((float)head_size * p->rope_partial_rotary_factor);

    if (rotary_partial > 0) {
        s->cos_cache = (float *)a_calloc((size_t)p->seq_len * rotary_partial * sizeof(float));
        s->sin_cache = (float *)a_calloc((size_t)p->seq_len * rotary_partial * sizeof(float));
        float theta = p->rope_theta;
        for (int pos = 0; pos < p->seq_len; pos++) {
            for (int i = 0; i < rotary_partial; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / rotary_partial);
                float val = pos * freq;
                s->cos_cache[pos * rotary_partial + i] = cosf(val);
                s->sin_cache[pos * rotary_partial + i] = sinf(val);
            }
        }
    } else {
        s->cos_cache = NULL;
        s->sin_cache = NULL;
    }

    if (! s->x || ! s->xb || ! s->xb2 || ! s->hb || ! s->hb2 || ! s->q || ! s->k || ! s->v || ! s->att || ! s->logits ||
            ! s->gate || ! s->xq.q || ! s->xq.s || ! s->hq.q || ! s->hq.s) {
        log_msg(stderr, "ERROR: Alloc failed!\n");
        exit(EXIT_FAILURE);
    }
    if (n_kv_layers > 0 && (! s->key_cache || ! s->value_cache)) {
        log_msg(stderr, "ERROR: alloc failed for KV cache!\n");
        exit(EXIT_FAILURE);
    }

    s->allocated = 1;
}

void free_state_q3_5(state_q3_5 *s) {
    if (! s->allocated) {
        return;
    }

    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->gate);
    free(s->key_cache);
    free(s->value_cache);
    free(s->qkv);
    free(s->z);
    free(s->beta);
    free(s->g);
    free(s->linear_out);
    free(s->conv_state);
    free(s->S);
    free(s->delta_S);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);

    if (s->cos_cache) {
        free(s->cos_cache);
    }
    if (s->sin_cache) {
        free(s->sin_cache);
    }

    s->allocated = 0;
}

void free_q3_5(Q3_5 *model_q3_5) {
    weights_q3_5 *w = &model_q3_5->weights;
    int n_full_attn = model_q3_5->config.n_full_attn_layers;
    int n_linear_attn = model_q3_5->config.n_linear_attn_layers;
    int n_layer = model_q3_5->config.n_layer;

    free_qt(&w->token_embedding_table);
    free(w->rms_att_weight);
    free_qt_array(w->wq, n_full_attn);
    free_qt_array(w->wk, n_full_attn);
    free_qt_array(w->wv, n_full_attn);
    free_qt_array(w->wo, n_full_attn);
    free(w->q_norm);
    free(w->k_norm);
    free_qt_array(w->in_proj_qkv, n_linear_attn);
    free_qt_array(w->in_proj_z, n_linear_attn);
    free(w->in_proj_b);
    free(w->in_proj_a);
    free(w->conv1d_weight);
    free(w->dt_bias);
    free(w->A_log);
    free(w->linear_norm);
    free_qt_array(w->out_proj, n_linear_attn);
    free(w->rms_ffn_weight);
    free_qt_array(w->w1, n_layer);
    free_qt_array(w->w2, n_layer);
    free_qt_array(w->w3, n_layer);
    free(w->rms_final_weight);

    if (! model_q3_5->config.tie_word_embeddings) {
        free_qt(&w->wcls);
    }

    free(model_q3_5->layer_types);
    free(model_q3_5->attn_layer_indices);
    free(model_q3_5->deltanet_layer_indices);

    if (model_q3_5->state.allocated) {
        free_state_q3_5(&model_q3_5->state);
    }
}
