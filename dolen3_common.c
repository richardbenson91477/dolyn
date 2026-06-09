#include "dolen3_common.h"


void alloc_state_qwen3(state_qwen3 *s, config_qwen3 *p) {
    int all_heads_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;

    s->x = a_calloc((size_t)p->dim * sizeof(float));
    s->xb = a_calloc((all_heads_dim > p->dim ? all_heads_dim : p->dim) * sizeof(float));
    s->hb = a_calloc((size_t)p->hidden_dim * sizeof(float));
    s->hb2 = a_calloc((size_t)p->hidden_dim * sizeof(float));

    int xq_size = all_heads_dim > p->dim ? all_heads_dim : p->dim;

    int xq_num_groups = (xq_size + GS - 1) / GS;
    int hq_num_groups = (p->hidden_dim + GS - 1) / GS;

    s->xq.q = a_calloc((size_t)xq_size * sizeof(int8_t));
    s->xq.s = a_calloc((size_t)xq_num_groups * sizeof(float));

    s->hq.q = a_calloc((size_t)p->hidden_dim * sizeof(int8_t));
    s->hq.s = a_calloc((size_t)hq_num_groups * sizeof(float));

    s->q = a_calloc((size_t)all_heads_dim * sizeof(float));
    s->k = a_calloc((size_t)kv_dim * sizeof(float));
    s->v = a_calloc((size_t)kv_dim * sizeof(float));
    s->att = a_calloc((size_t)p->n_heads * p->seq_len * sizeof(float));
    s->logits = a_calloc((size_t)p->vocab_size * sizeof(float));
    s->key_cache = a_calloc((size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float));
    s->value_cache = a_calloc((size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float));

    // Pre-compute RoPE cos/sin cache
    int rotary_half = p->head_dim / 2;
    if (rotary_half > 0) {
        s->cos_cache = (float *)a_calloc((size_t)p->seq_len * rotary_half * sizeof(float));
        s->sin_cache = (float *)a_calloc((size_t)p->seq_len * rotary_half * sizeof(float));
        for (int pos = 0; pos < p->seq_len; pos++) {
            float scaled_pos = pos / p->rope_scaling_factor;
            for (int i = 0; i < rotary_half; i++) {
                float freq = 1.0f / powf(p->rope_theta, (float)i / rotary_half);
                float val = scaled_pos * freq;
                s->cos_cache[pos * rotary_half + i] = cosf(val);
                s->sin_cache[pos * rotary_half + i] = sinf(val);
            }
        }
    } else {
        s->cos_cache = NULL;
        s->sin_cache = NULL;
    }

    if (! s->x || ! s->xb || ! s->hb || ! s->hb2 || ! s->xq.q || ! s->xq.s ||
            ! s->hq.q || ! s->hq.s ||
            ! s->q || ! s->k || ! s->v || ! s->att || ! s->logits ||
            ! s->key_cache || ! s->value_cache ||
            (rotary_half > 0 && (!s->cos_cache || !s->sin_cache))) {
        fprintf(stderr, "alloc failed!\n");
        exit(EXIT_FAILURE);
    }

    s->allocated = 1;
}

void free_state_qwen3(state_qwen3 *s) {
    if (! s->allocated) {
        return;
    }

    free(s->x);
    free(s->xb);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);

    if (s->cos_cache) {
        free(s->cos_cache);
    }
    if (s->sin_cache) {
        free(s->sin_cache);
    }

    s->allocated = 0;
}

void free_qwen3(Qwen3 *model_qwen3) {
    weights_qwen3 *w = &model_qwen3->weights;
    int n_layer = model_qwen3->config.n_layers;

    free_qt(&w->token_embedding_table);
    free(w->rms_att_weight);
    free(w->rms_ffn_weight);
    free(w->rms_final_weight);
    free(w->q_norm_weights);
    free(w->k_norm_weights);

    free_qt_array(w->wq, n_layer);
    free_qt_array(w->wk, n_layer);
    free_qt_array(w->wv, n_layer);
    free_qt_array(w->wo, n_layer);
    free_qt_array(w->w1, n_layer);
    free_qt_array(w->w2, n_layer);
    free_qt_array(w->w3, n_layer);

    if (! model_qwen3->config.shared_classifier) {
        free_qt(&w->wcls);
    }

    if (model_qwen3->state.allocated == 1) {
        free_state_qwen3(&model_qwen3->state);
    }
}

