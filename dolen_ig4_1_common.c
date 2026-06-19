#include "dolen_ig4_1_common.h"

void alloc_state_ig4_1(state_ig4_1 *s, config_ig4_1 *p) {
    int dim = p->dim;
    int head_size = p->d_head > 0 ? p->d_head : dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int hidden_dim = p->n_mlp;
    int attn_dim = p->n_heads * head_size;

    int max_act_dim = dim;
    if (attn_dim > max_act_dim) {
        max_act_dim = attn_dim;
    }
    if (hidden_dim > max_act_dim) {
        max_act_dim = hidden_dim;
    }

    s->x = a_calloc((size_t)dim * sizeof(float));
    s->xb = a_calloc((size_t)max_act_dim * sizeof(float));
    s->xb2 = a_calloc((size_t)dim * sizeof(float));
    s->hb = a_calloc((size_t)hidden_dim * sizeof(float));
    s->hb2 = a_calloc((size_t)hidden_dim * sizeof(float));
    s->q = a_calloc((size_t)attn_dim * sizeof(float));
    s->k = a_calloc((size_t)kv_dim * sizeof(float));
    s->v = a_calloc((size_t)kv_dim * sizeof(float));
    s->att = a_calloc((size_t)p->n_heads * p->seq_len * sizeof(float));
    s->logits = a_calloc((size_t)p->vocab_size * sizeof(float));

    int num_groups = (max_act_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    s->xq.q = (int8_t *)a_calloc((size_t)max_act_dim * sizeof(int8_t));
    s->xq.s = (float *)a_calloc((size_t)num_groups * sizeof(float));
    s->xq.rows = 1;
    s->xq.cols = max_act_dim;

    s->hq.q = (int8_t *)a_calloc((size_t)max_act_dim * sizeof(int8_t));
    s->hq.s = (float *)a_calloc((size_t)num_groups * sizeof(float));
    s->hq.rows = 1;
    s->hq.cols = max_act_dim;

    s->key_cache = a_calloc((size_t)p->n_layer * p->seq_len * kv_dim * sizeof(float));
    s->value_cache = a_calloc((size_t)p->n_layer * p->seq_len * kv_dim * sizeof(float));

    int rotary_dim = head_size;
    s->cos_cache = (float *)a_calloc((size_t)p->seq_len * rotary_dim * sizeof(float));
    s->sin_cache = (float *)a_calloc((size_t)p->seq_len * rotary_dim * sizeof(float));
    float theta = p->rope_theta;
    for (int pos = 0; pos < p->seq_len; pos++) {
        for (int i = 0; i < rotary_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / rotary_dim);
            float val = pos * freq;
            s->cos_cache[pos * rotary_dim + i] = cosf(val);
            s->sin_cache[pos * rotary_dim + i] = sinf(val);
        }
    }

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v ||
            !s->att || !s->logits || !s->xq.q || !s->xq.s || !s->hq.q || !s->hq.s ||
            !s->key_cache || !s->value_cache) {
        log_msg(stderr, "ERROR: Alloc failed!\n");
        exit(EXIT_FAILURE);
    }

    s->allocated = 1;
}

void free_state_ig4_1(state_ig4_1 *s) {
    if (!s->allocated) {
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
    free(s->key_cache);
    free(s->value_cache);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->cos_cache);
    free(s->sin_cache);

    s->allocated = 0;
}

void free_ig4_1(IG4_1 *model_ig4_1) {
    weights_ig4_1 *w = &model_ig4_1->weights;
    int n_layer = model_ig4_1->config.n_layer;

    free_qt(&w->token_embedding_table);
    free(w->rms_att_weight);
    free_qt_array(w->wq, n_layer);
    free_qt_array(w->wk, n_layer);
    free_qt_array(w->wv, n_layer);
    free_qt_array(w->wo, n_layer);
    free(w->rms_ffn_weight);
    free_qt_array(w->w1, n_layer);
    free_qt_array(w->w2, n_layer);
    free_qt_array(w->w3, n_layer);
    free(w->rms_final_weight);
    if (!model_ig4_1->config.tie_word_embeddings) {
        free_qt(&w->wcls);
    }

    if (model_ig4_1->state.allocated) {
        free_state_ig4_1(&model_ig4_1->state);
    }
}

