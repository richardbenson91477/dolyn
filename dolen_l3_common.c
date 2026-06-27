#include "dolen_l3_common.h"

void alloc_state_l3(state_l3 *s, config_l3 *p) {
    int dim = p->dim;
    int head_size = p->head_dim;
    int kv_dim = p->n_kv_heads * head_size;
    int hidden_dim = p->hidden_dim;
    int q_dim = p->n_heads * head_size;
    int attn_out_dim = p->n_heads * head_size;

    int max_act_dim = dim;
    if (q_dim > max_act_dim) {
        max_act_dim = q_dim;
    }
    if (attn_out_dim > max_act_dim) {
        max_act_dim = attn_out_dim;
    }
    if (hidden_dim > max_act_dim) {
        max_act_dim = hidden_dim;
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

    int num_groups = (max_act_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    s->xq.data = a_calloc((size_t)max_act_dim * sizeof(int8_t));
    s->xq.s = a_calloc((size_t)num_groups * sizeof(float));
    s->xq.type = Q_TYPE_Q8;
    s->xq.rows = 1;
    s->xq.cols = max_act_dim;

    s->hq.data = a_calloc((size_t)max_act_dim * sizeof(int8_t));
    s->hq.s = a_calloc((size_t)num_groups * sizeof(float));
    s->hq.type = Q_TYPE_Q8;
    s->hq.rows = 1;
    s->hq.cols = max_act_dim;

    s->n_layers = p->n_layers;
    s->key_cache = (float **)a_calloc((size_t)p->n_layers * sizeof(float *));
    s->value_cache = (float **)a_calloc((size_t)p->n_layers * sizeof(float *));
    
    if ((! s->key_cache) ||
            (! s->value_cache)) {
        log_msg(stderr, "ERROR: Alloc failed for KV cache pointer arrays\n");
        exit(EXIT_FAILURE);
    }

    for (int l = 0; l < p->n_layers; l++) {
        size_t cache_size = (size_t)p->seq_len * kv_dim * sizeof(float);
        s->key_cache[l] = a_calloc(cache_size);
        s->value_cache[l] = a_calloc(cache_size);
        
        // CRITICAL FIX: Check if KV cache allocation failed
        if ((! s->key_cache[l]) ||
                (! s->value_cache[l])) {
            log_msg(stderr, "ERROR: Alloc failed for KV cache layer %d!\n", l);
            log_msg(stderr, "       Requested size: %zu bytes (seq_len=%d, kv_dim=%d)\n", 
                    cache_size, p->seq_len, kv_dim);
            log_msg(stderr, "       This usually means you ran out of RAM, or seq_len/kv_dim are unexpectedly large.\n");
            exit(EXIT_FAILURE);
        }
    }

    int rotary_dim = head_size; 
    int half_rot = rotary_dim / 2;
    s->cos_cache = a_calloc((size_t)p->seq_len * half_rot * sizeof(float));
    s->sin_cache = a_calloc((size_t)p->seq_len * half_rot * sizeof(float));
    
    float theta = p->rope_theta;
    for (int pos = 0; pos < p->seq_len; pos++) {
        for (int i = 0; i < half_rot; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / rotary_dim);
            float val = (float)pos * freq;
            s->cos_cache[pos * half_rot + i] = cosf(val);
            s->sin_cache[pos * half_rot + i] = sinf(val);
        }
    }

    if ((! s->x) ||
            (! s->xb) ||
            (! s->xb2) ||
            (! s->hb) ||
            (! s->hb2) ||
            (! s->q) ||
            (! s->k) ||
            (! s->v) ||
            (! s->att) ||
            (! s->logits) ||
            (! s->xq.data) ||
            (! s->xq.s) ||
            (! s->hq.data) ||
            (! s->hq.s)) {
        log_msg(stderr, "ERROR: Alloc failed for state buffers!\n");
        exit(EXIT_FAILURE);
    }
    s->allocated = 1;
}

void free_state_l3(state_l3 *s) {
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
    
    if (s->key_cache) {
        for (int i = 0; i < s->n_layers; i++) {
            if (s->key_cache[i]) {
                free(s->key_cache[i]);
            }
        }
        free(s->key_cache);
    }
    if (s->value_cache) {
        for (int i = 0; i < s->n_layers; i++) {
            if (s->value_cache[i]) {
                free(s->value_cache[i]);
            }
        }
        free(s->value_cache);
    }

    free(s->xq.data);
    free(s->xq.s);
    free(s->hq.data);
    free(s->hq.s);
    if (s->cos_cache) {
        free(s->cos_cache);
    }
    if (s->sin_cache) {
        free(s->sin_cache);
    }

    s->allocated = 0;
}

void free_l3(L3 *model) {
    if (! model) {
        return;
    }
    config_l3 *p = &model->config;
    weights_l3 *w = &model->weights;

    free_qt(&w->embed_tokens_weight);
    free_qt_array(w->rms_att_weight, p->n_layers);
    free_qt_array(w->wq, p->n_layers);
    free_qt_array(w->wk, p->n_layers);
    free_qt_array(w->wv, p->n_layers);
    free_qt_array(w->wo, p->n_layers);
    free_qt_array(w->rms_ffn_weight, p->n_layers);
    free_qt_array(w->w1, p->n_layers);
    free_qt_array(w->w2, p->n_layers);
    free_qt_array(w->w3, p->n_layers);
    free_qt(&w->rms_final_weight);
    
    if (! p->tie_word_embeddings) {
        free_qt(&w->wcls);
    }

    if (model->state.allocated) {
        free_state_l3(&model->state);
    }

    free_tokenizer(&model->tokenizer);

    memset(model, 0, sizeof(L3));
}

