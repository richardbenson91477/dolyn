#include "dolen_ig4_1_common.h"

void alloc_state_ig4_1(state_ig4_1 *_state, config_ig4_1 *_config) {
    int dim = _config->dim;
    int head_size = _config->d_head > 0 ? _config->d_head : dim / _config->n_heads;
    int kv_dim = _config->n_kv_heads * head_size;
    int hidden_dim = _config->n_mlp;
    int attn_dim = _config->n_heads * head_size;

    int max_act_dim = dim;
    if (attn_dim > max_act_dim) {
        max_act_dim = attn_dim;
    }
    if (hidden_dim > max_act_dim) {
        max_act_dim = hidden_dim;
    }

    _state->_x = a_calloc((size_t)dim * sizeof(float));
    _state->_xb = a_calloc((size_t)max_act_dim * sizeof(float));
    _state->_xb2 = a_calloc((size_t)dim * sizeof(float));
    _state->_hb = a_calloc((size_t)hidden_dim * sizeof(float));
    _state->_hb2 = a_calloc((size_t)hidden_dim * sizeof(float));
    _state->_q = a_calloc((size_t)attn_dim * sizeof(float));
    _state->_k = a_calloc((size_t)kv_dim * sizeof(float));
    _state->_v = a_calloc((size_t)kv_dim * sizeof(float));
    _state->_att = a_calloc((size_t)_config->n_heads * _config->seq_len * sizeof(float));
    _state->_logits = a_calloc((size_t)_config->vocab_size * sizeof(float));

    int num_groups = (max_act_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    _state->xq._data = (int8_t *)a_calloc((size_t)max_act_dim * sizeof(int8_t));
    _state->xq._scales = (float *)a_calloc((size_t)num_groups * sizeof(float));
    _state->xq.type = Q_TYPE_Q8;
    _state->xq.rows = 1;
    _state->xq.cols = max_act_dim;

    _state->hq._data = (int8_t *)a_calloc((size_t)max_act_dim * sizeof(int8_t));
    _state->hq._scales = (float *)a_calloc((size_t)num_groups * sizeof(float));
    _state->hq.type = Q_TYPE_Q8;
    _state->hq.rows = 1;
    _state->hq.cols = max_act_dim;

    _state->_key_cache = a_calloc((size_t)_config->n_layer * _config->seq_len * kv_dim * sizeof(float));
    _state->_value_cache = a_calloc((size_t)_config->n_layer * _config->seq_len * kv_dim * sizeof(float));

    int rotary_dim = head_size;
    _state->_cos_cache = (float *)a_calloc((size_t)_config->seq_len * rotary_dim * sizeof(float));
    _state->_sin_cache = (float *)a_calloc((size_t)_config->seq_len * rotary_dim * sizeof(float));
    float theta = _config->rope_theta;
    for (int pos = 0; pos < _config->seq_len; pos++) {
        for (int i = 0; i < rotary_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / rotary_dim);
            float val = pos * freq;
            _state->_cos_cache[pos * rotary_dim + i] = cosf(val);
            _state->_sin_cache[pos * rotary_dim + i] = sinf(val);
        }
    }

    if ((!_state->_x) ||
            (!_state->_xb) ||
            (!_state->_xb2) ||
            (!_state->_hb) ||
            (!_state->_hb2) ||
            (!_state->_q) ||
            (!_state->_k) ||
            (!_state->_v) ||
            (!_state->_att) ||
            (!_state->_logits) ||
            (!_state->xq._data) ||
            (!_state->xq._scales) ||
            (!_state->hq._data) ||
            (!_state->hq._scales) ||
            (!_state->_key_cache) ||
            (!_state->_value_cache)) {
        log_msg(stderr, "ERROR: Alloc failed!\n");
        exit(EXIT_FAILURE);
    }

    _state->allocated = 1;
}

void free_state_ig4_1(state_ig4_1 *_state) {
    if (!_state->allocated) {
        return;
    }

    free(_state->_x);
    free(_state->_xb);
    free(_state->_xb2);
    free(_state->_hb);
    free(_state->_hb2);
    free(_state->_q);
    free(_state->_k);
    free(_state->_v);
    free(_state->_att);
    free(_state->_logits);
    free(_state->_key_cache);
    free(_state->_value_cache);
    free(_state->xq._data);
    free(_state->xq._scales);
    free(_state->hq._data);
    free(_state->hq._scales);
    free(_state->_cos_cache);
    free(_state->_sin_cache);

    _state->allocated = 0;
}

void free_ig4_1(IG4_1 *_model) {
    if (!_model) {
        return;
    }
    weights_ig4_1 *_weights = &(_model->weights);
    int n_layer = _model->config.n_layer;

    free_qt(&(_weights->embed_tokens_weight));
    free_qt_array(_weights->_rms_att_weight, n_layer);
    free_qt_array(_weights->_wq, n_layer);
    free_qt_array(_weights->_wk, n_layer);
    free_qt_array(_weights->_wv, n_layer);
    free_qt_array(_weights->_wo, n_layer);
    free_qt_array(_weights->_rms_ffn_weight, n_layer);
    free_qt_array(_weights->_w1, n_layer);
    free_qt_array(_weights->_w2, n_layer);
    free_qt_array(_weights->_w3, n_layer);
    free_qt(&(_weights->rms_final_weight));

    if (!_model->config.tie_word_embeddings) {
        free_qt(&(_weights->wcls));
    }

    if (_model->state.allocated) {
        free_state_ig4_1(&(_model->state));
    }

    free_tokenizer(&(_model->tokenizer));
}