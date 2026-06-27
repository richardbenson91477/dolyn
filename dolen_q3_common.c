#include "dolen_q3_common.h"


void alloc_state_q3(state_q3 *_state, config_q3 *_config) {
    int all_heads_dim = _config->n_heads * _config->head_dim;
    int kv_dim = _config->n_kv_heads * _config->head_dim;

    _state->_x = a_calloc((size_t)_config->dim * sizeof(float));
    _state->_xb = a_calloc((all_heads_dim > _config->dim ? all_heads_dim : _config->dim) * sizeof(float));
    _state->_hb = a_calloc((size_t)_config->hidden_dim * sizeof(float));
    _state->_hb2 = a_calloc((size_t)_config->hidden_dim * sizeof(float));

    int xq_size = all_heads_dim > _config->dim ? all_heads_dim : _config->dim;

    int xq_num_groups = (xq_size + GROUP_SIZE - 1) / GROUP_SIZE;
    int hq_num_groups = (_config->hidden_dim + GROUP_SIZE - 1) / GROUP_SIZE;

    _state->xq._data = a_calloc((size_t)xq_size * sizeof(int8_t));
    _state->xq._scales = a_calloc((size_t)xq_num_groups * sizeof(float));
    _state->xq.type = Q_TYPE_Q8;

    _state->hq._data = a_calloc((size_t)_config->hidden_dim * sizeof(int8_t));
    _state->hq._scales = a_calloc((size_t)hq_num_groups * sizeof(float));
    _state->hq.type = Q_TYPE_Q8;

    _state->_q = a_calloc((size_t)all_heads_dim * sizeof(float));
    _state->_k = a_calloc((size_t)kv_dim * sizeof(float));
    _state->_v = a_calloc((size_t)kv_dim * sizeof(float));
    _state->_att = a_calloc((size_t)_config->n_heads * _config->seq_len * sizeof(float));
    _state->_logits = a_calloc((size_t)_config->vocab_size * sizeof(float));
    _state->_key_cache = a_calloc((size_t)_config->n_layers * _config->seq_len * kv_dim * sizeof(float));
    _state->_value_cache = a_calloc((size_t)_config->n_layers * _config->seq_len * kv_dim * sizeof(float));

    int rotary_half = _config->head_dim / 2;
    if (rotary_half > 0) {
        _state->_cos_cache = (float *)a_calloc((size_t)_config->seq_len * rotary_half * sizeof(float));
        _state->_sin_cache = (float *)a_calloc((size_t)_config->seq_len * rotary_half * sizeof(float));
        for (int pos = 0; pos < _config->seq_len; pos++) {
            float scaled_pos = pos / _config->rope_scaling_factor;
            for (int i = 0; i < rotary_half; i++) {
                float freq = 1.0f / powf(_config->rope_theta, (float)i / rotary_half);
                float val = scaled_pos * freq;
                _state->_cos_cache[pos * rotary_half + i] = cosf(val);
                _state->_sin_cache[pos * rotary_half + i] = sinf(val);
            }
        }
    }
    else {
        _state->_cos_cache = NULL;
        _state->_sin_cache = NULL;
    }

    if ((! _state->_x) ||
            (! _state->_xb) ||
            (! _state->_hb) ||
            (! _state->_hb2) ||
            (! _state->xq._data) ||
            (! _state->xq._scales) ||
            (! _state->hq._data) ||
            (! _state->hq._scales) ||
            (! _state->_q) ||
            (! _state->_k) ||
            (! _state->_v) ||
            (! _state->_att) ||
            (! _state->_logits) ||
            (! _state->_key_cache) ||
            (! _state->_value_cache) ||
            ((rotary_half > 0) &&
                ((! _state->_cos_cache) ||
                 (! _state->_sin_cache)))) {
        log_msg(stderr, "ERROR: alloc failed!\n");
        exit(EXIT_FAILURE);
    }

    _state->allocated = 1;
}

void free_state_q3(state_q3 *_state) {
    if (! _state->allocated) {
        return;
    }

    free(_state->_x);
    free(_state->_xb);
    free(_state->_hb);
    free(_state->_hb2);
    free(_state->xq._data);
    free(_state->xq._scales);
    free(_state->hq._data);
    free(_state->hq._scales);
    free(_state->_q);
    free(_state->_k);
    free(_state->_v);
    free(_state->_att);
    free(_state->_logits);
    free(_state->_key_cache);
    free(_state->_value_cache);

    if (_state->_cos_cache) {
        free(_state->_cos_cache);
    }
    if (_state->_sin_cache) {
        free(_state->_sin_cache);
    }

    _state->allocated = 0;
}

void free_q3(Q3 *model_q3) {
    weights_q3 *_weights = &(model_q3->weights);
    int n_layer = model_q3->config.n_layers;

    free_qt(&(_weights->embed_tokens_weight));
    free_qt_array(_weights->_rms_att_weight, n_layer);
    free_qt_array(_weights->_rms_ffn_weight, n_layer);
    free_qt(&(_weights->rms_final_weight));

    free_qt_array(_weights->_q_norm, n_layer);
    free_qt_array(_weights->_k_norm, n_layer);

    free_qt_array(_weights->_wq, n_layer);
    free_qt_array(_weights->_wk, n_layer);
    free_qt_array(_weights->_wv, n_layer);
    free_qt_array(_weights->_wo, n_layer);
    free_qt_array(_weights->_w1, n_layer);
    free_qt_array(_weights->_w2, n_layer);
    free_qt_array(_weights->_w3, n_layer);

    if (! model_q3->config.shared_classifier) {
        free_qt(&(_weights->wcls));
    }

    if (model_q3->state.allocated == 1) {
        free_state_q3(&(model_q3->state));
    }

    free_tokenizer(&(model_q3->tokenizer1));
}

