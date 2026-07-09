#include "dolen_l3_common.h"


bool alloc_state_l3(L3 *_model, int seq_n) {
    state_l3 *_state = &(_model->state);
    config_l3 *_config = &(_model->config);

    _state->seq_n = seq_n;

    int32_t dim = _config->dim;
    int32_t head_size = _config->head_dim;
    int32_t kv_dim = _config->n_kv_heads * head_size;
    int32_t hidden_dim = _config->hidden_dim;
    int32_t q_dim = _config->n_heads * head_size;
    int32_t attn_out_dim = _config->n_heads * head_size;

    int32_t max_act_dim = dim;
    if (q_dim > max_act_dim) {
        max_act_dim = q_dim;
    }
    if (attn_out_dim > max_act_dim) {
        max_act_dim = attn_out_dim;
    }
    if (hidden_dim > max_act_dim) {
        max_act_dim = hidden_dim;
    }

    _state->_x = a_calloc((size_t)dim * sizeof(float));
    _state->_xb = a_calloc((size_t)max_act_dim * sizeof(float));
    _state->_xb2 = a_calloc((size_t)dim * sizeof(float));
    _state->_hb = a_calloc((size_t)hidden_dim * sizeof(float));
    _state->_hb2 = a_calloc((size_t)hidden_dim * sizeof(float));
    _state->_q = a_calloc((size_t)q_dim * sizeof(float));
    _state->_k = a_calloc((size_t)kv_dim * sizeof(float));
    _state->_v = a_calloc((size_t)kv_dim * sizeof(float));
    _state->_att = a_calloc((size_t)_config->n_heads * seq_n * sizeof(float));
    _state->_logits = a_calloc((size_t)_config->vocab_size * sizeof(float));

    int32_t num_groups = (max_act_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    _state->xq._data = a_calloc((size_t)max_act_dim * sizeof(int8_t));
    _state->xq._scales = a_calloc((size_t)num_groups * sizeof(float));
    _state->xq.type = Q_TYPE_Q8;
    _state->xq.rows = 1;
    _state->xq.cols = max_act_dim;

    _state->hq._data = a_calloc((size_t)max_act_dim * sizeof(int8_t));
    _state->hq._scales = a_calloc((size_t)num_groups * sizeof(float));
    _state->hq.type = Q_TYPE_Q8;
    _state->hq.rows = 1;
    _state->hq.cols = max_act_dim;

    _state->n_layers = _config->n_layers;
    _state->__key_cache = (float **)a_calloc((size_t)_config->n_layers * sizeof(float *));
    _state->__value_cache = (float **)a_calloc((size_t)_config->n_layers * sizeof(float *));
    
    if ((! _state->__key_cache) ||
            (! _state->__value_cache)) {
        log_msg(stderr, "ERROR: Alloc failed for KV cache pointer arrays\n");
        return false;
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        size_t cache_size = (size_t)seq_n * kv_dim * sizeof(float);
        _state->__key_cache[l] = a_calloc(cache_size);
        _state->__value_cache[l] = a_calloc(cache_size);
        
        if ((! _state->__key_cache[l]) ||
                (! _state->__value_cache[l])) {
            log_msg(stderr, "ERROR: Alloc failed for KV cache layer %d!\n", l);
            return false;
        }
    }

    int32_t rotary_dim = head_size;
    int32_t half_rot = rotary_dim / 2;
    _state->_cos_cache = a_calloc((size_t)seq_n * half_rot * sizeof(float));
    _state->_sin_cache = a_calloc((size_t)seq_n * half_rot * sizeof(float));
    
    float theta = _config->rope_theta;
    for (int32_t pos = 0; pos < seq_n; pos++) {
        for (int32_t i = 0; i < half_rot; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / rotary_dim);
            float val = (float)pos * freq;
            _state->_cos_cache[pos * half_rot + i] = cosf(val);
            _state->_sin_cache[pos * half_rot + i] = sinf(val);
        }
    }

    if ((! _state->_x) ||
            (! _state->_xb) ||
            (! _state->_xb2) ||
            (! _state->_hb) ||
            (! _state->_hb2) ||
            (! _state->_q) ||
            (! _state->_k) ||
            (! _state->_v) ||
            (! _state->_att) ||
            (! _state->_logits) ||
            (! _state->xq._data) ||
            (! _state->xq._scales) ||
            (! _state->hq._data) ||
            (! _state->hq._scales)) {
        log_msg(stderr, "ERROR: Alloc failed for state buffers!\n");
        return false;
    }

    _state->allocated = 1;
    return true;
}

void free_state_l3(state_l3 *_state) {
    if (! _state->allocated) {
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
    
    if (_state->__key_cache) {
        for (int32_t i = 0; i < _state->n_layers; i++) {
            if (_state->__key_cache[i]) {
                free(_state->__key_cache[i]);
            }
        }
        free(_state->__key_cache);
    }
    if (_state->__value_cache) {
        for (int32_t i = 0; i < _state->n_layers; i++) {
            if (_state->__value_cache[i]) {
                free(_state->__value_cache[i]);
            }
        }
        free(_state->__value_cache);
    }

    free(_state->xq._data);
    free(_state->xq._scales);
    free(_state->hq._data);
    free(_state->hq._scales);
    free(_state->_cos_cache);
    free(_state->_sin_cache);

    _state->allocated = 0;
}

void free_l3(L3 *_model) {
    if (! _model) {
        return;
    }
    config_l3 *_config = &(_model->config);
    weights_l3 *_weights = &(_model->weights);

    free_qt(&(_weights->embed_tokens_weight));
    free_qt_array(_weights->_rms_att_weight, _config->n_layers);
    free_qt_array(_weights->_wq, _config->n_layers);
    free_qt_array(_weights->_wk, _config->n_layers);
    free_qt_array(_weights->_wv, _config->n_layers);
    free_qt_array(_weights->_wo, _config->n_layers);
    free_qt_array(_weights->_rms_ffn_weight, _config->n_layers);
    free_qt_array(_weights->_w1, _config->n_layers);
    free_qt_array(_weights->_w2, _config->n_layers);
    free_qt_array(_weights->_w3, _config->n_layers);
    free_qt(&(_weights->rms_final_weight));
    
    if (! _config->tie_word_embeddings) {
        free_qt(&_weights->wcls);
    }

    if (_model->state.allocated) {
        free_state_l3(&(_model->state));
    }

    free_tokenizer(&(_model->tokenizer1));

    memset(_model, 0, sizeof(L3));
}

