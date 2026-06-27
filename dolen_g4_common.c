#include "dolen_g4_common.h"


void alloc_state_g4(state_g4 *_state, config_g4 *_config, weights_g4 *_weights, const int *_layer_types) {
    int max_head_dim = _config->head_dim > _config->global_head_dim ? _config->head_dim : _config->global_head_dim;
    int max_kv_heads = _config->n_kv_heads > _config->n_global_kv_heads ? _config->n_kv_heads : _config->n_global_kv_heads;
    int max_kv_dim = max_kv_heads * max_head_dim;
    int attn_out_dim = _config->n_heads * max_head_dim;

    int cache_stride_full = _config->global_head_dim / 2;
    int cache_stride_sliding = _config->head_dim / 2;

    int max_act_dim = _config->dim;
    if (attn_out_dim > max_act_dim) {
        max_act_dim = attn_out_dim;
    }
    if (_config->hidden_dim > max_act_dim) {
        max_act_dim = _config->hidden_dim;
    }

    _state->_x = a_calloc((size_t)_config->dim * sizeof(float));
    _state->_xb = a_calloc((size_t)max_act_dim * sizeof(float));
    _state->_hb = a_calloc((size_t)_config->hidden_dim * sizeof(float));
    _state->_hb2 = a_calloc((size_t)_config->hidden_dim * sizeof(float));
    _state->_q = a_calloc((size_t)attn_out_dim * sizeof(float));
    _state->_k = a_calloc((size_t)max_kv_dim * sizeof(float));
    _state->_k_raw = a_calloc((size_t)max_kv_dim * sizeof(float));
    _state->_v = a_calloc((size_t)max_kv_dim * sizeof(float));
    _state->_att = a_calloc((size_t)_config->n_heads * _config->seq_len * sizeof(float));
    _state->_logits = a_calloc((size_t)_config->vocab_size * sizeof(float));

    _state->n_layers = _config->n_layers;
    _state->__key_cache = (float **)a_calloc((size_t)_config->n_layers * sizeof(float *));
    _state->__value_cache = (float **)a_calloc((size_t)_config->n_layers * sizeof(float *));
    for (int l = 0; l < _config->n_layers; l++) {
        int is_full = _layer_types ? _layer_types[l] : 0;
        int kv_heads = is_full ? _config->n_global_kv_heads : _config->n_kv_heads;
        int head_dim = is_full ? _config->global_head_dim : _config->head_dim;
        int kv_dim = kv_heads * head_dim;
        _state->__key_cache[l] = a_calloc((size_t)_config->seq_len * kv_dim * sizeof(float));
        _state->__value_cache[l] = a_calloc((size_t)_config->seq_len * kv_dim * sizeof(float));
    }

    int num_groups_xq = (max_act_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    _state->xq._data = a_calloc((size_t)max_act_dim * sizeof(int8_t));
    _state->xq._scales = a_calloc((size_t)num_groups_xq * sizeof(float));
    _state->xq.type = Q_TYPE_Q8;
    _state->xq.rows = 1;
    _state->xq.cols = max_act_dim;

    int num_groups_hq = (_config->hidden_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    _state->hq._data = a_calloc((size_t)_config->hidden_dim * sizeof(int8_t));
    _state->hq._scales = a_calloc((size_t)num_groups_hq * sizeof(float));
    _state->hq.type = Q_TYPE_Q8;
    _state->hq.rows = 1;
    _state->hq.cols = _config->hidden_dim;

    int rotary_dim_full = (int)(_config->rope_partial_factor * _config->global_head_dim);

    if (cache_stride_full > 0) {
        _state->_cos_cache_full = a_calloc((size_t)_config->seq_len * cache_stride_full * sizeof(float));
        _state->_sin_cache_full = a_calloc((size_t)_config->seq_len * cache_stride_full * sizeof(float));

        for (int pos = 0; pos < _config->seq_len; pos++) {
            for (int i = 0; i < cache_stride_full; i++) {
                float cos_val, sin_val;
                if (i < rotary_dim_full / 2) {
                    float freq = 1.0f / powf(_config->rope_theta_full, (float)(2 * i) / _config->global_head_dim);
                    float val = (float)pos * freq;
                    cos_val = cosf(val);
                    sin_val = sinf(val);
                }
                else {
                    cos_val = 1.0f;
                    sin_val = 0.0f;
                }
                _state->_cos_cache_full[pos * cache_stride_full + i] = cos_val;
                _state->_sin_cache_full[pos * cache_stride_full + i] = sin_val;
            }
        }
    }

    if (cache_stride_sliding > 0) {
        _state->_cos_cache_sliding = a_calloc((size_t)_config->seq_len * cache_stride_sliding * sizeof(float));
        _state->_sin_cache_sliding = a_calloc((size_t)_config->seq_len * cache_stride_sliding * sizeof(float));
        for (int pos = 0; pos < _config->seq_len; pos++) {
            for (int i = 0; i < cache_stride_sliding; i++) {
                float freq = 1.0f / powf(_config->rope_theta_sliding, (float)(2 * i) / _config->head_dim);
                float val = (float)pos * freq;
                _state->_cos_cache_sliding[pos * cache_stride_sliding + i] = cosf(val);
                _state->_sin_cache_sliding[pos * cache_stride_sliding + i] = sinf(val);
            }
        }
    }
    else {
        _state->_cos_cache_sliding = NULL;
        _state->_sin_cache_sliding = NULL;
    }

    if ((! _state->_x) ||
            (! _state->_xb) ||
            (! _state->_hb) ||
            (! _state->_hb2) ||
            (! _state->_q) ||
            (! _state->_k) ||
            (! _state->_k_raw) ||
            (! _state->_v) ||
            (! _state->_att) ||
            (! _state->_logits) ||
            (! _state->__key_cache) ||
            (! _state->__value_cache) ||
            (! _state->xq._data) ||
            (! _state->xq._scales) ||
            (! _state->hq._data) ||
            (! _state->hq._scales)) {
        log_msg(stderr, "ERROR: Alloc failed!\n");
        exit(EXIT_FAILURE);
    }
    if (_config->seq_len > 1 &&
            (! _state->_cos_cache_full) ||
            (! _state->_sin_cache_full) ||
            (! _state->_cos_cache_sliding) ||
            (! _state->_sin_cache_sliding)) {
        log_msg(stderr, "ERROR: Alloc failed for RoPE cache!\n");
        exit(EXIT_FAILURE);
    }
    _state->allocated = 1;
}

void free_state_g4(state_g4 *_state) {
    if (! _state->allocated) {
        return;
    }

    free(_state->_x);
    free(_state->_xb);
    free(_state->_hb);
    free(_state->_hb2);
    free(_state->_q);
    free(_state->_k);
    free(_state->_k_raw);
    free(_state->_v);
    free(_state->_att);
    free(_state->_logits);

    if (_state->__key_cache) {
        for (int i = 0; i < _state->n_layers; i++) {
            free(_state->__key_cache[i]);
        }
        free(_state->__key_cache);
    }
    if (_state->__value_cache) {
        for (int i = 0; i < _state->n_layers; i++) {
            free(_state->__value_cache[i]);
        }
        free(_state->__value_cache);
    }

    free(_state->xq._data);
    free(_state->xq._scales);
    free(_state->hq._data);
    free(_state->hq._scales);
    free(_state->_cos_cache_full);
    free(_state->_sin_cache_full);
    free(_state->_cos_cache_sliding);
    free(_state->_sin_cache_sliding);

    _state->allocated = 0;
}

void free_g4(G4 *model) {
    if (! model) {
        return;
    }

    config_g4 *_config = &model->config;
    weights_g4 *_weights = &model->weights;

    free(model->_layer_types);

    free_qt(&_weights->embed_tokens_weight);
    free_qt(&_weights->rms_final_norm);
    free_qt(&_weights->rope_freqs_full);

    free_qt_array(_weights->_rms_input_layernorm, _config->n_layers);
    free_qt_array(_weights->_rms_post_attn_layernorm, _config->n_layers);
    free_qt_array(_weights->_rms_pre_ffn_layernorm, _config->n_layers);
    free_qt_array(_weights->_rms_post_ffn_layernorm, _config->n_layers);
    free_qt_array(_weights->_rms_q_norm, _config->n_layers);
    free_qt_array(_weights->_rms_k_norm, _config->n_layers);

    free_qt_array(_weights->_q_proj, _config->n_layers);
    free_qt_array(_weights->_k_proj, _config->n_layers);
    free_qt_array(_weights->_v_proj, _config->n_layers);
    free_qt_array(_weights->_o_proj, _config->n_layers);
    free_qt_array(_weights->_gate_proj, _config->n_layers);
    free_qt_array(_weights->_up_proj, _config->n_layers);
    free_qt_array(_weights->_down_proj, _config->n_layers);

    free(_weights->_layer_scalars);

    if (model->state.allocated == 1) {
        free_state_g4(&model->state);
    }

    free_tokenizer(&model->tokenizer);

    memset(model, 0, sizeof(G4));
}

