#include "dolen_g4e_common.h"

bool alloc_state_g4e(G4E *_model, int32_t seq_n) {
    state_g4e  *_state  = &(_model->state);
    config_g4e *_config = &(_model->config);

    _state->seq_n = seq_n;

    int32_t max_head_dim = _config->head_dim > _config->global_head_dim
                         ? _config->head_dim : _config->global_head_dim;
    int32_t max_kv_heads = _config->n_kv_heads > _config->n_global_kv_heads
                         ? _config->n_kv_heads : _config->n_global_kv_heads;
    int32_t max_kv_dim   = max_kv_heads * max_head_dim;
    int32_t attn_out_dim = _config->n_heads * max_head_dim;

    int32_t cache_stride_full    = _config->global_head_dim / 2;
    int32_t cache_stride_sliding = _config->head_dim / 2;

    int32_t ple_dim   = _config->hidden_size_per_layer_input;
    int32_t ple_total = _config->n_layers * ple_dim;

    int32_t max_act_dim = _config->dim;
    if (attn_out_dim       > max_act_dim) max_act_dim = attn_out_dim;
    if (_config->hidden_dim > max_act_dim) max_act_dim = _config->hidden_dim;
    if (ple_dim            > max_act_dim) max_act_dim = ple_dim;

    /* ---- base state buffers ---- */
    _state->_x      = a_calloc((size_t)_config->dim        * sizeof(float));
    _state->_xb     = a_calloc((size_t)max_act_dim         * sizeof(float));
    _state->_hb     = a_calloc((size_t)_config->hidden_dim * sizeof(float));
    _state->_hb2    = a_calloc((size_t)_config->hidden_dim * sizeof(float));
    _state->_q      = a_calloc((size_t)attn_out_dim        * sizeof(float));
    _state->_k      = a_calloc((size_t)max_kv_dim          * sizeof(float));
    _state->_k_raw  = a_calloc((size_t)max_kv_dim          * sizeof(float));
    _state->_v      = a_calloc((size_t)max_kv_dim          * sizeof(float));
    _state->_att    = a_calloc((size_t)_config->n_heads * seq_n * sizeof(float));
    _state->_logits = a_calloc((size_t)_config->vocab_size * sizeof(float));

    _state->n_layers      = _config->n_layers;
    _state->__key_cache   = (_Float16 **)a_calloc((size_t)_config->n_layers * sizeof(_Float16 *));
    _state->__value_cache = (_Float16 **)a_calloc((size_t)_config->n_layers * sizeof(_Float16 *));

    int32_t first_shared = _model->_first_kv_shared_layer_idx;
    for (int32_t l = 0; l < _config->n_layers; l++) {
        int32_t is_shared = (_config->num_kv_shared_layers > 0) && (l >= first_shared);
        if (is_shared) {
            _state->__key_cache[l]   = NULL;
            _state->__value_cache[l] = NULL;
        } else {
            int32_t is_full = _model->_layer_types ? _model->_layer_types[l] : 0;
            int32_t kv_heads = is_full ? _config->n_global_kv_heads : _config->n_kv_heads;
            int32_t hd       = is_full ? _config->global_head_dim   : _config->head_dim;
            int32_t kv_dim   = kv_heads * hd;
            _state->__key_cache[l]   = a_calloc((size_t)seq_n * kv_dim * sizeof(_Float16));
            _state->__value_cache[l] = a_calloc((size_t)seq_n * kv_dim * sizeof(_Float16));
        }
    }

    /* ---- shared KV caches ---- */
    if (_config->num_kv_shared_layers > 0) {
        int32_t full_kv_dim    = _config->n_global_kv_heads * _config->global_head_dim;
        int32_t sliding_kv_dim = _config->n_kv_heads * _config->head_dim;
        _state->_shared_key_full     = a_calloc((size_t)seq_n * full_kv_dim    * sizeof(_Float16));
        _state->_shared_value_full   = a_calloc((size_t)seq_n * full_kv_dim    * sizeof(_Float16));
        _state->_shared_key_sliding  = a_calloc((size_t)seq_n * sliding_kv_dim * sizeof(_Float16));
        _state->_shared_value_sliding= a_calloc((size_t)seq_n * sliding_kv_dim * sizeof(_Float16));
    } else {
        _state->_shared_key_full = _state->_shared_value_full = NULL;
        _state->_shared_key_sliding = _state->_shared_value_sliding = NULL;
    }

    /* ---- PLE scratch ---- */
    if (ple_dim > 0) {
        _state->_ple_combined  = a_calloc((size_t)ple_total           * sizeof(float));
        _state->_ple_proj_raw  = a_calloc((size_t)ple_total           * sizeof(float));
        _state->_ple_gate_out  = a_calloc((size_t)ple_dim             * sizeof(float));
        _state->_ple_proj_out  = a_calloc((size_t)_config->dim        * sizeof(float));
    } else {
        _state->_ple_combined = _state->_ple_proj_raw = NULL;
        _state->_ple_gate_out = _state->_ple_proj_out = NULL;
    }

    /* ---- quantized scratch ---- */
    int32_t num_groups_xq = (max_act_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    _state->xq._data   = a_calloc((size_t)max_act_dim * sizeof(int8_t));
    _state->xq._scales = a_calloc((size_t)num_groups_xq * sizeof(float));
    _state->xq.type    = Q_TYPE_Q8;
    _state->xq.rows    = 1;
    _state->xq.cols    = max_act_dim;

    int32_t num_groups_hq = (_config->hidden_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    _state->hq._data   = a_calloc((size_t)_config->hidden_dim * sizeof(int8_t));
    _state->hq._scales = a_calloc((size_t)num_groups_hq * sizeof(float));
    _state->hq.type    = Q_TYPE_Q8;
    _state->hq.rows    = 1;
    _state->hq.cols    = _config->hidden_dim;

    /* ---- RoPE caches (identical to G4) ---- */
    int32_t rotary_dim_full = (int32_t)(_config->rope_partial_factor * _config->global_head_dim);

    if (cache_stride_full > 0) {
        _state->_cos_cache_full = a_calloc((size_t)seq_n * cache_stride_full * sizeof(float));
        _state->_sin_cache_full = a_calloc((size_t)seq_n * cache_stride_full * sizeof(float));
        for (int32_t pos = 0; pos < seq_n; pos++) {
            for (int32_t i = 0; i < cache_stride_full; i++) {
                float cv, sv;
                if (i < rotary_dim_full / 2) {
                    float freq = 1.0f / powf(_config->rope_theta_full,
                                             (float)(2 * i) / _config->global_head_dim);
                    float val  = (float)pos * freq;
                    cv = cosf(val);
                    sv = sinf(val);
                } else {
                    cv = 1.0f;
                    sv = 0.0f;
                }
                _state->_cos_cache_full[pos * cache_stride_full + i] = cv;
                _state->_sin_cache_full[pos * cache_stride_full + i] = sv;
            }
        }
    }
    if (cache_stride_sliding > 0) {
        _state->_cos_cache_sliding = a_calloc((size_t)seq_n * cache_stride_sliding * sizeof(float));
        _state->_sin_cache_sliding = a_calloc((size_t)seq_n * cache_stride_sliding * sizeof(float));
        for (int32_t pos = 0; pos < seq_n; pos++) {
            for (int32_t i = 0; i < cache_stride_sliding; i++) {
                float freq = 1.0f / powf(_config->rope_theta_sliding,
                                         (float)(2 * i) / _config->head_dim);
                float val  = (float)pos * freq;
                _state->_cos_cache_sliding[pos * cache_stride_sliding + i] = cosf(val);
                _state->_sin_cache_sliding[pos * cache_stride_sliding + i] = sinf(val);
            }
        }
    } else {
        _state->_cos_cache_sliding = NULL;
        _state->_sin_cache_sliding = NULL;
    }

    /* ---- allocation check ---- */
    if (!_state->_x || !_state->_xb || !_state->_hb || !_state->_hb2 ||
        !_state->_q || !_state->_k || !_state->_k_raw || !_state->_v ||
        !_state->_att || !_state->_logits ||
        !_state->__key_cache || !_state->__value_cache ||
        !_state->xq._data || !_state->xq._scales ||
        !_state->hq._data || !_state->hq._scales) {
        log_msg(stderr, "ERROR: Alloc failed!\n");
        free_state_g4e(_state);
        return false;
    }
    return true;
}

void free_state_g4e(state_g4e *_state) {
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
        for (int32_t i = 0; i < _state->n_layers; i++)
            free(_state->__key_cache[i]);
        free(_state->__key_cache);
    }
    if (_state->__value_cache) {
        for (int32_t i = 0; i < _state->n_layers; i++)
            free(_state->__value_cache[i]);
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

    free(_state->_ple_combined);
    free(_state->_ple_proj_raw);
    free(_state->_ple_gate_out);
    free(_state->_ple_proj_out);

    free(_state->_shared_key_full);
    free(_state->_shared_value_full);
    free(_state->_shared_key_sliding);
    free(_state->_shared_value_sliding);
}

void free_g4e(G4E *_model) {
    if (!_model) return;

    config_g4e  *_config  = &(_model->config);
    weights_g4e *_weights = &(_model->weights);

    free(_model->_layer_types);

    free_qt(&_weights->embed_tokens_weight);
    free_qt(&_weights->rms_final_norm);
    free_qt(&_weights->rope_freqs_full);

    /* PLE weights */
    free_qt(&_weights->embed_tokens_per_layer);
    free_qt(&_weights->per_layer_model_projection);
    free_qt(&_weights->per_layer_projection_norm);
    free_qt_array(_weights->_ple_gate,       _config->n_layers);
    free_qt_array(_weights->_ple_projection, _config->n_layers);
    free_qt_array(_weights->_ple_post_norm,  _config->n_layers);

    free_qt_array(_weights->_rms_input_layernorm,      _config->n_layers);
    free_qt_array(_weights->_rms_post_attn_layernorm,  _config->n_layers);
    free_qt_array(_weights->_rms_pre_ffn_layernorm,    _config->n_layers);
    free_qt_array(_weights->_rms_post_ffn_layernorm,   _config->n_layers);
    free_qt_array(_weights->_rms_q_norm,               _config->n_layers);
    free_qt_array(_weights->_rms_k_norm,               _config->n_layers);

    free_qt_array(_weights->_q_proj,     _config->n_layers);
    free_qt_array(_weights->_k_proj,     _config->n_layers);
    free_qt_array(_weights->_v_proj,     _config->n_layers);
    free_qt_array(_weights->_o_proj,     _config->n_layers);
    free_qt_array(_weights->_gate_proj,  _config->n_layers);
    free_qt_array(_weights->_up_proj,    _config->n_layers);
    free_qt_array(_weights->_down_proj,  _config->n_layers);

    free(_weights->_layer_scalars);

    free_state_g4e(&(_model->state));
    free_tokenizer(&(_model->tokenizer));

    memset(_model, 0, sizeof(G4E));
}

