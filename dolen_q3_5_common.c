#include "dolen_q3_5_common.h"


void alloc_state_q3_5(state_q3_5 *_state, config_q3_5 *_config) {
    int32_t dim = _config->dim;
    int32_t head_size = _config->d_head > 0 ? _config->d_head : dim / _config->n_heads;
    int32_t kv_dim = _config->n_kv_heads * head_size;
    int32_t hidden_dim = _config->n_mlp;
    int32_t q_dim = _config->n_heads * head_size * 2;
    int32_t attn_dim = _config->n_heads * head_size;
    size_t n_kv_layers = (size_t)_config->n_full_attn_layers;
    size_t n_linear_layers = (size_t)_config->n_linear_attn_layers;
    int32_t value_dim = _config->n_linear_v_heads * _config->d_linear_v;

    int32_t max_act_dim = dim;
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

    _state->_x = a_calloc((size_t)dim * sizeof(float));
    _state->_xb = a_calloc((size_t)max_act_dim * sizeof(float));
    _state->_xb2 = a_calloc((size_t)dim * sizeof(float));
    _state->_hb = a_calloc((size_t)hidden_dim * sizeof(float));
    _state->_hb2 = a_calloc((size_t)hidden_dim * sizeof(float));
    _state->_q = a_calloc((size_t)q_dim * sizeof(float));
    _state->_k = a_calloc((size_t)kv_dim * sizeof(float));
    _state->_v = a_calloc((size_t)kv_dim * sizeof(float));
    _state->_att = a_calloc((size_t)_config->n_heads * _config->seq_len * sizeof(float));
    _state->_logits = a_calloc((size_t)_config->vocab_size * sizeof(float));
    _state->_gate = a_calloc((size_t)_config->n_heads * head_size * sizeof(float));

    int32_t num_groups = (max_act_dim + GROUP_SIZE - 1) / GROUP_SIZE;
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

    if (n_kv_layers > 0) {
        _state->_key_cache = a_calloc((size_t)_config->n_layer * (size_t)_config->seq_len * (size_t)kv_dim
                * sizeof(float));
        _state->_value_cache = a_calloc((size_t)_config->n_layer * (size_t)_config->seq_len * (size_t)kv_dim
                * sizeof(float));
    }

    if (n_linear_layers > 0) {
        int32_t key_dim = _config->n_linear_k_heads * _config->d_linear_k;
        int32_t conv_dim = key_dim * 2 + value_dim;

        _state->_qkv = a_calloc((size_t)conv_dim * sizeof(float));
        _state->_z = a_calloc((size_t)value_dim * sizeof(float));
        _state->_beta = a_calloc((size_t)_config->n_linear_v_heads * sizeof(float));
        _state->_g = a_calloc((size_t)_config->n_linear_v_heads * sizeof(float));
        _state->_linear_out = a_calloc((size_t)value_dim * sizeof(float));
        _state->_conv_state = a_calloc(n_linear_layers * conv_dim * _config->linear_conv_kernel * sizeof(float));
        _state->_S = a_calloc(n_linear_layers * _config->n_linear_v_heads * _config->d_linear_k * _config->d_linear_v *
                sizeof(float));
        _state->_delta_S = a_calloc((size_t)_config->n_linear_v_heads * _config->d_linear_v * sizeof(float));
    }

    int32_t rotary_partial = (int32_t)((float)head_size * _config->rope_partial_rotary_factor);

    if (rotary_partial > 0) {
        _state->_cos_cache = (float *)a_calloc((size_t)_config->seq_len * rotary_partial * sizeof(float));
        _state->_sin_cache = (float *)a_calloc((size_t)_config->seq_len * rotary_partial * sizeof(float));
        float theta = _config->rope_theta;
        for (int32_t pos = 0; pos < _config->seq_len; pos++) {
            for (int32_t i = 0; i < rotary_partial; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / rotary_partial);
                float val = pos * freq;
                _state->_cos_cache[pos * rotary_partial + i] = cosf(val);
                _state->_sin_cache[pos * rotary_partial + i] = sinf(val);
            }
        }
    }
    else {
        _state->_cos_cache = NULL;
        _state->_sin_cache = NULL;
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
            (! _state->_gate) ||
            (! _state->xq._data) ||
            (! _state->xq._scales) ||
            (! _state->hq._data) ||
            (! _state->hq._scales)) {
        log_msg(stderr, "ERROR: Alloc failed!\n");
        exit(EXIT_FAILURE);
    }
    if (n_kv_layers > 0 &&
            ((! _state->_key_cache) ||
             (! _state->_value_cache))) {
        log_msg(stderr, "ERROR: alloc failed for KV cache!\n");
        exit(EXIT_FAILURE);
    }

    _state->allocated = 1;
}

void free_state_q3_5(state_q3_5 *_state) {
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
    free(_state->_gate);
    free(_state->_key_cache);
    free(_state->_value_cache);
    free(_state->_qkv);
    free(_state->_z);
    free(_state->_beta);
    free(_state->_g);
    free(_state->_linear_out);
    free(_state->_conv_state);
    free(_state->_S);
    free(_state->_delta_S);
    free(_state->xq._data);
    free(_state->xq._scales);
    free(_state->hq._data);
    free(_state->hq._scales);

    if (_state->_cos_cache) {
        free(_state->_cos_cache);
    }
    if (_state->_sin_cache) {
        free(_state->_sin_cache);
    }

    _state->allocated = 0;
}

void free_q3_5(Q3_5 *model_q3_5) {
    weights_q3_5 *_weights = &model_q3_5->weights;
    int32_t n_full_attn = model_q3_5->config.n_full_attn_layers;
    int32_t n_linear_attn = model_q3_5->config.n_linear_attn_layers;
    int32_t n_layer = model_q3_5->config.n_layer;

    free_qt(&_weights->embed_tokens_weight);
    free_qt_array(_weights->_rms_att_weight, n_layer);
    free_qt_array(_weights->_wq, n_full_attn);
    free_qt_array(_weights->_wk, n_full_attn);
    free_qt_array(_weights->_wv, n_full_attn);
    free_qt_array(_weights->_wo, n_full_attn);
    free_qt_array(_weights->_q_norm, n_full_attn);
    free_qt_array(_weights->_k_norm, n_full_attn);
    free_qt_array(_weights->_in_proj_qkv, n_linear_attn);
    free_qt_array(_weights->_in_proj_z, n_linear_attn);
    free_qt_array(_weights->_in_proj_b, n_linear_attn);
    free_qt_array(_weights->_in_proj_a, n_linear_attn);
    free_qt_array(_weights->_conv1d_weight, n_linear_attn);
    free_qt_array(_weights->_dt_bias, n_linear_attn);
    free_qt_array(_weights->_A_log, n_linear_attn);
    free_qt_array(_weights->_linear_norm, n_linear_attn);
    free_qt_array(_weights->_out_proj, n_linear_attn);
    free_qt_array(_weights->_rms_ffn_weight, n_layer);
    free_qt_array(_weights->_w1, n_layer);
    free_qt_array(_weights->_w2, n_layer);
    free_qt_array(_weights->_w3, n_layer);
    free_qt(&(_weights->rms_final_weight));

    if (! model_q3_5->config.tie_word_embeddings) {
        free_qt(&(_weights->wcls));
    }

    free(model_q3_5->_layer_types);
    free(model_q3_5->_attn_layer_indices);
    free(model_q3_5->_deltanet_layer_indices);

    if (model_q3_5->state.allocated) {
        free_state_q3_5(&model_q3_5->state);
    }

    free_tokenizer(&model_q3_5->tokenizer1);
}

