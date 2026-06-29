#include "dolen_q3_5_common.h"


static const chat_template CHAT_TEMPLATE_Q3_5 = {
    ._system_s = "<|im_start|>system\n%s<|im_end|>\n",
    ._main_s = "<|im_start|>user\n%s<|im_end|>\n"
             "<|im_start|>assistant\n"
             "<think>\n\n</think>\n\n",
    ._end_turn_s = "<|im_end|>\n",
};

static const chat_template CHAT_TEMPLATE_THINK_Q3_5 = {
    ._system_s = "<|im_start|>system\n%s<|im_end|>\n",
    ._main_s = "<|im_start|>user\n%s<|im_end|>\n"
             "<|im_start|>assistant\n"
             "<think>",
    ._end_turn_s = "<|im_end|>\n",
};


int load_quantized_q3_5(const char *_file_path_s, Q3_5 *_model, int seq_n_max) {
    FILE *_file = fopen(_file_path_s, "rb");
    if (! _file) {
        log_msg(stderr, "ERROR: Failed to open %s for reading\n", _file_path_s);
        return -1;
    }

    memset(_model, 0, sizeof(Q3_5));

    uint64_t magic;
    uint32_t version;

    if ((fread(&magic, sizeof(uint64_t), 1, _file) != 1) ||
            (fread(&version, sizeof(uint32_t), 1, _file) != 1)) {
        log_msg(stderr, "ERROR: Failed to read header from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    if (magic != MAGIC_Q3_5) {
        log_msg(stderr, "ERROR: Invalid magic number in %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    if (version != 3) {
        log_msg(stderr, "ERROR: Unsupported version %d in %s\n", version, _file_path_s);
        fclose(_file);
        return -1;
    }

    if (fread(&_model->config, sizeof(config_q3_5), 1, _file) != 1) {
        log_msg(stderr, "ERROR: Failed to read config from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    if (tokenizer_read_from_file(_file, _model->config.vocab_size, &_model->tokenizer1)) {
        log_msg(stderr, "ERROR: Failed to read tokenizer from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    config_q3_5 *_config = &_model->config;
    weights_q3_5 *_weights = &_model->weights;

    if (seq_n_max) {
        _config->seq_len = seq_n_max;
    }

    _model->_layer_types = (int *)a_calloc((size_t)_config->n_layer * sizeof(int));
    _model->_attn_layer_indices = (int *)a_calloc((size_t)_config->n_layer * sizeof(int));
    _model->_deltanet_layer_indices = (int *)a_calloc((size_t)_config->n_layer * sizeof(int));

    if ((! _model->_layer_types) ||
            (! _model->_attn_layer_indices) ||
            (! _model->_deltanet_layer_indices)) {
        log_msg(stderr, "ERROR: Failed to allocate memory for layer indices\n");
        fclose(_file);
        return -1;
    }

    if (fread(_model->_layer_types, sizeof(int), (size_t)_config->n_layer, _file) != (size_t)_config->n_layer) {
        log_msg(stderr, "ERROR: Failed to read layer_types from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    int la = 0, ld = 0;
    for (int i = 0; i < _config->n_layer; i++) {
        if (_model->_layer_types[i] == 1) {
            _model->_deltanet_layer_indices[i] = ld++;
        } else {
            _model->_attn_layer_indices[i] = la++;
        }
    }

    _weights->_rms_att_weight = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    _weights->_q_norm = (qtensor *)a_calloc((size_t)_config->n_full_attn_layers * sizeof(qtensor));
    _weights->_k_norm = (qtensor *)a_calloc((size_t)_config->n_full_attn_layers * sizeof(qtensor));
    _weights->_rms_ffn_weight = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));

    if ((! _weights->_rms_att_weight) ||
            (! _weights->_q_norm) ||
            (! _weights->_k_norm) ||
            (! _weights->_rms_ffn_weight)) {
        log_msg(stderr, "ERROR: Failed to allocate norm weights\n");
        fclose(_file);
        return -1;
    }

    read_qt(_file, &_weights->embed_tokens_weight);

    for (int i = 0; i < _config->n_layer; i++) {
        read_qt(_file, &_weights->_rms_att_weight[i]);
    }

    _weights->_wq = (qtensor *)a_calloc((size_t)_config->n_full_attn_layers * sizeof(qtensor));
    _weights->_wk = (qtensor *)a_calloc((size_t)_config->n_full_attn_layers * sizeof(qtensor));
    _weights->_wv = (qtensor *)a_calloc((size_t)_config->n_full_attn_layers * sizeof(qtensor));
    _weights->_wo = (qtensor *)a_calloc((size_t)_config->n_full_attn_layers * sizeof(qtensor));

    if ((! _weights->_wq) ||
            (! _weights->_wk) ||
            (! _weights->_wv) ||
            (! _weights->_wo)) {
        log_msg(stderr, "ERROR: Failed to allocate attention weights\n");
        fclose(_file);
        return -1;
    }

    for (int i = 0; i < _config->n_full_attn_layers; i++) {
        read_qt(_file, &_weights->_wq[i]);
        read_qt(_file, &_weights->_wk[i]);
        read_qt(_file, &_weights->_wv[i]);
        read_qt(_file, &_weights->_wo[i]);
    }

    for (int i = 0; i < _config->n_full_attn_layers; i++) {
        read_qt(_file, &_weights->_q_norm[i]);
    }
    for (int i = 0; i < _config->n_full_attn_layers; i++) {
        read_qt(_file, &_weights->_k_norm[i]);
    }

    if (_config->n_linear_attn_layers > 0) {
        _weights->_in_proj_qkv = (qtensor *)a_calloc((size_t)_config->n_linear_attn_layers * sizeof(qtensor));
        _weights->_in_proj_z = (qtensor *)a_calloc((size_t)_config->n_linear_attn_layers * sizeof(qtensor));
        _weights->_in_proj_b = (qtensor *)a_calloc((size_t)_config->n_linear_attn_layers * sizeof(qtensor));
        _weights->_in_proj_a = (qtensor *)a_calloc((size_t)_config->n_linear_attn_layers * sizeof(qtensor));
        _weights->_conv1d_weight = (qtensor *)a_calloc((size_t)_config->n_linear_attn_layers * sizeof(qtensor));
        _weights->_dt_bias = (qtensor *)a_calloc((size_t)_config->n_linear_attn_layers * sizeof(qtensor));
        _weights->_A_log = (qtensor *)a_calloc((size_t)_config->n_linear_attn_layers * sizeof(qtensor));
        _weights->_linear_norm = (qtensor *)a_calloc((size_t)_config->n_linear_attn_layers * sizeof(qtensor));
        _weights->_out_proj = (qtensor *)a_calloc((size_t)_config->n_linear_attn_layers * sizeof(qtensor));

        if ((! _weights->_in_proj_qkv) ||
                (! _weights->_in_proj_z) ||
                (! _weights->_in_proj_b) ||
                (! _weights->_in_proj_a) ||
                (! _weights->_conv1d_weight) ||
                (! _weights->_dt_bias) ||
                (! _weights->_A_log) ||
                (! _weights->_linear_norm) ||
                (! _weights->_out_proj)) {
            log_msg(stderr, "ERROR: Failed to allocate linear attention weights\n");
            fclose(_file);
            return -1;
        }

        for (int i = 0; i < _config->n_linear_attn_layers; i++) {
            read_qt(_file, &_weights->_in_proj_qkv[i]);
            read_qt(_file, &_weights->_in_proj_z[i]);
        }
        for (int i = 0; i < _config->n_linear_attn_layers; i++) {
            read_qt(_file, &_weights->_in_proj_b[i]);
        }
        for (int i = 0; i < _config->n_linear_attn_layers; i++) {
            read_qt(_file, &_weights->_in_proj_a[i]);
        }
        for (int i = 0; i < _config->n_linear_attn_layers; i++) {
            read_qt(_file, &_weights->_conv1d_weight[i]);
        }
        for (int i = 0; i < _config->n_linear_attn_layers; i++) {
            read_qt(_file, &_weights->_dt_bias[i]);
        }
        for (int i = 0; i < _config->n_linear_attn_layers; i++) {
            read_qt(_file, &_weights->_A_log[i]);
        }
        for (int i = 0; i < _config->n_linear_attn_layers; i++) {
            read_qt(_file, &_weights->_linear_norm[i]);
        }
        for (int i = 0; i < _config->n_linear_attn_layers; i++) {
            read_qt(_file, &_weights->_out_proj[i]);
        }
    }

    for (int i = 0; i < _config->n_layer; i++) {
        read_qt(_file, &_weights->_rms_ffn_weight[i]);
    }

    _weights->_w1 = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    _weights->_w2 = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    _weights->_w3 = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    if ((! _weights->_w1) ||
            (! _weights->_w2) ||
            (! _weights->_w3)) {
        log_msg(stderr, "ERROR: Failed to allocate MLP weights\n");
        fclose(_file);
        return -1;
    }
    for (int i = 0; i < _config->n_layer; i++) {
        read_qt(_file, &_weights->_w1[i]);
        read_qt(_file, &_weights->_w2[i]);
        read_qt(_file, &_weights->_w3[i]);
    }

    read_qt(_file, &_weights->rms_final_weight);

    if (! _config->tie_word_embeddings) {
        read_qt(_file, &_weights->wcls);
    } else {
        _weights->wcls = _weights->embed_tokens_weight;
    }

    fclose(_file);
    log_msg(stdout, "INFO: Quantized model loaded from %s\n", _file_path_s);

    alloc_state_q3_5(&_model->state, &_model->config);

    return 0;
}

void forward_q3_5_attention_layer(Q3_5 *_model, int l, int la, int pos) {
    config_q3_5 *_config = &_model->config;
    weights_q3_5 *_weights = &_model->weights;
    state_q3_5 *_state = &_model->state;
    float *_x = _state->_x;
    int dim = _config->dim;
    int head_size = _config->d_head > 0 ? _config->d_head : dim / _config->n_heads;
    int kv_dim = _config->n_kv_heads * head_size;
    int attn_out_dim = _config->n_heads * head_size;
    int kv_mul = _config->n_heads / _config->n_kv_heads;
    long long loff = (long long)la * _config->seq_len * kv_dim;
    float eps = _config->rms_norm_eps;
    float *_key_cache_row = _state->_key_cache + loff + pos * kv_dim;
    float *_value_cache_row = _state->_value_cache + loff + pos * kv_dim;
    float *_rms_att_weight = (float *)_weights->_rms_att_weight[l]._data;
    float *_q_norm = (float *)_weights->_q_norm[la]._data;
    float *_k_norm = (float *)_weights->_k_norm[la]._data;

    rmsnorm_gemma(_state->_xb, _x, _rms_att_weight, dim, eps);

    quantize_vec(&_state->xq, _state->_xb, dim);
    matmul_qq(_state->_q, &_state->xq, &_weights->_wq[la]);
    matmul_qq(_state->_k, &_state->xq, &_weights->_wk[la]);
    matmul_qq(_state->_v, &_state->xq, &_weights->_wv[la]);

    for (int h = 0; h < _config->n_heads; h++) {
        float *_q_ptr = _state->_q + h * head_size;
        float *_gate_ptr = _state->_gate + h * head_size;
        for (int i = 0; i < head_size; i++) {
            _q_ptr[i] = _state->_q[h * head_size * 2 + i];
            _gate_ptr[i] = _state->_q[h * head_size * 2 + head_size + i];
        }
    }

#pragma omp parallel for
    for (int h = 0; h < _config->n_heads; h++) {
        float *_q_ptr = _state->_q + h * head_size;
        rmsnorm_gemma(_q_ptr, _q_ptr, _q_norm, head_size, eps);
    }

#pragma omp parallel for
    for (int h = 0; h < _config->n_kv_heads; h++) {
        float *_k_ptr = _state->_k + h * head_size;
        rmsnorm_gemma(_k_ptr, _k_ptr, _k_norm, head_size, eps);
    }

    int rotary_partial = (int)((float)head_size * _config->rope_partial_rotary_factor);

    if ((rotary_partial > 0) && _state->_cos_cache) {
        float *_cos_row = _state->_cos_cache + pos * rotary_partial;
        float *_sin_row = _state->_sin_cache + pos * rotary_partial;

#pragma omp parallel for
        for (int h = 0; h < _config->n_heads; h++) {
            float *_q = _state->_q + h * head_size;
            for (int i = 0; i < rotary_partial; i++) {
                float c = _cos_row[i];
                float sn = _sin_row[i];
                float q0 = _q[i];
                float q1 = _q[i + rotary_partial];
                _q[i] = q0 * c - q1 * sn;
                _q[i + rotary_partial] = q0 * sn + q1 * c;
            }
        }
#pragma omp parallel for
        for (int h = 0; h < _config->n_kv_heads; h++) {
            float *_k = _state->_k + h * head_size;
            for (int i = 0; i < rotary_partial; i++) {
                float c = _cos_row[i], sn = _sin_row[i];
                float k0 = _k[i];
                float k1 = _k[i + rotary_partial];
                _k[i] = k0 * c - k1 * sn;
                _k[i + rotary_partial] = k0 * sn + k1 * c;
            }
        }
    }

    memcpy(_key_cache_row, _state->_k, kv_dim * sizeof(float));
    memcpy(_value_cache_row, _state->_v, kv_dim * sizeof(float));

    float inv_sqrt_head = 1.0f / sqrtf((float)head_size);

#pragma omp parallel for
    for (int h = 0; h < _config->n_heads; h++) {
        float *_q = _state->_q + h * head_size;
        float *_att = _state->_att + h * _config->seq_len;

        for (int t = 0; t <= pos; t++) {
            float *_k = _state->_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float score = 0.0f;
#pragma omp simd reduction(+ : score)
            for (int i = 0; i < head_size; i++) {
                score += _q[i] * _k[i];
            }
            _att[t] = score * inv_sqrt_head;
        }

        softmax(_att, pos + 1);

        float *_xb = _state->_xb + h * head_size;
        memset(_xb, 0, head_size * sizeof(float));

        for (int t = 0; t <= pos; t++) {
            float *_v = _state->_value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float a = _att[t];
#pragma omp simd
            for (int i = 0; i < head_size; i++) {
                _xb[i] += a * _v[i];
            }
        }

        float *_gate_ptr = _state->_gate + h * head_size;

#pragma omp simd
        for (int i = 0; i < head_size; i++) {
            _xb[i] *= sigmoid(_gate_ptr[i]);
        }
    }

    quantize_vec(&_state->xq, _state->_xb, attn_out_dim);
    matmul_qq(_state->_xb2, &_state->xq, &_weights->_wo[la]);
    for (int i = 0; i < dim; i++) {
        _x[i] += _state->_xb2[i];
    }
}

void forward_q3_5_linear_attention_layer(Q3_5 *_model, int l, int ld, int pos) {
    config_q3_5 *_config = &_model->config;
    weights_q3_5 *_weights = &_model->weights;
    state_q3_5 *_state = &_model->state;
    float *_x = _state->_x;
    int dim = _config->dim;
    float eps = _config->rms_norm_eps;
    int n_k_heads = _config->n_linear_k_heads;
    int n_v_heads = _config->n_linear_v_heads;
    int d_k = _config->d_linear_k;
    int d_v = _config->d_linear_v;
    int key_dim = n_k_heads * d_k;
    int value_dim = n_v_heads * d_v;
    int conv_dim = key_dim * 2 + value_dim;
    int conv_kernel = _config->linear_conv_kernel;
    float *_rms_att_weight = (float *)_weights->_rms_att_weight[l]._data;
    float *_in_proj_b = (float *)_weights->_in_proj_b[ld]._data;
    float *_in_proj_a = (float *)_weights->_in_proj_a[ld]._data;
    float *_conv1d_weight = (float *)_weights->_conv1d_weight[ld]._data;
    float *_dt_bias = (float *)_weights->_dt_bias[ld]._data;
    float *_A_log = (float *)_weights->_A_log[ld]._data;
    float *_linear_norm = (float *)_weights->_linear_norm[ld]._data;
    float *_conv_state = _state->_conv_state + (long long)ld * conv_dim * conv_kernel;
    float *_S = _state->_S + (long long)ld * n_v_heads * d_k * d_v;

    if (! pos) {
        memset(_conv_state, 0, (size_t)conv_dim * conv_kernel * sizeof(float));
        memset(_S, 0, (size_t)n_v_heads * d_k * d_v * sizeof(float));
    }

    rmsnorm_gemma(_state->_xb, _x, _rms_att_weight, dim, eps);

    quantize_vec(&_state->xq, _state->_xb, dim);
    matmul_qq(_state->_qkv, &_state->xq, &_weights->_in_proj_qkv[ld]);
    matmul_qq(_state->_z, &_state->xq, &_weights->_in_proj_z[ld]);

#pragma omp parallel for
    for (int i = 0; i < n_v_heads; i++) {
        _state->_beta[i] = sigmoid(matmul_scalar(_state->_xb, _in_proj_b + i * dim, dim));
    }

#pragma omp parallel for
    for (int i = 0; i < n_v_heads; i++) {
        float a_val = matmul_scalar(_state->_xb, _in_proj_a + i * dim, dim);
        float A = -expf(_A_log[i]);
        _state->_g[i] = A * softplus(a_val + _dt_bias[i]);
    }

#pragma omp parallel for
    for (int i = 0; i < conv_dim; i++) {
        for (int j = 0; j < conv_kernel - 1; j++) {
            _conv_state[i * conv_kernel + j] = _conv_state[i * conv_kernel + j + 1];
        }
        _conv_state[i * conv_kernel + conv_kernel - 1] = _state->_qkv[i];
    }

    float *_qkv_conv = _state->_qkv;
#pragma omp parallel for
    for (int i = 0; i < conv_dim; i++) {
        float val = 0.0f;
        for (int j = 0; j < conv_kernel; j++) {
            val += _conv_state[i * conv_kernel + j] * _conv1d_weight[i * conv_kernel + j];
        }
        _qkv_conv[i] = silu(val);
    }

    float *_q = _qkv_conv;
    float *_k = _qkv_conv + key_dim;
    float *_v = _qkv_conv + key_dim * 2;

    float scale = 1.0f / sqrtf((float)d_k);
#pragma omp parallel for
    for (int h = 0; h < n_k_heads; h++) {
        float *_k_h = _k + h * d_k;
        float *_q_h = _q + h * d_k;
        l2norm(_k_h, d_k);
        l2norm(_q_h, d_k);
        for (int i = 0; i < d_k; i++) {
            _q_h[i] *= scale;
        }
    }

    int r = (n_v_heads > n_k_heads) ? n_v_heads / n_k_heads : 1;

#pragma omp parallel for
    for (int h = 0; h < n_v_heads; h++) {
        float g_t = expf(_state->_g[h]);
        float beta_t = _state->_beta[h];

        float *_S_h = _S + h * d_k * d_v;
        float *_q_h = _q + (h / r) * d_k;
        float *_k_h = _k + (h / r) * d_k;
        float *_v_h = _v + h * d_v;

        for (int i = 0; i < d_k * d_v; i++) {
            _S_h[i] *= g_t;
        }

        float *_delta = _state->_delta_S + h * d_v;
        for (int j = 0; j < d_v; j++) {
            float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
            for (int i = 0; i < d_k; i++) {
                dot += _S_h[i * d_v + j] * _k_h[i];
            }
            _delta[j] = (_v_h[j] - dot) * beta_t;
        }

        for (int i = 0; i < d_k; i++) {
#pragma omp simd
            for (int j = 0; j < d_v; j++) {
                _S_h[i * d_v + j] += _k_h[i] * _delta[j];
            }
        }

        float *_out_h = _state->_linear_out + h * d_v;
        for (int j = 0; j < d_v; j++) {
            float val = 0.0f;
#pragma omp simd reduction(+ : val)
            for (int i = 0; i < d_k; i++) {
                val += _S_h[i * d_v + j] * _q_h[i];
            }
            _out_h[j] = val;
        }
    }

    rmsnorm_gated(_state->_linear_out, _state->_linear_out, _state->_z, _linear_norm, n_v_heads, d_v, eps);

    quantize_vec(&_state->hq, _state->_linear_out, value_dim);
    matmul_qq(_state->_xb, &_state->hq, &_weights->_out_proj[ld]);
    for (int i = 0; i < dim; i++) {
        _x[i] += _state->_xb[i];
    }
}

void forward_q3_5_mlp_layer(Q3_5 *_model, int l) {
    config_q3_5 *_config = &_model->config;
    weights_q3_5 *_weights = &_model->weights;
    state_q3_5 *_state = &_model->state;
    float *_x = _state->_x;
    int dim = _config->dim;
    int hidden_dim = _config->n_mlp;
    float eps = _config->rms_norm_eps;
    float *_rms_ffn_weight = (float *)_weights->_rms_ffn_weight[l]._data;

    rmsnorm_gemma(_state->_xb, _x, _rms_ffn_weight, dim, eps);

    quantize_vec(&_state->xq, _state->_xb, dim);
    matmul_qq(_state->_hb, &_state->xq, &_weights->_w1[l]);
    matmul_qq(_state->_hb2, &_state->xq, &_weights->_w3[l]);

#pragma omp parallel for
    for (int i = 0; i < hidden_dim; i++) {
        float val = _state->_hb[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= _state->_hb2[i];
        _state->_hb[i] = val;
    }

    quantize_vec(&_state->hq, _state->_hb, hidden_dim);
    matmul_qq(_state->_xb, &_state->hq, &_weights->_w2[l]);
    for (int i = 0; i < dim; i++) {
        _x[i] += _state->_xb[i];
    }
}

float *forward_q3_5(Q3_5 *_model, int token, int pos) {
    config_q3_5 *_config = &_model->config;
    weights_q3_5 *_weights = &_model->weights;
    state_q3_5 *_state = &_model->state;
    float *_x = _state->_x;
    int dim = _config->dim;

    dequantize_row(_x, &_weights->embed_tokens_weight, token);

    for (int l = 0; l < _config->n_layer; l++) {
        if (_model->_layer_types[l] == 1) {
            forward_q3_5_linear_attention_layer(_model, l, _model->_deltanet_layer_indices[l], pos);
        } else {
            forward_q3_5_attention_layer(_model, l, _model->_attn_layer_indices[l], pos);
        }
        forward_q3_5_mlp_layer(_model, l);
    }

    rmsnorm_gemma(_x, _x, (float *)_weights->rms_final_weight._data, dim, _config->rms_norm_eps);

    if (_config->tie_word_embeddings) {
        matmul_qt(_state->_logits, _x, &_weights->embed_tokens_weight);
    } else {
        matmul_qt(_state->_logits, _x, &_weights->wcls);
    }

    return _state->_logits;
}

static float *forward_q3_5_wrap(void *_model, int token, int pos) {
    return forward_q3_5((Q3_5 *)_model, token, pos);
}

static void free_q3_5_wrap(void *_model) {
    free_q3_5((Q3_5 *)_model);
    free(_model);
}

model_iface *init_q3_5(const char *_model_path_s, int seq_n_max, bool think_) {
    Q3_5 *_model = a_calloc(1 * sizeof(Q3_5));

    if (load_quantized_q3_5(_model_path_s, _model, seq_n_max)) {
        free_q3_5(_model);
        free(_model);
        return NULL;
    }

    _model->tokenizer1.bos_id = _model->config.bos_token_id;
    _model->tokenizer1.eos_id = _model->config.eos_token_id;
    
    // Qwen3.5 HF config sets eos_token_id to <|endoftext|> (248044).
    // For ChatML, we still need <|im_end|> (248046) to halt generation correctly.
    _model->tokenizer1.im_end_id = 248046; 

    model_iface *_model_i = a_calloc(sizeof(model_iface));
    *_model_i = (model_iface){
        ._model = _model,
        .forward = forward_q3_5_wrap,
        .free_model = free_q3_5_wrap,
        .seq_n_max = seq_n_max ? seq_n_max : _model->config.seq_len,
        ._chat_template = think_ ? &CHAT_TEMPLATE_THINK_Q3_5 : &CHAT_TEMPLATE_Q3_5,
        ._tokenizer = &_model->tokenizer1,
    };
    return _model_i;
}

