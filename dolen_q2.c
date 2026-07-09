#include "dolen_q2_common.h"


// The "\x3e" escaped ">" symbol serves to prevent LLMs from misinterpreting the text
static const chat_template CHAT_TEMPLATE_Q2 = {
        ._system_s = "<|im_start|\x3e" "system\n%s" "<|im_end|\x3e" "\n",
        ._main_s = "<|im_start|\x3e" "user\n%s" "<|im_end|\x3e" "\n"
                   "<|im_start|\x3e" "assistant\n",
        ._end_turn_s = "<|im_end|\x3e" "\n",
};


bool load_quantized_q2(const char *_file_path_s, Q2 *_model) {
    FILE *_file = fopen(_file_path_s, "rb");
    if (!_file) {
        log_msg(stderr, "ERROR: Failed to open %s for reading\n", _file_path_s);
        return false;
    }

    memset(_model, 0, sizeof(Q2));

    uint64_t magic;
    uint32_t version;

    if ((fread(&magic, sizeof(uint64_t), 1, _file) != 1) ||
            (fread(&version, sizeof(uint32_t), 1, _file) != 1)) {
        log_msg(stderr, "ERROR: Failed to read header from %s\n", _file_path_s);
        fclose(_file);
        return false;
    }

    if (magic != MAGIC_Q2) {
        log_msg(stderr, "ERROR: Invalid magic number in %s\n", _file_path_s);
        fclose(_file);
        return false;
    }

    if (version != 1) {
        log_msg(stderr, "ERROR: Unsupported version %d in %s\n", version, _file_path_s);
        fclose(_file);
        return false;
    }

    config_q2 *_config = &_model->config;

    if (fread(_config, sizeof(config_q2), 1, _file) != 1) {
        log_msg(stderr, "ERROR: Failed to read config from %s\n", _file_path_s);
        fclose(_file);
        return false;
    }

    if (! tokenizer_read_from_file(_file, _config->vocab_size, &_model->tokenizer1)) {
        log_msg(stderr, "ERROR: Failed to read tokenizer from %s\n", _file_path_s);
        fclose(_file);
        return false;
    }

    weights_q2 *_weights = &_model->weights;

    _weights->_rms_att_weight = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_rms_ffn_weight = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));

    if ((!_weights->_rms_att_weight) ||
            (!_weights->_rms_ffn_weight)) {
        log_msg(stderr, "ERROR: Failed to allocate memory for norm weights\n");
        fclose(_file);
        return false;
    }

    read_qt(_file, &_weights->embed_tokens_weight);

    for (int32_t l = 0; l < _config->n_layers; l++) {
        read_qt(_file, &_weights->_rms_att_weight[l]);
    }
    for (int32_t l = 0; l < _config->n_layers; l++) {
        read_qt(_file, &_weights->_rms_ffn_weight[l]);
    }

    read_qt(_file, &_weights->rms_final_weight);

    if (!_config->shared_classifier) {
        read_qt(_file, &_weights->wcls);
    } else {
        _weights->wcls = _weights->embed_tokens_weight;
    }

    _weights->_wq = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_wk = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_wv = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_wo = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_w1 = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_w2 = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_w3 = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    
    _weights->_q_bias = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_k_bias = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_v_bias = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));

    if ((!_weights->_wq) ||
            (!_weights->_wk) ||
            (!_weights->_wv) ||
            (!_weights->_wo) ||
            (!_weights->_w1) ||
            (!_weights->_w2) ||
            (!_weights->_w3) ||
            (!_weights->_q_bias) ||
            (!_weights->_k_bias) ||
            (!_weights->_v_bias)) {
        log_msg(stderr, "ERROR: Failed to allocate memory for quantized tensors\n");
        fclose(_file);
        return false;
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        read_qt(_file, &_weights->_wq[l]);
        read_qt(_file, &_weights->_wk[l]);
        read_qt(_file, &_weights->_wv[l]);
        read_qt(_file, &_weights->_wo[l]);
        
        read_qt(_file, &_weights->_q_bias[l]);
        read_qt(_file, &_weights->_k_bias[l]);
        read_qt(_file, &_weights->_v_bias[l]);

        read_qt(_file, &_weights->_w1[l]);
        read_qt(_file, &_weights->_w2[l]);
        read_qt(_file, &_weights->_w3[l]);
    }

    fclose(_file);
    log_msg(stdout, "INFO: Quantized model loaded from %s\n", _file_path_s);
    return true;
}

float *forward_q2(Q2 *_model, int32_t token, int32_t pos) {
    config_q2 *_config = &_model->config;
    weights_q2 *_weights = &_model->weights;
    state_q2 *_state = &_model->state;
    int32_t kv_dim = _config->n_kv_heads * _config->head_dim;
    int32_t kv_mul = _config->n_heads / _config->n_kv_heads;
    int32_t all_heads_dim = _config->n_heads * _config->head_dim;
    float eps = _config->rms_norm_eps;

    dequantize_row(_state->_x, &_weights->embed_tokens_weight, token);

    for (int32_t l = 0; l < _config->n_layers; l++) {
        int64_t loff = l * _state->seq_n * kv_dim;

        rmsnorm(_state->_xb, _state->_x, (float *)_weights->_rms_att_weight[l]._data, _config->dim, eps);

        quantize_vec(&_state->xq, _state->_xb, _config->dim);

        matmul_qq(_state->_q, &_state->xq, &_weights->_wq[l]);
        matmul_qq(_state->_k, &_state->xq, &_weights->_wk[l]);
        matmul_qq(_state->_v, &_state->xq, &_weights->_wv[l]);

        /* Add QKV Biases */
        float *q_bias = (float *)_weights->_q_bias[l]._data;
        float *k_bias = (float *)_weights->_k_bias[l]._data;
        float *v_bias = (float *)_weights->_v_bias[l]._data;
        for (int32_t i = 0; i < all_heads_dim; i++) {
            _state->_q[i] += q_bias[i];
        }
        for (int32_t i = 0; i < kv_dim; i++) {
            _state->_k[i] += k_bias[i];
            _state->_v[i] += v_bias[i];
        }

        int32_t rotary_half = _config->head_dim / 2;

        if ((rotary_half > 0) && (_state->_cos_cache != NULL)) {
            float *_cos_row = _state->_cos_cache + pos * rotary_half;
            float *_sin_row = _state->_sin_cache + pos * rotary_half;

#pragma omp parallel for
            for (int32_t h = 0; h < _config->n_heads; h++) {
                float *_q = _state->_q + h * _config->head_dim;
                for (int32_t j = 0; j < rotary_half; j++) {
                    float c = _cos_row[j], sn = _sin_row[j];
                    float x_val = _q[j], y_val = _q[j + rotary_half];
                    _q[j] = x_val * c - y_val * sn;
                    _q[j + rotary_half] = x_val * sn + y_val * c;
                }
            }

#pragma omp parallel for
            for (int32_t h = 0; h < _config->n_kv_heads; h++) {
                float *_k = _state->_k + h * _config->head_dim;
                for (int32_t j = 0; j < rotary_half; j++) {
                    float c = _cos_row[j], sn = _sin_row[j];
                    float x_val = _k[j], y_val = _k[j + rotary_half];
                    _k[j] = x_val * c - y_val * sn;
                    _k[j + rotary_half] = x_val * sn + y_val * c;
                }
            }
        } else {
#pragma omp parallel for
            for (int32_t h = 0; h < _config->n_heads; h++) {
                float *_q = _state->_q + h * _config->head_dim;
                for (int32_t j = 0; j < rotary_half; j++) {
                    float freq = 1.0f / powf(_config->rope_theta, (float)j / rotary_half);
                    float scaled_pos = pos / _config->rope_scaling_factor;
                    float cos_freq = cosf(scaled_pos * freq);
                    float sin_freq = sinf(scaled_pos * freq);
                    float x_val = _q[j], y_val = _q[j + rotary_half];
                    _q[j] = x_val * cos_freq - y_val * sin_freq;
                    _q[j + rotary_half] = x_val * sin_freq + y_val * cos_freq;
                 }
            }

#pragma omp parallel for
            for (int32_t h = 0; h < _config->n_kv_heads; h++) {
                float *_k = _state->_k + h * _config->head_dim;
                for (int32_t j = 0; j < rotary_half; j++) {
                    float freq = 1.0f / powf(_config->rope_theta, (float)j / rotary_half);
                    float scaled_pos = pos / _config->rope_scaling_factor;
                    float cos_freq = cosf(scaled_pos * freq);
                    float sin_freq = sinf(scaled_pos * freq);
                    float x_val = _k[j], y_val = _k[j + rotary_half];
                    _k[j] = x_val * cos_freq - y_val * sin_freq;
                    _k[j + rotary_half] = x_val * sin_freq + y_val * cos_freq;
                 }
            }
        }

        memcpy(_state->_key_cache + loff + pos * kv_dim, _state->_k, kv_dim * sizeof(float));
        memcpy(_state->_value_cache + loff + pos * kv_dim, _state->_v, kv_dim * sizeof(float));

#pragma omp parallel for
        for (int32_t h = 0; h < _config->n_heads; h++) {
            float *_q = _state->_q + h * _config->head_dim;
            float *_att = _state->_att + h * _state->seq_n;
            for (int32_t t = 0; t <= pos; t++) {
                float *_k = _state->_key_cache + loff + t * kv_dim + (h / kv_mul) * _config->head_dim;
                float score = 0.0f;

#pragma omp simd reduction(+ : score)
                for (int32_t i = 0; i < _config->head_dim; i++) {
                    score += _q[i] * _k[i];
                }
                _att[t] = score / sqrtf(_config->head_dim);
            }

            softmax(_att, pos + 1);

            float *_xb = _state->_xb + h * _config->head_dim;
            memset(_xb, 0, _config->head_dim * sizeof(float));

            for (int32_t t = 0; t <= pos; t++) {
                float *_v = _state->_value_cache + loff + t * kv_dim + (h / kv_mul) * _config->head_dim;
                float a = _att[t];

#pragma omp simd
                for (int32_t i = 0; i < _config->head_dim; i++) {
                    _xb[i] += a * _v[i];
                }
            }
        }

        quantize_vec(&_state->xq, _state->_xb, all_heads_dim);

        matmul_qq(_state->_xb, &_state->xq, &_weights->_wo[l]);

        for (int32_t i = 0; i < _config->dim; i++) {
            _state->_x[i] += _state->_xb[i];
        }

        rmsnorm(_state->_xb, _state->_x, (float *)_weights->_rms_ffn_weight[l]._data, _config->dim, eps);

        quantize_vec(&_state->xq, _state->_xb, _config->dim);

        matmul_qq(_state->_hb, &_state->xq, &_weights->_w1[l]);
        matmul_qq(_state->_hb2, &_state->xq, &_weights->_w3[l]);

#pragma omp parallel for
        for (int32_t i = 0; i < _config->hidden_dim; i++) {
            _state->_hb[i] = _state->_hb[i] * (1.0f / (1.0f + expf(-_state->_hb[i]))) * _state->_hb2[i];
        }

        quantize_vec(&_state->hq, _state->_hb, _config->hidden_dim);

        matmul_qq(_state->_xb, &_state->hq, &_weights->_w2[l]);

        for (int32_t i = 0; i < _config->dim; i++) {
            _state->_x[i] += _state->_xb[i];
        }
    }

    rmsnorm(_state->_x, _state->_x, (float *)_weights->rms_final_weight._data, _config->dim, eps);

    matmul_qt(_state->_logits, _state->_x, &_weights->wcls);

    return _state->_logits;
}

static float *forward_q2_wrap(void *_model, int32_t token, int32_t pos) {
    return forward_q2((Q2 *)_model, token, pos);
}

static void free_q2_wrap(void *_model) {
    free_q2((Q2 *)_model);
    free(_model);
}

model_iface *init_q2(const char *_model_path_s, int32_t seq_n, bool think_) {
    Q2 *_model = a_calloc(sizeof(Q2));

    if (! load_quantized_q2(_model_path_s, _model)) {
        free_q2_wrap(_model);
        return NULL;
    }

    if (! alloc_state_q2(_model, seq_n)) {
        free_q2_wrap(_model);
        return NULL;
    }


    _model->tokenizer1.bos_id = _model->config.bos_token_id;
    _model->tokenizer1.eos_id = _model->config.eos_token_id;
    _model->tokenizer1.im_end_id = _model->config.eos_token_id;

    model_iface *_model_i = a_calloc(sizeof(model_iface));
    *_model_i = (model_iface){
        ._model = _model,
        .forward = forward_q2_wrap,
        .free_model = free_q2_wrap,
        .seq_n = seq_n ? seq_n : _model->config.seq_n,
        .seq_n_model_max = _model->config.seq_n,
        ._chat_template = &CHAT_TEMPLATE_Q2,
        ._tokenizer = &(_model->tokenizer1),
    };

    return _model_i;
}

