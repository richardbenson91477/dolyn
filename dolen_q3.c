#include "dolen_q3_common.h"

static token_map SPECIAL_TOKENS_Q3[] = {
    {"<|endoftext|>", 151643},
    {"<|im_start|>", 151644},
    {"<|im_end|>", 151645},
    {"<|object_ref_start|>", 151646},
    {"<|object_ref_end|>", 151647},
    {"<|box_start|>", 151648},
    {"<|box_end|>", 151649},
    {"<|quad_start|>", 151650},
    {"<|quad_end|>", 151651},
    {"<|vision_start|>", 151652},
    {"<|vision_end|>", 151653},
    {"<|vision_pad|>", 151654},
    {"<|image_pad|>", 151655},
    {"<|video_pad|>", 151656},
    {NULL, 0}
};

static const chat_template CHAT_TEMPLATE_Q3 = {
    ._system_s = "<|im_start|>system\n/no_think\n%s<|im_end|>\n",
    ._main_s = "<|im_start|>user\n%s<|im_end|>\n"
             "<|im_start|>assistant\n",
    ._end_turn_s = "<|im_end|>\n",
};

static const chat_template CHAT_TEMPLATE_THINK_Q3 = {
    ._system_s = "<|im_start|>system\n/think\n%s<|im_end|>\n",
    ._main_s = "<|im_start|>user\n%s<|im_end|>\n"
             "<|im_start|>assistant<think>\n",
    ._end_turn_s = "<|im_end|>\n",
};

int load_quantized_q3(const char *_file_path_s, Q3 *_model, int seq_n_max) {
    FILE *_file = fopen(_file_path_s, "rb");
    if (!_file) {
        log_msg(stderr, "ERROR: Failed to open %s for reading\n", _file_path_s);
        return -1;
    }

    memset(_model, 0, sizeof(Q3));

    uint64_t magic;
    uint32_t version;

    if ((fread(&magic, sizeof(uint64_t), 1, _file) != 1) ||
            (fread(&version, sizeof(uint32_t), 1, _file) != 1)) {
        log_msg(stderr, "ERROR: Failed to read header from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    if (magic != MAGIC_Q3) {
        log_msg(stderr, "ERROR: Invalid magic number in %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    if (version != 3) {
        log_msg(stderr, "ERROR: Unsupported version %d in %s\n", version, _file_path_s);
        fclose(_file);
        return -1;
    }

    config_q3 *_config = &_model->config;

    if (fread(_config, sizeof(config_q3), 1, _file) != 1) {
        log_msg(stderr, "ERROR: Failed to read config from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    if (tokenizer_read_from_file(_file, _config->vocab_size, &_model->tokenizer1)) {
        log_msg(stderr, "ERROR: Failed to read tokenizer from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    weights_q3 *_weights = &_model->weights;

    if (seq_n_max) {
        _config->seq_len = seq_n_max;
    }

    _weights->_rms_att_weight = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_rms_ffn_weight = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));

    _weights->_q_norm = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_k_norm = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));

    if ((!_weights->_rms_att_weight) ||
            (!_weights->_rms_ffn_weight) ||
            (!_weights->_q_norm) ||
            (!_weights->_k_norm)) {
        log_msg(stderr, "ERROR: Failed to allocate memory for weights\n");
        fclose(_file);
        return -1;
    }

    read_qt(_file, &_weights->embed_tokens_weight);

    for (int l = 0; l < _config->n_layers; l++) {
        read_qt(_file, &_weights->_rms_att_weight[l]);
    }
    for (int l = 0; l < _config->n_layers; l++) {
        read_qt(_file, &_weights->_rms_ffn_weight[l]);
    }

    read_qt(_file, &_weights->rms_final_weight);

    for (int l = 0; l < _config->n_layers; l++) {
        read_qt(_file, &_weights->_q_norm[l]);
    }
    for (int l = 0; l < _config->n_layers; l++) {
        read_qt(_file, &_weights->_k_norm[l]);
    }

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

    if ((!_weights->_wq) ||
            (!_weights->_wk) ||
            (!_weights->_wv) ||
            (!_weights->_wo) ||
            (!_weights->_w1) ||
            (!_weights->_w2) ||
            (!_weights->_w3)) {
        log_msg(stderr, "ERROR: Failed to allocate memory for quantized tensors\n");
        fclose(_file);
        return -1;
    }

    for (int l = 0; l < _config->n_layers; l++) {
        read_qt(_file, &_weights->_wq[l]);
        read_qt(_file, &_weights->_wk[l]);
        read_qt(_file, &_weights->_wv[l]);
        read_qt(_file, &_weights->_wo[l]);
        read_qt(_file, &_weights->_w1[l]);
        read_qt(_file, &_weights->_w2[l]);
        read_qt(_file, &_weights->_w3[l]);
    }

    fclose(_file);
    log_msg(stdout, "INFO: Quantized model loaded from %s\n", _file_path_s);

    alloc_state_q3(&(_model->state), &(_model->config));

    return 0;
}

float *forward_q3(Q3 *_model, int token, int pos) {
    config_q3 *_config = &_model->config;
    weights_q3 *_weights = &_model->weights;
    state_q3 *_state = &_model->state;
    int kv_dim = _config->n_kv_heads * _config->head_dim;
    int kv_mul = _config->n_heads / _config->n_kv_heads;
    int all_heads_dim = _config->n_heads * _config->head_dim;
    float eps = _config->rms_norm_eps;

    dequantize_row(_state->_x, &_weights->embed_tokens_weight, token);

    for (int l = 0; l < _config->n_layers; l++) {
        long long loff = l * _config->seq_len * kv_dim;

        rmsnorm(_state->_xb, _state->_x, (float *)_weights->_rms_att_weight[l]._data, _config->dim, eps);

        quantize_vec(&_state->xq, _state->_xb, _config->dim);

        matmul_qq(_state->_q, &_state->xq, &_weights->_wq[l]);
        matmul_qq(_state->_k, &_state->xq, &_weights->_wk[l]);
        matmul_qq(_state->_v, &_state->xq, &_weights->_wv[l]);

        int rotary_half = _config->head_dim / 2;

        if ((rotary_half > 0) && (_state->_cos_cache != NULL)) {
            float *_cos_row = _state->_cos_cache + pos * rotary_half;
            float *_sin_row = _state->_sin_cache + pos * rotary_half;

#pragma omp parallel for
            for (int h = 0; h < _config->n_heads; h++) {
                float *_q = _state->_q + h * _config->head_dim;

                rmsnorm(_q, _q, (float *)_weights->_q_norm[l]._data, _config->head_dim, eps);

                for (int j = 0; j < rotary_half; j++) {
                    float c = _cos_row[j], sn = _sin_row[j];
                    float x_val = _q[j], y_val = _q[j + rotary_half];
                    _q[j] = x_val * c - y_val * sn;
                    _q[j + rotary_half] = x_val * sn + y_val * c;
                }
            }

#pragma omp parallel for
            for (int h = 0; h < _config->n_kv_heads; h++) {
                float *_k = _state->_k + h * _config->head_dim;

                rmsnorm(_k, _k, (float *)_weights->_k_norm[l]._data, _config->head_dim, eps);

                for (int j = 0; j < rotary_half; j++) {
                    float c = _cos_row[j], sn = _sin_row[j];
                    float x_val = _k[j], y_val = _k[j + rotary_half];
                    _k[j] = x_val * c - y_val * sn;
                    _k[j + rotary_half] = x_val * sn + y_val * c;
                }
            }
        } else {
#pragma omp parallel for
            for (int h = 0; h < _config->n_heads; h++) {
                float *_q = _state->_q + h * _config->head_dim;
                rmsnorm(_q, _q, (float *)_weights->_q_norm[l]._data, _config->head_dim, eps);
                for (int j = 0; j < rotary_half; j++) {
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
            for (int h = 0; h < _config->n_kv_heads; h++) {
                float *_k = _state->_k + h * _config->head_dim;
                rmsnorm(_k, _k, (float *)_weights->_k_norm[l]._data, _config->head_dim, eps);
                for (int j = 0; j < rotary_half; j++) {
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
        for (int h = 0; h < _config->n_heads; h++) {
            float *_q = _state->_q + h * _config->head_dim;
            float *_att = _state->_att + h * _config->seq_len;
            for (int t = 0; t <= pos; t++) {
                float *_k = _state->_key_cache + loff + t * kv_dim + (h / kv_mul) * _config->head_dim;
                float score = 0.0f;

#pragma omp simd reduction(+ : score)
                for (int i = 0; i < _config->head_dim; i++) {
                    score += _q[i] * _k[i];
                }
                _att[t] = score / sqrtf(_config->head_dim);
            }

            softmax(_att, pos + 1);

            float *_xb = _state->_xb + h * _config->head_dim;
            memset(_xb, 0, _config->head_dim * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                float *_v = _state->_value_cache + loff + t * kv_dim + (h / kv_mul) * _config->head_dim;
                float a = _att[t];

#pragma omp simd
                for (int i = 0; i < _config->head_dim; i++) {
                    _xb[i] += a * _v[i];
                }
            }
        }

        quantize_vec(&_state->xq, _state->_xb, all_heads_dim);

        matmul_qq(_state->_xb, &_state->xq, &_weights->_wo[l]);

        for (int i = 0; i < _config->dim; i++) {
            _state->_x[i] += _state->_xb[i];
        }

        rmsnorm(_state->_xb, _state->_x, (float *)_weights->_rms_ffn_weight[l]._data, _config->dim, eps);

        quantize_vec(&_state->xq, _state->_xb, _config->dim);

        matmul_qq(_state->_hb, &_state->xq, &_weights->_w1[l]);
        matmul_qq(_state->_hb2, &_state->xq, &_weights->_w3[l]);

#pragma omp parallel for
        for (int i = 0; i < _config->hidden_dim; i++) {
            _state->_hb[i] = _state->_hb[i] * (1.0f / (1.0f + expf(-_state->_hb[i]))) * _state->_hb2[i];
        }

        quantize_vec(&_state->hq, _state->_hb, _config->hidden_dim);

        matmul_qq(_state->_xb, &_state->hq, &_weights->_w2[l]);

        for (int i = 0; i < _config->dim; i++) {
            _state->_x[i] += _state->_xb[i];
        }
    }

    rmsnorm(_state->_x, _state->_x, (float *)_weights->rms_final_weight._data, _config->dim, eps);

    matmul_qt(_state->_logits, _state->_x, &_weights->wcls);

    return _state->_logits;
}

static float *forward_q3_wrap(void *_model, int token, int pos) {
    return forward_q3((Q3 *)_model, token, pos);
}

static void free_q3_wrap(void *_model) {
    free_q3((Q3 *)_model);
    free(_model);
}

model_iface *init_q3(const char *_model_path_s, int seq_n_max, bool think_) {
    Q3 *_model = a_calloc(1 * sizeof(Q3));

    if (load_quantized_q3(_model_path_s, _model, seq_n_max)) {
        free_q3(_model);
        free(_model);
        return NULL;
    }

    // Map dynamically from config instead of hardcoding Qwen3 specific IDs
    _model->tokenizer1._tokens_special = SPECIAL_TOKENS_Q3;
    _model->tokenizer1.bos_id = _model->config.bos_token_id;
    _model->tokenizer1.eos_id = _model->config.eos_token_id;
    _model->tokenizer1.im_end_id = _model->config.eos_token_id;

    model_iface *_model_i = a_calloc(sizeof(model_iface));
    *_model_i = (model_iface){
        ._model = _model,
        .forward = forward_q3_wrap,
        .free_model = free_q3_wrap,
        .seq_n_max = seq_n_max ? seq_n_max : _model->config.seq_len,
        ._chat_template = think_ ? &CHAT_TEMPLATE_THINK_Q3 : &CHAT_TEMPLATE_Q3,
        ._tokenizer = &_model->tokenizer1,
    };
    return _model_i;
}