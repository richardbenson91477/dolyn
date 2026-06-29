#include "dolen_l3_common.h"

static const chat_template CHAT_TEMPLATE_L3 = {
    ._system_s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>",
    ._main_s = "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ._end_turn_s = "<|eot_id|>",
};

static const chat_template CHAT_TEMPLATE_THINK_L3 = {
    ._system_s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>",
    ._main_s = "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ._end_turn_s = "<|eot_id|>",
};

int load_quantized_l3(const char *_file_path_s, L3 *_model, int seq_n_max) {
    FILE *_file = fopen(_file_path_s, "rb");
    if (! _file) {
        log_msg(stderr, "ERROR: Failed to open %s\n", _file_path_s);
        return -1;
    }

    memset(_model, 0, sizeof(L3));

    uint64_t magic;
    uint32_t version;

    if ((fread(&magic, sizeof(magic), 1, _file) != 1) ||
            (fread(&version, sizeof(version), 1, _file) != 1)) {
        log_msg(stderr, "ERROR: Failed to read header\n");
        fclose(_file);
        return -1;
    }

    if (magic != MAGIC_L3) {
        log_msg(stderr, "ERROR: Invalid magic number\n");
        fclose(_file);
        return -1;
    }

    if (version != 2) {
        log_msg(stderr, "ERROR: Unsupported version %u (expected 2)\n", version);
        fclose(_file);
        return -1;
    }

    config_l3 *_config = &_model->config;
    if (fread(_config, sizeof(config_l3), 1, _file) != 1) {
        log_msg(stderr, "ERROR: Failed to read config\n");
        fclose(_file);
        return -1;
    }

    if (tokenizer_read_from_file(_file, _config->vocab_size, &_model->tokenizer1)) {
        log_msg(stderr, "ERROR: Failed to read tokenizer\n");
        fclose(_file);
        return -1;
    }

    if (seq_n_max) {
        _config->seq_len = seq_n_max;
    }

    weights_l3 *_weights = &_model->weights;

    _weights->_rms_att_weight = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_rms_ffn_weight = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_wq = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_wk = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_wv = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_wo = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_w1 = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_w2 = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_w3 = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));

    if ((! _weights->_rms_att_weight) ||
            (! _weights->_rms_ffn_weight) ||
            (! _weights->_wq) ||
            (! _weights->_wk) ||
            (! _weights->_wv) ||
            (! _weights->_wo) ||
            (! _weights->_w1) ||
            (! _weights->_w2) ||
            (! _weights->_w3)) {
        log_msg(stderr, "ERROR: Alloc failed\n");
        fclose(_file);
        return -1;
    }

    read_qt(_file, &_weights->embed_tokens_weight);

    for (int i = 0; i < _config->n_layers; i++) {
        read_qt(_file, &_weights->_rms_att_weight[i]);
    }

    for (int i = 0; i < _config->n_layers; i++) {
        read_qt(_file, &_weights->_wq[i]);
        read_qt(_file, &_weights->_wk[i]);
        read_qt(_file, &_weights->_wv[i]);
        read_qt(_file, &_weights->_wo[i]);
    }    

    for (int i = 0; i < _config->n_layers; i++) {
        read_qt(_file, &_weights->_rms_ffn_weight[i]);
    }

    for (int i = 0; i < _config->n_layers; i++) {
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
    log_msg(stdout, "INFO: Quantized L3 loaded from %s\n", _file_path_s);

    alloc_state_l3(&_model->state, _config);
    return 0;
}

static void apply_rope(float *_vec, float *_cos, float *_sin, int rotary_dim, int pos) {
    int half_rot = rotary_dim / 2;
    float *_cos_row = _cos + pos * half_rot;
    float *_sin_row = _sin + pos * half_rot;

    for (int i = 0; i < half_rot; i++) {
        float c = _cos_row[i];
        float sn = _sin_row[i];
        float v0 = _vec[i];
        float v1 = _vec[i + half_rot];
        _vec[i] = v0 * c - v1 * sn;
        _vec[i + half_rot] = v0 * sn + v1 * c;
    }
}

float *forward_l3(L3 *_model, int token, int pos) {
    config_l3 *_config = &_model->config;
    weights_l3 *_weights = &_model->weights;
    state_l3 *_state = &_model->state;
    float *_x = _state->_x;
    int dim = _config->dim;
    int head_size = _config->head_dim;
    int kv_dim = _config->n_kv_heads * head_size;
    int kv_mul = _config->n_heads / _config->n_kv_heads;
    float eps = _config->rms_norm_eps;

    dequantize_row(_x, &_weights->embed_tokens_weight, token);

    for (int l = 0; l < _config->n_layers; l++) {
        float *_rms_att = (float *)_weights->_rms_att_weight[l]._data;
        rmsnorm(_state->_xb, _x, _rms_att, dim, eps);

        quantize_vec(&_state->xq, _state->_xb, dim);
        matmul_qq(_state->_q, &_state->xq, &_weights->_wq[l]);
        matmul_qq(_state->_k, &_state->xq, &_weights->_wk[l]);
        matmul_qq(_state->_v, &_state->xq, &_weights->_wv[l]);

        for (int h = 0; h < _config->n_heads; h++) {
            apply_rope(_state->_q + h * head_size, _state->_cos_cache, _state->_sin_cache, head_size, pos);
        }
        for (int h = 0; h < _config->n_kv_heads; h++) {
            apply_rope(_state->_k + h * head_size, _state->_cos_cache, _state->_sin_cache, head_size, pos);
        }

        memcpy(_state->__key_cache[l] + (long long)pos * kv_dim, _state->_k, kv_dim * sizeof(float));
        memcpy(_state->__value_cache[l] + (long long)pos * kv_dim, _state->_v, kv_dim * sizeof(float));

        float inv_sqrt_head = 1.0f / sqrtf((float)head_size);
#pragma omp parallel for
        for (int h = 0; h < _config->n_heads; h++) {
            float *_q = _state->_q + h * head_size;
            float *_att = _state->_att + h * _config->seq_len;
            int kv_head = h / kv_mul;

            for (int t = 0; t <= pos; t++) {
                float *_k = _state->__key_cache[l] + (long long)t * kv_dim + (long long)kv_head * head_size;
                float score = 0.0f;
#pragma omp simd reduction(+ : score)
                for (int i = 0; i < head_size; i++) {
                    score += _q[i] * _k[i];
                }
                _att[t] = score * inv_sqrt_head;
            }
            softmax(_att, pos + 1); 

            float *_x_out = _state->_xb + h * head_size; 
            memset(_x_out, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *_v = _state->__value_cache[l] + (long long)t * kv_dim + (long long)kv_head * head_size;
                float a = _att[t];
#pragma omp simd
                for (int i = 0; i < head_size; i++) {
                    _x_out[i] += a * _v[i];
                }
            }
        }

        quantize_vec(&_state->xq, _state->_xb, _config->n_heads * head_size);
        matmul_qq(_state->_xb2, &_state->xq, &_weights->_wo[l]);

        for (int i = 0; i < dim; i++) {
            _x[i] += _state->_xb2[i];
        }

        float *_rms_ffn = (float *)_weights->_rms_ffn_weight[l]._data;
        rmsnorm(_state->_xb, _x, _rms_ffn, dim, eps);

        quantize_vec(&(_state->xq), _state->_xb, dim);
        matmul_qq(_state->_hb, &_state->xq, &_weights->_w1[l]);
        matmul_qq(_state->_hb2, &_state->xq, &_weights->_w3[l]);

#pragma omp parallel for
        for (int i = 0; i < _config->hidden_dim; i++) {
            float val = _state->_hb[i];
            // Swish / SiLU Activation
            val *= (1.0f / (1.0f + expf(-val)));
            val *= _state->_hb2[i];
            _state->_hb[i] = val;
        }

        quantize_vec(&_state->hq, _state->_hb, _config->hidden_dim);
        matmul_qq(_state->_xb, &_state->hq, &_weights->_w2[l]);

        for (int i = 0; i < dim; i++) {
            _x[i] += _state->_xb[i];
        }
    }

    rmsnorm(_x, _x, (float *)_weights->rms_final_weight._data, dim, eps);

    if (_config->tie_word_embeddings) {
        matmul_qt(_state->_logits, _x, &_weights->embed_tokens_weight);
    } else {
        matmul_qt(_state->_logits, _x, &_weights->wcls);
    }

    return _state->_logits;
}

static float *forward_l3_wrap(void *_model, int token, int pos) {
    return forward_l3((L3 *)_model, token, pos);
}

static void free_l3_wrap(void *_model) {
    free_l3((L3 *)_model);
    free(_model);
}

model_iface *init_l3(const char *_model_path_s, int seq_n_max, bool think_) {
    L3 *_model = a_calloc(sizeof(L3));
    if (load_quantized_l3(_model_path_s, _model, seq_n_max)) {
        free_l3(_model);
        free(_model);
        return NULL;
    }

    // Dynamically map Token IDs from the embedded configuration
    _model->tokenizer1.bos_id = _model->config.bos_token_id;
    _model->tokenizer1.eos_id = _model->config.eos_token_id;
    _model->tokenizer1.im_end_id = _model->config.eos_token_id;

    model_iface *_model_i = a_calloc(sizeof(model_iface));
    *_model_i = (model_iface){
        ._model = _model,
        .forward = forward_l3_wrap,
        .free_model = free_l3_wrap,
        .seq_n_max = seq_n_max ? seq_n_max : _model->config.seq_len,
        ._chat_template = think_ ? &CHAT_TEMPLATE_THINK_L3 : &CHAT_TEMPLATE_L3,
        ._tokenizer = &_model->tokenizer1
    };

    return _model_i;
}
