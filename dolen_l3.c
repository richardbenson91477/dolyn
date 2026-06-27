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


int load_quantized_l3(const char *filepath, L3 *_model, int seq_n_max) {
    FILE *f = fopen(filepath, "rb");
    if (! f) {
        log_msg(stderr, "ERROR: Failed to open %s\n", filepath);
        return -1;
    }

    memset(_model, 0, sizeof(L3));

    uint64_t magic;
    uint32_t version;

    if ((fread(&magic, sizeof(magic), 1, f) != 1) ||
            (fread(&version, sizeof(version), 1, f) != 1)) {
        log_msg(stderr, "ERROR: Failed to read header\n");
        fclose(f);
        return -1;
    }

    if (magic != MAGIC_L3) {
        log_msg(stderr, "ERROR: Invalid magic number\n");
        fclose(f);
        return -1;
    }

    if (version != 1) {
        log_msg(stderr, "ERROR: Unsupported version %u (expected 1)\n", version);
        fclose(f);
        return -1;
    }

    config_l3 *_config = &_model->config;
    if (fread(_config, sizeof(config_l3), 1, f) != 1) {
        log_msg(stderr, "ERROR: Failed to read config\n");
        fclose(f);
        return -1;
    }

    if (tokenizer_read_from_file(f, _config->vocab_size, &_model->tokenizer1)) {
        log_msg(stderr, "ERROR: Failed to read tokenizer\n");
        fclose(f);
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
        fclose(f);
        return -1;
    }

    read_qt(f, &_weights->embed_tokens_weight);

    for (int i = 0; i < _config->n_layers; i++) {
        read_qt(f, &_weights->_rms_att_weight[i]);
    }

    for (int i = 0; i < _config->n_layers; i++) {
        read_qt(f, &_weights->_wq[i]);
        read_qt(f, &_weights->_wk[i]);
        read_qt(f, &_weights->_wv[i]);
        read_qt(f, &_weights->_wo[i]);
    }    

    for (int i = 0; i < _config->n_layers; i++) {
        read_qt(f, &_weights->_rms_ffn_weight[i]);
    }

    for (int i = 0; i < _config->n_layers; i++) {
        read_qt(f, &_weights->_w1[i]);
        read_qt(f, &_weights->_w2[i]);
        read_qt(f, &_weights->_w3[i]);
    }

    read_qt(f, &_weights->rms_final_weight);

    if (! _config->tie_word_embeddings) {
        read_qt(f, &_weights->wcls);
    } else {
        _weights->wcls = _weights->embed_tokens_weight;
    }

    fclose(f);
    log_msg(stdout, "INFO: Quantized L3 loaded from %s\n", filepath);

    alloc_state_l3(&_model->state, _config);
    return 0;
}

static void apply_rope(float *vec, float *cos, float *sin, int rotary_dim, int pos) {
    int half_rot = rotary_dim / 2;
    float *cos_row = cos + pos * half_rot;
    float *sin_row = sin + pos * half_rot;

    for (int i = 0; i < half_rot; i++) {
        float c = cos_row[i], sn = sin_row[i];
        float v0 = vec[i], v1 = vec[i + half_rot];
        vec[i] = v0 * c - v1 * sn;
        vec[i + half_rot] = v0 * sn + v1 * c;
    }
}

float *forward_l3(L3 *_model, int token, int pos) {
    config_l3 *_config = &_model->config;
    weights_l3 *_weights = &_model->weights;
    state_l3 *_state = &_model->state;
    float *x = _state->_x;
    int dim = _config->dim;
    int head_size = _config->head_dim;
    int kv_dim = _config->n_kv_heads * head_size;
    int kv_mul = _config->n_heads / _config->n_kv_heads;
    float eps = _config->rms_norm_eps;

    dequantize_row(x, &_weights->embed_tokens_weight, token);

    for (int l = 0; l < _config->n_layers; l++) {
        float *rms_att = (float *)_weights->_rms_att_weight[l]._data;
        rmsnorm(_state->_xb, x, rms_att, dim, eps);

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
            float *q = _state->_q + h * head_size;
            float *_att = _state->_att + h * _config->seq_len;
            int kv_head = h / kv_mul;

            for (int t = 0; t <= pos; t++) {
                float *k = _state->__key_cache[l] + (long long)t * kv_dim + (long long)kv_head * head_size;
                float score = 0.0f;
#pragma omp simd reduction(+ : score)
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                _att[t] = score * inv_sqrt_head;
            }
            softmax(_att, pos + 1);

            float *out = _state->_xb + h * head_size; 
            memset(out, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v = _state->__value_cache[l] + (long long)t * kv_dim + (long long)kv_head * head_size;
                float a = _att[t];
#pragma omp simd
                for (int i = 0; i < head_size; i++) {
                    out[i] += a * v[i];
                }
            }
        }

        quantize_vec(&_state->xq, _state->_xb, _config->n_heads * head_size);
        matmul_qq(_state->_xb2, &_state->xq, &_weights->_wo[l]);

        for (int i = 0; i < dim; i++) {
            x[i] += _state->_xb2[i];
        }

        float *rms_ffn = (float *)_weights->_rms_ffn_weight[l]._data;
        rmsnorm(_state->_xb, x, rms_ffn, dim, eps);

        quantize_vec(&(_state->xq), _state->_xb, dim);
        matmul_qq(_state->_hb, &_state->xq, &_weights->_w1[l]);
        matmul_qq(_state->_hb2, &_state->xq, &_weights->_w3[l]);

#pragma omp parallel for
        for (int i = 0; i < _config->hidden_dim; i++) {
            float val = _state->_hb[i];
            val *= (1.0f / (1.0f + expf(-val))); // Swish / SiLU Activation
            val *= _state->_hb2[i];
            _state->_hb[i] = val;
        }

        quantize_vec(&_state->hq, _state->_hb, _config->hidden_dim);
        matmul_qq(_state->_xb, &_state->hq, &_weights->_w2[l]);

        for (int i = 0; i < dim; i++) {
            x[i] += _state->_xb[i];
        }
    }

    rmsnorm(x, x, (float *)_weights->rms_final_weight._data, dim, eps);

    if (_config->tie_word_embeddings) {
        matmul_qt(_state->_logits, x, &_weights->embed_tokens_weight);
    } else {
        matmul_qt(_state->_logits, x, &_weights->wcls);
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

model_iface *init_l3(const char *model_path, int seq_n_max, bool think_) {
    L3 *_model = a_calloc(sizeof(L3));
    if (load_quantized_l3(model_path, _model, seq_n_max)) {
        free_l3(_model);
        free(_model);
        return NULL;
    }

    // Llama 3 Token mappings
    _model->tokenizer1.bos_id = 128000; // <|begin_of_text|>
    _model->tokenizer1.eos_id = 128001; // <|end_of_text|>
    _model->tokenizer1.im_end_id = 128009;    // <|eot_id|>

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

