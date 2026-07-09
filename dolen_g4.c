#include "dolen_g4_common.h"


// The "\x3e" escaped ">" symbol serves to prevent LLMs from misinterpreting the text
static const chat_template CHAT_TEMPLATE_G4 = {
    ._system_s = "<|turn|\x3e" "system\n%s" "<turn|\x3e" "\n",
    ._main_s = "<|turn|\x3e" "user\n%s" "<turn|\x3e" "\n"
             "<|turn|\x3e" "model\n"
             "<|channel|\x3e" "thought\n<channel|\x3e",
    ._end_turn_s = "<turn|\x3e" "\n",
};

static const chat_template CHAT_TEMPLATE_THINK_G4 = {
    ._system_s = "<|turn|\x3e" "system\n<|think|\x3e" "%s" "<turn|\x3e" "\n",
    ._main_s = "<|turn|\x3e" "user\n%s" "<turn|\x3e" "\n"
             "<|turn|\x3e" "model\n",
    ._end_turn_s = "<turn|\x3e" "\n",
};


bool load_quantized_g4(const char *_path_s, G4 *_model) {
    FILE *_file = fopen(_path_s, "rb");
    if (! _file) {
        log_msg(stderr, "ERROR: Failed to open %s\n", _path_s);
        return false;
    }

    memset(_model, 0, sizeof(G4));

    uint64_t magic;
    uint32_t version;

    if (fread(&magic, sizeof(magic), 1, _file) != 1 ||
            fread(&version, sizeof(version), 1, _file) != 1) {
        log_msg(stderr, "ERROR: Failed to read header\n");
        fclose(_file);
        return false;
    }

    if (magic != MAGIC_G4) {
        log_msg(stderr, "ERROR: Invalid magic number\n");
        fclose(_file);
        return false;
    }

    if (version != 6) {
        log_msg(stderr, "ERROR: Unsupported version %u (expected 6). RE-RUN QUANTIZER.\n", version);
        fclose(_file);
        return false;
    }

    config_g4 *_config = &_model->config;

    if (fread(_config, sizeof(config_g4), 1, _file) != 1) {
        log_msg(stderr, "ERROR: Failed to read config\n");
        fclose(_file);
        return false;
    }

    if (! tokenizer_read_from_file(_file, _config->vocab_size, &_model->tokenizer)) {
        log_msg(stderr, "ERROR: Failed to read tokenizer from %s\n", _path_s);
        fclose(_file);
        return false;
    }

    _model->_layer_types = (int32_t *)a_calloc((size_t)_config->n_layers * sizeof(int32_t));
    if (fread(_model->_layer_types, sizeof(int32_t), (size_t)_config->n_layers, _file) != (size_t)_config->n_layers) {
        log_msg(stderr, "ERROR: Failed to read layer_types\n");
        fclose(_file);
        return false;
    }

    weights_g4 *_weights = &_model->weights;

    _weights->_rms_input_layernorm = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_rms_post_attn_layernorm = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_rms_pre_ffn_layernorm = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_rms_post_ffn_layernorm = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_rms_q_norm = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_rms_k_norm = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));

    if (! _weights->_rms_input_layernorm ||
            (! _weights->_rms_post_attn_layernorm) ||
            (! _weights->_rms_pre_ffn_layernorm) ||
            (! _weights->_rms_post_ffn_layernorm) ||
            (! _weights->_rms_q_norm) ||
            (! _weights->_rms_k_norm)) {
        log_msg(stderr, "ERROR: Alloc failed\n");
        fclose(_file);
        return false;
    }

    read_qt(_file, &_weights->embed_tokens_weight);

    for (int32_t i = 0; i < _config->n_layers; i++) {
        read_qt(_file, &_weights->_rms_input_layernorm[i]);
    }
    for (int32_t i = 0; i < _config->n_layers; i++) {
        read_qt(_file, &_weights->_rms_post_attn_layernorm[i]);
    }
    for (int32_t i = 0; i < _config->n_layers; i++) {
        read_qt(_file, &_weights->_rms_pre_ffn_layernorm[i]);
    }
    for (int32_t i = 0; i < _config->n_layers; i++) {
        read_qt(_file, &_weights->_rms_post_ffn_layernorm[i]);
    }
    for (int32_t i = 0; i < _config->n_layers; i++) {
        read_qt(_file, &_weights->_rms_q_norm[i]);
    }
    for (int32_t i = 0; i < _config->n_layers; i++) {
        read_qt(_file, &_weights->_rms_k_norm[i]);
    }

    read_qt(_file, &_weights->rms_final_norm);

    _weights->_q_proj = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_k_proj = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_v_proj = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_o_proj = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_gate_proj = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_up_proj = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_down_proj = (qtensor *)a_calloc((size_t)_config->n_layers * sizeof(qtensor));
    _weights->_layer_scalars = (float *)a_calloc((size_t)_config->n_layers * sizeof(float));

    for (int32_t i = 0; i < _config->n_layers; i++) {
        _weights->_layer_scalars[i] = 1.0f;
    }

    for (int32_t i = 0; i < _config->n_layers; i++) {
        read_qt(_file, &_weights->_q_proj[i]);
        read_qt(_file, &_weights->_k_proj[i]);
        read_qt(_file, &_weights->_v_proj[i]);
        read_qt(_file, &_weights->_o_proj[i]);
        read_qt(_file, &_weights->_gate_proj[i]);
        read_qt(_file, &_weights->_up_proj[i]);
        read_qt(_file, &_weights->_down_proj[i]);
    }

    fread(_weights->_layer_scalars, sizeof(float), (size_t)_config->n_layers, _file);

    if (_config->use_rope_freqs) {
        read_qt(_file, &_weights->rope_freqs_full);
    }

    fclose(_file);

    log_msg(stdout, "INFO: Quantized G4 loaded from %s\n", _path_s);
    return true;
}

static void apply_rope(float *_vec, float *_cos, float *_sin, int32_t rotary_dim, int32_t vec_dim, int32_t pos) {
    if (rotary_dim <= 0) {
        return;
    }

    int32_t half_rot = rotary_dim / 2;
    int32_t cache_stride = vec_dim / 2;
    float *_cos_row = _cos + pos * cache_stride;
    float *_sin_row = _sin + pos * cache_stride;

    for (int32_t i = 0; i < half_rot; i++) {
        float c = _cos_row[i];
        float sn = _sin_row[i];
        float v0 = _vec[i];
        float v1 = _vec[i + cache_stride];
        _vec[i] = v0 * c - v1 * sn;
        _vec[i + cache_stride] = v0 * sn + v1 * c;
    }
}

float *forward_g4(G4 *_model, int32_t token, int32_t pos) {
    config_g4 *_config = &_model->config;
    weights_g4 *_weights = &_model->weights;
    state_g4 *_state = &_model->state;
    float *_x = _state->_x;
    int32_t dim = _config->dim;
    float eps = _config->rms_norm_eps;
    float embed_scale = sqrtf((float)dim);

    if (token < 0 || token >= _config->vocab_size) {
        log_msg(stderr, "ERROR: token %d is outside vocabulary [0, %d)\n", token, _config->vocab_size);
        return NULL;
    }

    if ((pos < 0) || 
            (pos >= _state->seq_n)) {
        log_msg(stderr, "ERROR: position %d is outside KV cache [0, %d)\n", pos, _state->seq_n);
        return NULL;
    }

    dequantize_row(_x, &_weights->embed_tokens_weight, token);

#pragma omp simd
    for (int32_t i = 0; i < dim; i++) {
        _x[i] *= embed_scale;
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        int32_t is_full = _model->_layer_types[l];
        int32_t use_alternative_attention = is_full &&
                _config->attention_k_eq_v;
        int32_t head_dim = is_full ? _config->global_head_dim : _config->head_dim;
        int32_t kv_heads = use_alternative_attention ? _config->n_global_kv_heads : _config->n_kv_heads;
        int32_t kv_dim = kv_heads * head_dim;
        int32_t rotary_dim = is_full ?
            (int32_t)(_config->rope_partial_factor * _config->global_head_dim) : _config->head_dim;
        float *_cos_cache = is_full ? _state->_cos_cache_full : _state->_cos_cache_sliding;
        float *_sin_cache = is_full ? _state->_sin_cache_full : _state->_sin_cache_sliding;

        int32_t layer_attn_out_dim = _config->n_heads * head_dim;

        float *_rms_in = (float *)_weights->_rms_input_layernorm[l]._data;
        float *_rms_post_a = (float *)_weights->_rms_post_attn_layernorm[l]._data;
        float *_rms_pre_f = (float *)_weights->_rms_pre_ffn_layernorm[l]._data;
        float *_rms_post_f = (float *)_weights->_rms_post_ffn_layernorm[l]._data;
        float *_rms_q = (float *)_weights->_rms_q_norm[l]._data;
        float *_rms_k = (float *)_weights->_rms_k_norm[l]._data;

        rmsnorm_g4(_state->_xb, _x, _rms_in, dim, eps, true);

        quantize_vec(&_state->xq, _state->_xb, dim);
        matmul_qq(_state->_q, &_state->xq, &_weights->_q_proj[l]);
        matmul_qq(_state->_k_raw, &_state->xq, &_weights->_k_proj[l]);

        if (use_alternative_attention) {
            memcpy(_state->_v, _state->_k_raw, kv_dim * sizeof(float));
        } else {
            matmul_qq(_state->_v, &_state->xq, &_weights->_v_proj[l]);
        }

#pragma omp parallel for
        for (int32_t h = 0; h < _config->n_heads; h++) {
            float *_qh = _state->_q + h * head_dim;
            rmsnorm_g4(_qh, _qh, _rms_q, head_dim, eps, true);
            if ((rotary_dim > 0) &&
                    _cos_cache) {
                apply_rope(_qh, _cos_cache, _sin_cache, rotary_dim, head_dim, pos);
            }
        }

#pragma omp parallel for
        for (int32_t h = 0; h < kv_heads; h++) {
            float *_kh = _state->_k_raw + h * head_dim;
            rmsnorm_g4(_kh, _kh, _rms_k, head_dim, eps, true);
            if ((rotary_dim > 0) &&
                    _cos_cache) {
                apply_rope(_kh, _cos_cache, _sin_cache, rotary_dim, head_dim, pos);
            }
        }

        memcpy(_state->_k, _state->_k_raw, kv_dim * sizeof(float));

#pragma omp parallel for
        for (int32_t h = 0; h < kv_heads; h++) {
            rmsnorm_g4(_state->_v + h * head_dim, _state->_v + h * head_dim, NULL, head_dim, eps, false);
        }

        memcpy(_state->__key_cache[l] + (int64_t)pos * kv_dim, _state->_k, kv_dim * sizeof(float));
        memcpy(_state->__value_cache[l] + (int64_t)pos * kv_dim, _state->_v, kv_dim * sizeof(float));

        int32_t start_t = is_full ? 0 : fmax(0, pos - _config->sliding_window + 1);

#pragma omp parallel for
        for (int32_t h = 0; h < _config->n_heads; h++) {
            float *_q = _state->_q + h * head_dim;
            float *_att = _state->_att + h * _state->seq_n;
            int32_t kv_head = h / (_config->n_heads / kv_heads);

            for (int32_t t = 0; t < start_t; t++) {
                _att[t] = -1e9f;
            }

            float attn_scale = 1.0f;
            for (int32_t t = start_t; t <= pos; t++) {
                float *_k = _state->__key_cache[l] + (int64_t)t * kv_dim + (int64_t)kv_head * head_dim;
                float score = 0.0f;

#pragma omp simd reduction(+ : score)
                for (int32_t i = 0; i < head_dim; i++) {
                    score += _q[i] * _k[i];
                }
                _att[t] = score * attn_scale;
            }
            softmax(_att, pos + 1);

            float *_out = _state->_hb + h * head_dim;
            memset(_out, 0, head_dim * sizeof(float));
            for (int32_t t = start_t; t <= pos; t++) {
                float *_v = _state->__value_cache[l] + (int64_t)t * kv_dim + (int64_t)kv_head * head_dim;
                float a = _att[t];

#pragma omp simd
                for (int32_t i = 0; i < head_dim; i++) {
                    _out[i] += a * _v[i];
                }
            }
        }

        quantize_vec(&_state->xq, _state->_hb, layer_attn_out_dim);
        matmul_qq(_state->_xb, &_state->xq, &_weights->_o_proj[l]);
        rmsnorm_g4(_state->_xb, _state->_xb, _rms_post_a, dim, eps, true);

#pragma omp simd
        for (int32_t i = 0; i < dim; i++) {
            _x[i] += _state->_xb[i];
        }

        rmsnorm_g4(_state->_xb, _x, _rms_pre_f, dim, eps, true);

        quantize_vec(&_state->xq, _state->_xb, dim);
        matmul_qq(_state->_hb, &_state->xq, &_weights->_gate_proj[l]);
        matmul_qq(_state->_hb2, &_state->xq, &_weights->_up_proj[l]);

        int32_t ffn_dim = _weights->_gate_proj[l].rows;

#pragma omp parallel for
        for (int32_t i = 0; i < ffn_dim; i++) {
            _state->_hb[i] = gelu(_state->_hb[i]) * _state->_hb2[i];
        }

        quantize_vec(&_state->hq, _state->_hb, ffn_dim);
        matmul_qq(_state->_xb, &_state->hq, &_weights->_down_proj[l]);
        rmsnorm_g4(_state->_xb, _state->_xb, _rms_post_f, dim, eps, true);

#pragma omp simd
        for (int32_t i = 0; i < dim; i++) {
            _x[i] += _state->_xb[i];
        }

        if (_weights->_layer_scalars[l] != 1.0f) {
            float scale = _weights->_layer_scalars[l];
#pragma omp simd
            for (int32_t i = 0; i < dim; i++) {
                _x[i] *= scale;
            }
        }
    }

    rmsnorm_g4(_x, _x, (float *)_weights->rms_final_norm._data, dim, eps, true);

    matmul_qt(_state->_logits, _state->_x, &_weights->embed_tokens_weight);

    if (_config->final_logit_softcapping > 0.0f) {
        float cap = _config->final_logit_softcapping;
        float inv = 1.0f / cap;

#pragma omp parallel for
        for (int32_t i = 0; i < _config->vocab_size; i++) {
            _state->_logits[i] = tanhf(_state->_logits[i] * inv) * cap;
        }
    }

    return _state->_logits;
}

static float *forward_g4_wrap(void *_model, int32_t token, int32_t pos) {
    return forward_g4((G4 *)_model, token, pos);
}

static void free_g4_wrap(void *_model) {
    free_g4((G4 *)_model);
    free(_model);
}

model_iface *init_g4(const char *_model_path_s, int32_t seq_n, bool think_) {
    G4 *_model = a_calloc(1 * sizeof(G4));
    if (! load_quantized_g4(_model_path_s, _model)) {
        free_g4_wrap(_model);
        return NULL;
    }

    if (! alloc_state_g4(_model, seq_n)) {
        free_g4_wrap(_model);
        return NULL;
    }


    _model->tokenizer.bos_id = _model->config.bos_token_id;
    _model->tokenizer.eos_id = _model->config.eos_token_id;
    _model->tokenizer.im_end_id = 106;

    model_iface *_model_i = a_calloc(sizeof(model_iface));
    *_model_i = (model_iface) {
        ._model = _model,
        .forward = forward_g4_wrap,
        .free_model = free_g4_wrap,
        .seq_n = seq_n ? seq_n : _model->config.seq_n,
        .seq_n_model_max = _model->config.seq_n,
        ._chat_template = think_ ? &CHAT_TEMPLATE_THINK_G4 : &CHAT_TEMPLATE_G4,
        ._tokenizer = &(_model->tokenizer),
    };

    return _model_i;
}

