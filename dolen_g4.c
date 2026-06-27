#include "dolen_g4_common.h"


static const chat_template CHAT_TEMPLATE_G4 = {
    ._system_s = "<|turn\x3e" "system\n%s<turn|\x3e" "\n",
    ._main_s = "<|turn\x3e" "user\n%s<turn|\x3e" "\n"
            "<|turn\x3e" "model\n"
            "<|channel\x3e" "thought\n<channel|\x3e",
    ._end_turn_s = "<turn|\x3e" "\n",
};

static const chat_template CHAT_TEMPLATE_THINK_G4 = {
    ._system_s = "<|turn\x3e" "system\n<|think|\x3e" "%s<turn|\x3e" "\n",
    ._main_s = "<|turn\x3e" "user\n%s<turn|\x3e" "\n"
            "<|turn\x3e" "model\n",
    ._end_turn_s = "<turn|\x3e" "\n",
};


int load_quantized_g4(const char *filepath, G4 *_model, int seq_n_max) {
    FILE *f = fopen(filepath, "rb");
    if (! f) {
        log_msg(stderr, "ERROR: Failed to open %s\n", filepath);
        return -1;
    }

    memset(_model, 0, sizeof(G4));

    uint64_t magic;
    uint32_t version;

    if (fread(&magic, sizeof(magic), 1, f) != 1 ||
            fread(&version, sizeof(version), 1, f) != 1) {
        log_msg(stderr, "ERROR: Failed to read header\n");
        fclose(f);
        return -1;
    }

    if (magic != MAGIC_G4) {
        log_msg(stderr, "ERROR: Invalid magic number\n");
        fclose(f);
        return -1;
    }

    if (version != 5) {
        log_msg(stderr, "ERROR: Unsupported version %u (expected 5). RE-RUN QUANTIZER.\n", version);
        fclose(f);
        return -1;
    }

    config_g4 *p = &_model->config;

    if (fread(p, sizeof(config_g4), 1, f) != 1) {
        log_msg(stderr, "ERROR: Failed to read config\n");
        fclose(f);
        return -1;
    }

    if (tokenizer_read_from_file(f, p->vocab_size, &_model->tokenizer)) {
        log_msg(stderr, "ERROR: Failed to read tokenizer from %s\n", filepath);
        fclose(f);
        return -1;
    }

    if (seq_n_max) {
        p->seq_len = seq_n_max;
    }

    _model->_layer_types = (int *)a_calloc((size_t)p->n_layers * sizeof(int));
    if (fread(_model->_layer_types, sizeof(int), (size_t)p->n_layers, f) != (size_t)p->n_layers) {
        log_msg(stderr, "ERROR: Failed to read layer_types\n");
        fclose(f);
        return -1;
    }

    weights_g4 *w = &_model->weights;

    w->_rms_input_layernorm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_rms_post_attn_layernorm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_rms_pre_ffn_layernorm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_rms_post_ffn_layernorm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_rms_q_norm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_rms_k_norm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));

    if (! w->_rms_input_layernorm ||
            (! w->_rms_post_attn_layernorm) ||
            (! w->_rms_pre_ffn_layernorm) ||
            (! w->_rms_post_ffn_layernorm) ||
            (! w->_rms_q_norm) ||
            (! w->_rms_k_norm)) {
        log_msg(stderr, "ERROR: Alloc failed\n");
        fclose(f);
        return -1;
    }

    read_qt(f, &w->embed_tokens_weight);

    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->_rms_input_layernorm[i]);
    }
    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->_rms_post_attn_layernorm[i]);
    }
    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->_rms_pre_ffn_layernorm[i]);
    }
    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->_rms_post_ffn_layernorm[i]);
    }
    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->_rms_q_norm[i]);
    }
    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->_rms_k_norm[i]);
    }

    read_qt(f, &w->rms_final_norm);

    w->_q_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_k_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_v_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_o_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_gate_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_up_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_down_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->_layer_scalars = (float *)a_calloc((size_t)p->n_layers * sizeof(float));

    for (int i = 0; i < p->n_layers; i++) {
        w->_layer_scalars[i] = 1.0f;
    }

    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->_q_proj[i]);
        read_qt(f, &w->_k_proj[i]);
        read_qt(f, &w->_v_proj[i]);
        read_qt(f, &w->_o_proj[i]);
        read_qt(f, &w->_gate_proj[i]);
        read_qt(f, &w->_up_proj[i]);
        read_qt(f, &w->_down_proj[i]);
    }

    fread(w->_layer_scalars, sizeof(float), (size_t)p->n_layers, f);

    if (p->use_rope_freqs) {
        read_qt(f, &w->rope_freqs_full);
    }

    fclose(f);
    log_msg(stdout, "INFO: Quantized G4 loaded from %s\n", filepath);

    alloc_state_g4(&_model->state, p, w, _model->_layer_types);

    return 0;
}

static void apply_rope(float *vec, float *cos, float *sin, int rotary_dim, int vec_dim, int pos) {
    if (rotary_dim <= 0) {
        return;
    }

    int half_rot = rotary_dim / 2;
    int cache_stride = vec_dim / 2;
    float *cos_row = cos + pos * cache_stride;
    float *sin_row = sin + pos * cache_stride;

    for (int i = 0; i < half_rot; i++) {
        float c = cos_row[i], sn = sin_row[i];
        float v0 = vec[i], v1 = vec[i + cache_stride];
        vec[i] = v0 * c - v1 * sn;
        vec[i + cache_stride] = v0 * sn + v1 * c;
    }
}

float *forward_g4(G4 *_model, int token, int pos) {
    config_g4 *p = &_model->config;
    weights_g4 *w = &_model->weights;
    state_g4 *s = &_model->state;
    float *_x = s->_x;
    int dim = p->dim;
    float eps = p->rms_norm_eps;
    float embed_scale = sqrtf((float)dim);

    if (token < 0 ||
            token >= p->vocab_size) {
        log_msg(stderr, "ERROR: token %d is outside vocabulary [0, %d)\n", token, p->vocab_size);
        exit(EXIT_FAILURE);
    }

    if (pos < 0 ||
            pos >= p->seq_len) {
        log_msg(stderr, "ERROR: position %d is outside KV cache [0, %d)\n", pos, p->seq_len);
        exit(EXIT_FAILURE);
    }

    dequantize_row(_x, &w->embed_tokens_weight, token);

#pragma omp simd
    for (int i = 0; i < dim; i++) {
        _x[i] *= embed_scale;
    }

    for (int l = 0; l < p->n_layers; l++) {
        int is_full = _model->_layer_types[l];
        int use_alternative_attention = is_full && p->attention_k_eq_v;
        int head_dim = is_full ? p->global_head_dim : p->head_dim;
        int kv_heads = use_alternative_attention ? p->n_global_kv_heads : p->n_kv_heads;
        int kv_dim = kv_heads * head_dim;
        int rotary_dim = is_full ? (int)(p->rope_partial_factor * p->global_head_dim) : p->head_dim;
        float *_cos_cache = is_full ? s->_cos_cache_full : s->_cos_cache_sliding;
        float *_sin_cache = is_full ? s->_sin_cache_full : s->_sin_cache_sliding;

        int layer_attn_out_dim = p->n_heads * head_dim;

        float *_rms_in = (float *)w->_rms_input_layernorm[l]._data;
        float *_rms_post_a = (float *)w->_rms_post_attn_layernorm[l]._data;
        float *_rms_pre_f = (float *)w->_rms_pre_ffn_layernorm[l]._data;
        float *_rms_post_f = (float *)w->_rms_post_ffn_layernorm[l]._data;
        float *_rms_q = (float *)w->_rms_q_norm[l]._data;
        float *_rms_k = (float *)w->_rms_k_norm[l]._data;

        rmsnorm_g4(s->_xb, _x, _rms_in, dim, eps, 1);

        quantize_vec(&s->xq, s->_xb, dim);
        matmul_qq(s->_q, &s->xq, &w->_q_proj[l]);
        matmul_qq(s->_k_raw, &s->xq, &w->_k_proj[l]);

        if (use_alternative_attention) {
            memcpy(s->_v, s->_k_raw, kv_dim * sizeof(float));
        }
        else {
            matmul_qq(s->_v, &s->xq, &w->_v_proj[l]);
        }

#pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float *qh = s->_q + h * head_dim;
            rmsnorm_g4(qh, qh, _rms_q, head_dim, eps, 1);
            if ((rotary_dim > 0) &&
                    _cos_cache) {
                apply_rope(qh, _cos_cache, _sin_cache, rotary_dim, head_dim, pos);
            }
        }

#pragma omp parallel for
        for (int h = 0; h < kv_heads; h++) {
            float *kh = s->_k_raw + h * head_dim;
            rmsnorm_g4(kh, kh, _rms_k, head_dim, eps, 1);
            if ((rotary_dim > 0) &&
                    _cos_cache) {
                apply_rope(kh, _cos_cache, _sin_cache, rotary_dim, head_dim, pos);
            }
        }

        memcpy(s->_k, s->_k_raw, kv_dim * sizeof(float));

#pragma omp parallel for
        for (int h = 0; h < kv_heads; h++) {
            rmsnorm_g4(s->_v + h * head_dim, s->_v + h * head_dim, NULL, head_dim, eps, 0);
        }

        memcpy(s->__key_cache[l] + (long long)pos * kv_dim, s->_k, kv_dim * sizeof(float));
        memcpy(s->__value_cache[l] + (long long)pos * kv_dim, s->_v, kv_dim * sizeof(float));

        int start_t = is_full ? 0 : fmax(0, pos - p->sliding_window + 1);

#pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float *_q = s->_q + h * head_dim;
            float *_att = s->_att + h * p->seq_len;
            int kv_head = h / (p->n_heads / kv_heads);

            for (int t = 0; t < start_t; t++) {
                _att[t] = -1e9f;
            }

            float attn_scale = 1.0f;
            for (int t = start_t; t <= pos; t++) {
                float *_k = s->__key_cache[l] + (long long)t * kv_dim + (long long)kv_head * head_dim;
                float score = 0.0f;

#pragma omp simd reduction(+ : score)
                for (int i = 0; i < head_dim; i++) {
                    score += _q[i] * _k[i];
                }
                _att[t] = score * attn_scale;
            }
            softmax(_att, pos + 1);

            float *out = s->_hb + h * head_dim;
            memset(out, 0, head_dim * sizeof(float));
            for (int t = start_t; t <= pos; t++) {
                float *v = s->__value_cache[l] + (long long)t * kv_dim + (long long)kv_head * head_dim;
                float a = _att[t];

#pragma omp simd
                for (int i = 0; i < head_dim; i++) {
                    out[i] += a * v[i];
                }
            }
        }

        quantize_vec(&s->xq, s->_hb, layer_attn_out_dim);
        matmul_qq(s->_xb, &s->xq, &w->_o_proj[l]);
        rmsnorm_g4(s->_xb, s->_xb, _rms_post_a, dim, eps, 1);

#pragma omp simd
        for (int i = 0; i < dim; i++) {
            _x[i] += s->_xb[i];
        }

        rmsnorm_g4(s->_xb, _x, _rms_pre_f, dim, eps, 1);

        quantize_vec(&s->xq, s->_xb, dim);
        matmul_qq(s->_hb, &s->xq, &w->_gate_proj[l]);
        matmul_qq(s->_hb2, &s->xq, &w->_up_proj[l]);

        int ffn_dim = w->_gate_proj[l].rows;

#pragma omp parallel for
        for (int i = 0; i < ffn_dim; i++) {
            s->_hb[i] = gelu(s->_hb[i]) * s->_hb2[i];
        }

        quantize_vec(&s->hq, s->_hb, ffn_dim);
        matmul_qq(s->_xb, &s->hq, &w->_down_proj[l]);
        rmsnorm_g4(s->_xb, s->_xb, _rms_post_f, dim, eps, 1);

#pragma omp simd
        for (int i = 0; i < dim; i++) {
            _x[i] += s->_xb[i];
        }

        if (w->_layer_scalars[l] != 1.0f) {
            float scale = w->_layer_scalars[l];
#pragma omp simd
            for (int i = 0; i < dim; i++) {
                _x[i] *= scale;
            }
        }
    }

    rmsnorm_g4(_x, _x, (float *)w->rms_final_norm._data, dim, eps, 1);

    matmul_qt(s->_logits, s->_x, &w->embed_tokens_weight);


    if (p->final_logit_softcapping > 0.0f) {
        float cap = p->final_logit_softcapping;
        float inv = 1.0f / cap;

#pragma omp parallel for
        for (int i = 0; i < p->vocab_size; i++) {
            s->_logits[i] = tanhf(s->_logits[i] * inv) * cap;
        }
    }
    return s->_logits;
}

static float *forward_g4_wrap(void *_model, int token, int pos) {
    return forward_g4((G4 *)_model, token, pos);
}

static void free_g4_wrap(void *_model) {
    free_g4((G4 *)_model);
    free(_model);
}

model_iface *init_g4(const char *_model_path_s, int seq_n_max, bool think_) {
    G4 *_model = a_calloc(1 * sizeof(G4));
    if (load_quantized_g4(_model_path_s, _model, seq_n_max)) {
        free_g4(_model);
        free(_model);
        return NULL;
    }

    _model->tokenizer._tokens_special = NULL;
    _model->tokenizer.bos_id = 2;
    _model->tokenizer.eos_id = 1;
    _model->tokenizer.im_end_id = 106;

    model_iface *_model_i = a_calloc(sizeof(model_iface));
    *_model_i = (model_iface) {
        ._model = _model,
        .forward = forward_g4_wrap,
        .free_model = free_g4_wrap,
        .seq_n_max = seq_n_max ? seq_n_max : _model->config.seq_len,
        ._chat_template = think_ ? &CHAT_TEMPLATE_THINK_G4 : &CHAT_TEMPLATE_G4,
        ._tokenizer = &_model->tokenizer
    };

    return _model_i;
}

