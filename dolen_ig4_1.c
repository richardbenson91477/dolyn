#include "dolen_ig4_1_common.h"


// The "\x3e" escaped ">" symbol serves to prevent LLMs from misinterpreting the text
static const chat_template CHAT_TEMPLATE_IG4_1 = {
    ._system_s = "<|start_of_role|\x3e" "system<|end_of_role|\x3e" "%s" "<|end_of_text|\x3e" "\n",
    ._main_s = "<|start_of_role|\x3e" "user<|end_of_role|\x3e" "%s" "<|end_of_text|\x3e" "\n"
            "<|start_of_role|\x3e" "assistant<|end_of_role|\x3e",
    ._end_turn_s = "<|end_of_text|\x3e" "\n",
};


int32_t load_quantized_ig4_1(const char *_file_path_s, IG4_1 *_model) {
    FILE *_file = fopen(_file_path_s, "rb");
    if (! _file) {
        log_msg(stderr, "ERROR: Failed to open %s for reading\n", _file_path_s);
        return -1;
    }

    memset(_model, 0, sizeof(IG4_1));

    uint64_t magic;
    uint32_t version;

    if ((fread(&magic, sizeof(uint64_t), 1, _file) != 1) ||
            (fread(&version, sizeof(uint32_t), 1, _file) != 1)) {
        log_msg(stderr, "ERROR: Failed to read header from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    if (magic != MAGIC_IG4_1) {
        log_msg(stderr, "ERROR: Invalid magic number in %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    if (version != 3) {
        log_msg(stderr, "ERROR: Unsupported version %d in %s\n", version, _file_path_s);
        fclose(_file);
        return -1;
    }

    if (fread(&(_model->config), sizeof(config_ig4_1), 1, _file) != 1) {
        log_msg(stderr, "ERROR: Failed to read config from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    if (! tokenizer_read_from_file(_file, _model->config.vocab_size, &(_model->tokenizer))) {
        log_msg(stderr, "ERROR: Failed to read tokenizer from %s\n", _file_path_s);
        fclose(_file);
        return -1;
    }

    config_ig4_1 *_config = &(_model->config);
    weights_ig4_1 *_weights = &(_model->weights);

    if (! (_config->rope_theta > 1.0f)) {
        log_msg(stderr, "ERROR: Invalid rope_theta %.9g in quantized model\n", _config->rope_theta);
        fclose(_file);
        return -1;
    }

    read_qt(_file, &(_weights->embed_tokens_weight));

    _weights->_rms_att_weight = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    if (! _weights->_rms_att_weight) {
        log_msg(stderr, "ERROR: Failed to allocate rms_att_weight\n");
        fclose(_file);
        return -1;
    }
    for (int32_t i = 0; i < _config->n_layer; i++) {
        read_qt(_file, &(_weights->_rms_att_weight[i]));
    }

    _weights->_wq = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    _weights->_wk = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    _weights->_wv = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    _weights->_wo = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));

    for (int32_t i = 0; i < _config->n_layer; i++) {
        read_qt(_file, &(_weights->_wq[i]));
        read_qt(_file, &(_weights->_wk[i]));
        read_qt(_file, &(_weights->_wv[i]));
        read_qt(_file, &(_weights->_wo[i]));
    }

    _weights->_rms_ffn_weight = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    if (! _weights->_rms_ffn_weight) {
        log_msg(stderr, "ERROR: Failed to allocate rms_ffn_weight\n");
        fclose(_file);
        return -1;
    }
    for (int32_t i = 0; i < _config->n_layer; i++) {
        read_qt(_file, &(_weights->_rms_ffn_weight[i]));
    }

    _weights->_w1 = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    _weights->_w2 = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    _weights->_w3 = (qtensor *)a_calloc((size_t)_config->n_layer * sizeof(qtensor));
    for (int32_t i = 0; i < _config->n_layer; i++) {
        read_qt(_file, &(_weights->_w1[i]));
        read_qt(_file, &(_weights->_w2[i]));
        read_qt(_file, &(_weights->_w3[i]));
    }

    read_qt(_file, &(_weights->rms_final_weight));

    if (! _config->tie_word_embeddings) {
        read_qt(_file, &(_weights->wcls));
    }
    else {
        _weights->wcls = _weights->embed_tokens_weight;
    }

    fclose(_file);
    log_msg(stdout, "INFO: Quantized model loaded from %s\n", _file_path_s);
    return 0;
}

void forward_ig4_1_attention_layer(IG4_1 *_model, int32_t l, int32_t pos) {
    config_ig4_1 *_config = &(_model->config);
    weights_ig4_1 *_weights = &(_model->weights);
    state_ig4_1 *_state = &(_model->state);
    float *_x = _state->_x;
    int32_t dim = _config->dim;
    int32_t head_size = _config->d_head > 0 ? _config->d_head : dim / _config->n_heads;
    int32_t kv_dim = _config->n_kv_heads * head_size;
    int32_t attn_out_dim = _config->n_heads * head_size;
    int32_t kv_mul = _config->n_heads / _config->n_kv_heads;
    int64_t loff = (int64_t)l * _state->seq_n * kv_dim;
    float eps = _config->rms_norm_eps;
    float *_key_cache_row = _state->_key_cache + loff + pos * kv_dim;
    float *_value_cache_row = _state->_value_cache + loff + pos * kv_dim;
    float *_rms_att_weight = (float *)_weights->_rms_att_weight[l]._data;

    rmsnorm(_state->_xb, _x, _rms_att_weight, dim, eps);

    quantize_vec(&(_state->xq), _state->_xb, dim);
    matmul_qq(_state->_q, &(_state->xq), &(_weights->_wq[l]));
    matmul_qq(_state->_k, &(_state->xq), &(_weights->_wk[l]));
    matmul_qq(_state->_v, &(_state->xq), &(_weights->_wv[l]));

    int32_t rotary_dim = head_size;

    if (_state->_cos_cache) {
        float *_cos_row = _state->_cos_cache + pos * rotary_dim;
        float *_sin_row = _state->_sin_cache + pos * rotary_dim;

#pragma omp parallel for
        for (int32_t h = 0; h < _config->n_heads; h++) {
            float *_q = _state->_q + h * head_size;
            for (int32_t i = 0; i < rotary_dim / 2; i++) {
                float c = _cos_row[i], sn = _sin_row[i];
                float q0 = _q[i], q1 = _q[i + rotary_dim / 2];
                _q[i] = q0 * c - q1 * sn;
                _q[i + rotary_dim / 2] = q0 * sn + q1 * c;
            }
        }

#pragma omp parallel for
        for (int32_t h = 0; h < _config->n_kv_heads; h++) {
            float *_k = _state->_k + h * head_size;
            for (int32_t i = 0; i < rotary_dim / 2; i++) {
                float c = _cos_row[i], sn = _sin_row[i];
                float k0 = _k[i], k1 = _k[i + rotary_dim / 2];
                _k[i] = k0 * c - k1 * sn;
                _k[i + rotary_dim / 2] = k0 * sn + k1 * c;
            }
        }
    }

    memcpy(_key_cache_row, _state->_k, kv_dim * sizeof(float));
    memcpy(_value_cache_row, _state->_v, kv_dim * sizeof(float));

    float attn_scale = _config->attention_multiplier;
    if (attn_scale == 0.0f) {
        attn_scale = 1.0f / sqrtf((float)head_size);
    }

#pragma omp parallel for
    for (int32_t h = 0; h < _config->n_heads; h++) {
        float *_q = _state->_q + h * head_size;
        float *_att = _state->_att + h * _state->seq_n;

        for (int32_t t = 0; t <= pos; t++) {
            float *_k = _state->_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float score = 0.0f;

#pragma omp simd reduction(+ : score)
            for (int32_t i = 0; i < head_size; i++) {
                score += _q[i] * _k[i];
            }
            _att[t] = score * attn_scale;
        }

        softmax(_att, pos + 1);

        float *_xb = _state->_xb + h * head_size;
        memset(_xb, 0, head_size * sizeof(float));

        for (int32_t t = 0; t <= pos; t++) {
            float *_v = _state->_value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float a = _att[t];

#pragma omp simd
            for (int32_t i = 0; i < head_size; i++) {
                _xb[i] += a * _v[i];
            }
        }
    }

    quantize_vec(&(_state->xq), _state->_xb, attn_out_dim);
    matmul_qq(_state->_xb2, &(_state->xq), &(_weights->_wo[l]));

    float res_mult = _config->residual_multiplier;
    for (int32_t i = 0; i < dim; i++) {
        _x[i] += _state->_xb2[i] * res_mult;
    }
}

void forward_ig4_1_mlp_layer(IG4_1 *_model, int32_t l) {
    config_ig4_1 *_config = &(_model->config);
    weights_ig4_1 *_weights = &(_model->weights);
    state_ig4_1 *_state = &(_model->state);
    float *_x = _state->_x;
    int32_t dim = _config->dim;
    int32_t hidden_dim = _config->n_mlp;
    float eps = _config->rms_norm_eps;
    float *_rms_ffn_weight = (float *)_weights->_rms_ffn_weight[l]._data;

    rmsnorm(_state->_xb, _x, _rms_ffn_weight, dim, eps);

    quantize_vec(&(_state->xq), _state->_xb, dim);
    matmul_qq(_state->_hb, &(_state->xq), &(_weights->_w1[l]));
    matmul_qq(_state->_hb2, &(_state->xq), &(_weights->_w3[l]));

#pragma omp parallel for
    for (int32_t i = 0; i < hidden_dim; i++) {
        float val = _state->_hb[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= _state->_hb2[i];
        _state->_hb[i] = val;
    }

    quantize_vec(&(_state->hq), _state->_hb, hidden_dim);
    matmul_qq(_state->_xb, &(_state->hq), &(_weights->_w2[l]));

    float res_mult = _config->residual_multiplier;
    for (int32_t i = 0; i < dim; i++) {
        _x[i] += _state->_xb[i] * res_mult;
    }
}

float *forward_ig4_1(IG4_1 *_model, int32_t token, int32_t pos) {
    config_ig4_1 *_config = &(_model->config);
    weights_ig4_1 *_weights = &(_model->weights);
    state_ig4_1 *_state = &(_model->state);
    float *_x = _state->_x;
    int32_t dim = _config->dim;

    dequantize_row(_x, &(_weights->embed_tokens_weight), token);

    float emb_mult = _config->embedding_multiplier;
    if (emb_mult != 1.0f) {
#pragma omp simd
        for (int32_t i = 0; i < dim; i++) {
            _x[i] *= emb_mult;
        }
    }

    for (int32_t l = 0; l < _config->n_layer; l++) {
        forward_ig4_1_attention_layer(_model, l, pos);
        forward_ig4_1_mlp_layer(_model, l);
    }

    rmsnorm(_x, _x, (float *)_weights->rms_final_weight._data, dim, _config->rms_norm_eps);

    if (_config->tie_word_embeddings) {
        matmul_qt(_state->_logits, _x, &(_weights->embed_tokens_weight));
    }
    else {
        matmul_qt(_state->_logits, _x, &(_weights->wcls));
    }

    float logit_scale = _config->logits_scaling;
    if ((logit_scale != 0.0f) && (logit_scale != 1.0f)) {
#pragma omp simd
        for (int32_t i = 0; i < _config->vocab_size; i++) {
            _state->_logits[i] /= logit_scale;
        }
    }

    return _state->_logits;
}

static float *forward_ig4_1_wrap(void *_model, int32_t token, int32_t pos) {
    return forward_ig4_1((IG4_1 *)_model, token, pos);
}

static void free_ig4_1_wrap(void *_model) {
    free_ig4_1((IG4_1 *)_model);
    free(_model);
}

model_iface *init_ig4_1(const char *_model_path_s, int32_t seq_n, bool think_) {
    IG4_1 *_model = a_calloc(1 * sizeof(IG4_1));

    if (think_) {
        log_msg(stderr, "WARNING: Think mode requested but not supported.\n");
    }

    if (load_quantized_ig4_1(_model_path_s, _model)) {
        return NULL;
    }

    if (! alloc_state_ig4_1(_model, seq_n)) {
        return NULL;
    }


    _model->tokenizer.bos_id = _model->config.bos_token_id;
    _model->tokenizer.eos_id = _model->config.eos_token_id;
    _model->tokenizer.im_end_id = _model->config.eos_token_id;

    model_iface *_model_i = a_calloc(sizeof(model_iface));
    *_model_i = (model_iface){
        ._model = _model,
        .forward = forward_ig4_1_wrap,
        .free_model = free_ig4_1_wrap,
        .seq_n = seq_n ? seq_n : _model->config.seq_n,
        .seq_n_model_max = _model->config.seq_n,
        ._chat_template = &CHAT_TEMPLATE_IG4_1,
        ._tokenizer = &(_model->tokenizer),
    };

    return _model_i;
}

