#include "dolen_quantize_common.h"
#include "dolen_q3_5_common.h"


static int32_t get_layer_type(int32_t layer_idx, const JsonValue *_layer_types) {
    if ((! _layer_types) ||
            (_layer_types->type != JSON_ARRAY)) {
        return 0;
    }

    if (layer_idx >= (int32_t)_layer_types->data.array.count) {
        return 0;
    }

    JsonValue *_js_lt = json_array_get(_layer_types, layer_idx);
    if ((! _js_lt) ||
            (_js_lt->type != JSON_STRING)) {
        return 0;
    }

    const char *_type_s = _js_lt->data.string;
    if (! strcmp(_type_s, "linear_attention")) {
        return 1;
    }

    return 0;
}

int32_t load_config_q3_5(Q3_5 *_model, const char *_model_dir_s) {
    config_q3_5 *_config = &_model->config;

    char _config_path_s[PATH_MAX];
    snprintf(_config_path_s, sizeof(_config_path_s), "%s/config.json", _model_dir_s);

    FILE *_file = fopen(_config_path_s, "rb");
    if (! _file) {
        log_msg(stderr, "ERROR: Could not open config.json at %s\n", _config_path_s);
        return -1;
    }
    fseek(_file, 0, SEEK_END);
    int64_t size = ftell(_file);
    fseek(_file, 0, SEEK_SET);
    char *_json_str = (char *)a_calloc(size + 1);
    if ((! _json_str) ||
            (fread(_json_str, 1, size, _file) != (size_t)size)) {
        free(_json_str);
        fclose(_file);
        return -1;
    }
    _json_str[size] = '\0';

    fclose(_file);

    char _error_s[256] = {0};

    JsonValue *_js_root = json_parse(_json_str, size, _error_s, sizeof(_error_s));
    free(_json_str);
    if (! _js_root) {
        log_msg(stderr, "ERROR: Failed to parse config.json: %s\n", _error_s);
        return -1;
    }

    JsonValue *_js_cfg = json_object_get(_js_root, "text_config");
    if (! _js_cfg) {
        _js_cfg = _js_root;
    }
    
    memset(_config, 0, sizeof(config_q3_5));
    _config->dim = json_get_int(json_object_get(_js_cfg, "hidden_size"), 896);
    _config->n_heads = json_get_int(json_object_get(_js_cfg, "num_attention_heads"), 14);
    _config->n_kv_heads = json_get_int(json_object_get(_js_cfg, "num_key_value_heads"), _config->n_heads);
    _config->n_layer = json_get_int(json_object_get(_js_cfg, "num_hidden_layers"), 24);
    _config->n_mlp = json_get_int(json_object_get(_js_cfg, "intermediate_size"), 4864);
    if (! _config->n_mlp) {
        _config->n_mlp = json_get_int(json_object_get(_js_cfg, "shared_expert_intermediate_size"), 4864);
    }
    _config->vocab_size = json_get_int(json_object_get(_js_cfg, "vocab_size"), 151936);
    _config->seq_n = json_get_int(json_object_get(_js_cfg, "max_position_embeddings"), 262144);
    JsonValue *_js_rope_params = json_object_get(_js_cfg, "rope_parameters");
    _config->rope_theta = json_get_double(json_object_get(_js_rope_params, "rope_theta"), 10000.0);
    _config->rope_partial_rotary_factor = json_get_double(json_object_get(_js_rope_params, "partial_rotary_factor"), 1.0);
    _config->rms_norm_eps = json_get_double(json_object_get(_js_cfg, "rms_norm_eps"), 1e-6);
    _config->tie_word_embeddings = json_get_bool(json_object_get(_js_root, "tie_word_embeddings"), 0);
    _config->d_head = json_get_int(json_object_get(_js_cfg, "head_dim"), _config->dim / _config->n_heads);
    _config->n_linear_k_heads = json_get_int(json_object_get(_js_cfg, "linear_num_key_heads"), 0);
    _config->n_linear_v_heads = json_get_int(json_object_get(_js_cfg, "linear_num_value_heads"), 0);
    _config->d_linear_k = json_get_int(json_object_get(_js_cfg, "linear_key_head_dim"), 0);
    _config->d_linear_v = json_get_int(json_object_get(_js_cfg, "linear_value_head_dim"), 0);
    _config->linear_conv_kernel = json_get_int(json_object_get(_js_cfg, "linear_conv_kernel_dim"), 4);
    _config->bos_token_id = json_get_int(json_object_get(_js_cfg, "bos_token_id"), 0);
    _config->eos_token_id = json_get_int(json_object_get(_js_cfg, "eos_token_id"), 248044);
    _model->_layer_types = (int32_t *)a_calloc((size_t)_config->n_layer * sizeof(int32_t));
    _model->_attn_layer_indices = (int32_t *)a_calloc((size_t)_config->n_layer * sizeof(int32_t));
    _model->_deltanet_layer_indices = (int32_t *)a_calloc((size_t)_config->n_layer * sizeof(int32_t));
    if ((! _model->_layer_types) ||
            (! _model->_attn_layer_indices) ||
            (! _model->_deltanet_layer_indices)) {
        log_msg(stderr, "ERROR: Failed to allocate layer type arrays\n");
        free(_model->_layer_types);
        free(_model->_attn_layer_indices);
        free(_model->_deltanet_layer_indices);
        json_free(_js_root);
        return -1;
    }

    JsonValue *_js_layer_types = json_object_get(_js_cfg, "layer_types");
    int32_t la = 0, ld = 0;
    _config->n_full_attn_layers = 0;
    _config->n_linear_attn_layers = 0;
    for (int32_t i = 0; i < _config->n_layer; i++) {
        int32_t is_linear = get_layer_type(i, _js_layer_types);
        _model->_layer_types[i] = is_linear;
        if (is_linear == 1) {
            _model->_deltanet_layer_indices[i] = ld++;
            _config->n_linear_attn_layers++;
        } else {
            _model->_attn_layer_indices[i] = la++;
            _config->n_full_attn_layers++;
        }
    }

    json_free(_js_root);

    log_msg(stdout, "INFO: Model config loaded\n");
    return 0;
}

static int32_t write_layer_tensor(quantize_ctx *_qt_ctx, FILE *_file, int32_t layer, const char *_suffix_s,
        int32_t rows, int32_t cols, q_type_t type) {
    char _name_s[256];
    snprintf(_name_s, sizeof(_name_s), "model.language_model.layers.%d.%s", layer, _suffix_s);

    if (quantize_write_tensor_or_empty(_qt_ctx, _file, _name_s, rows, cols, type)) {
        log_msg(stderr, "ERROR: Failed quantizing %s\n", _name_s);
        return -1;
    }

    return 0;
}

int32_t quantize_q3_5_to_file(const char *_model_dir_s, const char *_file_path_s,
        const quant_preset_t *_preset, const char *_tokenizer_path_s) {
    Q3_5 model;
    memset(&model, 0, sizeof(model));
    if (load_config_q3_5(&model, _model_dir_s)) {
        return -1;
    }

    quantize_ctx qt_ctx;
    if (quantize_ctx_open(&qt_ctx, _model_dir_s)) {
        log_msg(stderr, "ERROR: Could not load safetensors metadata from %s\n", _model_dir_s);
        free(model._layer_types);
        free(model._attn_layer_indices);
        free(model._deltanet_layer_indices);
        return -1;
    }

    FILE *_file = fopen(_file_path_s, "wb");
    if (! _file) {
        log_msg(stderr, "ERROR: Failed to open %s for writing\n", _file_path_s);
        quantize_ctx_close(&qt_ctx);
        free(model._layer_types);
        free(model._attn_layer_indices);
        free(model._deltanet_layer_indices);
        return -1;
    }

    config_q3_5 *_config = &model.config;
    int32_t head_size = _config->d_head > 0 ? _config->d_head : (_config->dim / _config->n_heads);
    int32_t kv_dim = _config->n_kv_heads * head_size;
    int32_t key_dim = _config->n_linear_k_heads * _config->d_linear_k;
    int32_t value_dim = _config->n_linear_v_heads * _config->d_linear_v;
    int32_t conv_dim = key_dim * 2 + value_dim;
    int32_t q_dim = _config->n_heads * head_size * 2;
    int32_t attn_out_dim = _config->n_heads * head_size;
    uint64_t magic = MAGIC_Q3_5;
    uint32_t version = 3;
    int32_t failed = 0;
    if (quantize_write_bytes(_file, &magic, sizeof(magic), 1) ||
            quantize_write_bytes(_file, &version, sizeof(version), 1) ||
            quantize_write_bytes(_file, _config, sizeof(config_q3_5), 1)) {
        failed = 1;
        goto cleanup;
    }

    tokenizer tokenizer1;
    memset(&tokenizer1, 0, sizeof(tokenizer1));

    build_tokenizer(&tokenizer1, _tokenizer_path_s, _config->vocab_size);

    if (tokenizer_write_to_file(_file, &tokenizer1)) {
        log_msg(stderr, "ERROR: Failed to write tokenizer\n");
        failed = 1;
        goto cleanup;
    }

    if (quantize_write_bytes(_file, model._layer_types, sizeof(int32_t), _config->n_layer)) {
        failed = 1;
        goto cleanup;
    }

    if (quantize_write_tensor(&qt_ctx, _file, "model.language_model.embed_tokens.weight",
        _config->vocab_size, _config->dim, _preset->embed)) {
        failed = 1;
        goto cleanup;
    }

    for (int32_t l = 0; l < _config->n_layer; l++) {
        if (write_layer_tensor(&qt_ctx, _file, l, "input_layernorm.weight",
            1, _config->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int32_t l = 0; l < _config->n_layer; l++) {
        if (model._layer_types[l]) {
            continue;
        }
        if (write_layer_tensor(&qt_ctx, _file, l, "self_attn.q_proj.weight",
                    q_dim, _config->dim, _preset->attn) ||
                write_layer_tensor(&qt_ctx, _file, l, "self_attn.k_proj.weight",
                    kv_dim, _config->dim, _preset->attn) ||
                write_layer_tensor(&qt_ctx, _file, l, "self_attn.v_proj.weight",
                    kv_dim, _config->dim, _preset->attn) ||
                write_layer_tensor(&qt_ctx, _file, l, "self_attn.o_proj.weight",
                    _config->dim, attn_out_dim, _preset->attn)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int32_t l = 0; l < _config->n_layer; l++) {
        if (! model._layer_types[l] &&
            write_layer_tensor(&qt_ctx, _file, l, "self_attn.q_norm.weight",
            1, head_size, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int32_t l = 0; l < _config->n_layer; l++) {
        if (! model._layer_types[l] &&
            write_layer_tensor(&qt_ctx, _file, l, "self_attn.k_norm.weight",
            1, head_size, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (_config->n_linear_attn_layers > 0) {
        for (int32_t l = 0; l < _config->n_layer; l++) {
            if (model._layer_types[l] != 1) {
                continue;
            }
            if (write_layer_tensor(&qt_ctx, _file, l, "linear_attn.in_proj_qkv.weight",
                        conv_dim, _config->dim, _preset->attn) ||
                    write_layer_tensor(&qt_ctx, _file, l, "linear_attn.in_proj_z.weight",
                        value_dim, _config->dim, _preset->attn)) {
                failed = 1;
                goto cleanup;
            }
        }

        for (int32_t l = 0; l < _config->n_layer; l++) {
            if ((model._layer_types[l] == 1) &&
                    write_layer_tensor(&qt_ctx, _file, l, "linear_attn.in_proj_b.weight",
                        1, _config->n_linear_v_heads * _config->dim, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int32_t l = 0; l < _config->n_layer; l++) {
            if (model._layer_types[l] == 1 &&
                    write_layer_tensor(&qt_ctx, _file, l, "linear_attn.in_proj_a.weight",
                        1, _config->n_linear_v_heads * _config->dim, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int32_t l = 0; l < _config->n_layer; l++) {
            if (model._layer_types[l] == 1 &&
                    write_layer_tensor(&qt_ctx, _file, l, "linear_attn.conv1d.weight",
                        1, conv_dim * _config->linear_conv_kernel, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int32_t l = 0; l < _config->n_layer; l++) {
            if (model._layer_types[l] == 1 &&
                    write_layer_tensor(&qt_ctx, _file, l, "linear_attn.dt_bias",
                        1, _config->n_linear_v_heads, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int32_t l = 0; l < _config->n_layer; l++) {
            if (model._layer_types[l] == 1 &&
                    write_layer_tensor(&qt_ctx, _file, l, "linear_attn.A_log",
                        1, _config->n_linear_v_heads, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int32_t l = 0; l < _config->n_layer; l++) {
            if (model._layer_types[l] == 1 &&
                    write_layer_tensor(&qt_ctx, _file, l, "linear_attn.norm.weight",
                        1, _config->d_linear_v, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int32_t l = 0; l < _config->n_layer; l++) {
            if (model._layer_types[l] == 1 &&
                    write_layer_tensor(&qt_ctx, _file, l, "linear_attn.out_proj.weight",
                        _config->dim, value_dim, _preset->attn)) {
                failed = 1;
                goto cleanup;
            }
        }
    }
    for (int32_t l = 0; l < _config->n_layer; l++) {
        if (write_layer_tensor(&qt_ctx, _file, l, "post_attention_layernorm.weight",
                1, _config->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }
    for (int32_t l = 0; l < _config->n_layer; l++) {
        if (write_layer_tensor(&qt_ctx, _file, l, "mlp.gate_proj.weight",
                    _config->n_mlp, _config->dim, _preset->mlp) ||
                write_layer_tensor(&qt_ctx, _file, l, "mlp.down_proj.weight",
                    _config->dim, _config->n_mlp, _preset->mlp) ||
                write_layer_tensor(&qt_ctx, _file, l, "mlp.up_proj.weight",
                    _config->n_mlp, _config->dim, _preset->mlp)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (quantize_write_tensor_or_empty(&qt_ctx, _file, "model.language_model.norm.weight",
            1, _config->dim, Q_TYPE_F32)) {
        failed = 1;
        goto cleanup;
    }

    if ((! _config->tie_word_embeddings) &&
            quantize_write_tensor(&qt_ctx, _file, "lm_head.weight",
                _config->vocab_size, _config->dim, _preset->embed)) {
        failed = 1;
        goto cleanup;
    }

cleanup:
    free_tokenizer(&tokenizer1);

    if (fclose(_file)) {
        failed = 1;
    }

    quantize_ctx_close(&qt_ctx);

    free(model._layer_types);
    free(model._attn_layer_indices);
    free(model._deltanet_layer_indices);

    if (failed) {
        remove(_file_path_s);
        return -1;
    }

    log_msg(stdout, "INFO: Quantized model saved to %s\n", _file_path_s);
    return 0;
}

