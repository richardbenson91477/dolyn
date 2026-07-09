#include "dolen_quantize_common.h"
#include "dolen_g4_common.h"


int32_t load_config_g4(G4 *_model, const char *_model_dir_s) {
    config_g4 *_config = &_model->config;

    char config_path[PATH_MAX];
    snprintf(config_path, sizeof(config_path), "%s/config.json", _model_dir_s);
    FILE *_file = fopen(config_path, "rb");
    if (! _file) {
        log_msg(stderr, "ERROR: Could not open config.json at %s\n", config_path);
        return -1;
    }
    fseek(_file, 0, SEEK_END);
    int64_t size = ftell(_file);
    fseek(_file, 0, SEEK_SET);

    char *_json_s = (char *)a_calloc(size + 1);
    if ((! _json_s) ||
            (fread(_json_s, 1, size, _file) != (size_t)size)) {
        free(_json_s);
        fclose(_file);
        return -1;
    }
    _json_s[size] = '\0';

    fclose(_file);

    char _error_s[256] = {0};

    JsonValue *_js_root = json_parse(_json_s, size, _error_s, sizeof(_error_s));
    free(_json_s);
    if (! _js_root) {
        log_msg(stderr, "ERROR: Failed to parse config.json: %s\n", _error_s);
        return -1;
    }

    JsonValue *_js_cfg = json_object_get(_js_root, "text_config");
    if (! _js_cfg) {
        _js_cfg = _js_root;
    }

    memset(_config, 0, sizeof(config_g4));
    _config->dim = json_get_int(json_object_get(_js_cfg, "hidden_size"), 0);
    _config->hidden_dim = json_get_int(json_object_get(_js_cfg, "intermediate_size"), 0);
    _config->n_layers = json_get_int(json_object_get(_js_cfg, "num_hidden_layers"), 0);
    _config->n_heads = json_get_int(json_object_get(_js_cfg, "num_attention_heads"), 0);
    _config->n_kv_heads = json_get_int(json_object_get(_js_cfg, "num_key_value_heads"), 0);
    _config->n_global_kv_heads = json_get_int(json_object_get(_js_cfg, "num_global_key_value_heads"), _config->n_kv_heads);
    _config->vocab_size = json_get_int(json_object_get(_js_cfg, "vocab_size"), 0);
    _config->seq_n = json_get_int(json_object_get(_js_cfg, "max_position_embeddings"), 262144);
    _config->head_dim = json_get_int(json_object_get(_js_cfg, "head_dim"), 0);
    _config->global_head_dim = json_get_int(json_object_get(_js_cfg, "global_head_dim"), _config->head_dim);
    _config->sliding_window = json_get_int(json_object_get(_js_cfg, "sliding_window"), 1024);
    _config->tie_word_embeddings = json_get_bool(json_object_get(_js_cfg, "tie_word_embeddings"), 0);
    _config->rms_norm_eps = json_get_double(json_object_get(_js_cfg, "rms_norm_eps"), 1e-6);
    _config->final_logit_softcapping = json_get_double(json_object_get(_js_cfg, "final_logit_softcapping"), 30.0);
    _config->attention_k_eq_v = json_get_bool(json_object_get(_js_cfg, "attention_k_eq_v"), 0);
    _config->original_max_pos_embeds = json_get_int(json_object_get(_js_cfg, "original_max_position_embeddings"), 8192);
    _config->bos_token_id = json_get_int(json_object_get(_js_cfg, "bos_token_id"), 2);
    _config->eos_token_id = json_get_int(json_object_get(_js_cfg, "eos_token_id"), 1);
    JsonValue *_js_rope_params = json_object_get(_js_cfg, "rope_parameters");
    JsonValue *_js_full_rope = json_object_get(_js_rope_params, "full_attention");
    JsonValue *_js_slide_rope = json_object_get(_js_rope_params, "sliding_attention");
    _config->rope_theta_full = json_get_double(json_object_get(_js_full_rope, "rope_theta"), 1000000.0);
    _config->rope_partial_factor = json_get_double(json_object_get(_js_full_rope, "partial_rotary_factor"), 0.25);
    _config->rope_theta_sliding = json_get_double(json_object_get(_js_slide_rope, "rope_theta"), 10000.0);
    const char *_rope_type_s = json_get_string(json_object_get(_js_full_rope, "rope_type"), "proportional");
    _config->use_rope_freqs = 0;
    if (_rope_type_s && (! strcmp(_rope_type_s, "proportional"))) {
        log_msg(stdout, "INFO: Full attention uses config-derived proportional RoPE\n");
    }
    JsonValue *_js_layer_types = json_object_get(_js_cfg, "layer_types");
    _model->_layer_types = (int32_t *)a_calloc((size_t)_config->n_layers * sizeof(int32_t));
    if (! _model->_layer_types) {
        log_msg(stderr, "ERROR: Failed to allocate layer_types\n");
        json_free(_js_root);
        return -1;
    }
    if (_js_layer_types && (_js_layer_types->type == JSON_ARRAY)) {
        for (int32_t i = 0; i < _config->n_layers; i++) {
            JsonValue *_js_layer_type = json_array_get(_js_layer_types, i);
            if (_js_layer_type && (_js_layer_type->type == JSON_STRING)) {
                _model->_layer_types[i] = (strcmp(_js_layer_type->data.string, "full_attention")) ? 0 : 1;
            } else {
                _model->_layer_types[i] = 0;
            }
        }
    } else {
        for (int32_t i = 0; i < _config->n_layers; i++) {
            _model->_layer_types[i] = ((i + 1) % 6) ? 0 : 1;
        }
    }
    json_free(_js_root);
    log_msg(stdout, "INFO: G4 config loaded\n");
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

int32_t quantize_g4_to_file(const char *_model_dir_s, const char *_out_path_s,
        const quant_preset_t *_preset, const char *_tokenizer_path_s) {
    G4 model;
    memset(&model, 0, sizeof(model));
    if (load_config_g4(&model, _model_dir_s)) {
        return -1;
    }

    quantize_ctx qt_ctx;
    if (quantize_ctx_open(&qt_ctx, _model_dir_s)) {
        log_msg(stderr, "ERROR: Could not load safetensors metadata from %s\n", _model_dir_s);
        free(model._layer_types);
        return -1;
    }

    FILE *_file = fopen(_out_path_s, "wb");
    if (! _file) {
        log_msg(stderr, "ERROR: Failed to open %s\n", _out_path_s);
        quantize_ctx_close(&qt_ctx);
        free(model._layer_types);
        return -1;
    }

    config_g4 *_config = &model.config;
    uint64_t magic = MAGIC_G4;
    uint32_t version = 6;
    int32_t failed = 0;
    if (quantize_write_bytes(_file, &magic, sizeof(magic), 1) ||
            quantize_write_bytes(_file, &version, sizeof(version), 1) ||
            quantize_write_bytes(_file, _config, sizeof(config_g4), 1)) {
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

    if (quantize_write_bytes(_file, model._layer_types, sizeof(int32_t), _config->n_layers)) {
        failed = 1;
        goto cleanup;
    }
    
    if (quantize_write_tensor(&qt_ctx, _file, "model.language_model.embed_tokens.weight", 
            _config->vocab_size, _config->dim, _preset->embed)) {
        log_msg(stderr, "ERROR: Failed quantizing embedding weights\n");
        failed = 1;
        goto cleanup;
    }
    
    if (! _config->tie_word_embeddings) {
        if (quantize_write_tensor(&qt_ctx, _file, "lm_head.weight", 
                _config->vocab_size, _config->dim, _preset->embed)) {
            log_msg(stderr, "ERROR: Failed quantizing lm_head weights\n");
            failed = 1;
            goto cleanup;
        }
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        if (write_layer_tensor(&qt_ctx, _file, l, "input_layernorm.weight",
                1, _config->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        if (write_layer_tensor(&qt_ctx, _file, l, "post_attention_layernorm.weight",
                1, _config->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        if (write_layer_tensor(&qt_ctx, _file, l, "pre_feedforward_layernorm.weight",
                1, _config->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        if (write_layer_tensor(&qt_ctx, _file, l, "post_feedforward_layernorm.weight",
                1, _config->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        int32_t hd = model._layer_types[l] ? _config->global_head_dim : _config->head_dim;
        if (write_layer_tensor(&qt_ctx, _file, l, "self_attn.q_norm.weight",
                1, hd, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        int32_t hd = model._layer_types[l] ? _config->global_head_dim : _config->head_dim;
        if (write_layer_tensor(&qt_ctx, _file, l, "self_attn.k_norm.weight",
                1, hd, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (quantize_write_tensor_or_empty(&qt_ctx, _file, "model.language_model.norm.weight",
            1, _config->dim, Q_TYPE_F32)) {
        failed = 1;
        goto cleanup;
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        int32_t is_full = model._layer_types[l];
        int32_t use_alternative_attention = is_full && _config->attention_k_eq_v;
        int32_t hd = is_full ? _config->global_head_dim : _config->head_dim;
        int32_t kv_heads = use_alternative_attention ? _config->n_global_kv_heads : _config->n_kv_heads;

        if (write_layer_tensor(&qt_ctx, _file, l, "self_attn.q_proj.weight",
            _config->n_heads * hd, _config->dim, _preset->attn)) {
            failed = 1;
            goto cleanup;
        }

        if (write_layer_tensor(&qt_ctx, _file, l, "self_attn.k_proj.weight",
            kv_heads * hd, _config->dim, _preset->attn)) {
            failed = 1;
            goto cleanup;
        }

        if (use_alternative_attention) {
            if (quantize_write_empty_tensor(_file)) {
                failed = 1;
                goto cleanup;
            }
        }
        else {
            if (write_layer_tensor(&qt_ctx, _file, l, "self_attn.v_proj.weight",
                    kv_heads * hd, _config->dim, _preset->attn)) {
                failed = 1;
                goto cleanup;
            }
        }

        if (write_layer_tensor(&qt_ctx, _file, l, "self_attn.o_proj.weight",
                _config->dim, _config->n_heads * hd, _preset->attn)) {
            failed = 1;
            goto cleanup;
        }

        if (write_layer_tensor(&qt_ctx, _file, l, "mlp.gate_proj.weight",
                _config->hidden_dim, _config->dim, _preset->mlp)) {
            failed = 1;
            goto cleanup;
        }

        if (write_layer_tensor(&qt_ctx, _file, l, "mlp.up_proj.weight",
                _config->hidden_dim, _config->dim, _preset->mlp)) {
            failed = 1;
            goto cleanup;
        }

        if (write_layer_tensor(&qt_ctx, _file, l, "mlp.down_proj.weight",
                _config->dim, _config->hidden_dim, _preset->mlp)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        char scalar0[256], scalar1[256], scalar2[256];
        snprintf(scalar0, sizeof(scalar0), "model.language_model.layers.%d.layer_scalar", l);
        snprintf(scalar1, sizeof(scalar1), "model.language_model.layers.%d.layer_scalar.weight", l);
        snprintf(scalar2, sizeof(scalar2), "model.language_model.layers.%d.layer_output_scale.weight", l);
        const char *__names[] = {scalar0, scalar1, scalar2};
        if (quantize_write_scalar_or_default(&qt_ctx, _file, __names, 3, 1.0f)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (_config->use_rope_freqs) {
        if (quantize_write_tensor(&qt_ctx, _file, "model.language_model.layers.0.self_attn.rope_freqs.weight",
                1, _config->global_head_dim / 2, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

cleanup:
    free_tokenizer(&tokenizer1);

    if (fclose(_file)) {
        failed = 1;
    }

    quantize_ctx_close(&qt_ctx);

    free(model._layer_types);

    if (failed) {
        remove(_out_path_s);
        return -1;
    }

    log_msg(stdout, "INFO: Quantized G4 saved to %s\n", _out_path_s);
    return 0;
}

