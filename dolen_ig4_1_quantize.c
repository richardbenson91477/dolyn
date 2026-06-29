#include "dolen_quantize_common.h"
#include "dolen_ig4_1_common.h"


int load_config_ig4_1(IG4_1 *_model, const char *_model_dir_s) {
    config_ig4_1 *_config = &_model->config;

    char config_path[PATH_MAX];
    snprintf(config_path, sizeof(config_path), "%s/config.json", _model_dir_s);
    FILE *_file = fopen(config_path, "rb");
    if (! _file) {
        log_msg(stderr, "ERROR: Could not open config.json at %s\n", config_path);
        return -1;
    }
    fseek(_file, 0, SEEK_END);
    long size = ftell(_file);
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

    memset(_config, 0, sizeof(config_ig4_1));

    _config->dim = json_get_int(json_object_get(_js_root, "hidden_size"), 4096);
    _config->n_heads = json_get_int(json_object_get(_js_root, "num_attention_heads"), 32);
    _config->n_kv_heads = json_get_int(json_object_get(_js_root, "num_key_value_heads"), _config->n_heads);
    _config->n_layer = json_get_int(json_object_get(_js_root, "num_hidden_layers"), 32);
    _config->n_mlp = json_get_int(json_object_get(_js_root, "intermediate_size"), 11008);
    _config->vocab_size = json_get_int(json_object_get(_js_root, "vocab_size"), 32000);
    _config->seq_len = json_get_int(json_object_get(_js_root, "max_position_embeddings"), 2048);

    JsonValue *_js_rope_theta_value = NULL;
    JsonValue *_js_rope_params = json_object_get(_js_root, "rope_parameters");
    if (_js_rope_params && (_js_rope_params->type == JSON_OBJECT)) {
        _js_rope_theta_value = json_object_get(_js_rope_params, "rope_theta");
    }
    if (! _js_rope_theta_value) {
        _js_rope_theta_value = json_object_get(_js_root, "rope_theta");
    }
    if (! _js_rope_theta_value) {
        log_msg(stderr, "ERROR: config.json has no rope_theta\n");
        json_free(_js_root);
        return -1;
    }
    _config->rope_theta = json_get_double(_js_rope_theta_value, 0.0);
    if (! (_config->rope_theta > 1.0f)) {
        log_msg(stderr, "ERROR: Invalid rope_theta %.9g in config.json\n", _config->rope_theta);
        json_free(_js_root);
        return -1;
    }

    _config->rms_norm_eps = json_get_double(json_object_get(_js_root, "rms_norm_eps"), 1e-6);
    _config->tie_word_embeddings = json_get_bool(json_object_get(_js_root, "tie_word_embeddings"), 0);
    _config->d_head = json_get_int(json_object_get(_js_root, "head_dim"), _config->dim / _config->n_heads);

    _config->embedding_multiplier = json_get_double(json_object_get(_js_root, "embedding_multiplier"), 1.0);
    _config->attention_multiplier = json_get_double(json_object_get(_js_root, "attention_multiplier"), 0.0);
    _config->residual_multiplier = json_get_double(json_object_get(_js_root, "residual_multiplier"), 1.0);
    _config->logits_scaling = json_get_double(json_object_get(_js_root, "logits_scaling"), 1.0);

    // Extract Token IDs dynamically
    _config->bos_token_id = json_get_int(json_object_get(_js_root, "bos_token_id"), 100257);
    _config->eos_token_id = json_get_int(json_object_get(_js_root, "eos_token_id"), 100257);

    json_free(_js_root);
    log_msg(stdout, "INFO: Granite config loaded\n");
    return 0;
}

static int write_layer_tensor(quantize_ctx *_qt_ctx, FILE *_file, int layer, const char *_suffix_s,
        int rows, int cols, q_type_t type) {
    char _name_s[256];
    snprintf(_name_s, sizeof(_name_s), "model.layers.%d.%s", layer, _suffix_s);
    if (quantize_write_tensor_or_empty(_qt_ctx, _file, _name_s, rows, cols, type)) {
        log_msg(stderr, "ERROR: Failed quantizing %s\n", _name_s);
        return -1;
    }
    return 0;
}

int quantize_ig4_1_to_file(const char *_model_dir_s, const char *_file_path_s,
        q_type_t embed_type, q_type_t attn_type, q_type_t mlp_type, const char *_tokenizer_path_s) {
    IG4_1 model;
    memset(&model, 0, sizeof(model));
    if (load_config_ig4_1(&model, _model_dir_s)) {
        return -1;
    }

    quantize_ctx _qt_ctx;
    if (quantize_ctx_open(&_qt_ctx, _model_dir_s)) {
        log_msg(stderr, "ERROR: Could not load safetensors metadata from %s\n", _model_dir_s);
        return -1;
    }

    FILE *_file = fopen(_file_path_s, "wb");
    if (! _file) {
        log_msg(stderr, "ERROR: Failed to open %s for writing\n", _file_path_s);
        quantize_ctx_close(&_qt_ctx);
        return -1;
    }

    config_ig4_1 *_config = &model.config;

    int head_size = _config->d_head > 0 ? _config->d_head : _config->dim / _config->n_heads;
    int kv_dim = _config->n_kv_heads * head_size;
    int attn_out_dim = _config->n_heads * head_size;

    uint64_t magic = MAGIC_IG4_1;
    uint32_t version = 3; // Bumped from 2 to 3

    int failed = 0;

    if (quantize_write_bytes(_file, &magic, sizeof(magic), 1) ||
            quantize_write_bytes(_file, &version, sizeof(version), 1) ||
            quantize_write_bytes(_file, _config, sizeof(*_config), 1)) {
        failed = 1;
        goto cleanup;
    }

    tokenizer tokenizer1;
    memset(&tokenizer1, 0, sizeof(tokenizer1));
    build_tokenizer(&tokenizer1, _tokenizer_path_s, _config->vocab_size, NULL);

    if (tokenizer_write_to_file(_file, &tokenizer1)) {
        log_msg(stderr, "ERROR: Failed to write tokenizer\n");
        failed = 1;
        goto cleanup;
    }

    if (quantize_write_tensor(&_qt_ctx, _file, "model.embed_tokens.weight", _config->vocab_size, _config->dim, embed_type)) {
        failed = 1;
        goto cleanup;
    }

    for (int l = 0; l < _config->n_layer; l++) {
        if (write_layer_tensor(&_qt_ctx, _file, l, "input_layernorm.weight", 1, _config->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < _config->n_layer; l++) {
        if (write_layer_tensor(&_qt_ctx, _file, l, "self_attn.q_proj.weight",
                    _config->n_heads * head_size, _config->dim, attn_type) ||
                write_layer_tensor(&_qt_ctx, _file, l, "self_attn.k_proj.weight",
                        kv_dim, _config->dim, attn_type) ||
                write_layer_tensor(&_qt_ctx, _file, l, "self_attn.v_proj.weight",
                        kv_dim, _config->dim, attn_type) ||
                write_layer_tensor(&_qt_ctx, _file, l, "self_attn.o_proj.weight",
                        _config->dim, attn_out_dim, attn_type)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < _config->n_layer; l++) {
        if (write_layer_tensor(&_qt_ctx, _file, l, "post_attention_layernorm.weight", 1, _config->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < _config->n_layer; l++) {
        if (write_layer_tensor(&_qt_ctx, _file, l, "mlp.gate_proj.weight", _config->n_mlp, _config->dim, mlp_type) ||
                write_layer_tensor(&_qt_ctx, _file, l, "mlp.down_proj.weight", _config->dim, _config->n_mlp, mlp_type) ||
                write_layer_tensor(&_qt_ctx, _file, l, "mlp.up_proj.weight", _config->n_mlp, _config->dim, mlp_type)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (quantize_write_tensor_or_empty(&_qt_ctx, _file, "model.norm.weight", 1, _config->dim, Q_TYPE_F32)) {
        failed = 1;
        goto cleanup;
    }

    if ((! _config->tie_word_embeddings) &&
            quantize_write_tensor(&_qt_ctx, _file, "lm_head.weight", _config->vocab_size, _config->dim, embed_type)) {
        failed = 1;
        goto cleanup;
    }

cleanup:
    free_tokenizer(&tokenizer1);
    if (fclose(_file)) {
        failed = 1;
    }
    quantize_ctx_close(&_qt_ctx);

    if (failed) {
        remove(_file_path_s);
        return -1;
    }

    log_msg(stdout, "INFO: Quantized model saved to %s\n", _file_path_s);
    return 0;
}

