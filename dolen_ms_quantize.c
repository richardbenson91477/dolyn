#include "dolen_quantize_common.h"
#include "dolen_ms_common.h"

int32_t load_config_ms(MS *_model, const char *_model_dir_s) {
    config_ms *_config = &_model->config;

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

    JsonValue *_js_cfg = _js_root;

    memset(_config, 0, sizeof(config_ms));
    _config->dim = json_get_int(json_object_get(_js_cfg, "hidden_size"), 0);
    _config->hidden_dim = json_get_int(json_object_get(_js_cfg, "intermediate_size"), 0);
    _config->n_layers = json_get_int(json_object_get(_js_cfg, "num_hidden_layers"), 0);
    _config->n_heads = json_get_int(json_object_get(_js_cfg, "num_attention_heads"), 0);
    _config->n_kv_heads = json_get_int(json_object_get(_js_cfg, "num_key_value_heads"), _config->n_heads);
    _config->vocab_size = json_get_int(json_object_get(_js_cfg, "vocab_size"), 0);
    _config->seq_len = json_get_int(json_object_get(_js_cfg, "max_position_embeddings"), 32768);
    _config->head_dim = json_get_int(json_object_get(_js_cfg, "head_dim"), _config->dim / _config->n_heads);
    _config->shared_classifier = json_get_bool(json_object_get(_js_cfg, "tie_word_embeddings"), 0);
    // Mistral Defaults
    _config->rope_theta = get_json_float_val(json_object_get(_js_cfg, "rope_theta"), 10000.0f);
    _config->rms_norm_eps = get_json_float_val(json_object_get(_js_cfg, "rms_norm_eps"), 1e-5f);
    _config->sliding_window = json_get_int(json_object_get(_js_cfg, "sliding_window"), 0);
    _config->bos_token_id = json_get_int(json_object_get(_js_cfg, "bos_token_id"), 1);
    _config->eos_token_id = json_get_int(json_object_get(_js_cfg, "eos_token_id"), 2);

    json_free(_js_root);

    log_msg(stdout, "INFO: Model config loaded\n");
    return 0;
}

static int32_t write_layer_tensor(quantize_ctx *_qt_ctx, FILE *_file, int32_t layer, const char *_suffix_s,
        int32_t rows, int32_t cols, q_type_t type) {
    char _name_s[256];
    snprintf(_name_s, sizeof(_name_s), "model.layers.%d.%s", layer, _suffix_s);
    if (quantize_write_tensor_or_empty(_qt_ctx, _file, _name_s, rows, cols, type)) {
        log_msg(stderr, "ERROR: Failed quantizing %s\n", _name_s);
        return -1;
    }
    return 0;
}

int32_t quantize_ms_to_file(const char *_model_dir_s, const char *_file_path_s,
        const quant_preset_t *_preset, const char *_tokenizer_path_s) {
    MS model;
    memset(&model, 0, sizeof(model));
    if (load_config_ms(&model, _model_dir_s)) {
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

    config_ms *_config = &model.config;
    int32_t head_size = _config->head_dim;
    int32_t kv_dim = _config->n_kv_heads * head_size;
    int32_t all_heads_dim = _config->n_heads * head_size;
    uint64_t magic = MAGIC_MS;
    uint32_t version = 1;
    int32_t failed = 0;

    if (quantize_write_bytes(_file, &magic, sizeof(magic), 1) ||
            quantize_write_bytes(_file, &version, sizeof(version), 1) ||
            quantize_write_bytes(_file, _config, sizeof(config_ms), 1)) {
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

    if (quantize_write_tensor(&_qt_ctx, _file, "model.embed_tokens.weight",
                _config->vocab_size, _config->dim, _preset->embed)) {
        failed = 1;
        goto cleanup;
    }
    for (int32_t l = 0; l < _config->n_layers; l++) {
        if (write_layer_tensor(&_qt_ctx, _file, l, "input_layernorm.weight", 1, _config->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }
    for (int32_t l = 0; l < _config->n_layers; l++) {
        if (write_layer_tensor(&_qt_ctx, _file, l, "post_attention_layernorm.weight", 1, _config->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (quantize_write_tensor_or_empty(&_qt_ctx, _file, "model.norm.weight", 1, _config->dim, Q_TYPE_F32)) {
        failed = 1;
        goto cleanup;
    }

    if ((! _config->shared_classifier) &&
            quantize_write_tensor(&_qt_ctx, _file, "lm_head.weight",
                _config->vocab_size, _config->dim, _preset->lm_head)) {
        failed = 1;
        goto cleanup;
    }

    for (int32_t l = 0; l < _config->n_layers; l++) {
        if (write_layer_tensor(&_qt_ctx, _file, l, "self_attn.q_proj.weight",
                    all_heads_dim, _config->dim, _preset->attn) ||
                write_layer_tensor(&_qt_ctx, _file, l, "self_attn.k_proj.weight",
                    kv_dim, _config->dim, _preset->attn) ||
                write_layer_tensor(&_qt_ctx, _file, l, "self_attn.v_proj.weight",
                    kv_dim, _config->dim, _preset->attn) ||
                write_layer_tensor(&_qt_ctx, _file, l, "self_attn.o_proj.weight",
                    _config->dim, all_heads_dim, _preset->attn) ||
                /* NO QKV Biases for Mistral */
                write_layer_tensor(&_qt_ctx, _file, l, "mlp.gate_proj.weight",
                    _config->hidden_dim, _config->dim, _preset->mlp) ||
                write_layer_tensor(&_qt_ctx, _file, l, "mlp.down_proj.weight",
                    _config->dim, _config->hidden_dim, _preset->mlp) ||
                write_layer_tensor(&_qt_ctx, _file, l, "mlp.up_proj.weight",
                    _config->hidden_dim, _config->dim, _preset->mlp)) {
            failed = 1;
            goto cleanup;
        }
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

