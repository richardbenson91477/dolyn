#include "dolen_quantize_common.h"
#include "dolen_ig4_1_common.h"


int load_config_ig4_1(IG4_1 *model, const char *_model_dir_s) {
    config_ig4_1 *p = &model->config;

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

    char _error_s[256] = { 0 };
    JsonValue *_js_root = json_parse(_json_s, size, _error_s, sizeof(_error_s));
    free(_json_s);
    if (! _js_root) {
        log_msg(stderr, "ERROR: Failed to parse config.json: %s\n", _error_s);
        return -1;
    }

    memset(p, 0, sizeof(config_ig4_1));

    p->dim = json_get_int(json_object_get(_js_root, "hidden_size"), 4096);
    p->n_heads = json_get_int(json_object_get(_js_root, "num_attention_heads"), 32);
    p->n_kv_heads = json_get_int(json_object_get(_js_root, "num_key_value_heads"), p->n_heads);
    p->n_layer = json_get_int(json_object_get(_js_root, "num_hidden_layers"), 32);
    p->n_mlp = json_get_int(json_object_get(_js_root, "intermediate_size"), 11008);
    p->vocab_size = json_get_int(json_object_get(_js_root, "vocab_size"), 32000);
    p->seq_len = json_get_int(json_object_get(_js_root, "max_position_embeddings"), 2048);

    JsonValue *_js_rope_theta_value = NULL;
    JsonValue *_js_rope_params = json_object_get(_js_root, "rope_parameters");
    if (_js_rope_params &&
            (_js_rope_params->type == JSON_OBJECT)) {
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
    p->rope_theta = json_get_double(_js_rope_theta_value, 0.0);
    if (! (p->rope_theta > 1.0f)) {
        log_msg(stderr, "ERROR: Invalid rope_theta %.9g in config.json\n", p->rope_theta);
        json_free(_js_root);
        return -1;
    }

    p->rms_norm_eps = json_get_double(json_object_get(_js_root, "rms_norm_eps"), 1e-6);
    p->tie_word_embeddings = json_get_bool(json_object_get(_js_root, "tie_word_embeddings"), 0);
    p->d_head = json_get_int(json_object_get(_js_root, "head_dim"), p->dim / p->n_heads);

    p->embedding_multiplier = json_get_double(json_object_get(_js_root, "embedding_multiplier"), 1.0);
    p->attention_multiplier = json_get_double(json_object_get(_js_root, "attention_multiplier"), 0.0);
    p->residual_multiplier = json_get_double(json_object_get(_js_root, "residual_multiplier"), 1.0);
    p->logits_scaling = json_get_double(json_object_get(_js_root, "logits_scaling"), 1.0);

    json_free(_js_root);
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

    config_ig4_1 *p = &model.config;

    int head_size = p->d_head > 0 ? p->d_head : p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int attn_out_dim = p->n_heads * head_size;

    uint64_t magic = MAGIC_IG4_1;
    uint32_t version = 2;

    int failed = 0;

    if (quantize_write_bytes(_file, &magic, sizeof(magic), 1) ||
            quantize_write_bytes(_file, &version, sizeof(version), 1) ||
            quantize_write_bytes(_file, p, sizeof(*p), 1)) {
        failed = 1;
        goto cleanup;
    }

    tokenizer tokenizer1;
    memset(&tokenizer1, 0, sizeof(tokenizer));
    build_tokenizer(&tokenizer1, _tokenizer_path_s, p->vocab_size, NULL);

    if (tokenizer_write_to_file(_file, &tokenizer1)) {
        log_msg(stderr, "ERROR: Failed to write tokenizer\n");
        failed = 1;
        goto cleanup;
    }

    if (quantize_write_tensor(&_qt_ctx, _file, "model.embed_tokens.weight", p->vocab_size, p->dim, embed_type)) {
        failed = 1;
        goto cleanup;
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (write_layer_tensor(&_qt_ctx, _file, l, "input_layernorm.weight", 1, p->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (write_layer_tensor(&_qt_ctx, _file, l, "self_attn.q_proj.weight", p->n_heads * head_size, p->dim, attn_type) ||
                write_layer_tensor(&_qt_ctx, _file, l, "self_attn.k_proj.weight", kv_dim, p->dim, attn_type) ||
                write_layer_tensor(&_qt_ctx, _file, l, "self_attn.v_proj.weight", kv_dim, p->dim, attn_type) ||
                write_layer_tensor(&_qt_ctx, _file, l, "self_attn.o_proj.weight", p->dim, attn_out_dim, attn_type)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (write_layer_tensor(&_qt_ctx, _file, l, "post_attention_layernorm.weight", 1, p->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (write_layer_tensor(&_qt_ctx, _file, l, "mlp.gate_proj.weight", p->n_mlp, p->dim, mlp_type) ||
                write_layer_tensor(&_qt_ctx, _file, l, "mlp.down_proj.weight", p->dim, p->n_mlp, mlp_type) ||
                write_layer_tensor(&_qt_ctx, _file, l, "mlp.up_proj.weight", p->n_mlp, p->dim, mlp_type)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (quantize_write_tensor_or_empty(&_qt_ctx, _file, "model.norm.weight", 1, p->dim, Q_TYPE_F32)) {
        failed = 1;
        goto cleanup;
    }

    if ((! p->tie_word_embeddings) &&
            quantize_write_tensor(&_qt_ctx, _file, "lm_head.weight", p->vocab_size, p->dim, embed_type)) {
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

int main(int argc, char *__argv[]) {
    if (argc < 3) {
        log_msg(stdout, "Usage: %s <model_dir> <output_file> [--type TYPE] [--embed TYPE] [--attn TYPE] [--mlp TYPE] [--tokenizer PATH]\n", __argv[0]);
        return EXIT_FAILURE;
    }

    q_type_t embed_type = Q_TYPE_Q8, attn_type = Q_TYPE_Q8, mlp_type = Q_TYPE_Q8;
    char *_tokenizer_path_s = "tokenizer.bin";
    for (int i = 3; i < argc; i++) {
        if ((! strcmp(__argv[i], "--type")) &&
                (i + 1 < argc)) {
            i += 1;
            q_type_t t = parse_q_type(__argv[i]);
            embed_type = attn_type = mlp_type = t;
        }
        else if ((! strcmp(__argv[i], "--embed")) &&
                (i + 1 < argc)) {
            i += 1;
            embed_type = parse_q_type(__argv[i]);
        }
        else if ((! strcmp(__argv[i], "--attn")) &&
                (i + 1 < argc)) {
            i += 1;
            attn_type = parse_q_type(__argv[i]);
        }
        else if ((! strcmp(__argv[i], "--mlp")) &&
                (i + 1 < argc)) {
            i += 1;
            mlp_type = parse_q_type(__argv[i]);
        }
        else if ((! strcmp(__argv[i], "--tokenizer")) &&
                (i + 1 < argc)) {
            i += 1;
            _tokenizer_path_s = __argv[i];
        }
    }

    return quantize_ig4_1_to_file(__argv[1], __argv[2], embed_type, attn_type, mlp_type, _tokenizer_path_s) \
        ? EXIT_FAILURE : EXIT_SUCCESS;
}

