#include "dolen_quantize_common.h"
#include "dolen_l3_common.h"

int load_config_l3(L3 *_model, const char *_model_dir_s) {
    config_l3 *p = &_model->config;

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

    char error[256] = {0};
    JsonValue *root = json_parse(_json_s, size, error, sizeof(error));
    free(_json_s);
    if (! root) {
        log_msg(stderr, "ERROR: Failed to parse config.json: %s\n", error);
        return -1;
    }

    JsonValue *cfg = json_object_get(root, "text_config");
    if (! cfg) {
        cfg = root;
    }

    memset(p, 0, sizeof(config_l3));
    p->dim = json_get_int(json_object_get(cfg, "hidden_size"), 4096);
    p->hidden_dim = json_get_int(json_object_get(cfg, "intermediate_size"), 11008);
    p->n_layers = json_get_int(json_object_get(cfg, "num_hidden_layers"), 32);
    p->n_heads = json_get_int(json_object_get(cfg, "num_attention_heads"), 32);
    p->n_kv_heads = json_get_int(json_object_get(cfg, "num_key_value_heads"), p->n_heads);
    p->vocab_size = json_get_int(json_object_get(cfg, "vocab_size"), 32000);
    p->seq_len = json_get_int(json_object_get(cfg, "max_position_embeddings"), 2048);
    p->rms_norm_eps = json_get_double(json_object_get(cfg, "rms_norm_eps"), 1e-6);
    p->tie_word_embeddings = json_get_bool(json_object_get(cfg, "tie_word_embeddings"), 0);
    
    p->head_dim = json_get_int(json_object_get(cfg, "head_dim"), p->dim / p->n_heads);
    
    // Handle different ways RoPE parameters can be stored in config.json
    p->rope_theta = json_get_double(json_object_get(cfg, "rope_theta"), 10000.0);
    if (p->rope_theta == 10000.0) {
        JsonValue *rope_params = json_object_get(cfg, "rope_parameters");
        if (rope_params) {
            p->rope_theta = json_get_double(json_object_get(rope_params, "rope_theta"), 10000.0);
        }
    }

    json_free(root);
    log_msg(stdout, "INFO: L3 config loaded\n");
    return 0;
}

static int write_layer_tensor(quantize_ctx *_qt_ctx, FILE *_file, int layer, const char *suffix,
        int rows, int cols, q_type_t type) {
    char name[256];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    if (quantize_write_tensor_or_empty(_qt_ctx, _file, name, rows, cols, type)) {
        log_msg(stderr, "ERROR: Failed quantizing %s\n", name);
        return -1;
    }
    return 0;
}

int quantize_l3_to_file(const char *_model_dir_s, const char *_file_path_s,
        q_type_t embed_type, q_type_t attn_type, q_type_t mlp_type, const char *_tokenizer_path_s) {
    L3 model;
    memset(&model, 0, sizeof(model));
    if (load_config_l3(&model, _model_dir_s)) {
        return -1;
    }

    quantize_ctx qt_ctx;
    if (quantize_ctx_open(&qt_ctx, _model_dir_s)) {
        log_msg(stderr, "ERROR: Could not load safetensors metadata from %s\n", _model_dir_s);
        return -1;
    }

    FILE *_file = fopen(_file_path_s, "wb");
    if (! _file) {
        log_msg(stderr, "ERROR: Failed to open %s\n", _file_path_s);
        quantize_ctx_close(&qt_ctx);
        return -1;
    }

    config_l3 *p = &model.config;
    int head_size = p->head_dim;
    int kv_dim = p->n_kv_heads * head_size;
    int q_dim = p->n_heads * head_size;

    uint64_t magic = MAGIC_L3;
    uint32_t version = 1;
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

    if (quantize_write_tensor(&qt_ctx, _file, "model.embed_tokens.weight", p->vocab_size, p->dim, embed_type)) {
        log_msg(stderr, "ERROR: Failed quantizing embedding weights\n");
        failed = 1;
        goto cleanup;
    }

    for (int l = 0; l < p->n_layers; l++) {
        if (write_layer_tensor(&qt_ctx, _file, l, "input_layernorm.weight", 1, p->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layers; l++) {
        if (write_layer_tensor(&qt_ctx, _file, l, "self_attn.q_proj.weight", q_dim, p->dim, attn_type) ||
                write_layer_tensor(&qt_ctx, _file, l, "self_attn.k_proj.weight", kv_dim, p->dim, attn_type) ||
                write_layer_tensor(&qt_ctx, _file, l, "self_attn.v_proj.weight", kv_dim, p->dim, attn_type) ||
                write_layer_tensor(&qt_ctx, _file, l, "self_attn.o_proj.weight", p->dim, q_dim, attn_type)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layers; l++) {
        if (write_layer_tensor(&qt_ctx, _file, l, "post_attention_layernorm.weight", 1, p->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layers; l++) {
        if (write_layer_tensor(&qt_ctx, _file, l, "mlp.gate_proj.weight", p->hidden_dim, p->dim, mlp_type) ||
                write_layer_tensor(&qt_ctx, _file, l, "mlp.down_proj.weight", p->dim, p->hidden_dim, mlp_type) ||
                write_layer_tensor(&qt_ctx, _file, l, "mlp.up_proj.weight", p->hidden_dim, p->dim, mlp_type)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (quantize_write_tensor_or_empty(&qt_ctx, _file, "model.norm.weight", 1, p->dim, Q_TYPE_F32)) {
        failed = 1;
        goto cleanup;
    }

    if (! p->tie_word_embeddings &&
            quantize_write_tensor(&qt_ctx, _file, "lm_head.weight", p->vocab_size, p->dim, embed_type)) {
        failed = 1;
        goto cleanup;
    }

cleanup:
    free_tokenizer(&tokenizer1);
    if (fclose(_file)) {
        failed = 1;
    }
    quantize_ctx_close(&qt_ctx);

    if (failed) {
        remove(_file_path_s);
        return -1;
    }

    log_msg(stdout, "INFO: Quantized L3 saved to %s\n", _file_path_s);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        log_msg(stdout, "Usage: %s <model_dir> <output_file> [--type TYPE] [--embed TYPE] [--attn TYPE] [--mlp TYPE] [--tokenizer PATH]\n", argv[0]);
        return EXIT_FAILURE;
    }

    q_type_t embed_type = Q_TYPE_Q8, attn_type = Q_TYPE_Q8, mlp_type = Q_TYPE_Q8;
    char *_tokenizer_path_s = "tokenizer.bin";
    for (int i = 3; i < argc; i++) {
        if ((! strcmp(argv[i], "--type")) &&
                ((i + 1) < argc)) {
            i += 1;
            q_type_t t = parse_q_type(argv[i]);
            embed_type = attn_type = mlp_type = t;
        }
        else if ((! strcmp(argv[i], "--embed") &&
                    ((i + 1) < argc))) {
            i += 1;
            embed_type = parse_q_type(argv[i]);
        }
        else if ((! strcmp(argv[i], "--attn") &&
                    ((i + 1) < argc))) {
            i += 1;
            attn_type = parse_q_type(argv[i]);
        }
        else if ((! strcmp(argv[i], "--mlp") &&
                    ((i + 1) < argc))) {
            i += 1;
            mlp_type = parse_q_type(argv[i]);
        }
        else if ((! strcmp(argv[i], "--tokenizer") &&
                    ((i + 1) < argc))) {
            i += 1;
            _tokenizer_path_s = argv[i];
        }
    }

    return quantize_l3_to_file(argv[1], argv[2], embed_type, attn_type, mlp_type, _tokenizer_path_s) \
        ? EXIT_FAILURE : EXIT_SUCCESS;
}

