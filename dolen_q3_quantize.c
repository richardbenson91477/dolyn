#include "dolen_quantize_common.h"
#include "dolen_q3_common.h"

int load_config_q3(const char *model_dir, config_q3 *config) {
    char config_path[PATH_MAX];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

    FILE *f = fopen(config_path, "rb");
    if (!f) {
        log_msg(stderr, "ERROR: Could not open config.json at %s\n", config_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json_str = (char *)a_calloc(size + 1);
    if (!json_str) { fclose(f); return -1; }
    if (fread(json_str, 1, size, f) != (size_t)size) { free(json_str); fclose(f); return -1; }
    json_str[size] = '\0';
    fclose(f);

    char error[256] = {0};
    JsonValue *root = json_parse(json_str, size, error, sizeof(error));
    free(json_str);
    if (!root) {
        log_msg(stderr, "ERROR: Failed to parse config.json: %s\n", error);
        return -1;
    }

    JsonValue *cfg = json_object_get(root, "text_config");
    if (!cfg) cfg = root;

    memset(config, 0, sizeof(config_q3));
    config->dim = json_get_int(json_object_get(cfg, "hidden_size"), 0);
    config->hidden_dim = json_get_int(json_object_get(cfg, "intermediate_size"), 0);
    config->n_layers = json_get_int(json_object_get(cfg, "num_hidden_layers"), 0);
    config->n_heads = json_get_int(json_object_get(cfg, "num_attention_heads"), 0);
    config->n_kv_heads = json_get_int(json_object_get(cfg, "num_key_value_heads"), config->n_heads);
    config->vocab_size = json_get_int(json_object_get(cfg, "vocab_size"), 0);
    config->seq_len = json_get_int(json_object_get(cfg, "max_position_embeddings"), 262144);
    config->head_dim = json_get_int(json_object_get(cfg, "head_dim"), config->dim / config->n_heads);
    config->shared_classifier = json_get_bool(json_object_get(cfg, "tie_word_embeddings"), 0);
    config->rope_theta = get_json_float_val(json_object_get(cfg, "rope_theta"), 1000000.0f);
    config->rms_norm_eps = get_json_float_val(json_object_get(cfg, "rms_norm_eps"), 1e-6f);

    JsonValue *rope_scaling = json_object_get(cfg, "rope_scaling");
    if (rope_scaling && (rope_scaling->type == JSON_OBJECT)) {
        config->rope_scaling_factor = get_json_float_val(json_object_get(rope_scaling, "factor"), 1.0f);
    } else {
        config->rope_scaling_factor = 1.0f;
    }

    json_free(root);
    log_msg(stderr, "INFO: Model config loaded\n");
    return 0;
}

static int write_layer_tensor(quantize_ctx *ctx, FILE *out, int layer, const char *suffix, int rows, int cols, q_type_t type) {
    char name[256];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    if (quantize_write_tensor_or_empty(ctx, out, name, rows, cols, type)) {
        log_msg(stderr, "ERROR: Failed quantizing %s\n", name);
        return -1;
    }
    return 0;
}

int quantize_q3_to_file(const char *model_dir, const char *output_file) {
    config_q3 config;
    if (load_config_q3(model_dir, &config)) return -1;

    quantize_ctx ctx;
    if (quantize_ctx_open(&ctx, model_dir)) {
        log_msg(stderr, "ERROR: Could not load safetensors metadata from %s\n", model_dir);
        return -1;
    }

    FILE *out = fopen(output_file, "wb");
    if (!out) {
        log_msg(stderr, "ERROR: Failed to open %s for writing\n", output_file);
        quantize_ctx_close(&ctx);
        return -1;
    }

    uint32_t magic = 0x30335751;
    uint32_t version = 2;
    int failed = 0;
    int head_size = config.head_dim;
    int kv_dim = config.n_kv_heads * head_size;
    int all_heads_dim = config.n_heads * head_size;

    if (quantize_write_bytes(out, &magic, sizeof(magic), 1) ||
            quantize_write_bytes(out, &version, sizeof(version), 1) ||
            quantize_write_bytes(out, &config, sizeof(config), 1) ||
            quantize_write_tensor(&ctx, out, "model.embed_tokens.weight", config.vocab_size, config.dim, Q_TYPE_Q8)) {
        failed = 1;
        goto cleanup;
    }

    for (int l = 0; l < config.n_layers; l++) {
        if (write_layer_tensor(&ctx, out, l, "input_layernorm.weight", 1, config.dim, Q_TYPE_F32)) {
            failed = 1; goto cleanup;
        }
    }
    for (int l = 0; l < config.n_layers; l++) {
        if (write_layer_tensor(&ctx, out, l, "post_attention_layernorm.weight", 1, config.dim, Q_TYPE_F32)) {
            failed = 1; goto cleanup;
        }
    }
    if (quantize_write_tensor_or_empty(&ctx, out, "model.norm.weight", 1, config.dim, Q_TYPE_F32)) {
        failed = 1; goto cleanup;
    }
    for (int l = 0; l < config.n_layers; l++) {
        if (write_layer_tensor(&ctx, out, l, "self_attn.q_norm.weight", 1, head_size, Q_TYPE_F32)) {
            failed = 1; goto cleanup;
        }
    }
    for (int l = 0; l < config.n_layers; l++) {
        if (write_layer_tensor(&ctx, out, l, "self_attn.k_norm.weight", 1, head_size, Q_TYPE_F32)) {
            failed = 1; goto cleanup;
        }
    }

    if (!config.shared_classifier &&
            quantize_write_tensor(&ctx, out, "lm_head.weight", config.vocab_size, config.dim, Q_TYPE_Q8)) {
        failed = 1; goto cleanup;
    }

    for (int l = 0; l < config.n_layers; l++) {
        if (write_layer_tensor(&ctx, out, l, "self_attn.q_proj.weight", all_heads_dim, config.dim, Q_TYPE_Q8) ||
                write_layer_tensor(&ctx, out, l, "self_attn.k_proj.weight", kv_dim, config.dim, Q_TYPE_Q8) ||
                write_layer_tensor(&ctx, out, l, "self_attn.v_proj.weight", kv_dim, config.dim, Q_TYPE_Q8) ||
                write_layer_tensor(&ctx, out, l, "self_attn.o_proj.weight", config.dim, all_heads_dim, Q_TYPE_Q8) ||
                write_layer_tensor(&ctx, out, l, "mlp.gate_proj.weight", config.hidden_dim, config.dim, Q_TYPE_Q8) ||
                write_layer_tensor(&ctx, out, l, "mlp.down_proj.weight", config.dim, config.hidden_dim, Q_TYPE_Q8) ||
                write_layer_tensor(&ctx, out, l, "mlp.up_proj.weight", config.hidden_dim, config.dim, Q_TYPE_Q8)) {
            failed = 1;
            goto cleanup;
        }
    }

cleanup:
    if (fclose(out) != 0) failed = 1;
    quantize_ctx_close(&ctx);

    if (failed) {
        remove(output_file);
        return -1;
    }

    log_msg(stderr, "INFO: Quantized model saved to %s\n", output_file);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        log_msg(stderr, "Usage: dolen3_quantize <model_dir> <output_file>\n");
        return EXIT_FAILURE;
    }
    return quantize_q3_to_file(argv[1], argv[2]) ? EXIT_FAILURE : EXIT_SUCCESS;
}