#include "dolen_quantize_common.h"
#include "dolen_ig4_1_common.h"


int load_config_ig4_1(const char *model_dir, config_ig4_1 *config) {
    char config_path[PATH_MAX];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

    FILE *f = fopen(config_path, "rb");
    if (! f) {
        log_msg(stderr, "ERROR: Could not open config.json at %s\n", config_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json_str = (char *)a_calloc(size + 1);
    if (fread(json_str, 1, size, f) != (size_t)size) {
        free(json_str);
        fclose(f);
        return -1;
    }
    json_str[size] = '\0';
    fclose(f);

    char error[256] = { 0 };
    JsonValue *root = json_parse(json_str, size, error, sizeof(error));
    free(json_str);
    if (! root) {
        return -1;
    }

    memset(config, 0, sizeof(config_ig4_1));

    config->dim = json_get_int(json_object_get(root, "hidden_size"), 4096);
    config->n_heads = json_get_int(json_object_get(root, "num_attention_heads"), 32);
    config->n_kv_heads = json_get_int(json_object_get(root, "num_key_value_heads"), config->n_heads);
    config->n_layer = json_get_int(json_object_get(root, "num_hidden_layers"), 32);
    config->n_mlp = json_get_int(json_object_get(root, "intermediate_size"), 11008);
    config->vocab_size = json_get_int(json_object_get(root, "vocab_size"), 32000);
    config->seq_len = json_get_int(json_object_get(root, "max_position_embeddings"), 2048);

    JsonValue *rope_theta_value = NULL;
    JsonValue *rope_params = json_object_get(root, "rope_parameters");
    if (rope_params &&
            (rope_params->type == JSON_OBJECT)) {
        rope_theta_value = json_object_get(rope_params, "rope_theta");
    }
    if (! rope_theta_value) {
        rope_theta_value = json_object_get(root, "rope_theta");
    }
    if (! rope_theta_value) {
        log_msg(stderr, "ERROR: config.json has no rope_theta\n");
        json_free(root);
        return -1;
    }
    config->rope_theta = json_get_double(rope_theta_value, 0.0);
    if (! (config->rope_theta > 1.0f)) {
        log_msg(stderr, "ERROR: Invalid rope_theta %.9g in config.json\n", config->rope_theta);
        json_free(root);
        return -1;
    }

    config->rms_norm_eps = json_get_double(json_object_get(root, "rms_norm_eps"), 1e-6);
    config->tie_word_embeddings = json_get_bool(json_object_get(root, "tie_word_embeddings"), 0);
    config->d_head = json_get_int(json_object_get(root, "head_dim"), config->dim / config->n_heads);

    config->embedding_multiplier = json_get_double(json_object_get(root, "embedding_multiplier"), 1.0);
    config->attention_multiplier = json_get_double(json_object_get(root, "attention_multiplier"), 0.0);
    config->residual_multiplier = json_get_double(json_object_get(root, "residual_multiplier"), 1.0);
    config->logits_scaling = json_get_double(json_object_get(root, "logits_scaling"), 1.0);

    log_msg(stderr,
            "INFO: Model config loaded: dim=%d heads=%d kv_heads=%d head_dim=%d "
            "layers=%d seq_len=%d rope_theta=%.9g attn_mult=%.9g "
            "emb_mult=%.9g residual_mult=%.9g logits_scaling=%.9g\n",
            config->dim, config->n_heads, config->n_kv_heads, config->d_head, config->n_layer, config->seq_len,
            config->rope_theta, config->attention_multiplier, config->embedding_multiplier, config->residual_multiplier,
            config->logits_scaling);
    json_free(root);
    return 0;
}

static int write_layer_f32(quantize_ctx *ctx, FILE *out, int layer, const char *suffix, int cols) {
    char name[256];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    if (quantize_write_tensor_or_empty(ctx, out, name, 1, cols, Q_TYPE_F32)) {
        log_msg(stderr, "ERROR: Failed writing %s\n", name);
        return -1;
    }
    return 0;
}

static int write_layer_qt(quantize_ctx *ctx, FILE *out, int layer, const char *suffix, int rows, int cols) {
    char name[256];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    if (quantize_write_tensor_or_empty(ctx, out, name, rows, cols, Q_TYPE_Q8)) {
        log_msg(stderr, "ERROR: Failed quantizing %s\n", name);
        return -1;
    }
    return 0;
}

int quantize_ig4_1_to_file(const char *model_dir, const char *output_file) {
    config_ig4_1 config;
    if (load_config_ig4_1(model_dir, &config)) {
        return -1;
    }

    quantize_ctx ctx;
    if (quantize_ctx_open(&ctx, model_dir)) {
        log_msg(stderr, "ERROR: Could not load safetensors metadata from %s\n", model_dir);
        return -1;
    }

    FILE *out = fopen(output_file, "wb");
    if (! out) {
        log_msg(stderr, "ERROR: Failed to open %s for writing\n", output_file);
        quantize_ctx_close(&ctx);
        return -1;
    }

    uint32_t magic = 0x31344749;
    uint32_t version = 2; // Bumped version for unified qtensor formats
    int failed = 0;
    int head_size = config.d_head > 0 ? config.d_head : config.dim / config.n_heads;
    int kv_dim = config.n_kv_heads * head_size;
    int attn_out_dim = config.n_heads * head_size;

    if (quantize_write_bytes(out, &magic, sizeof(magic), 1) ||
            quantize_write_bytes(out, &version, sizeof(version), 1) ||
            quantize_write_bytes(out, &config, sizeof(config), 1) ||
            quantize_write_tensor(&ctx, out, "model.embed_tokens.weight", config.vocab_size, config.dim, Q_TYPE_Q8)) {
        failed = 1;
        goto cleanup;
    }

    for (int l = 0; l < config.n_layer; l++) {
        if (write_layer_f32(&ctx, out, l, "input_layernorm.weight", config.dim)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < config.n_layer; l++) {
        if (write_layer_qt(&ctx, out, l, "self_attn.q_proj.weight", config.n_heads * head_size, config.dim) ||
                write_layer_qt(&ctx, out, l, "self_attn.k_proj.weight", kv_dim, config.dim) ||
                write_layer_qt(&ctx, out, l, "self_attn.v_proj.weight", kv_dim, config.dim) ||
                write_layer_qt(&ctx, out, l, "self_attn.o_proj.weight", config.dim, attn_out_dim)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < config.n_layer; l++) {
        if (write_layer_f32(&ctx, out, l, "post_attention_layernorm.weight", config.dim)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < config.n_layer; l++) {
        if (write_layer_qt(&ctx, out, l, "mlp.gate_proj.weight", config.n_mlp, config.dim) ||
                write_layer_qt(&ctx, out, l, "mlp.down_proj.weight", config.dim, config.n_mlp) ||
                write_layer_qt(&ctx, out, l, "mlp.up_proj.weight", config.n_mlp, config.dim)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (quantize_write_tensor_or_empty(&ctx, out, "model.norm.weight", 1, config.dim, Q_TYPE_F32)) {
        failed = 1;
        goto cleanup;
    }

    if ((! config.tie_word_embeddings) &&
            quantize_write_tensor(&ctx, out, "lm_head.weight", config.vocab_size, config.dim, Q_TYPE_Q8)) {
        failed = 1;
        goto cleanup;
    }

cleanup:
    if (fclose(out) != 0) {
        failed = 1;
    }
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
        log_msg(stderr, "Usage: dolen_ig4_1_quantize <model_dir> <output_file>\n");
        return EXIT_FAILURE;
    }
    return quantize_ig4_1_to_file(argv[1], argv[2]) ? EXIT_FAILURE : EXIT_SUCCESS;
}

