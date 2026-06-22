#include "dolen_quantize_common.h"
#include "dolen_ig4_1_common.h"


int load_config_ig4_1(IG4_1 *model, const char *model_dir) {
    config_ig4_1 *p = &model->config;

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
    if ((! json_str) ||
            (fread(json_str, 1, size, f) != (size_t)size)) {
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
        log_msg(stderr, "ERROR: Failed to parse config.json: %s\n", error);
        return -1;
    }

    memset(p, 0, sizeof(config_ig4_1));

    p->dim = json_get_int(json_object_get(root, "hidden_size"), 4096);
    p->n_heads = json_get_int(json_object_get(root, "num_attention_heads"), 32);
    p->n_kv_heads = json_get_int(json_object_get(root, "num_key_value_heads"), p->n_heads);
    p->n_layer = json_get_int(json_object_get(root, "num_hidden_layers"), 32);
    p->n_mlp = json_get_int(json_object_get(root, "intermediate_size"), 11008);
    p->vocab_size = json_get_int(json_object_get(root, "vocab_size"), 32000);
    p->seq_len = json_get_int(json_object_get(root, "max_position_embeddings"), 2048);

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
    p->rope_theta = json_get_double(rope_theta_value, 0.0);
    if (! (p->rope_theta > 1.0f)) {
        log_msg(stderr, "ERROR: Invalid rope_theta %.9g in config.json\n", p->rope_theta);
        json_free(root);
        return -1;
    }

    p->rms_norm_eps = json_get_double(json_object_get(root, "rms_norm_eps"), 1e-6);
    p->tie_word_embeddings = json_get_bool(json_object_get(root, "tie_word_embeddings"), 0);
    p->d_head = json_get_int(json_object_get(root, "head_dim"), p->dim / p->n_heads);

    p->embedding_multiplier = json_get_double(json_object_get(root, "embedding_multiplier"), 1.0);
    p->attention_multiplier = json_get_double(json_object_get(root, "attention_multiplier"), 0.0);
    p->residual_multiplier = json_get_double(json_object_get(root, "residual_multiplier"), 1.0);
    p->logits_scaling = json_get_double(json_object_get(root, "logits_scaling"), 1.0);

    json_free(root);
    return 0;
}

static int write_layer_tensor(
        quantize_ctx *ctx, FILE *out, int layer, const char *suffix, int rows, int cols, q_type_t type) {
    char name[256];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    if (quantize_write_tensor_or_empty(ctx, out, name, rows, cols, type)) {
        log_msg(stderr, "ERROR: Failed quantizing %s\n", name);
        return -1;
    }
    return 0;
}

int quantize_ig4_1_to_file(
        const char *model_dir, const char *output_file, q_type_t embed_type, q_type_t attn_type, q_type_t mlp_type) {
    IG4_1 model;
    memset(&model, 0, sizeof(model));
    if (load_config_ig4_1(&model, model_dir)) {
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

    config_ig4_1 *p = &model.config;

    int head_size = p->d_head > 0 ? p->d_head : p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int attn_out_dim = p->n_heads * head_size;

    uint64_t magic = MAGIC_IG4_1;
    uint32_t version = 2;

    int failed = 0;

    if (quantize_write_bytes(out, &magic, sizeof(magic), 1) ||
            quantize_write_bytes(out, &version, sizeof(version), 1) ||
            quantize_write_bytes(out, p, sizeof(*p), 1) ||
            quantize_write_tensor(&ctx, out, "model.embed_tokens.weight", p->vocab_size, p->dim, embed_type)) {
        failed = 1;
        goto cleanup;
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (write_layer_tensor(&ctx, out, l, "input_layernorm.weight", 1, p->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (write_layer_tensor(&ctx, out, l, "self_attn.q_proj.weight", p->n_heads * head_size, p->dim, attn_type) ||
                write_layer_tensor(&ctx, out, l, "self_attn.k_proj.weight", kv_dim, p->dim, attn_type) ||
                write_layer_tensor(&ctx, out, l, "self_attn.v_proj.weight", kv_dim, p->dim, attn_type) ||
                write_layer_tensor(&ctx, out, l, "self_attn.o_proj.weight", p->dim, attn_out_dim, attn_type)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (write_layer_tensor(&ctx, out, l, "post_attention_layernorm.weight", 1, p->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (write_layer_tensor(&ctx, out, l, "mlp.gate_proj.weight", p->n_mlp, p->dim, mlp_type) ||
                write_layer_tensor(&ctx, out, l, "mlp.down_proj.weight", p->dim, p->n_mlp, mlp_type) ||
                write_layer_tensor(&ctx, out, l, "mlp.up_proj.weight", p->n_mlp, p->dim, mlp_type)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (quantize_write_tensor_or_empty(&ctx, out, "model.norm.weight", 1, p->dim, Q_TYPE_F32)) {
        failed = 1;
        goto cleanup;
    }

    if ((! p->tie_word_embeddings) &&
            quantize_write_tensor(&ctx, out, "lm_head.weight", p->vocab_size, p->dim, embed_type)) {
        failed = 1;
        goto cleanup;
    }

cleanup:
    if (fclose(out)) {
        failed = 1;
    }
    quantize_ctx_close(&ctx);

    if (failed) {
        remove(output_file);
        return -1;
    }

    log_msg(stdout, "INFO: Quantized model saved to %s\n", output_file);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        log_msg(stdout, "Usage: %s <model_dir> <output_file> [--type TYPE] [--embed TYPE] [--attn TYPE] [--mlp TYPE]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    q_type_t embed_type = Q_TYPE_Q8, attn_type = Q_TYPE_Q8, mlp_type = Q_TYPE_Q8;
    for (int i = 3; i < argc; i++) {
        if ((! strcmp(argv[i], "--type")) &&
                (i + 1 < argc)) {
            i += 1;
            q_type_t t = parse_q_type(argv[i]);
            embed_type = attn_type = mlp_type = t;
        }
        else if ((! strcmp(argv[i], "--embed")) &&
                (i + 1 < argc)) {
            i += 1;
            embed_type = parse_q_type(argv[i]);
        }
        else if ((! strcmp(argv[i], "--attn")) &&
                (i + 1 < argc)) {
            i += 1;
            attn_type = parse_q_type(argv[i]);
        }
        else if ((! strcmp(argv[i], "--mlp")) &&
                (i + 1 < argc)) {
            i += 1;
            mlp_type = parse_q_type(argv[i]);
        }
    }

    return quantize_ig4_1_to_file(argv[1], argv[2], embed_type, attn_type, mlp_type) ? EXIT_FAILURE : EXIT_SUCCESS;
}

