#include "dolen_quantize_common.h"
#include "dolen_q3_5_common.h"


static int get_layer_type(int layer_idx, const JsonValue *layer_types) {
    if ((! layer_types) ||
            (layer_types->type != JSON_ARRAY)) {
        return 0;
    }
    if (layer_idx >= (int)layer_types->data.array.count) {
        return 0;
    }
    JsonValue *lt = json_array_get(layer_types, layer_idx);
    if ((! lt) ||
            (lt->type != JSON_STRING)) {
        return 0;
    }
    const char *type_str = lt->data.string;
    if (! strcmp(type_str, "linear_attention")) {
        return 1;
    }
    return 0;
}

int load_config_q3_5(Q3_5 *model, const char *model_dir) {
    config_q3_5 *p = &model->config;

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

    JsonValue *cfg = json_object_get(root, "text_config");
    if (! cfg) {
        cfg = root;
    }

    memset(p, 0, sizeof(config_q3_5));

    p->dim = json_get_int(json_object_get(cfg, "hidden_size"), 896);
    p->n_heads = json_get_int(json_object_get(cfg, "num_attention_heads"), 14);
    p->n_kv_heads = json_get_int(json_object_get(cfg, "num_key_value_heads"), p->n_heads);
    p->n_layer = json_get_int(json_object_get(cfg, "num_hidden_layers"), 24);
    p->n_mlp = json_get_int(json_object_get(cfg, "intermediate_size"), 4864);
    if (! p->n_mlp) {
        p->n_mlp = json_get_int(json_object_get(cfg, "shared_expert_intermediate_size"), 4864);
    }
    p->vocab_size = json_get_int(json_object_get(cfg, "vocab_size"), 151936);
    p->seq_len = json_get_int(json_object_get(cfg, "max_position_embeddings"), 262144);

    JsonValue *rope_params = json_object_get(cfg, "rope_parameters");
    p->rope_theta = json_get_double(json_object_get(rope_params, "rope_theta"), 10000.0);
    p->rope_partial_rotary_factor = json_get_double(json_object_get(rope_params, "partial_rotary_factor"), 1.0);

    p->rms_norm_eps = json_get_double(json_object_get(cfg, "rms_norm_eps"), 1e-6);
    p->tie_word_embeddings = json_get_bool(json_object_get(root, "tie_word_embeddings"), 0);
    p->d_head = json_get_int(json_object_get(cfg, "head_dim"), p->dim / p->n_heads);
    p->n_linear_k_heads = json_get_int(json_object_get(cfg, "linear_num_key_heads"), 0);
    p->n_linear_v_heads = json_get_int(json_object_get(cfg, "linear_num_value_heads"), 0);
    p->d_linear_k = json_get_int(json_object_get(cfg, "linear_key_head_dim"), 0);
    p->d_linear_v = json_get_int(json_object_get(cfg, "linear_value_head_dim"), 0);
    p->linear_conv_kernel = json_get_int(json_object_get(cfg, "linear_conv_kernel_dim"), 4);

    model->layer_types = (int *)a_calloc((size_t)p->n_layer * sizeof(int));
    model->attn_layer_indices = (int *)a_calloc((size_t)p->n_layer * sizeof(int));
    model->deltanet_layer_indices = (int *)a_calloc((size_t)p->n_layer * sizeof(int));

    if ((! model->layer_types) ||
            (! model->attn_layer_indices) ||
            (! model->deltanet_layer_indices)) {
        log_msg(stderr, "ERROR: Failed to allocate layer type arrays\n");
        free(model->layer_types);
        free(model->attn_layer_indices);
        free(model->deltanet_layer_indices);
        json_free(root);
        return -1;
    }

    JsonValue *layer_types_json = json_object_get(cfg, "layer_types");
    int la = 0, ld = 0;
    p->n_full_attn_layers = 0;
    p->n_linear_attn_layers = 0;

    for (int i = 0; i < p->n_layer; i++) {
        int is_linear = get_layer_type(i, layer_types_json);
        model->layer_types[i] = is_linear;
        if (is_linear == 1) {
            model->deltanet_layer_indices[i] = ld++;
            p->n_linear_attn_layers++;
        } else {
            model->attn_layer_indices[i] = la++;
            p->n_full_attn_layers++;
        }
    }

    json_free(root);
    log_msg(stdout, "INFO: Model config loaded\n");
    return 0;
}

static int write_layer_tensor(
        quantize_ctx *ctx, FILE *out, int layer, const char *suffix, int rows, int cols, q_type_t type) {
    char name[256];
    snprintf(name, sizeof(name), "model.language_model.layers.%d.%s", layer, suffix);
    if (quantize_write_tensor_or_empty(ctx, out, name, rows, cols, type)) {
        log_msg(stderr, "ERROR: Failed quantizing %s\n", name);
        return -1;
    }
    log_msg(stdout, "INFO: wrote \"%s\", %d rows, %d cols, %d type\n", name, rows, cols, (int)type);
    return 0;
}

int quantize_q3_5_to_file(
        const char *model_dir, const char *output_file, q_type_t embed_type, q_type_t attn_type, q_type_t mlp_type) {
    Q3_5 model;
    memset(&model, 0, sizeof(model));
    if (load_config_q3_5(&model, model_dir)) {
        return -1;
    }

    quantize_ctx ctx;
    if (quantize_ctx_open(&ctx, model_dir)) {
        log_msg(stderr, "ERROR: Could not load safetensors metadata from %s\n", model_dir);
        free(model.layer_types);
        free(model.attn_layer_indices);
        free(model.deltanet_layer_indices);
        return -1;
    }

    FILE *out = fopen(output_file, "wb");
    if (! out) {
        log_msg(stderr, "ERROR: Failed to open %s for writing\n", output_file);
        quantize_ctx_close(&ctx);
        free(model.layer_types);
        free(model.attn_layer_indices);
        free(model.deltanet_layer_indices);
        return -1;
    }

    config_q3_5 *p = &model.config;
    uint32_t magic = 0x35335751;
    uint32_t version = 2;
    int failed = 0;
    int head_size = p->d_head > 0 ? p->d_head : p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int key_dim = p->n_linear_k_heads * p->d_linear_k;
    int value_dim = p->n_linear_v_heads * p->d_linear_v;
    int conv_dim = key_dim * 2 + value_dim;
    int q_dim = p->n_heads * head_size * 2;
    int attn_out_dim = p->n_heads * head_size;

    if (quantize_write_bytes(out, &magic, sizeof(magic), 1) ||
            quantize_write_bytes(out, &version, sizeof(version), 1) ||
            quantize_write_bytes(out, p, sizeof(*p), 1) ||
            quantize_write_bytes(out, model.layer_types, sizeof(int), p->n_layer) ||
            quantize_write_tensor(
                    &ctx, out, "model.language_model.embed_tokens.weight", p->vocab_size, p->dim, embed_type)) {
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
        if (model.layer_types[l]) {
            continue;
        }
        if (write_layer_tensor(&ctx, out, l, "self_attn.q_proj.weight", q_dim, p->dim, attn_type) ||
                write_layer_tensor(&ctx, out, l, "self_attn.k_proj.weight", kv_dim, p->dim, attn_type) ||
                write_layer_tensor(&ctx, out, l, "self_attn.v_proj.weight", kv_dim, p->dim, attn_type) ||
                write_layer_tensor(&ctx, out, l, "self_attn.o_proj.weight", p->dim, attn_out_dim, attn_type)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (! model.layer_types[l] &&
                write_layer_tensor(&ctx, out, l, "self_attn.q_norm.weight", 1, head_size, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }
    for (int l = 0; l < p->n_layer; l++) {
        if (! model.layer_types[l] &&
                write_layer_tensor(&ctx, out, l, "self_attn.k_norm.weight", 1, head_size, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (p->n_linear_attn_layers > 0) {
        for (int l = 0; l < p->n_layer; l++) {
            if (model.layer_types[l] != 1) {
                continue;
            }
            if (write_layer_tensor(&ctx, out, l, "linear_attn.in_proj_qkv.weight", conv_dim, p->dim, attn_type) ||
                    write_layer_tensor(&ctx, out, l, "linear_attn.in_proj_z.weight", value_dim, p->dim, attn_type)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if (model.layer_types[l] == 1 && write_layer_tensor(&ctx, out, l, "linear_attn.in_proj_b.weight", 1,
                    p->n_linear_v_heads * p->dim, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if (model.layer_types[l] == 1 && write_layer_tensor(&ctx, out, l, "linear_attn.in_proj_a.weight", 1,
                    p->n_linear_v_heads * p->dim, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if (model.layer_types[l] == 1 && write_layer_tensor(&ctx, out, l, "linear_attn.conv1d.weight", 1,
                    conv_dim * p->linear_conv_kernel, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if (model.layer_types[l] == 1 &&
                    write_layer_tensor(&ctx, out, l, "linear_attn.dt_bias", 1, p->n_linear_v_heads, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if (model.layer_types[l] == 1 &&
                    write_layer_tensor(&ctx, out, l, "linear_attn.A_log", 1, p->n_linear_v_heads, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if (model.layer_types[l] == 1 &&
                    write_layer_tensor(&ctx, out, l, "linear_attn.norm.weight", 1, p->d_linear_v, Q_TYPE_F32)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if (model.layer_types[l] == 1 &&
                    write_layer_tensor(&ctx, out, l, "linear_attn.out_proj.weight", p->dim, value_dim, attn_type)) {
                failed = 1;
                goto cleanup;
            }
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

    if (quantize_write_tensor_or_empty(&ctx, out, "model.language_model.norm.weight", 1, p->dim, Q_TYPE_F32)) {
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
    free(model.layer_types);
    free(model.attn_layer_indices);
    free(model.deltanet_layer_indices);

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
        if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            q_type_t t = parse_q_type(argv[++i]);
            embed_type = attn_type = mlp_type = t;
        } else if (strcmp(argv[i], "--embed") == 0 && i + 1 < argc) {
            embed_type = parse_q_type(argv[++i]);
        }
        else if (strcmp(argv[i], "--attn") == 0 && i + 1 < argc) {
            attn_type = parse_q_type(argv[++i]);
        }
        else if (strcmp(argv[i], "--mlp") == 0 && i + 1 < argc) {
            mlp_type = parse_q_type(argv[++i]);
        }
    }

    return quantize_q3_5_to_file(argv[1], argv[2], embed_type, attn_type, mlp_type) ? EXIT_FAILURE : EXIT_SUCCESS;
}
