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

int load_config_q3_5(const char *model_dir, config_q3_5 *config) {
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
    if (! json_str) {
        fclose(f);
        return -1;
    }
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
        log_msg(stderr, "ERROR: Failed to parse config.json: %s\n", error);
        return -1;
    }

    JsonValue *cfg = json_object_get(root, "text_config");
    if (! cfg) {
        cfg = root;
    }

    memset(config, 0, sizeof(config_q3_5));

    config->dim = json_get_int(json_object_get(cfg, "hidden_size"), 896);
    config->n_heads = json_get_int(json_object_get(cfg, "num_attention_heads"), 14);
    config->n_kv_heads = json_get_int(json_object_get(cfg, "num_key_value_heads"), config->n_heads);
    config->n_layer = json_get_int(json_object_get(cfg, "num_hidden_layers"), 24);
    config->n_mlp = json_get_int(json_object_get(cfg, "intermediate_size"), 4864);
    if (! config->n_mlp) {
        config->n_mlp = json_get_int(json_object_get(cfg, "shared_expert_intermediate_size"), 4864);
    }
    config->vocab_size = json_get_int(json_object_get(cfg, "vocab_size"), 151936);
    config->seq_len = json_get_int(json_object_get(cfg, "max_position_embeddings"), 262144);

    JsonValue *rope_params = json_object_get(cfg, "rope_parameters");
    config->rope_theta = json_get_double(json_object_get(rope_params, "rope_theta"), 10000.0);
    config->rope_partial_rotary_factor = json_get_double(json_object_get(rope_params, "partial_rotary_factor"), 1.0);

    config->rms_norm_eps = json_get_double(json_object_get(cfg, "rms_norm_eps"), 1e-6);
    config->tie_word_embeddings = json_get_bool(json_object_get(root, "tie_word_embeddings"), 0);
    config->d_head = json_get_int(json_object_get(cfg, "head_dim"), config->dim / config->n_heads);
    config->n_linear_k_heads = json_get_int(json_object_get(cfg, "linear_num_key_heads"), 0);
    config->n_linear_v_heads = json_get_int(json_object_get(cfg, "linear_num_value_heads"), 0);
    config->d_linear_k = json_get_int(json_object_get(cfg, "linear_key_head_dim"), 0);
    config->d_linear_v = json_get_int(json_object_get(cfg, "linear_value_head_dim"), 0);
    config->linear_conv_kernel = json_get_int(json_object_get(cfg, "linear_conv_kernel_dim"), 4);

    JsonValue *layer_types = json_object_get(cfg, "layer_types");
    config->n_full_attn_layers = 0;
    config->n_linear_attn_layers = 0;
    for (int i = 0; i < config->n_layer; i++) {
        if (get_layer_type(i, layer_types) == 1) {
            config->n_linear_attn_layers++;
        }
        else {
            config->n_full_attn_layers++;
        }
    }

    json_free(root);
    log_msg(stderr, "INFO: Model config loaded\n");
    return 0;
}

void load_q3_5_layer_types(Q3_5 *model_q3_5, const char *model_path) {
    char config_path[PATH_MAX];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_path);

    FILE *f = fopen(config_path, "rb");
    if (! f) {
        return;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json_str = (char *)a_calloc(size + 1);
    if ((! json_str) ||
            (fread(json_str, 1, size, f) != (size_t)size)) {
        if (json_str) {
            free(json_str);
        }
        fclose(f);
        return;
    }
    json_str[size] = '\0';
    fclose(f);

    char error[256] = { 0 };
    JsonValue *root = json_parse(json_str, size, error, sizeof(error));
    free(json_str);
    if (! root) {
        return;
    }

    JsonValue *cfg = json_object_get(root, "text_config");
    if (! cfg) {
        cfg = root;
    }
    JsonValue *layer_types_json = json_object_get(cfg, "layer_types");

    int la = 0, ld = 0;
    for (int i = 0; i < model_q3_5->config.n_layer; i++) {
        model_q3_5->layer_types[i] = get_layer_type(i, layer_types_json);
        if (model_q3_5->layer_types[i] == 1) {
            model_q3_5->deltanet_layer_indices[i] = ld++;
        }
        else {
            model_q3_5->attn_layer_indices[i] = la++;
        }
    }
    json_free(root);
}

static int write_layer_f32(quantize_ctx *ctx, FILE *out, int layer, const char *suffix, int cols) {
    char name[256];
    snprintf(name, sizeof(name), "model.language_model.layers.%d.%s", layer, suffix);
    if (quantize_write_tensor_or_empty(ctx, out, name, 1, cols, Q_TYPE_F32)) {
        log_msg(stderr, "ERROR: Failed writing %s\n", name);
        return -1;
    }
    return 0;
}

static int write_layer_qt(quantize_ctx *ctx, FILE *out, int layer, const char *suffix, int rows, int cols) {
    char name[256];
    snprintf(name, sizeof(name), "model.language_model.layers.%d.%s", layer, suffix);
    if (quantize_write_tensor_or_empty(ctx, out, name, rows, cols, Q_TYPE_Q8)) {
        log_msg(stderr, "ERROR: Failed quantizing %s\n", name);
        return -1;
    }
    return 0;
}

int quantize_q3_5_to_file(const char *model_dir, const char *output_file) {
    Q3_5 model;
    memset(&model, 0, sizeof(model));
    if (load_config_q3_5(model_dir, &model.config)) {
        return -1;
    }

    config_q3_5 *p = &model.config;
    model.layer_types = (int *)a_calloc((size_t)p->n_layer * sizeof(int));
    model.attn_layer_indices = (int *)a_calloc((size_t)p->n_layer * sizeof(int));
    model.deltanet_layer_indices = (int *)a_calloc((size_t)p->n_layer * sizeof(int));
    if ((! model.layer_types) ||
            (! model.attn_layer_indices) ||
            (! model.deltanet_layer_indices)) {
        free(model.layer_types);
        free(model.attn_layer_indices);
        free(model.deltanet_layer_indices);
        return -1;
    }
    load_q3_5_layer_types(&model, model_dir);

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

    uint32_t magic = 0x35335751;
    uint32_t version = 2; // Bumped version for unified qtensor formats
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
            quantize_write_tensor(&ctx, out, "model.language_model.embed_tokens.weight",
                    p->vocab_size, p->dim, Q_TYPE_Q8)) {
        failed = 1;
        goto cleanup;
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (write_layer_f32(&ctx, out, l, "input_layernorm.weight", p->dim)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (model.layer_types[l]) {
            continue;
        }
        if (write_layer_qt(&ctx, out, l, "self_attn.q_proj.weight", q_dim, p->dim) ||
                write_layer_qt(&ctx, out, l, "self_attn.k_proj.weight", kv_dim, p->dim) ||
                write_layer_qt(&ctx, out, l, "self_attn.v_proj.weight", kv_dim, p->dim) ||
                write_layer_qt(&ctx, out, l, "self_attn.o_proj.weight", p->dim, attn_out_dim)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        if ((! model.layer_types[l]) &&
                write_layer_f32(&ctx, out, l, "self_attn.q_norm.weight", head_size)) {
            failed = 1;
            goto cleanup;
        }
    }
    for (int l = 0; l < p->n_layer; l++) {
        if ((! model.layer_types[l]) &&
                write_layer_f32(&ctx, out, l, "self_attn.k_norm.weight", head_size)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (p->n_linear_attn_layers > 0) {
        for (int l = 0; l < p->n_layer; l++) {
            if (model.layer_types[l] != 1) {
                continue;
            }
            if (write_layer_qt(&ctx, out, l, "linear_attn.in_proj_qkv.weight", conv_dim, p->dim) ||
                    write_layer_qt(&ctx, out, l, "linear_attn.in_proj_z.weight", value_dim, p->dim)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if ((model.layer_types[l] == 1) &&
                    write_layer_f32(&ctx, out, l, "linear_attn.in_proj_b.weight", p->n_linear_v_heads * p->dim)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if ((model.layer_types[l] == 1) &&
                    write_layer_f32(&ctx, out, l, "linear_attn.in_proj_a.weight", p->n_linear_v_heads * p->dim)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if ((model.layer_types[l] == 1) &&
                    write_layer_f32(&ctx, out, l, "linear_attn.conv1d.weight", conv_dim * p->linear_conv_kernel)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if ((model.layer_types[l] == 1) &&
                    write_layer_f32(&ctx, out, l, "linear_attn.dt_bias", p->n_linear_v_heads)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if ((model.layer_types[l] == 1) &&
                    write_layer_f32(&ctx, out, l, "linear_attn.A_log", p->n_linear_v_heads)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if ((model.layer_types[l] == 1) &&
                    write_layer_f32(&ctx, out, l, "linear_attn.norm.weight", p->d_linear_v)) {
                failed = 1;
                goto cleanup;
            }
        }
        for (int l = 0; l < p->n_layer; l++) {
            if ((model.layer_types[l] == 1) &&
                    write_layer_qt(&ctx, out, l, "linear_attn.out_proj.weight", p->dim, value_dim)) {
                failed = 1;
                goto cleanup;
            }
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (write_layer_f32(&ctx, out, l, "post_attention_layernorm.weight", p->dim)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        if (write_layer_qt(&ctx, out, l, "mlp.gate_proj.weight", p->n_mlp, p->dim) ||
                write_layer_qt(&ctx, out, l, "mlp.down_proj.weight", p->dim, p->n_mlp) ||
                write_layer_qt(&ctx, out, l, "mlp.up_proj.weight", p->n_mlp, p->dim)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (quantize_write_tensor_or_empty(&ctx, out, "model.language_model.norm.weight", 1, p->dim, Q_TYPE_F32)) {
        failed = 1;
        goto cleanup;
    }

    if ((! p->tie_word_embeddings) &&
            quantize_write_tensor(&ctx, out, "lm_head.weight", p->vocab_size, p->dim, Q_TYPE_Q8)) {
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

    log_msg(stderr, "INFO: Quantized model saved to %s\n", output_file);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        log_msg(stderr, "Usage: dolen3_5_quantize <model_dir> <output_file>\n");
        return EXIT_FAILURE;
    }
    return quantize_q3_5_to_file(argv[1], argv[2]) ? EXIT_FAILURE : EXIT_SUCCESS;
}

