#include "dolen_quantize_common.h"
#include "dolen_g4u_common.h"


int load_config_g4u(G4U *model, const char *model_dir) {
    config_g4u *p = &model->config;

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

    memset(p, 0, sizeof(config_g4u));
    p->dim = json_get_int(json_object_get(cfg, "hidden_size"), 0);
    p->hidden_dim = json_get_int(json_object_get(cfg, "intermediate_size"), 0);
    p->n_layers = json_get_int(json_object_get(cfg, "num_hidden_layers"), 0);
    p->n_heads = json_get_int(json_object_get(cfg, "num_attention_heads"), 0);
    p->n_kv_heads = json_get_int(json_object_get(cfg, "num_key_value_heads"), 0);
    p->n_global_kv_heads = json_get_int(json_object_get(cfg, "num_global_key_value_heads"), p->n_kv_heads);

    p->vocab_size = json_get_int(json_object_get(cfg, "vocab_size"), 0);
    p->seq_len = json_get_int(json_object_get(cfg, "max_position_embeddings"), 262144);
    p->head_dim = json_get_int(json_object_get(cfg, "head_dim"), 0);
    p->global_head_dim = json_get_int(json_object_get(cfg, "global_head_dim"), p->head_dim);
    p->sliding_window = json_get_int(json_object_get(cfg, "sliding_window"), 1024);
    p->tie_word_embeddings = json_get_bool(json_object_get(cfg, "tie_word_embeddings"), 0);
    p->rms_norm_eps = json_get_double(json_object_get(cfg, "rms_norm_eps"), 1e-6);
    p->final_logit_softcapping = json_get_double(json_object_get(cfg, "final_logit_softcapping"), 30.0);
    p->attention_k_eq_v = json_get_bool(json_object_get(cfg, "attention_k_eq_v"), 0);
    p->original_max_seq_len = json_get_int(json_object_get(cfg, "original_max_position_embeddings"), 8192);

    JsonValue *rope_params = json_object_get(cfg, "rope_parameters");
    JsonValue *full_rope = json_object_get(rope_params, "full_attention");
    JsonValue *slide_rope = json_object_get(rope_params, "sliding_attention");
    p->rope_theta_full = json_get_double(json_object_get(full_rope, "rope_theta"), 1000000.0);
    p->rope_partial_factor = json_get_double(json_object_get(full_rope, "partial_rotary_factor"), 0.25);
    p->rope_theta_sliding = json_get_double(json_object_get(slide_rope, "rope_theta"), 10000.0);

    const char *rope_type = json_get_string(json_object_get(full_rope, "rope_type"), "proportional");
    p->use_rope_freqs = 0;
    if (rope_type && (! strcmp(rope_type, "proportional"))) {
        log_msg(stdout, "INFO: Full attention uses config-derived proportional RoPE\n");
    }

    JsonValue *layer_types_json = json_object_get(cfg, "layer_types");
    model->layer_types = (int *)a_calloc((size_t)p->n_layers * sizeof(int));
    if (! model->layer_types) {
        log_msg(stderr, "ERROR: Failed to allocate layer_types\n");
        json_free(root);
        return -1;
    }

    if (layer_types_json && layer_types_json->type == JSON_ARRAY) {
        for (int i = 0; i < p->n_layers; i++) {
            JsonValue *lt = json_array_get(layer_types_json, i);
            if (lt && (lt->type == JSON_STRING)) {
                model->layer_types[i] = (strcmp(lt->data.string, "full_attention")) ? 0 : 1;
            } else {
                model->layer_types[i] = 0;
            }
        }
    } else {
        for (int i = 0; i < p->n_layers; i++) {
            model->layer_types[i] = ((i + 1) % 6) ? 0 : 1;
        }
    }

    json_free(root);
    log_msg(stdout, "INFO: G4U config loaded\n");
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
    return 0;
}

int quantize_g4u_to_file(
        const char *model_dir, const char *output_file, q_type_t embed_type, q_type_t attn_type, q_type_t mlp_type) {
    G4U model;
    memset(&model, 0, sizeof(model));
    if (load_config_g4u(&model, model_dir)) {
        return -1;
    }

    quantize_ctx ctx;
    if (quantize_ctx_open(&ctx, model_dir)) {
        log_msg(stderr, "ERROR: Could not load safetensors metadata from %s\n", model_dir);
        free(model.layer_types);
        return -1;
    }

    FILE *out = fopen(output_file, "wb");
    if (! out) {
        log_msg(stderr, "ERROR: Failed to open %s\n", output_file);
        quantize_ctx_close(&ctx);
        free(model.layer_types);
        return -1;
    }

    config_g4u *p = &model.config;
    uint32_t magic = 0x55344D47;
    uint32_t version = 5;
    int failed = 0;

    if (quantize_write_bytes(out, &magic, sizeof(magic), 1) ||
            quantize_write_bytes(out, &version, sizeof(version), 1) ||
            quantize_write_bytes(out, p, sizeof(*p), 1) ||
            quantize_write_bytes(out, model.layer_types, sizeof(int), p->n_layers)) {
        failed = 1;
        goto cleanup;
    }

    const char *embed_names[2] = { "model.language_model.embed_tokens.weight", "lm_head.weight" };
    size_t n_embed_names = p->tie_word_embeddings ? 1 : 2;
    const weightmap_entry *embed = quantize_find_last_tensor(&ctx, embed_names, n_embed_names);
    // FIXME: (?) 
    if (quantize_write_tensor_entry(&ctx, out, "(?)", embed, p->vocab_size, p->dim, embed_type)) {
        log_msg(stderr, "ERROR: Failed quantizing embedding/classifier weights\n");
        failed = 1;
        goto cleanup;
    }

    for (int l = 0; l < p->n_layers; l++) {
        if (write_layer_tensor(&ctx, out, l, "input_layernorm.weight", 1, p->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }
    for (int l = 0; l < p->n_layers; l++) {
        if (write_layer_tensor(&ctx, out, l, "post_attention_layernorm.weight", 1, p->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }
    for (int l = 0; l < p->n_layers; l++) {
        if (write_layer_tensor(&ctx, out, l, "pre_feedforward_layernorm.weight", 1, p->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }
    for (int l = 0; l < p->n_layers; l++) {
        if (write_layer_tensor(&ctx, out, l, "post_feedforward_layernorm.weight", 1, p->dim, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }
    for (int l = 0; l < p->n_layers; l++) {
        int hd = model.layer_types[l] ? p->global_head_dim : p->head_dim;
        if (write_layer_tensor(&ctx, out, l, "self_attn.q_norm.weight", 1, hd, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }
    for (int l = 0; l < p->n_layers; l++) {
        int hd = model.layer_types[l] ? p->global_head_dim : p->head_dim;
        if (write_layer_tensor(&ctx, out, l, "self_attn.k_norm.weight", 1, hd, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }
    if (quantize_write_tensor_or_empty(&ctx, out, "model.language_model.norm.weight", 1, p->dim, Q_TYPE_F32)) {
        failed = 1;
        goto cleanup;
    }

    for (int l = 0; l < p->n_layers; l++) {
        int is_full = model.layer_types[l];
        int use_alternative_attention = is_full && p->attention_k_eq_v;
        int hd = is_full ? p->global_head_dim : p->head_dim;
        int kv_heads = use_alternative_attention ? p->n_global_kv_heads : p->n_kv_heads;

        if (write_layer_tensor(&ctx, out, l, "self_attn.q_proj.weight", p->n_heads * hd, p->dim, attn_type)) {
            failed = 1;
            goto cleanup;
        }
        if (write_layer_tensor(&ctx, out, l, "self_attn.k_proj.weight", kv_heads * hd, p->dim, attn_type)) {
            failed = 1;
            goto cleanup;
        }

        if (use_alternative_attention) {
            if (quantize_write_empty_tensor(out)) {
                failed = 1;
                goto cleanup;
            }
        } else {
            if (write_layer_tensor(&ctx, out, l, "self_attn.v_proj.weight", kv_heads * hd, p->dim, attn_type)) {
                failed = 1;
                goto cleanup;
            }
        }

        if (write_layer_tensor(&ctx, out, l, "self_attn.o_proj.weight", p->dim, p->n_heads * hd, attn_type)) {
            failed = 1;
            goto cleanup;
        }
        if (write_layer_tensor(&ctx, out, l, "mlp.gate_proj.weight", p->hidden_dim, p->dim, mlp_type)) {
            failed = 1;
            goto cleanup;
        }
        if (write_layer_tensor(&ctx, out, l, "mlp.up_proj.weight", p->hidden_dim, p->dim, mlp_type)) {
            failed = 1;
            goto cleanup;
        }
        if (write_layer_tensor(&ctx, out, l, "mlp.down_proj.weight", p->dim, p->hidden_dim, mlp_type)) {
            failed = 1;
            goto cleanup;
        }
    }

    for (int l = 0; l < p->n_layers; l++) {
        char scalar0[256], scalar1[256], scalar2[256];
        snprintf(scalar0, sizeof(scalar0), "model.language_model.layers.%d.layer_scalar", l);
        snprintf(scalar1, sizeof(scalar1), "model.language_model.layers.%d.layer_scalar.weight", l);
        snprintf(scalar2, sizeof(scalar2), "model.language_model.layers.%d.layer_output_scale.weight", l);
        const char *names[] = {
            scalar0,
            scalar1,
            scalar2
        };
        if (quantize_write_scalar_or_default(&ctx, out, names, 3, 1.0f)) {
            failed = 1;
            goto cleanup;
        }
    }

    if (p->use_rope_freqs) {
        if (quantize_write_tensor(&ctx, out, "model.language_model.layers.0.self_attn.rope_freqs.weight", 1,
                    p->global_head_dim / 2, Q_TYPE_F32)) {
            failed = 1;
            goto cleanup;
        }
    }

cleanup:
    if (fclose(out)) {
        failed = 1;
    }
    quantize_ctx_close(&ctx);
    free(model.layer_types);

    if (failed) {
        remove(output_file);
        return -1;
    }

    log_msg(stdout, "INFO: Quantized G4U saved to %s\n", output_file);
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
        if ((! strcmp(argv[i], "--type")) && ((i + 1) < argc)) {
            i += 1;
            q_type_t t = parse_q_type(argv[i]);
            embed_type = attn_type = mlp_type = t;
        } else if ((! strcmp(argv[i], "--embed")) && ((i + 1) < argc)) {
            i += 1;
            embed_type = parse_q_type(argv[i]);
        }
        else if ((! strcmp(argv[i], "--attn")) && ((i + 1) < argc)) {
            i += 1;
            attn_type = parse_q_type(argv[i]);
        }
        else if ((! strcmp(argv[i], "--mlp")) && ((i + 1) < argc)) {
            i += 1;
            mlp_type = parse_q_type(argv[i]);
        }
    }

    return quantize_g4u_to_file(argv[1], argv[2], embed_type, attn_type, mlp_type) ? EXIT_FAILURE : EXIT_SUCCESS;
}

