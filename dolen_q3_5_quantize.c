#include "dolen_q3_5_common.h"
#include "dolen_quantize_common.h"


static int get_layer_type(int layer_idx, const JsonValue *layer_types) {
    if (! layer_types || layer_types->type != JSON_ARRAY) {
        return 0;
    }
    if (layer_idx >= (int)layer_types->data.array.count) {
        return 0;
    }
    JsonValue *lt = json_array_get(layer_types, layer_idx);
    if (! lt || lt->type != JSON_STRING) {
        return 0;
    }
    const char *type_str = lt->data.string;
    if (strcmp(type_str, "linear_attention") == 0) {
        return 1;
    }
    return 0;
}

int load_config_qwen3_5(const char *model_dir, config_qwen3_5 *config) {
    char config_path[4096];
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

    char error[256] = {0};
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

    memset(config, 0, sizeof(config_qwen3_5));

    config->dim = json_get_int(json_object_get(cfg, "hidden_size"), 896);
    config->n_heads = json_get_int(json_object_get(cfg, "num_attention_heads"), 14);
    config->n_kv_heads = json_get_int(json_object_get(cfg, "num_key_value_heads"), config->n_heads);
    config->n_layer = json_get_int(json_object_get(cfg, "num_hidden_layers"), 24);
    config->n_mlp = json_get_int(json_object_get(cfg, "intermediate_size"), 4864);
    if (config->n_mlp == 0)
        config->n_mlp = json_get_int(json_object_get(cfg, "shared_expert_intermediate_size"), 4864);
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
        if (get_layer_type(i, layer_types) == 1)
            config->n_linear_attn_layers++;
        else
            config->n_full_attn_layers++;
    }

    json_free(root);

    log_msg(stderr, "INFO: Model config loaded\n");
    return 0;
}

void load_qwen3_5_layer_types(Qwen3_5 *model_qwen3_5, const char *model_path) {
    char config_path[4096];

    snprintf(config_path, sizeof(config_path), "%s/config.json", model_path);

    FILE *f = fopen(config_path, "rb");
    if (! f) {
        return;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json_str = (char *)a_calloc(size + 1);
    if ((! json_str) || fread(json_str, 1, size, f) != (size_t)size) {
        if (json_str) {
            free(json_str);
        }
        fclose(f);
        return;
    }
    json_str[size] = '\0';
    fclose(f);
    
    char error[256] = {0};
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
    for (int i = 0; i < model_qwen3_5->config.n_layer; i++) {
        model_qwen3_5->layer_types[i] = get_layer_type(i, layer_types_json);
        if (model_qwen3_5->layer_types[i] == 1) {
            model_qwen3_5->deltanet_layer_indices[i] = ld++;
        } else {
            model_qwen3_5->attn_layer_indices[i] = la++;
        }
    }
    json_free(root);
}

static float *load_layer_tensor_from_handle(csafetensors_t *st, int l, const char *suffix,
        float *dest, size_t expected_size) {
    char name[256];
    snprintf(name, sizeof(name), "model.language_model.layers.%d.%s", l, suffix);
    return load_tensor_from_handle(st, name, dest, expected_size);
}

static void process_qwen3_5_safetensors_file(Qwen3_5 *model_qwen3_5, safetensors_idx *idx, const char *filename) {
    config_qwen3_5 *p = &model_qwen3_5->config;
    weights_qwen3_5 *w = &model_qwen3_5->weights;

    char filepath[4096];
    snprintf(filepath, sizeof(filepath), "%s/%s", idx->model_dir, filename);

    log_msg(stderr, "INFO: Loading shard: %s\n", filename);
    csafetensors_t st;
    if (csafetensors_load_from_file(filepath, &st) != CSAFETENSORS_SUCCESS) {
        log_msg(stderr, "ERROR: Failed to load %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    int head_size = p->d_head > 0 ? p->d_head : p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int key_dim = p->n_linear_k_heads * p->d_linear_k;
    int value_dim = p->n_linear_v_heads * p->d_linear_v;
    int conv_dim = key_dim * 2 + value_dim;
    int q_dim = p->n_heads * head_size * 2;
    int attn_out_dim  = p->n_heads * head_size;

    for (size_t i = 0; i < idx->n_entries; i++) {
        if (strcmp(idx->entries[i].filename, filename) != 0) {
            continue;
        }

        const char *tname = idx->entries[i].tensor_name;

        if (strcmp(tname, "model.language_model.embed_tokens.weight") == 0) {
            float *f = load_tensor_from_handle(&st, tname, NULL, 0);
            if (f) {
                quantize_group(&w->token_embedding_table, f, p->vocab_size, p->dim);
                free(f);
            }
        }
        else if (strcmp(tname, "lm_head.weight") == 0) {
            if (! p->tie_word_embeddings) {
                float *f = load_tensor_from_handle(&st, tname, NULL, 0);
                if (f) {
                    quantize_group(&w->wcls, f, p->vocab_size, p->dim);
                    free(f);
                }
            }
        }
        else if (strcmp(tname, "model.language_model.norm.weight") == 0) {
            load_tensor_from_handle(&st, tname, w->rms_final_weight, p->dim);
        }
        else if (strncmp(tname, "model.language_model.layers.", 28) == 0) {
            int l = atoi(tname + 28);
            if (l < 0 || l >= p->n_layer) {
                continue;
            }

            const char *suffix = strstr(tname + 28, ".");
            if (! suffix) {
                continue;
            }
            suffix++;

            if (strcmp(suffix, "input_layernorm.weight") == 0) {
                load_layer_tensor_from_handle(&st, l, suffix, w->rms_att_weight + l * p->dim, p->dim);
            }
            else if (strcmp(suffix, "post_attention_layernorm.weight") == 0) {
                load_layer_tensor_from_handle(&st, l, suffix, w->rms_ffn_weight + l * p->dim, p->dim);
            }
            else if (model_qwen3_5->layer_types[l] == 0) {
                int la = model_qwen3_5->attn_layer_indices[l];
                if (strcmp(suffix, "self_attn.q_proj.weight") == 0)
                    load_and_quantize_from_handle(&st, tname, &w->wq[la], q_dim, p->dim);
                else if (strcmp(suffix, "self_attn.k_proj.weight") == 0)
                    load_and_quantize_from_handle(&st, tname, &w->wk[la], kv_dim, p->dim);
                else if (strcmp(suffix, "self_attn.v_proj.weight") == 0)
                    load_and_quantize_from_handle(&st, tname, &w->wv[la], kv_dim, p->dim);
                else if (strcmp(suffix, "self_attn.o_proj.weight") == 0)
                    load_and_quantize_from_handle(&st, tname, &w->wo[la], p->dim, attn_out_dim);
                else if (strcmp(suffix, "self_attn.q_norm.weight") == 0)
                    load_layer_tensor_from_handle(&st, l, suffix, w->q_norm + la * head_size, head_size);
                else if (strcmp(suffix, "self_attn.k_norm.weight") == 0)
                    load_layer_tensor_from_handle(&st, l, suffix, w->k_norm + la * head_size, head_size);
            }
            else if (model_qwen3_5->layer_types[l] == 1) {
                int ld = model_qwen3_5->deltanet_layer_indices[l];
                if (strcmp(suffix, "linear_attn.in_proj_qkv.weight") == 0)
                    load_and_quantize_from_handle(&st, tname, &w->in_proj_qkv[ld], conv_dim, p->dim);
                else if (strcmp(suffix, "linear_attn.in_proj_z.weight") == 0)
                    load_and_quantize_from_handle(&st, tname, &w->in_proj_z[ld], value_dim, p->dim);
                else if (strcmp(suffix, "linear_attn.in_proj_b.weight") == 0)
                    load_layer_tensor_from_handle(&st, l, suffix,
                            w->in_proj_b + ld * p->n_linear_v_heads * p->dim,
                            (size_t)p->n_linear_v_heads * p->dim);
                else if (strcmp(suffix, "linear_attn.in_proj_a.weight") == 0)
                    load_layer_tensor_from_handle(&st, l, suffix,
                            w->in_proj_a + ld * p->n_linear_v_heads * p->dim,
                            (size_t)p->n_linear_v_heads * p->dim);
                else if (strcmp(suffix, "linear_attn.conv1d.weight") == 0)
                    load_layer_tensor_from_handle(&st, l, suffix,
                            w->conv1d_weight + ld * conv_dim * p->linear_conv_kernel,
                            (size_t)conv_dim * p->linear_conv_kernel);
                else if (strcmp(suffix, "linear_attn.dt_bias") == 0)
                    load_layer_tensor_from_handle(&st, l, suffix,
                            w->dt_bias + ld * p->n_linear_v_heads,
                            p->n_linear_v_heads);
                else if (strcmp(suffix, "linear_attn.A_log") == 0)
                    load_layer_tensor_from_handle(&st, l, suffix,
                            w->A_log + ld * p->n_linear_v_heads,
                            p->n_linear_v_heads);
                else if (strcmp(suffix, "linear_attn.norm.weight") == 0)
                    load_layer_tensor_from_handle(&st, l, suffix,
                            w->linear_norm + ld * p->d_linear_v,
                            p->d_linear_v);
                else if (strcmp(suffix, "linear_attn.out_proj.weight") == 0)
                    load_and_quantize_from_handle(&st, tname, &w->out_proj[ld], p->dim, value_dim);
            }

            if (strcmp(suffix, "mlp.gate_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->w1[l], p->n_mlp, p->dim);
            else if (strcmp(suffix, "mlp.down_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->w2[l], p->dim, p->n_mlp);
            else if (strcmp(suffix, "mlp.up_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->w3[l], p->n_mlp, p->dim);
        }
    }

    csafetensors_free(&st);
    log_msg(stderr, "INFO: Finished shard: %s\n", filename);
}

int load_qwen3_5_from_safetensors(Qwen3_5 *model_qwen3_5, const char *model_dir) {
    config_qwen3_5 *p = &model_qwen3_5->config;
    weights_qwen3_5 *w = &model_qwen3_5->weights;
    safetensors_idx idx;

    if (load_safetensors_index(&idx, model_dir) != 0) {
        log_msg(stderr, "ERROR: Could not find model.safetensors.index.json in %s\n", model_dir);
        return -1;
    }

    log_msg(stderr, "INFO: Found %d safetensors shards\n", idx.n_unique_files);
    int head_size = p->d_head > 0 ? p->d_head : p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int key_dim = p->n_linear_k_heads * p->d_linear_k;
    int value_dim = p->n_linear_v_heads * p->d_linear_v;
    int conv_dim = key_dim * 2 + value_dim;
    int n_full_attn = p->n_full_attn_layers;
    int n_linear_attn = p->n_linear_attn_layers;

    w->rms_att_weight = (float *)a_calloc((size_t)p->n_layer * p->dim * sizeof(float));
    w->wq = (qtensor *)a_calloc((size_t)n_full_attn * sizeof(qtensor));
    w->wk = (qtensor *)a_calloc((size_t)n_full_attn * sizeof(qtensor));
    w->wv = (qtensor *)a_calloc((size_t)n_full_attn * sizeof(qtensor));
    w->wo = (qtensor *)a_calloc((size_t)n_full_attn * sizeof(qtensor));
    w->q_norm = (float *)a_calloc((size_t)n_full_attn * head_size * sizeof(float));
    w->k_norm = (float *)a_calloc((size_t)n_full_attn * head_size * sizeof(float));

    if (n_linear_attn > 0) {
        w->in_proj_qkv = (qtensor *)a_calloc((size_t)n_linear_attn * sizeof(qtensor));
        w->in_proj_z = (qtensor *)a_calloc((size_t)n_linear_attn * sizeof(qtensor));
        w->in_proj_b = (float *)a_calloc((size_t)n_linear_attn * p->n_linear_v_heads * p->dim * sizeof(float));
        w->in_proj_a = (float *)a_calloc((size_t)n_linear_attn * p->n_linear_v_heads * p->dim * sizeof(float));
        w->conv1d_weight = (float *)a_calloc((size_t)n_linear_attn * conv_dim * p->linear_conv_kernel * sizeof(float));
        w->dt_bias = (float *)a_calloc((size_t)n_linear_attn * p->n_linear_v_heads * sizeof(float));
        w->A_log = (float *)a_calloc((size_t)n_linear_attn * p->n_linear_v_heads * sizeof(float));
        w->linear_norm = (float *)a_calloc((size_t)n_linear_attn * p->d_linear_v * sizeof(float));
        w->out_proj = (qtensor *)a_calloc((size_t)n_linear_attn * sizeof(qtensor));
    }

    w->rms_ffn_weight = (float *)a_calloc((size_t)p->n_layer * p->dim * sizeof(float));
    w->w1 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->w2 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->w3 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));

    w->rms_final_weight = (float *)a_calloc(p->dim * sizeof(float));

    memset(&w->token_embedding_table, 0, sizeof(qtensor));
    memset(&w->wcls, 0, sizeof(qtensor));

    for (int i = 0; i < idx.n_unique_files; i++) {
        process_qwen3_5_safetensors_file(model_qwen3_5, &idx, idx.unique_filenames[i]);
    }

    if (p->tie_word_embeddings) {
        w->wcls = w->token_embedding_table;
    } else if (w->wcls.q == NULL) {
         log_msg(stderr, "ERROR: lm_head.weight was not found and tie_word_embeddings is false\n");
         return -1;
    }

    if (w->token_embedding_table.q == NULL) {
        log_msg(stderr, "ERROR: embed_tokens.weight was not found\n");
        return -1;
    }

    if (w->rms_final_weight == NULL) {
        log_msg(stderr, "ERROR: model.language_model.norm.weight was not found\n");
        return -1;
    }

    log_msg(stderr, "INFO: Weights loaded successfully\n");

    free_safetensors_index(&idx);

    return 0;
}

void build_qwen3_5(Qwen3_5 *model_qwen3_5, char *model_path) {
    memset(model_qwen3_5, 0, sizeof(Qwen3_5));

    if (load_config_qwen3_5(model_path, &model_qwen3_5->config) != 0) {
        exit(EXIT_FAILURE);
    }

    model_qwen3_5->layer_types = a_calloc(model_qwen3_5->config.n_layer * sizeof(int));
    model_qwen3_5->attn_layer_indices = a_calloc(model_qwen3_5->config.n_layer * sizeof(int));
    model_qwen3_5->deltanet_layer_indices = a_calloc(model_qwen3_5->config.n_layer * sizeof(int));
    load_qwen3_5_layer_types(model_qwen3_5, model_path);

    if (load_qwen3_5_from_safetensors(model_qwen3_5, model_path) != 0) {
        exit(EXIT_FAILURE);
    }
}

void save_quantized_qwen3_5(const char *filepath, Qwen3_5* model_qwen3_5) {
    FILE *f = fopen(filepath, "wb");
    if (! f) {
        log_msg(stderr, "ERROR: Failed to open %s for writing\n", filepath);
        exit(EXIT_FAILURE);
    }
    
    uint32_t magic = 0x35335751; // 'QW35'
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);

    fwrite(&model_qwen3_5->config, sizeof(config_qwen3_5), 1, f);

    fwrite(model_qwen3_5->layer_types, sizeof(int), model_qwen3_5->config.n_layer, f);

    config_qwen3_5 *p = &model_qwen3_5->config;
    weights_qwen3_5 *w = &model_qwen3_5->weights;
    
    write_qt(f, &w->token_embedding_table);
    fwrite(w->rms_att_weight, sizeof(float), (size_t)p->n_layer * p->dim, f);
    
    for (int i = 0; i < p->n_full_attn_layers; i++) {
        write_qt(f, &w->wq[i]);
        write_qt(f, &w->wk[i]);
        write_qt(f, &w->wv[i]);
        write_qt(f, &w->wo[i]);
    }
    
    int head_size = p->d_head > 0 ? p->d_head : p->dim / p->n_heads;
    fwrite(w->q_norm, sizeof(float), (size_t)p->n_full_attn_layers * head_size, f);
    fwrite(w->k_norm, sizeof(float), (size_t)p->n_full_attn_layers * head_size, f);
    
    if (p->n_linear_attn_layers > 0) {
        int key_dim = p->n_linear_k_heads * p->d_linear_k;
        int value_dim = p->n_linear_v_heads * p->d_linear_v;
        int conv_dim = key_dim * 2 + value_dim;
        
        for (int i = 0; i < p->n_linear_attn_layers; i++) {
            write_qt(f, &w->in_proj_qkv[i]);
            write_qt(f, &w->in_proj_z[i]);
        }
        fwrite(w->in_proj_b, sizeof(float), (size_t)p->n_linear_attn_layers * p->n_linear_v_heads * p->dim, f);
        fwrite(w->in_proj_a, sizeof(float), (size_t)p->n_linear_attn_layers * p->n_linear_v_heads * p->dim, f);
        fwrite(w->conv1d_weight, sizeof(float), (size_t)p->n_linear_attn_layers * conv_dim * p->linear_conv_kernel, f);
        fwrite(w->dt_bias, sizeof(float), (size_t)p->n_linear_attn_layers * p->n_linear_v_heads, f);
        fwrite(w->A_log, sizeof(float), (size_t)p->n_linear_attn_layers * p->n_linear_v_heads, f);
        fwrite(w->linear_norm, sizeof(float), (size_t)p->n_linear_attn_layers * p->d_linear_v, f);
        for (int i = 0; i < p->n_linear_attn_layers; i++) {
            write_qt(f, &w->out_proj[i]);
        }
    }
    
    fwrite(w->rms_ffn_weight, sizeof(float), (size_t)p->n_layer * p->dim, f);
    
    for (int i = 0; i < p->n_layer; i++) {
        write_qt(f, &w->w1[i]);
        write_qt(f, &w->w2[i]);
        write_qt(f, &w->w3[i]);
    }
    
    fwrite(w->rms_final_weight, sizeof(float), p->dim, f);
    
    if (! p->tie_word_embeddings) {
        write_qt(f, &w->wcls);
    }
    
    fclose(f);
    log_msg(stderr, "INFO: Quantized model saved to %s\n", filepath);
}

int main(int argc, char *argv[]) {
    char *model_arg = NULL;
    char *output_file = NULL;

    if (argc >= 3) {
        model_arg = argv[1];
        output_file = argv[2];
    } else {
        log_msg(stderr, "Usage: dolen3_5_quantize <model_dir> <output_file>\n");
        exit(EXIT_FAILURE);
    }

    Qwen3_5 model_qwen3_5;
    build_qwen3_5(&model_qwen3_5, model_arg);

    save_quantized_qwen3_5(output_file, &model_qwen3_5);

    free_qwen3_5(&model_qwen3_5);
    
    return 0;
}

