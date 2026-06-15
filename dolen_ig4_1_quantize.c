#include "dolen_ig4_1_common.h"
#include "dolen_quantize_common.h"

int load_config_ig4_1(const char *model_dir, config_ig4_1 *config) {
    char config_path[4096];
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
    if (!root) {
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

    JsonValue *rope_params = json_object_get(root, "rope_parameters");
    if (rope_params) {
        config->rope_theta = json_get_double(json_object_get(rope_params, "rope_theta"), 10000.0);
    } else {
        config->rope_theta = json_get_double(json_object_get(root, "rope_theta"), 10000.0);
    }
    
    config->rms_norm_eps = json_get_double(json_object_get(root, "rms_norm_eps"), 1e-6);
    config->tie_word_embeddings = json_get_bool(json_object_get(root, "tie_word_embeddings"), 0);
    config->d_head = json_get_int(json_object_get(root, "head_dim"), config->dim / config->n_heads);

    config->embedding_multiplier = json_get_double(json_object_get(root, "embedding_multiplier"), 1.0);
    config->attention_multiplier = json_get_double(json_object_get(root, "attention_multiplier"), 0.0);
    config->residual_multiplier = json_get_double(json_object_get(root, "residual_multiplier"), 1.0);
    config->logits_scaling = json_get_double(json_object_get(root, "logits_scaling"), 1.0);

    json_free(root);
    log_msg(stderr, "INFO: Model config loaded\n");
    return 0;
}

static float *load_layer_tensor_from_handle(csafetensors_t *st, int l, const char *suffix, float *dest, size_t expected_size) {
    char name[256];
    snprintf(name, sizeof(name), "model.layers.%d.%s", l, suffix);
    return load_tensor_from_handle(st, name, dest, expected_size);
}

static void process_ig4_1_safetensors_file(IG4_1 *model_ig4_1, safetensors_idx *idx, const char *filename) {
    config_ig4_1 *p = &model_ig4_1->config;
    weights_ig4_1 *w = &model_ig4_1->weights;

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
    int attn_out_dim = p->n_heads * head_size;

    for (size_t i = 0; i < idx->n_entries; i++) {
        if (strcmp(idx->entries[i].filename, filename)) {
            continue;
        }

        const char *tname = idx->entries[i].tensor_name;

        if (strcmp(tname, "model.embed_tokens.weight") == 0) {
            float *f = load_tensor_from_handle(&st, tname, NULL, 0);
            if (f) {
                quantize_group(&w->token_embedding_table, f, p->vocab_size, p->dim);
                free(f);
            }
        }
        else if (strcmp(tname, "lm_head.weight") == 0) {
            if (!p->tie_word_embeddings) {
                float *f = load_tensor_from_handle(&st, tname, NULL, 0);
                if (f) {
                    quantize_group(&w->wcls, f, p->vocab_size, p->dim);
                    free(f);
                }
            }
        }
        else if (strcmp(tname, "model.norm.weight") == 0) {
            load_tensor_from_handle(&st, tname, w->rms_final_weight, p->dim);
        }
        else if (strncmp(tname, "model.layers.", 13) == 0) {
            int l = atoi(tname + 13);
            if (l < 0 || l >= p->n_layer) {
                continue;
            }

            const char *suffix = strstr(tname + 13, ".");
            if (!suffix) {
                continue;
            }
            suffix++;

            if (strcmp(suffix, "input_layernorm.weight") == 0) {
                load_layer_tensor_from_handle(&st, l, suffix, w->rms_att_weight + l * p->dim, p->dim);
            }
            else if (strcmp(suffix, "post_attention_layernorm.weight") == 0) {
                load_layer_tensor_from_handle(&st, l, suffix, w->rms_ffn_weight + l * p->dim, p->dim);
            }
            else if (strcmp(suffix, "self_attn.q_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->wq[l], p->n_heads * head_size, p->dim);
            else if (strcmp(suffix, "self_attn.k_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->wk[l], kv_dim, p->dim);
            else if (strcmp(suffix, "self_attn.v_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->wv[l], kv_dim, p->dim);
            else if (strcmp(suffix, "self_attn.o_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->wo[l], p->dim, attn_out_dim);
            else if (strcmp(suffix, "mlp.gate_proj.weight") == 0)
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

int load_ig4_1_from_safetensors(IG4_1 *model_ig4_1, const char *model_dir) {
    config_ig4_1 *p = &model_ig4_1->config;
    weights_ig4_1 *w = &model_ig4_1->weights;
    safetensors_idx idx;

    if (load_safetensors_index(&idx, model_dir)) {
        log_msg(stderr, "ERROR: Could not find model.safetensors.index.json in %s\n", model_dir);
        return -1;
    }

    int head_size = p->d_head > 0 ? p->d_head : p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int attn_out_dim = p->n_heads * head_size;

    w->rms_att_weight = (float *)a_calloc((size_t)p->n_layer * p->dim * sizeof(float));
    w->wq = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->wk = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->wv = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->wo = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));

    w->rms_ffn_weight = (float *)a_calloc((size_t)p->n_layer * p->dim * sizeof(float));
    w->w1 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->w2 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->w3 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));

    w->rms_final_weight = (float *)a_calloc(p->dim * sizeof(float));

    memset(&w->token_embedding_table, 0, sizeof(qtensor));
    memset(&w->wcls, 0, sizeof(qtensor));

    for (int i = 0; i < idx.n_unique_files; i++) {
        process_ig4_1_safetensors_file(model_ig4_1, &idx, idx.unique_filenames[i]);
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
        log_msg(stderr, "ERROR: model.norm.weight was not found\n");
        return -1;
    }

    log_msg(stderr, "INFO: Weights loaded successfully\n");
    free_safetensors_index(&idx);
    return 0;
}

void build_ig4_1(IG4_1 *model_ig4_1, char *model_path) {
    memset(model_ig4_1, 0, sizeof(IG4_1));

    if (load_config_ig4_1(model_path, &model_ig4_1->config)) {
        exit(EXIT_FAILURE);
    }

    if (load_ig4_1_from_safetensors(model_ig4_1, model_path)) {
        exit(EXIT_FAILURE);
    }
}

void save_quantized_ig4_1(const char *filepath, IG4_1* model_ig4_1) {
    FILE *f = fopen(filepath, "wb");
    if (!f) {
        log_msg(stderr, "ERROR: Failed to open %s for writing\n", filepath);
        exit(EXIT_FAILURE);
    }
    
    uint32_t magic = 0x31344749; // 'IG4_1'
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);

    fwrite(&model_ig4_1->config, sizeof(config_ig4_1), 1, f);

    config_ig4_1 *p = &model_ig4_1->config;
    weights_ig4_1 *w = &model_ig4_1->weights;
    
    write_qt(f, &w->token_embedding_table);
    fwrite(w->rms_att_weight, sizeof(float), (size_t)p->n_layer * p->dim, f);
    
    for (int i = 0; i < p->n_layer; i++) {
        write_qt(f, &w->wq[i]);
        write_qt(f, &w->wk[i]);
        write_qt(f, &w->wv[i]);
        write_qt(f, &w->wo[i]);
    }
    
    fwrite(w->rms_ffn_weight, sizeof(float), (size_t)p->n_layer * p->dim, f);
    
    for (int i = 0; i < p->n_layer; i++) {
        write_qt(f, &w->w1[i]);
        write_qt(f, &w->w2[i]);
        write_qt(f, &w->w3[i]);
    }
    
    fwrite(w->rms_final_weight, sizeof(float), p->dim, f);
    
    if (!p->tie_word_embeddings) {
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
        log_msg(stderr, "Usage: dolen_ig4_1_quantize <model_dir> <output_file>\n");
        exit(EXIT_FAILURE);
    }

    IG4_1 model_ig4_1;
    build_ig4_1(&model_ig4_1, model_arg);

    save_quantized_ig4_1(output_file, &model_ig4_1);

    free_ig4_1(&model_ig4_1);
    
    return 0;
}

