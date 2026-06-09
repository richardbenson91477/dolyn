#include "dolen3_common.h"
#include "dolen_q_common.h"


static float get_json_float_val(JsonValue *v, float def) {
    if (! v) {
        return def;
    }
    if (v->type == JSON_NUMBER) {
        return (float)v->data.number;
    }
    if (v->type == JSON_STRING) {
        return (float)atof(v->data.string);
    }
    return def;
}

int load_config_qwen3(const char *model_dir, config_qwen3 *config) {
    char config_path[4096];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

    FILE *f = fopen(config_path, "rb");
    if (! f) {
        fprintf(stderr, "ERROR: Could not open config.json at %s\n", config_path);
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
        fprintf(stderr, "ERROR: Failed to parse config.json: %s\n", error);
        return -1;
    }

    JsonValue *cfg = json_object_get(root, "text_config");
    if (! cfg) {
        cfg = root;
    }

    memset(config, 0, sizeof(config_qwen3));
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

    fprintf(stderr, "INFO: Model config loaded\n");
    return 0;
}

static void process_qwen3_safetensors_file(Qwen3 *model_qwen3, safetensors_idx *idx, const char *filename) {
    config_qwen3 *p = &model_qwen3->config;
    weights_qwen3 *w = &model_qwen3->weights;

    char filepath[4096];
    snprintf(filepath, sizeof(filepath), "%s/%s", idx->model_dir, filename);

    fprintf(stderr, "INFO: Loading shard: %s\n", filename);
    csafetensors_t st;
    if (csafetensors_load_from_file(filepath, &st) != CSAFETENSORS_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to load %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    int head_size = p->head_dim;
    int kv_dim = p->n_kv_heads * head_size;
    int all_heads_dim = p->n_heads * head_size;

    for (size_t i = 0; i < idx->n_entries; i++) {
        if (strcmp(idx->entries[i].filename, filename) != 0) {
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
            if (! p->shared_classifier) {
                float *f = load_tensor_from_handle(&st, tname, NULL, 0);
                if (! f) {
                    fprintf(stderr, "FATAL: failed to load lm_head.weight\n");
                    exit(EXIT_FAILURE);
                }
                quantize_group(&w->wcls, f, p->vocab_size, p->dim);
                free(f);
            }
        }
        else if (strcmp(tname, "model.norm.weight") == 0) {
            load_tensor_from_handle(&st, tname, w->rms_final_weight, p->dim);
        }
        else if (strncmp(tname, "model.layers.", 13) == 0) {
            int l = atoi(tname + 13);
            if (l < 0 || l >= p->n_layers) {
                continue;
            }

            const char *suffix = strstr(tname + 13, ".");
            if (! suffix) {
                continue;
            }
            suffix++;

            if (strcmp(suffix, "input_layernorm.weight") == 0)
                load_tensor_from_handle(&st, tname, w->rms_att_weight + l * p->dim, p->dim);
            else if (strcmp(suffix, "post_attention_layernorm.weight") == 0)
                load_tensor_from_handle(&st, tname, w->rms_ffn_weight + l * p->dim, p->dim);
            else if (strcmp(suffix, "self_attn.q_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->wq[l], all_heads_dim, p->dim);
            else if (strcmp(suffix, "self_attn.k_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->wk[l], kv_dim, p->dim);
            else if (strcmp(suffix, "self_attn.v_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->wv[l], kv_dim, p->dim);
            else if (strcmp(suffix, "self_attn.o_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->wo[l], p->dim, all_heads_dim);
            else if (strcmp(suffix, "self_attn.q_norm.weight") == 0)
                load_tensor_from_handle(&st, tname, w->q_norm_weights + l * head_size, head_size);
            else if (strcmp(suffix, "self_attn.k_norm.weight") == 0)
                load_tensor_from_handle(&st, tname, w->k_norm_weights + l * head_size, head_size);
            else if (strcmp(suffix, "mlp.gate_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->w1[l], p->hidden_dim, p->dim);
            else if (strcmp(suffix, "mlp.down_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->w2[l], p->dim, p->hidden_dim);
            else if (strcmp(suffix, "mlp.up_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->w3[l], p->hidden_dim, p->dim);
        }
    }
    csafetensors_free(&st);
    fprintf(stderr, "INFO: Finished shard: %s\n", filename);
}

int load_qwen3_from_safetensors(Qwen3 *model_qwen3, const char *model_dir) {
    config_qwen3 *p = &model_qwen3->config;
    weights_qwen3 *w = &model_qwen3->weights;
    safetensors_idx idx;

    if (load_safetensors_index(&idx, model_dir) != 0) {
        fprintf(stderr, "ERROR: Could not find model.safetensors.index.json in %s\n", model_dir);
        return -1;
    }

    fprintf(stderr, "INFO: Found %d safetensors shards\n", idx.n_unique_files);

    w->rms_att_weight = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_ffn_weight = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_final_weight = (float *)a_calloc(p->dim * sizeof(float));
    w->q_norm_weights = (float *)a_calloc((size_t)p->n_layers * p->head_dim * sizeof(float));
    w->k_norm_weights = (float *)a_calloc((size_t)p->n_layers * p->head_dim * sizeof(float));

    memset(&w->token_embedding_table, 0, sizeof(qtensor));
    memset(&w->wcls, 0, sizeof(qtensor));

    w->wq = (qtensor *)a_calloc(p->n_layers * sizeof(qtensor));
    w->wk = (qtensor *)a_calloc(p->n_layers * sizeof(qtensor));
    w->wv = (qtensor *)a_calloc(p->n_layers * sizeof(qtensor));
    w->wo = (qtensor *)a_calloc(p->n_layers * sizeof(qtensor));
    w->w1 = (qtensor *)a_calloc(p->n_layers * sizeof(qtensor));
    w->w2 = (qtensor *)a_calloc(p->n_layers * sizeof(qtensor));
    w->w3 = (qtensor *)a_calloc(p->n_layers * sizeof(qtensor));

    for (int i = 0; i < idx.n_unique_files; i++) {
        process_qwen3_safetensors_file(model_qwen3, &idx, idx.unique_filenames[i]);
    }

    if (p->shared_classifier) {
        w->wcls = w->token_embedding_table;
    } else if (w->wcls.q == NULL) {
        fprintf(stderr, "ERROR: lm_head.weight was not found and tie_word_embeddings is false\n");
        return -1;
    }

    if (w->token_embedding_table.q == NULL) {
        fprintf(stderr, "ERROR: embed_tokens.weight was not found\n");
        return -1;
    }

    fprintf(stderr, "INFO: Weights loaded successfully\n");

    free_safetensors_index(&idx);

    return 0;
}

void build_qwen3(Qwen3 *model_qwen3, char *model_path) {
    memset(model_qwen3, 0, sizeof(Qwen3));

    if (load_config_qwen3(model_path, &model_qwen3->config) != 0) {
        exit(EXIT_FAILURE);
    }

    if (load_qwen3_from_safetensors(model_qwen3, model_path) != 0) {
        exit(EXIT_FAILURE);
    }
}

void save_quantized_qwen3(const char *filepath, Qwen3* model_qwen3) {
    FILE *f = fopen(filepath, "wb");
    if (! f) {
        fprintf(stderr, "ERROR: Failed to open %s for writing\n", filepath);
        exit(EXIT_FAILURE);
    }

    uint32_t magic = 0x30335751; // 'QW30'
    uint32_t version = 2;

    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&model_qwen3->config, sizeof(config_qwen3), 1, f);
    
    config_qwen3 *p = &model_qwen3->config;
    weights_qwen3 *w = &model_qwen3->weights;
    
    write_qt(f, &w->token_embedding_table);
    fwrite(w->rms_att_weight, sizeof(float), p->n_layers * p->dim, f);
    fwrite(w->rms_ffn_weight, sizeof(float), p->n_layers * p->dim, f);
    fwrite(w->rms_final_weight, sizeof(float), p->dim, f);
    fwrite(w->q_norm_weights, sizeof(float), p->n_layers * p->head_dim, f);
    fwrite(w->k_norm_weights, sizeof(float), p->n_layers * p->head_dim, f);
    
    if (! p->shared_classifier) {
        write_qt(f, &w->wcls);
    }
    
    for (int l = 0; l < p->n_layers; l++) {
        write_qt(f, &w->wq[l]);
        write_qt(f, &w->wk[l]);
        write_qt(f, &w->wv[l]);
        write_qt(f, &w->wo[l]);
        write_qt(f, &w->w1[l]);
        write_qt(f, &w->w2[l]);
        write_qt(f, &w->w3[l]);
    }
    
    fclose(f);
    fprintf(stderr, "INFO: Quantized model saved to %s\n", filepath);
}

int main(int argc, char *argv[]) {
    char *model_arg = NULL;
    char *output_file = NULL;

    if (argc >= 3) {
        model_arg = argv[1];
        output_file = argv[2];
    } else {
        fprintf(stderr, "Usage:   dolen3_quantize <model> <output_file>\n");
        exit(EXIT_FAILURE);
    }
    
    Qwen3 model_qwen3;
    build_qwen3(&model_qwen3, model_arg);
    
    save_quantized_qwen3(output_file, &model_qwen3);
    
    free_qwen3(&model_qwen3);

    return 0;
}

