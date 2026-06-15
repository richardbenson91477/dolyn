#include "dolen_g4u_common.h"
#include "dolen_quantize_common.h"

int load_config_gemma4u(Gemma4Unified *model, const char *model_dir) {
    config_gemma4u *p = &model->config;
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
    if (!json_str || fread(json_str, 1, size, f) != (size_t)size) {
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
        log_msg(stderr, "ERROR: Failed to parse config.json: %s\n", error);
        return -1;
    }

    JsonValue *cfg = json_object_get(root, "text_config");
    if (!cfg) {
        cfg = root;
    }

    memset(p, 0, sizeof(config_gemma4u));
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
    // In the Hugging Face Gemma 4 Unified checkpoint, proportional RoPE is
    // derived from config and is not a learned safetensors parameter.
    p->use_rope_freqs = 0;
    if (rope_type && strcmp(rope_type, "proportional") == 0) {
        log_msg(stderr, "INFO: Full attention uses config-derived proportional RoPE\n");
    }

    JsonValue *layer_types_json = json_object_get(cfg, "layer_types");
    model->layer_types = a_calloc((size_t)p->n_layers * sizeof(int));
    if (layer_types_json && layer_types_json->type == JSON_ARRAY) {
        for (int i = 0; i < p->n_layers; i++) {
            JsonValue *lt = json_array_get(layer_types_json, i);
            if (lt && lt->type == JSON_STRING) {
                model->layer_types[i] = (strcmp(lt->data.string, "full_attention") == 0) ? 1 : 0;
            } else {
                model->layer_types[i] = 0;
            }
        }
    } else {
        for (int i = 0; i < p->n_layers; i++) {
            model->layer_types[i] = ((i + 1) % 6 == 0) ? 1 : 0;
        }
    }

    json_free(root);
    log_msg(stderr, "INFO: Gemma4Unified config loaded\n");
    return 0;
}

static void process_gemma4u_safetensors_file(Gemma4Unified *model, safetensors_idx *idx, const char *filename) {
    config_gemma4u *p = &model->config;
    weights_gemma4u *w = &model->weights;
    char filepath[4096];
    snprintf(filepath, sizeof(filepath), "%s/%s", idx->model_dir, filename);
    csafetensors_t st;
    if (csafetensors_load_from_file(filepath, &st) != CSAFETENSORS_SUCCESS) {
        log_msg(stderr, "ERROR: Failed to load %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < idx->n_entries; i++) {
        if (strcmp(idx->entries[i].filename, filename) != 0) {
            continue;
        }
        const char *tname = idx->entries[i].tensor_name;

        if (strcmp(tname, "model.language_model.embed_tokens.weight") == 0) {
            float *f = extract_tensor_from_handle(&st, tname, NULL, 0);
            if (f) {
                quantize_group(&w->embed_tokens, f, p->vocab_size, p->dim);
                free(f);
            }
        }
        else if (strcmp(tname, "lm_head.weight") == 0 && !p->tie_word_embeddings) {
            float *f = extract_tensor_from_handle(&st, tname, NULL, 0);
            if (f) {
                quantize_group(&w->embed_tokens, f, p->vocab_size, p->dim);
                free(f);
            }
        }
        else if (strcmp(tname, "model.language_model.norm.weight") == 0) {
            load_tensor_from_handle(&st, tname, w->rms_final_norm, p->dim);
        }
        else if (strncmp(tname, "model.language_model.layers.", 28) == 0) {
            int l = atoi(tname + 28);
            if (l < 0 || l >= p->n_layers) {
                continue;
            }
            const char *suffix = strstr(tname + 28, ".");
            if (!suffix) {
                continue;
            }
            suffix++;

            if (strcmp(suffix, "input_layernorm.weight") == 0)
                load_tensor_from_handle(&st, tname, w->rms_input_layernorm + l * p->dim, p->dim);
            else if (strcmp(suffix, "post_attention_layernorm.weight") == 0)
                load_tensor_from_handle(&st, tname, w->rms_post_attn_layernorm + l * p->dim, p->dim);
            else if (strcmp(suffix, "pre_feedforward_layernorm.weight") == 0)
                load_tensor_from_handle(&st, tname, w->rms_pre_ffn_layernorm + l * p->dim, p->dim);
            else if (strcmp(suffix, "post_feedforward_layernorm.weight") == 0)
                load_tensor_from_handle(&st, tname, w->rms_post_ffn_layernorm + l * p->dim, p->dim);
            else if (strcmp(suffix, "self_attn.q_norm.weight") == 0) {
                int hd = model->layer_types[l] ? p->global_head_dim : p->head_dim;
                load_tensor_from_handle(&st, tname, w->rms_q_norm + w->norm_offsets[l], hd);
            }
            else if (strcmp(suffix, "self_attn.k_norm.weight") == 0) {
                int hd = model->layer_types[l] ? p->global_head_dim : p->head_dim;
                load_tensor_from_handle(&st, tname, w->rms_k_norm + w->norm_offsets[l], hd);
            }
            else if (strcmp(suffix, "self_attn.rope_freqs.weight") == 0) {
                // Some converted formats materialize RoPE factors, but the
                // source safetensors model computes proportional RoPE from
                // config. Do not treat these factors as raw inverse freqs.
                continue;
            }
            else if (strcmp(suffix, "self_attn.q_proj.weight") == 0) {
                int hd = model->layer_types[l] ? p->global_head_dim : p->head_dim;
                load_and_quantize_from_handle(&st, tname, &w->q_proj[l], p->n_heads * hd, p->dim);
            }
            else if (strcmp(suffix, "self_attn.k_proj.weight") == 0) {
                int is_full = model->layer_types[l];
                int hd = is_full ? p->global_head_dim : p->head_dim;
                int kv_heads = (is_full && p->attention_k_eq_v) ? p->n_global_kv_heads : p->n_kv_heads;
                load_and_quantize_from_handle(&st, tname, &w->k_proj[l], kv_heads * hd, p->dim);
            }
            else if (strcmp(suffix, "self_attn.v_proj.weight") == 0) {
                int is_full = model->layer_types[l];
                int use_alternative_attention = is_full && p->attention_k_eq_v;
                if (use_alternative_attention) {
                    log_msg(stderr, "INFO: Skipping v_proj.weight for full-attention layer %d (K=V)\n", l);
                } else {
                    int hd = is_full ? p->global_head_dim : p->head_dim;
                    int kv_heads = p->n_kv_heads;
                    load_and_quantize_from_handle(&st, tname, &w->v_proj[l], kv_heads * hd, p->dim);
                }
            }
            else if (strcmp(suffix, "self_attn.o_proj.weight") == 0) {
                int hd = model->layer_types[l] ? p->global_head_dim : p->head_dim;
                load_and_quantize_from_handle(&st, tname, &w->o_proj[l], p->dim, p->n_heads * hd);
            }
            else if (strcmp(suffix, "layer_scalar") == 0 ||
                     strcmp(suffix, "layer_scalar.weight") == 0 ||
                     strcmp(suffix, "layer_output_scale.weight") == 0) {
                float *f = extract_tensor_from_handle(&st, tname, NULL, 0);
                if (f) {
                    w->layer_scalars[l] = f[0];
                    free(f);
                }
            }
            else if (strcmp(suffix, "mlp.gate_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->gate_proj[l], p->hidden_dim, p->dim);
            else if (strcmp(suffix, "mlp.up_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->up_proj[l], p->hidden_dim, p->dim);
            else if (strcmp(suffix, "mlp.down_proj.weight") == 0)
                load_and_quantize_from_handle(&st, tname, &w->down_proj[l], p->dim, p->hidden_dim);
        }
    }
    csafetensors_free(&st);
}

int load_gemma4u_from_safetensors(Gemma4Unified *model, const char *model_dir) {
    config_gemma4u *p = &model->config;
    safetensors_idx idx;
    if (load_safetensors_index(&idx, model_dir) != 0) {
        log_msg(stderr, "ERROR: Could not find model.safetensors.index.json in %s\n", model_dir);
        return -1;
    }

    int total_norm_dim = 0;
    model->weights.norm_offsets = (int *)a_calloc((size_t)p->n_layers * sizeof(int));
    for (int i = 0; i < p->n_layers; i++) {
        model->weights.norm_offsets[i] = total_norm_dim;
        int hd = model->layer_types[i] ? p->global_head_dim : p->head_dim;
        total_norm_dim += hd;
    }

    model->weights.rms_input_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    model->weights.rms_post_attn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    model->weights.rms_pre_ffn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    model->weights.rms_post_ffn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    model->weights.rms_q_norm = (float *)a_calloc((size_t)total_norm_dim * sizeof(float));
    model->weights.rms_k_norm = (float *)a_calloc((size_t)total_norm_dim * sizeof(float));
    model->weights.rms_final_norm = (float *)a_calloc((size_t)p->dim * sizeof(float));
    model->weights.layer_scalars = (float *)a_calloc((size_t)p->n_layers * sizeof(float));
    for (int i = 0; i < p->n_layers; i++) {
        model->weights.layer_scalars[i] = 1.0f;
    }
    model->weights.rope_freqs_full = NULL;

    if (!model->weights.rms_input_layernorm || !model->weights.rms_post_attn_layernorm ||
            !model->weights.rms_pre_ffn_layernorm || !model->weights.rms_post_ffn_layernorm ||
            !model->weights.rms_q_norm || !model->weights.rms_k_norm || !model->weights.rms_final_norm) {
        log_msg(stderr, "ERROR: Alloc failed for RMS norms\n");
        free_safetensors_index(&idx);
        return -1;
    }

    model->weights.q_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    model->weights.k_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    model->weights.v_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    model->weights.o_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    model->weights.gate_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    model->weights.up_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    model->weights.down_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));

    if (!model->weights.q_proj || !model->weights.k_proj) {
        log_msg(stderr, "ERROR: Alloc failed\n");
        free_safetensors_index(&idx);
        return -1;
    }

    for (int i = 0; i < idx.n_unique_files; i++) {
        process_gemma4u_safetensors_file(model, &idx, idx.unique_filenames[i]);
    }

    for (int l = 0; l < p->n_layers; l++) {
        int is_full = model->layer_types[l];
        int use_alternative_attention = is_full && p->attention_k_eq_v;
        int hd = is_full ? p->global_head_dim : p->head_dim;
        int kv_heads = use_alternative_attention ? p->n_global_kv_heads : p->n_kv_heads;

        if (model->weights.q_proj[l].rows != p->n_heads * hd || model->weights.q_proj[l].cols != p->dim ||
            model->weights.k_proj[l].rows != kv_heads * hd || model->weights.k_proj[l].cols != p->dim ||
            model->weights.o_proj[l].rows != p->dim || model->weights.o_proj[l].cols != p->n_heads * hd ||
            model->weights.gate_proj[l].rows != p->hidden_dim || model->weights.gate_proj[l].cols != p->dim ||
            model->weights.up_proj[l].rows != p->hidden_dim || model->weights.up_proj[l].cols != p->dim ||
            model->weights.down_proj[l].rows != p->dim || model->weights.down_proj[l].cols != p->hidden_dim) {
            log_msg(stderr, "ERROR: Missing or incorrectly shaped weights in layer %d\n", l);
            free_safetensors_index(&idx);
            return -1;
        }

        if (!use_alternative_attention &&
            (model->weights.v_proj[l].rows != kv_heads * hd || model->weights.v_proj[l].cols != p->dim)) {
            log_msg(stderr, "ERROR: Missing sliding-attention v_proj in layer %d\n", l);
            free_safetensors_index(&idx);
            return -1;
        }
    }

    if (model->weights.embed_tokens.q == NULL) {
        log_msg(stderr, "ERROR: embed_tokens.weight was not found\n");
        free_safetensors_index(&idx);
        return -1;
    }

    log_msg(stderr, "INFO: Gemma4Unified weights loaded successfully\n");
    free_safetensors_index(&idx);
    return 0;
}

void build_gemma4u(Gemma4Unified *model, char *model_path) {
    memset(model, 0, sizeof(Gemma4Unified));
    if (load_config_gemma4u(model, model_path) != 0) {
        exit(EXIT_FAILURE);
    }
    if (load_gemma4u_from_safetensors(model, model_path) != 0) {
        exit(EXIT_FAILURE);
    }
}

void save_quantized_gemma4u(const char *filepath, Gemma4Unified* model) {
    FILE *f = fopen(filepath, "wb");
    if (!f) {
        log_msg(stderr, "ERROR: Failed to open %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    uint32_t magic = 0x55344D47;
    // Version 4 fixes sliding-attention V projection serialization and
    // config-derived proportional RoPE.
    uint32_t version = 4;

    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&model->config, sizeof(config_gemma4u), 1, f);
    fwrite(model->layer_types, sizeof(int), model->config.n_layers, f);

    config_gemma4u *p = &model->config;
    weights_gemma4u *w = &model->weights;

    write_qt(f, &w->embed_tokens);
    fwrite(w->rms_input_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    fwrite(w->rms_post_attn_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    fwrite(w->rms_pre_ffn_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    fwrite(w->rms_post_ffn_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    
    int total_norm_dim = 0;
    for (int i = 0; i < p->n_layers; i++) {
        int hd = model->layer_types[i] ? p->global_head_dim : p->head_dim;
        total_norm_dim += hd;
    }
    fwrite(w->rms_q_norm, sizeof(float), total_norm_dim, f);
    fwrite(w->rms_k_norm, sizeof(float), total_norm_dim, f);
    fwrite(w->rms_final_norm, sizeof(float), (size_t)p->dim, f);

    for (int i = 0; i < p->n_layers; i++) {
        write_qt(f, &w->q_proj[i]);
        write_qt(f, &w->k_proj[i]);
        write_qt(f, &w->v_proj[i]);
        write_qt(f, &w->o_proj[i]);
        write_qt(f, &w->gate_proj[i]);
        write_qt(f, &w->up_proj[i]);
        write_qt(f, &w->down_proj[i]);
    }

    fwrite(w->layer_scalars, sizeof(float), (size_t)p->n_layers, f);

    if (p->use_rope_freqs && w->rope_freqs_full) {
        int freq_dim = p->global_head_dim / 2;
        fwrite(w->rope_freqs_full, sizeof(float), freq_dim, f);
        log_msg(stderr, "INFO: Saved rope_freqs (%d floats)\n", freq_dim);
    }

    fclose(f);
    log_msg(stderr, "INFO: Quantized Gemma4Unified saved to %s\n", filepath);
}

int main(int argc, char *argv[]) {
    char *model_arg = NULL;
    char *output_file = NULL;
    if (argc >= 3) {
        model_arg = argv[1];
        output_file = argv[2];
    }
    else {
        log_msg(stderr, "Usage: dolen_g4u_quantize <model_dir> <output_file>\n");
        exit(EXIT_FAILURE);
    }

    Gemma4Unified model;
    build_gemma4u(&model, model_arg);
    save_quantized_gemma4u(output_file, &model);
    free_gemma4u(&model);
    return 0;
}
