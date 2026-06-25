#include "dolen_g4_common.h"


static const chat_template CHAT_TEMPLATE_G4 = {
    .system = "<|turn\x3e" "system\n%s<turn|\x3e" "\n",
    .main = "<|turn\x3e" "user\n%s<turn|\x3e" "\n"
            "<|turn\x3e" "model\n"
            "<|channel\x3e" "thought\n<channel|\x3e",
    .end_turn = "<turn|\x3e" "\n",
};

static const chat_template CHAT_TEMPLATE_THINK_G4 = {
    .system = "<|turn\x3e" "system\n<|think|\x3e" "%s<turn|\x3e" "\n",
    .main = "<|turn\x3e" "user\n%s<turn|\x3e" "\n"
            "<|turn\x3e" "model\n",
    .end_turn = "<turn|\x3e" "\n",
};


int load_quantized_g4(const char *filepath, G4 *model, int seq_n_max) {
    FILE *f = fopen(filepath, "rb");
    if (! f) {
        log_msg(stderr, "ERROR: Failed to open %s\n", filepath);
        return -1;
    }

    memset(model, 0, sizeof(G4));

    uint64_t magic;
    uint32_t version;

    if (fread(&magic, sizeof(magic), 1, f) != 1 ||
            fread(&version, sizeof(version), 1, f) != 1) {
        log_msg(stderr, "ERROR: Failed to read header\n");
        fclose(f);
        return -1;
    }

    if (magic != MAGIC_G4) {
        log_msg(stderr, "ERROR: Invalid magic number\n");
        fclose(f);
        return -1;
    }

    if (version != 5) {
        log_msg(stderr, "ERROR: Unsupported version %u (expected 5). RE-RUN QUANTIZER.\n", version);
        fclose(f);
        return -1;
    }

    config_g4 *p = &model->config;

    if (fread(p, sizeof(config_g4), 1, f) != 1) {
        log_msg(stderr, "ERROR: Failed to read config\n");
        fclose(f);
        return -1;
    }

    if (tokenizer_read_from_file(f, p->vocab_size, &model->tokenizer)) {
        log_msg(stderr, "ERROR: Failed to read tokenizer from %s\n", filepath);
        fclose(f);
        return -1;
    }

    if (seq_n_max) {
        p->seq_len = seq_n_max;
    }

    model->layer_types = (int *)a_calloc((size_t)p->n_layers * sizeof(int));
    if (fread(model->layer_types, sizeof(int), (size_t)p->n_layers, f) != (size_t)p->n_layers) {
        log_msg(stderr, "ERROR: Failed to read layer_types\n");
        fclose(f);
        return -1;
    }

    weights_g4 *w = &model->weights;

    w->rms_input_layernorm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->rms_post_attn_layernorm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->rms_pre_ffn_layernorm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->rms_post_ffn_layernorm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->rms_q_norm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->rms_k_norm = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));

    if (! w->rms_input_layernorm ||
            (! w->rms_post_attn_layernorm) ||
            (! w->rms_pre_ffn_layernorm) ||
            (! w->rms_post_ffn_layernorm) ||
            (! w->rms_q_norm) ||
            (! w->rms_k_norm)) {
        log_msg(stderr, "ERROR: Alloc failed\n");
        fclose(f);
        return -1;
    }

    read_qt(f, &w->embed_tokens_weight);

    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->rms_input_layernorm[i]);
    }
    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->rms_post_attn_layernorm[i]);
    }
    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->rms_pre_ffn_layernorm[i]);
    }
    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->rms_post_ffn_layernorm[i]);
    }
    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->rms_q_norm[i]);
    }
    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->rms_k_norm[i]);
    }

    read_qt(f, &w->rms_final_norm);

    w->q_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->k_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->v_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->o_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->gate_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->up_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->down_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->layer_scalars = (float *)a_calloc((size_t)p->n_layers * sizeof(float));

    for (int i = 0; i < p->n_layers; i++) {
        w->layer_scalars[i] = 1.0f;
    }

    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->q_proj[i]);
        read_qt(f, &w->k_proj[i]);
        read_qt(f, &w->v_proj[i]);
        read_qt(f, &w->o_proj[i]);
        read_qt(f, &w->gate_proj[i]);
        read_qt(f, &w->up_proj[i]);
        read_qt(f, &w->down_proj[i]);
    }

    fread(w->layer_scalars, sizeof(float), (size_t)p->n_layers, f);

    if (p->use_rope_freqs) {
        read_qt(f, &w->rope_freqs_full);
    }

    fclose(f);
    log_msg(stdout, "INFO: Quantized G4 loaded from %s\n", filepath);

    alloc_state_g4(&model->state, p, w, model->layer_types);

    return 0;
}

static void apply_rope(float *vec, float *cos, float *sin, int rotary_dim, int vec_dim, int pos) {
    if (rotary_dim <= 0) {
        return;
    }

    int half_rot = rotary_dim / 2;
    int cache_stride = vec_dim / 2;
    float *cos_row = cos + pos * cache_stride;
    float *sin_row = sin + pos * cache_stride;

    for (int i = 0; i < half_rot; i++) {
        float c = cos_row[i], sn = sin_row[i];
        float v0 = vec[i], v1 = vec[i + cache_stride];
        vec[i] = v0 * c - v1 * sn;
        vec[i + cache_stride] = v0 * sn + v1 * c;
    }
}

float *forward_g4(G4 *model, int token, int pos) {
    config_g4 *p = &model->config;
    weights_g4 *w = &model->weights;
    state_g4 *s = &model->state;
    float *x = s->x;
    int dim = p->dim;
    float eps = p->rms_norm_eps;
    float embed_scale = sqrtf((float)dim);

    if (token < 0 ||
            token >= p->vocab_size) {
        log_msg(stderr, "ERROR: token %d is outside vocabulary [0, %d)\n", token, p->vocab_size);
        exit(EXIT_FAILURE);
    }

    if (pos < 0 ||
            pos >= p->seq_len) {
        log_msg(stderr, "ERROR: position %d is outside KV cache [0, %d)\n", pos, p->seq_len);
        exit(EXIT_FAILURE);
    }

    dequantize_row(x, &w->embed_tokens_weight, token);

#pragma omp simd
    for (int i = 0; i < dim; i++) {
        x[i] *= embed_scale;
    }

    for (int l = 0; l < p->n_layers; l++) {
        int is_full = model->layer_types[l];
        int use_alternative_attention = is_full && p->attention_k_eq_v;
        int head_dim = is_full ? p->global_head_dim : p->head_dim;
        int kv_heads = use_alternative_attention ? p->n_global_kv_heads : p->n_kv_heads;
        int kv_dim = kv_heads * head_dim;
        int rotary_dim = is_full ? (int)(p->rope_partial_factor * p->global_head_dim) : p->head_dim;
        float *cos_cache = is_full ? s->cos_cache_full : s->cos_cache_sliding;
        float *sin_cache = is_full ? s->sin_cache_full : s->sin_cache_sliding;

        int layer_attn_out_dim = p->n_heads * head_dim;

        float *rms_in = (float *)w->rms_input_layernorm[l].data;
        float *rms_post_a = (float *)w->rms_post_attn_layernorm[l].data;
        float *rms_pre_f = (float *)w->rms_pre_ffn_layernorm[l].data;
        float *rms_post_f = (float *)w->rms_post_ffn_layernorm[l].data;
        float *rms_q = (float *)w->rms_q_norm[l].data;
        float *rms_k = (float *)w->rms_k_norm[l].data;

        rmsnorm_g4(s->xb, x, rms_in, dim, eps, 1);

        quantize_vec(&s->xq, s->xb, dim);
        matmul_qq(s->q, &s->xq, &w->q_proj[l]);
        matmul_qq(s->k_raw, &s->xq, &w->k_proj[l]);

        if (use_alternative_attention) {
            memcpy(s->v, s->k_raw, kv_dim * sizeof(float));
        }
        else {
            matmul_qq(s->v, &s->xq, &w->v_proj[l]);
        }

#pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float *qh = s->q + h * head_dim;
            rmsnorm_g4(qh, qh, rms_q, head_dim, eps, 1);
            if ((rotary_dim > 0) &&
                    cos_cache) {
                apply_rope(qh, cos_cache, sin_cache, rotary_dim, head_dim, pos);
            }
        }

#pragma omp parallel for
        for (int h = 0; h < kv_heads; h++) {
            float *kh = s->k_raw + h * head_dim;
            rmsnorm_g4(kh, kh, rms_k, head_dim, eps, 1);
            if ((rotary_dim > 0) &&
                    cos_cache) {
                apply_rope(kh, cos_cache, sin_cache, rotary_dim, head_dim, pos);
            }
        }

        memcpy(s->k, s->k_raw, kv_dim * sizeof(float));

#pragma omp parallel for
        for (int h = 0; h < kv_heads; h++) {
            rmsnorm_g4(s->v + h * head_dim, s->v + h * head_dim, NULL, head_dim, eps, 0);
        }

        memcpy(s->key_cache[l] + (long long)pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache[l] + (long long)pos * kv_dim, s->v, kv_dim * sizeof(float));

        int start_t = is_full ? 0 : fmax(0, pos - p->sliding_window + 1);

#pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_dim;
            float *att = s->att + h * p->seq_len;
            int kv_head = h / (p->n_heads / kv_heads);

            for (int t = 0; t < start_t; t++) {
                att[t] = -1e9f;
            }

            float attn_scale = 1.0f;
            for (int t = start_t; t <= pos; t++) {
                float *k = s->key_cache[l] + (long long)t * kv_dim + (long long)kv_head * head_dim;
                float score = 0.0f;

#pragma omp simd reduction(+ : score)
                for (int i = 0; i < head_dim; i++) {
                    score += q[i] * k[i];
                }
                att[t] = score * attn_scale;
            }
            softmax(att, pos + 1);

            float *out = s->hb + h * head_dim;
            memset(out, 0, head_dim * sizeof(float));
            for (int t = start_t; t <= pos; t++) {
                float *v = s->value_cache[l] + (long long)t * kv_dim + (long long)kv_head * head_dim;
                float a = att[t];

#pragma omp simd
                for (int i = 0; i < head_dim; i++) {
                    out[i] += a * v[i];
                }
            }
        }

        quantize_vec(&s->xq, s->hb, layer_attn_out_dim);
        matmul_qq(s->xb, &s->xq, &w->o_proj[l]);
        rmsnorm_g4(s->xb, s->xb, rms_post_a, dim, eps, 1);

#pragma omp simd
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }

        rmsnorm_g4(s->xb, x, rms_pre_f, dim, eps, 1);

        quantize_vec(&s->xq, s->xb, dim);
        matmul_qq(s->hb, &s->xq, &w->gate_proj[l]);
        matmul_qq(s->hb2, &s->xq, &w->up_proj[l]);

        int ffn_dim = w->gate_proj[l].rows;

#pragma omp parallel for
        for (int i = 0; i < ffn_dim; i++) {
            s->hb[i] = gelu(s->hb[i]) * s->hb2[i];
        }

        quantize_vec(&s->hq, s->hb, ffn_dim);
        matmul_qq(s->xb, &s->hq, &w->down_proj[l]);
        rmsnorm_g4(s->xb, s->xb, rms_post_f, dim, eps, 1);

#pragma omp simd
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }

        if (w->layer_scalars[l] != 1.0f) {
            float scale = w->layer_scalars[l];
#pragma omp simd
            for (int i = 0; i < dim; i++) {
                x[i] *= scale;
            }
        }
    }

    rmsnorm_g4(x, x, (float *)w->rms_final_norm.data, dim, eps, 1);

    matmul_qt(s->logits, s->x, &w->embed_tokens_weight);


    if (p->final_logit_softcapping > 0.0f) {
        float cap = p->final_logit_softcapping;
        float inv = 1.0f / cap;

#pragma omp parallel for
        for (int i = 0; i < p->vocab_size; i++) {
            s->logits[i] = tanhf(s->logits[i] * inv) * cap;
        }
    }
    return s->logits;
}

static float *forward_g4_wrap(void *model, int token, int pos) {
    return forward_g4((G4 *)model, token, pos);
}

static void free_g4_wrap(void *model) {
    free_g4((G4 *)model);
    free(model);
}

static model_iface *init_g4(const char *model_path, int seq_n_max, bool think_) {
    G4 *model = a_calloc(1 * sizeof(G4));
    if (load_quantized_g4(model_path, model, seq_n_max)) {
        free_g4(model);
        free(model);
        return NULL;
    }

    model->tokenizer.special_tokens = NULL;
    model->tokenizer.bos_token_id = 2;
    model->tokenizer.eos_token_id = 1;
    model->tokenizer.im_end_id = 106;

    model_iface *model_i = a_calloc(sizeof(model_iface));
    *model_i = (model_iface){ .model = model,
        .forward = forward_g4_wrap,
        .free_model = free_g4_wrap,
        .seq_n_max = seq_n_max ? seq_n_max : model->config.seq_len,
        .chat_template = think_ ? &CHAT_TEMPLATE_THINK_G4 : &CHAT_TEMPLATE_G4,
        .tokenizer = &model->tokenizer };
    return model_i;
}

int main(int argc, char *argv[]) {
    return common_main(argc, argv, init_g4, "dolen_g4");
}

