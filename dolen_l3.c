#include "dolen_l3_common.h"

static const chat_template CHAT_TEMPLATE_L3 = {
    .system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>",
    .main = "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    .end_turn = "<|eot_id|>",
};

static const chat_template CHAT_TEMPLATE_THINK_L3 = {
    .system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>",
    .main = "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    .end_turn = "<|eot_id|>",
};

int load_quantized_l3(const char *filepath, L3 *model, int seq_n_max) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        log_msg(stderr, "ERROR: Failed to open %s\n", filepath);
        return -1;
    }

    memset(model, 0, sizeof(L3));

    uint64_t magic;
    uint32_t version;

    if (fread(&magic, sizeof(magic), 1, f) != 1 || fread(&version, sizeof(version), 1, f) != 1) {
        log_msg(stderr, "ERROR: Failed to read header\n");
        fclose(f);
        return -1;
    }

    if (magic != MAGIC_L3) {
        log_msg(stderr, "ERROR: Invalid magic number\n");
        fclose(f);
        return -1;
    }

    if (version != 1) {
        log_msg(stderr, "ERROR: Unsupported version %u (expected 1)\n", version);
        fclose(f);
        return -1;
    }

    config_l3 *p = &model->config;
    if (fread(p, sizeof(config_l3), 1, f) != 1) {
        log_msg(stderr, "ERROR: Failed to read config\n");
        fclose(f);
        return -1;
    }

    if (tokenizer_read_from_file(f, p->vocab_size, &model->tokenizer)) {
        log_msg(stderr, "ERROR: Failed to read tokenizer\n");
        fclose(f);
        return -1;
    }

    if (seq_n_max) p->seq_len = seq_n_max;

    weights_l3 *w = &model->weights;

    w->rms_att_weight = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->rms_ffn_weight = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->wq = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->wk = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->wv = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->wo = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->w1 = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->w2 = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->w3 = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));

    if (!w->rms_att_weight || !w->rms_ffn_weight || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3) {
        log_msg(stderr, "ERROR: Alloc failed\n");
        fclose(f);
        return -1;
    }

    read_qt(f, &w->embed_tokens_weight);

    for (int i = 0; i < p->n_layers; i++) read_qt(f, &w->rms_att_weight[i]);

    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->wq[i]);
        read_qt(f, &w->wk[i]);
        read_qt(f, &w->wv[i]);
        read_qt(f, &w->wo[i]);
    }    

    for (int i = 0; i < p->n_layers; i++) read_qt(f, &w->rms_ffn_weight[i]);

    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->w1[i]);
        read_qt(f, &w->w2[i]);
        read_qt(f, &w->w3[i]);
    }

    read_qt(f, &w->rms_final_weight);

    if (!p->tie_word_embeddings) {
        read_qt(f, &w->wcls);
    } else {
        w->wcls = w->embed_tokens_weight;
    }

    fclose(f);
    log_msg(stdout, "INFO: Quantized L3 loaded from %s\n", filepath);

    alloc_state_l3(&model->state, p);
    return 0;
}

static void apply_rope(float *vec, float *cos, float *sin, int rotary_dim, int pos) {
    int half_rot = rotary_dim / 2;
    float *cos_row = cos + pos * half_rot;
    float *sin_row = sin + pos * half_rot;

    for (int i = 0; i < half_rot; i++) {
        float c = cos_row[i], sn = sin_row[i];
        float v0 = vec[i], v1 = vec[i + half_rot];
        vec[i] = v0 * c - v1 * sn;
        vec[i + half_rot] = v0 * sn + v1 * c;
    }
}

float *forward_l3(L3 *model, int token, int pos) {
    config_l3 *p = &model->config;
    weights_l3 *w = &model->weights;
    state_l3 *s = &model->state;
    float *x = s->x;
    int dim = p->dim;
    int head_size = p->head_dim;
    int kv_dim = p->n_kv_heads * head_size;
    int kv_mul = p->n_heads / p->n_kv_heads;
    float eps = p->rms_norm_eps;

    dequantize_row(x, &w->embed_tokens_weight, token);

    for (int l = 0; l < p->n_layers; l++) {
        float *rms_att = (float *)w->rms_att_weight[l].data;
        rmsnorm(s->xb, x, rms_att, dim, eps);

        quantize_vec(&s->xq, s->xb, dim);
        matmul_qq(s->q, &s->xq, &w->wq[l]);
        matmul_qq(s->k, &s->xq, &w->wk[l]);
        matmul_qq(s->v, &s->xq, &w->wv[l]);

        for (int h = 0; h < p->n_heads; h++) {
            apply_rope(s->q + h * head_size, s->cos_cache, s->sin_cache, head_size, pos);
        }
        for (int h = 0; h < p->n_kv_heads; h++) {
            apply_rope(s->k + h * head_size, s->cos_cache, s->sin_cache, head_size, pos);
        }

        memcpy(s->key_cache[l] + (long long)pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache[l] + (long long)pos * kv_dim, s->v, kv_dim * sizeof(float));

        float inv_sqrt_head = 1.0f / sqrtf((float)head_size);
#pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_size;
            float *att = s->att + h * p->seq_len;
            int kv_head = h / kv_mul;

            for (int t = 0; t <= pos; t++) {
                float *k = s->key_cache[l] + (long long)t * kv_dim + (long long)kv_head * head_size;
                float score = 0.0f;
#pragma omp simd reduction(+ : score)
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                att[t] = score * inv_sqrt_head;
            }
            softmax(att, pos + 1);

            float *out = s->xb + h * head_size; 
            memset(out, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v = s->value_cache[l] + (long long)t * kv_dim + (long long)kv_head * head_size;
                float a = att[t];
#pragma omp simd
                for (int i = 0; i < head_size; i++) {
                    out[i] += a * v[i];
                }
            }
        }

        quantize_vec(&s->xq, s->xb, p->n_heads * head_size);
        matmul_qq(s->xb2, &s->xq, &w->wo[l]);

        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        float *rms_ffn = (float *)w->rms_ffn_weight[l].data;
        rmsnorm(s->xb, x, rms_ffn, dim, eps);

        quantize_vec(&s->xq, s->xb, dim);
        matmul_qq(s->hb, &s->xq, &w->w1[l]);
        matmul_qq(s->hb2, &s->xq, &w->w3[l]);

#pragma omp parallel for
        for (int i = 0; i < p->hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val))); // Swish / SiLU Activation
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        quantize_vec(&s->hq, s->hb, p->hidden_dim);
        matmul_qq(s->xb, &s->hq, &w->w2[l]);

        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    rmsnorm(x, x, (float *)w->rms_final_weight.data, dim, eps);

    if (p->tie_word_embeddings) {
        matmul_qt(s->logits, x, &w->embed_tokens_weight);
    } else {
        matmul_qt(s->logits, x, &w->wcls);
    }

    return s->logits;
}

static float *forward_l3_wrap(void *model, int token, int pos) {
    return forward_l3((L3 *)model, token, pos);
}

static void free_l3_wrap(void *model) {
    free_l3((L3 *)model);
    free(model);
}

static model_iface *init_l3(const char *model_path, int seq_n_max, bool think_) {
    L3 *model = a_calloc(sizeof(L3));
    if (load_quantized_l3(model_path, model, seq_n_max)) {
        free_l3(model);
        free(model);
        return NULL;
    }

    // Llama 3 Token mappings
    model->tokenizer.bos_token_id = 128000; // <|begin_of_text|>
    model->tokenizer.eos_token_id = 128001; // <|end_of_text|>
    model->tokenizer.im_end_id = 128009;    // <|eot_id|>

    model_iface *model_i = a_calloc(sizeof(model_iface));
    *model_i = (model_iface){
        .model = model,
        .forward = forward_l3_wrap,
        .free_model = free_l3_wrap,
        .seq_n_max = seq_n_max ? seq_n_max : model->config.seq_len,
        .chat_template = think_ ? &CHAT_TEMPLATE_THINK_L3 : &CHAT_TEMPLATE_L3,
        .tokenizer = &model->tokenizer
    };
    return model_i;
}

int main(int argc, char *argv[]) {
    return common_main(argc, argv, init_l3, "dolen_l3");
}
