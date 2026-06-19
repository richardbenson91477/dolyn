#include "dolen_ig4_1_common.h"

int load_quantized_ig4_1(const char *filepath, IG4_1 *model_ig4_1, int seq_n_max) {
    FILE *f = fopen(filepath, "rb");
    if (! f) {
        log_msg(stderr, "ERROR: Failed to open %s for reading\n", filepath);
        return -1;
    }

    memset(model_ig4_1, 0, sizeof(IG4_1));

    uint32_t magic, version;
    if ((fread(&magic, sizeof(uint32_t), 1, f) != 1) || (fread(&version, sizeof(uint32_t), 1, f) != 1)) {
        log_msg(stderr, "ERROR: Failed to read header from %s\n", filepath);
        fclose(f);
        return -1;
    }

    if (magic != 0x31344749) { // 'IG4_1'
        log_msg(stderr, "ERROR: Invalid magic number in %s\n", filepath);
        fclose(f);
        return -1;
    }

    if (version != 1) {
        log_msg(stderr, "ERROR: Unsupported version %d in %s\n", version, filepath);
        fclose(f);
        return -1;
    }

    if (fread(&model_ig4_1->config, sizeof(config_ig4_1), 1, f) != 1) {
        log_msg(stderr, "ERROR: Failed to read config from %s\n", filepath);
        fclose(f);
        return -1;
    }

    config_ig4_1 *p = &model_ig4_1->config;
    weights_ig4_1 *w = &model_ig4_1->weights;

    log_msg(stderr,
            "INFO: Granite config: dim=%d heads=%d kv_heads=%d head_dim=%d "
            "layers=%d seq_len=%d rope_theta=%.9g attn_mult=%.9g "
            "emb_mult=%.9g residual_mult=%.9g logits_scaling=%.9g\n",
            p->dim, p->n_heads, p->n_kv_heads, p->d_head, p->n_layer, p->seq_len, p->rope_theta,
            p->attention_multiplier, p->embedding_multiplier, p->residual_multiplier, p->logits_scaling);

    if (! (p->rope_theta > 1.0f)) {
        log_msg(stderr, "ERROR: Invalid rope_theta %.9g in quantized model\n", p->rope_theta);
        fclose(f);
        return -1;
    }

    if (seq_n_max) {
        p->seq_len = seq_n_max;
    }

    read_qt(f, &w->token_embedding_table);

    w->rms_att_weight = (float *)a_calloc((size_t)p->n_layer * p->dim * sizeof(float));
    if (fread(w->rms_att_weight, sizeof(float), (size_t)p->n_layer * p->dim, f) != (size_t)p->n_layer * p->dim) {
        log_msg(stderr, "ERROR: Failed to read rms_att_weight\n");
        fclose(f);
        return -1;
    }

    w->wq = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->wk = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->wv = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->wo = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));

    for (int i = 0; i < p->n_layer; i++) {
        read_qt(f, &w->wq[i]);
        read_qt(f, &w->wk[i]);
        read_qt(f, &w->wv[i]);
        read_qt(f, &w->wo[i]);
    }

    w->rms_ffn_weight = (float *)a_calloc((size_t)p->n_layer * p->dim * sizeof(float));
    if (fread(w->rms_ffn_weight, sizeof(float), (size_t)p->n_layer * p->dim, f) != (size_t)p->n_layer * p->dim) {
        log_msg(stderr, "ERROR: Failed to read rms_ffn_weight\n");
        fclose(f);
        return -1;
    }

    w->w1 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->w2 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->w3 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    for (int i = 0; i < p->n_layer; i++) {
        read_qt(f, &w->w1[i]);
        read_qt(f, &w->w2[i]);
        read_qt(f, &w->w3[i]);
    }

    w->rms_final_weight = (float *)a_calloc((size_t)p->dim * sizeof(float));
    if (fread(w->rms_final_weight, sizeof(float), (size_t)p->dim, f) != (size_t)p->dim) {
        log_msg(stderr, "ERROR: Failed to read rms_final_weight\n");
        fclose(f);
        return -1;
    }

    if (! p->tie_word_embeddings) {
        read_qt(f, &w->wcls);
    } else {
        w->wcls = w->token_embedding_table;
    }

    fclose(f);
    log_msg(stderr, "INFO: Quantized model loaded from %s\n", filepath);
    alloc_state_ig4_1(&(model_ig4_1->state), &(model_ig4_1->config));
    return 0;
}

void forward_ig4_1_attention_layer(IG4_1 *model_ig4_1, int l, int pos) {
    config_ig4_1 *p = &model_ig4_1->config;
    weights_ig4_1 *w = &model_ig4_1->weights;
    state_ig4_1 *s = &model_ig4_1->state;
    float *x = s->x;
    int dim = p->dim;
    int head_size = p->d_head > 0 ? p->d_head : dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int attn_out_dim = p->n_heads * head_size;
    int kv_mul = p->n_heads / p->n_kv_heads;
    long long loff = (long long)l * p->seq_len * kv_dim;
    float eps = p->rms_norm_eps;
    float *key_cache_row = s->key_cache + loff + pos * kv_dim;
    float *value_cache_row = s->value_cache + loff + pos * kv_dim;
    float *rms_att_weight = w->rms_att_weight + (long long)l * dim;

    rmsnorm(s->xb, x, rms_att_weight, dim, eps);

    quantize_vec(&s->xq, s->xb, dim);
    matmul_qq(s->q, &s->xq, &w->wq[l]);
    matmul_qq(s->k, &s->xq, &w->wk[l]);
    matmul_qq(s->v, &s->xq, &w->wv[l]);

    int rotary_dim = head_size;

    if (s->cos_cache) {
        float *cos_row = s->cos_cache + pos * rotary_dim;
        float *sin_row = s->sin_cache + pos * rotary_dim;

#pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_size;
            for (int i = 0; i < rotary_dim / 2; i++) {
                float c = cos_row[i], sn = sin_row[i];
                float q0 = q[i], q1 = q[i + rotary_dim / 2];
                q[i] = q0 * c - q1 * sn;
                q[i + rotary_dim / 2] = q0 * sn + q1 * c;
            }
        }

#pragma omp parallel for
        for (int h = 0; h < p->n_kv_heads; h++) {
            float *k = s->k + h * head_size;
            for (int i = 0; i < rotary_dim / 2; i++) {
                float c = cos_row[i], sn = sin_row[i];
                float k0 = k[i], k1 = k[i + rotary_dim / 2];
                k[i] = k0 * c - k1 * sn;
                k[i + rotary_dim / 2] = k0 * sn + k1 * c;
            }
        }
    }

    memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
    memcpy(value_cache_row, s->v, kv_dim * sizeof(float));

    float attn_scale = p->attention_multiplier;
    if (attn_scale == 0.0f) {
        attn_scale = 1.0f / sqrtf((float)head_size);
    }

#pragma omp parallel for
    for (int h = 0; h < p->n_heads; h++) {
        float *q = s->q + h * head_size;
        float *att = s->att + h * p->seq_len;

        for (int t = 0; t <= pos; t++) {
            float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float score = 0.0f;

#pragma omp simd reduction(+ : score)
            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
            att[t] = score * attn_scale;
        }

        softmax(att, pos + 1);

        float *xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));

        for (int t = 0; t <= pos; t++) {
            float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float a = att[t];

#pragma omp simd
            for (int i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
            }
        }
    }

    quantize_vec(&s->xq, s->xb, attn_out_dim);
    matmul_qq(s->xb2, &s->xq, &w->wo[l]);

    float res_mult = p->residual_multiplier;
    for (int i = 0; i < dim; i++) {
        x[i] += s->xb2[i] * res_mult;
    }
}

void forward_ig4_1_mlp_layer(IG4_1 *model_ig4_1, int l) {
    config_ig4_1 *p = &model_ig4_1->config;
    weights_ig4_1 *w = &model_ig4_1->weights;
    state_ig4_1 *s = &model_ig4_1->state;
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim = p->n_mlp;
    float eps = p->rms_norm_eps;
    float *rms_ffn_weight = w->rms_ffn_weight + (long long)l * dim;

    rmsnorm(s->xb, x, rms_ffn_weight, dim, eps);

    quantize_vec(&s->xq, s->xb, dim);
    matmul_qq(s->hb, &s->xq, &w->w1[l]);
    matmul_qq(s->hb2, &s->xq, &w->w3[l]);

#pragma omp parallel for
    for (int i = 0; i < hidden_dim; i++) {
        float val = s->hb[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= s->hb2[i];
        s->hb[i] = val;
    }

    quantize_vec(&s->hq, s->hb, hidden_dim);
    matmul_qq(s->xb, &s->hq, &w->w2[l]);

    float res_mult = p->residual_multiplier;
    for (int i = 0; i < dim; i++) {
        x[i] += s->xb[i] * res_mult;
    }
}

float *forward_ig4_1(IG4_1 *model_ig4_1, int token, int pos) {
    config_ig4_1 *p = &model_ig4_1->config;
    weights_ig4_1 *w = &model_ig4_1->weights;
    state_ig4_1 *s = &model_ig4_1->state;
    float *x = s->x;
    int dim = p->dim;

    dequantize_row(x, &w->token_embedding_table, token);

    float emb_mult = p->embedding_multiplier;
    if (emb_mult != 1.0f) {
#pragma omp simd
        for (int i = 0; i < dim; i++) {
            x[i] *= emb_mult;
        }
    }

    for (int l = 0; l < p->n_layer; l++) {
        forward_ig4_1_attention_layer(model_ig4_1, l, pos);
        forward_ig4_1_mlp_layer(model_ig4_1, l);
    }

    rmsnorm(x, x, w->rms_final_weight, dim, p->rms_norm_eps);

    if (p->tie_word_embeddings)
        matmul_qt(s->logits, x, &w->token_embedding_table);
    else
        matmul_qt(s->logits, x, &w->wcls);

    float logit_scale = p->logits_scaling;
    if (logit_scale != 0.0f && logit_scale != 1.0f) {
#pragma omp simd
        for (int i = 0; i < p->vocab_size; i++) {
            s->logits[i] /= logit_scale;
        }
    }

    return s->logits;
}

static float *forward_ig4_1_wrap(void *model, int token, int pos) {
    return forward_ig4_1((IG4_1 *)model, token, pos);
}

static void free_ig4_1_wrap(void *model) {
    free_ig4_1((IG4_1 *)model);
    free(model);
}

static const chat_template CHAT_TEMPLATE_IG4_1 = {
    .system = "<|start_of_role|>system<|end_of_role|>%s<|end_of_text|>\n",
    .main = "<|start_of_role|>user<|end_of_role|>%s<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>",
    .end_turn = "<|end_of_text|>\n",
};

static token_map SPECIAL_TOKENS_IG4_1[] = {
    { "<|end_of_text|>", 100257 },
    { "<|start_of_role|>", 100264 },
    { "<|end_of_role|>", 100265 },
    { NULL, 0 },
};

static model_iface *init_ig4_1(const char *model_path, int seq_n_max, bool _think) {
    IG4_1 *model = a_calloc(1 * sizeof(IG4_1));

    if (_think) {
        log_msg(stderr, "WARNING: Think mode requested but not supported.\n");
    }

    if (load_quantized_ig4_1(model_path, model, seq_n_max)) {
        free_ig4_1(model);
        free(model);
        return NULL;
    }

    model_iface *model_i = a_calloc(sizeof(model_iface));
    *model_i = (model_iface){ .model = model,
        .forward = forward_ig4_1_wrap,
        .free_model = free_ig4_1_wrap,
        .seq_n_max = (seq_n_max != 0) ? seq_n_max : model->config.seq_len,
        .vocab_size = model->config.vocab_size,
        .bos_token_id = 0,
        .eos_token_id = 100257,
        .im_end_id = 100257,
        .special_tokens = SPECIAL_TOKENS_IG4_1,
        .chat_template = &CHAT_TEMPLATE_IG4_1 };
    return model_i;
}

int main(int argc, char *argv[]) {
    return common_main(argc, argv, init_ig4_1, "dolen_ig4_1");
}
