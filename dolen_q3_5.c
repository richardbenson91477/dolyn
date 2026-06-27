#include "dolen_q3_5_common.h"


static token_map SPECIAL_TOKENS_Q3_5[] = {
    {"<|endoftext|\x3e", 248044},
    {"<|im_start|\x3e", 248045},
    {"<|im_end|\x3e", 248046},
    {"<|object_ref_start|\x3e", 248047},
    {"<|object_ref_end|\x3e", 248048},
    {"<|box_start|\x3e", 248049},
    {"<|box_end|\x3e", 248050},
    {"<|quad_start|\x3e", 248051},
    {"<|quad_end|\x3e", 248052},
    {"<|vision_start|\x3e", 248053},
    {"<|vision_end|\x3e", 248054},
    {"<|vision_pad|\x3e", 24805},
    {"<|image_pad|\x3e", 248056},
    {"<|video_pad|\x3e", 248057},
    {NULL, 0} 
};

static const chat_template CHAT_TEMPLATE_Q3_5 = {
    .system = "<|im_start|\x3e" "system\n%s<|im_end|\x3e" "\n",
    .main = "<|im_start|\x3e" "user\n%s<|im_end|\x3e" "\n"
            "<|im_start|\x3e" "assistant\n"
            "<think\x3e" "\n\n</think\x3e" "\n\n",
    .end_turn = "<|im_end|\x3e" "\n",
};

static const chat_template CHAT_TEMPLATE_THINK_Q3_5 = {
    .system = "<|im_start|\x3e" "system\n%s<|im_end|\x3e" "\n",
    .main = "<|im_start|\x3e" "user\n%s<|im_end|\x3e" "\n"
            "<|im_start|\x3e" "assistant\n"
            "<think\x3e",
    .end_turn = "<|im_end|\x3e" "\n",
};


int load_quantized_q3_5(const char *filepath, Q3_5 *model_q3_5, int seq_n_max) {
    FILE *f = fopen(filepath, "rb");
    if (! f) {
        log_msg(stderr, "ERROR: Failed to open %s for reading\n", filepath);
        return -1;
    }

    memset(model_q3_5, 0, sizeof(Q3_5));

    uint64_t magic;
    uint32_t version;

    if ((fread(&magic, sizeof(uint64_t), 1, f) != 1) ||
            (fread(&version, sizeof(uint32_t), 1, f) != 1)) {
        log_msg(stderr, "ERROR: Failed to read header from %s\n", filepath);
        fclose(f);
        return -1;
    }

    if (magic != MAGIC_Q3_5) {
        log_msg(stderr, "ERROR: Invalid magic number in %s\n", filepath);
        fclose(f);
        return -1;
    }

    if (version != 2) {
        log_msg(stderr, "ERROR: Unsupported version %d in %s\n", version, filepath);
        fclose(f);
        return -1;
    }

    if (fread(&model_q3_5->config, sizeof(config_q3_5), 1, f) != 1) {
        log_msg(stderr, "ERROR: Failed to read config from %s\n", filepath);
        fclose(f);
        return -1;
    }

    if (tokenizer_read_from_file(f, model_q3_5->config.vocab_size, &model_q3_5->tokenizer)) {
        log_msg(stderr, "ERROR: Failed to read tokenizer from %s\n", filepath);
        fclose(f);
        return -1;
    }

    config_q3_5 *p = &model_q3_5->config;
    weights_q3_5 *w = &model_q3_5->weights;

    if (seq_n_max) {
        p->seq_len = seq_n_max;
    }

    model_q3_5->layer_types = (int *)a_calloc((size_t)p->n_layer * sizeof(int));
    model_q3_5->attn_layer_indices = (int *)a_calloc((size_t)p->n_layer * sizeof(int));
    model_q3_5->deltanet_layer_indices = (int *)a_calloc((size_t)p->n_layer * sizeof(int));

    if ((! model_q3_5->layer_types) ||
            (! model_q3_5->attn_layer_indices) ||
            (! model_q3_5->deltanet_layer_indices)) {
        log_msg(stderr, "ERROR: Failed to allocate memory for layer indices\n");
        fclose(f);
        return -1;
    }

    if (fread(model_q3_5->layer_types, sizeof(int), (size_t)p->n_layer, f) != (size_t)p->n_layer) {
        log_msg(stderr, "ERROR: Failed to read layer_types from %s\n", filepath);
        fclose(f);
        return -1;
    }

    int la = 0, ld = 0;
    for (int i = 0; i < p->n_layer; i++) {
        if (model_q3_5->layer_types[i] == 1) {
            model_q3_5->deltanet_layer_indices[i] = ld++;
        }
        else {
            model_q3_5->attn_layer_indices[i] = la++;
        }
    }

    w->rms_att_weight = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->q_norm = (qtensor *)a_calloc((size_t)p->n_full_attn_layers * sizeof(qtensor));
    w->k_norm = (qtensor *)a_calloc((size_t)p->n_full_attn_layers * sizeof(qtensor));
    w->rms_ffn_weight = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));

    if ((! w->rms_att_weight) ||
            (! w->q_norm) ||
            (! w->k_norm) ||
            (! w->rms_ffn_weight)) {
        log_msg(stderr, "ERROR: Failed to allocate norm weights\n");
        fclose(f);
        return -1;
    }

    read_qt(f, &w->embed_tokens_weight);

    for (int i = 0; i < p->n_layer; i++) {
        read_qt(f, &w->rms_att_weight[i]);
    }

    w->wq = (qtensor *)a_calloc((size_t)p->n_full_attn_layers * sizeof(qtensor));
    w->wk = (qtensor *)a_calloc((size_t)p->n_full_attn_layers * sizeof(qtensor));
    w->wv = (qtensor *)a_calloc((size_t)p->n_full_attn_layers * sizeof(qtensor));
    w->wo = (qtensor *)a_calloc((size_t)p->n_full_attn_layers * sizeof(qtensor));

    if ((! w->wq) ||
            (! w->wk) ||
            (! w->wv) ||
            (! w->wo)) {
        log_msg(stderr, "ERROR: Failed to allocate attention weights\n");
        fclose(f);
        return -1;
    }

    for (int i = 0; i < p->n_full_attn_layers; i++) {
        read_qt(f, &w->wq[i]);
        read_qt(f, &w->wk[i]);
        read_qt(f, &w->wv[i]);
        read_qt(f, &w->wo[i]);
    }

    for (int i = 0; i < p->n_full_attn_layers; i++) {
        read_qt(f, &w->q_norm[i]);
    }
    for (int i = 0; i < p->n_full_attn_layers; i++) {
        read_qt(f, &w->k_norm[i]);
    }

    if (p->n_linear_attn_layers > 0) {
        w->in_proj_qkv = (qtensor *)a_calloc((size_t)p->n_linear_attn_layers * sizeof(qtensor));
        w->in_proj_z = (qtensor *)a_calloc((size_t)p->n_linear_attn_layers * sizeof(qtensor));
        w->in_proj_b = (qtensor *)a_calloc((size_t)p->n_linear_attn_layers * sizeof(qtensor));
        w->in_proj_a = (qtensor *)a_calloc((size_t)p->n_linear_attn_layers * sizeof(qtensor));
        w->conv1d_weight = (qtensor *)a_calloc((size_t)p->n_linear_attn_layers * sizeof(qtensor));
        w->dt_bias = (qtensor *)a_calloc((size_t)p->n_linear_attn_layers * sizeof(qtensor));
        w->A_log = (qtensor *)a_calloc((size_t)p->n_linear_attn_layers * sizeof(qtensor));
        w->linear_norm = (qtensor *)a_calloc((size_t)p->n_linear_attn_layers * sizeof(qtensor));
        w->out_proj = (qtensor *)a_calloc((size_t)p->n_linear_attn_layers * sizeof(qtensor));

        if ((! w->in_proj_qkv) ||
                (! w->in_proj_z) ||
                (! w->in_proj_b) ||
                (! w->in_proj_a) ||
                (! w->conv1d_weight) ||
                (! w->dt_bias) ||
                (! w->A_log) ||
                (! w->linear_norm) ||
                (! w->out_proj)) {
            log_msg(stderr, "ERROR: Failed to allocate linear attention weights\n");
            fclose(f);
            return -1;
        }

        for (int i = 0; i < p->n_linear_attn_layers; i++) {
            read_qt(f, &w->in_proj_qkv[i]);
            read_qt(f, &w->in_proj_z[i]);
        }
        for (int i = 0; i < p->n_linear_attn_layers; i++) {
            read_qt(f, &w->in_proj_b[i]);
        }
        for (int i = 0; i < p->n_linear_attn_layers; i++) {
            read_qt(f, &w->in_proj_a[i]);
        }
        for (int i = 0; i < p->n_linear_attn_layers; i++) {
            read_qt(f, &w->conv1d_weight[i]);
        }
        for (int i = 0; i < p->n_linear_attn_layers; i++) {
            read_qt(f, &w->dt_bias[i]);
        }
        for (int i = 0; i < p->n_linear_attn_layers; i++) {
            read_qt(f, &w->A_log[i]);
        }
        for (int i = 0; i < p->n_linear_attn_layers; i++) {
            read_qt(f, &w->linear_norm[i]);
        }
        for (int i = 0; i < p->n_linear_attn_layers; i++) {
            read_qt(f, &w->out_proj[i]);
        }
    }

    for (int i = 0; i < p->n_layer; i++) {
        read_qt(f, &w->rms_ffn_weight[i]);
    }

    w->w1 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->w2 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    w->w3 = (qtensor *)a_calloc((size_t)p->n_layer * sizeof(qtensor));
    if ((! w->w1) ||
            (! w->w2) ||
            (! w->w3)) {
        log_msg(stderr, "ERROR: Failed to allocate MLP weights\n");
        fclose(f);
        return -1;
    }
    for (int i = 0; i < p->n_layer; i++) {
        read_qt(f, &w->w1[i]);
        read_qt(f, &w->w2[i]);
        read_qt(f, &w->w3[i]);
    }

    read_qt(f, &w->rms_final_weight);

    if (! p->tie_word_embeddings) {
        read_qt(f, &w->wcls);
    }
    else {
        w->wcls = w->embed_tokens_weight;
    }

    fclose(f);
    log_msg(stdout, "INFO: Quantized model loaded from %s\n", filepath);

    alloc_state_q3_5(&(model_q3_5->state), &(model_q3_5->config));

    return 0;
}

void forward_q3_5_attention_layer(Q3_5 *model_q3_5, int l, int la, int pos) {
    config_q3_5 *p = &model_q3_5->config;
    weights_q3_5 *w = &model_q3_5->weights;
    state_q3_5 *s = &model_q3_5->state;
    float *x = s->x;
    int dim = p->dim;
    int head_size = p->d_head > 0 ? p->d_head : dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int attn_out_dim = p->n_heads * head_size;
    int kv_mul = p->n_heads / p->n_kv_heads;
    long long loff = (long long)la * p->seq_len * kv_dim;
    float eps = p->rms_norm_eps;
    float *key_cache_row = s->key_cache + loff + pos * kv_dim;
    float *value_cache_row = s->value_cache + loff + pos * kv_dim;
    float *rms_att_weight = (float *)w->rms_att_weight[l].data;
    float *q_norm = (float *)w->q_norm[la].data;
    float *k_norm = (float *)w->k_norm[la].data;

    rmsnorm_gemma(s->xb, x, rms_att_weight, dim, eps);

    quantize_vec(&s->xq, s->xb, dim);
    matmul_qq(s->q, &s->xq, &w->wq[la]);
    matmul_qq(s->k, &s->xq, &w->wk[la]);
    matmul_qq(s->v, &s->xq, &w->wv[la]);

    for (int h = 0; h < p->n_heads; h++) {
        float *q_ptr = s->q + h * head_size;
        float *gate_ptr = s->gate + h * head_size;
        for (int i = 0; i < head_size; i++) {
            q_ptr[i] = s->q[h * head_size * 2 + i];
            gate_ptr[i] = s->q[h * head_size * 2 + head_size + i];
        }
    }

#pragma omp parallel for
    for (int h = 0; h < p->n_heads; h++) {
        float *q_ptr = s->q + h * head_size;
        rmsnorm_gemma(q_ptr, q_ptr, q_norm, head_size, eps);
    }

#pragma omp parallel for
    for (int h = 0; h < p->n_kv_heads; h++) {
        float *k_ptr = s->k + h * head_size;
        rmsnorm_gemma(k_ptr, k_ptr, k_norm, head_size, eps);
    }

    int rotary_partial = (int)((float)head_size * p->rope_partial_rotary_factor);

    if ((rotary_partial > 0) &&
            s->cos_cache) {
        float *cos_row = s->cos_cache + pos * rotary_partial;
        float *sin_row = s->sin_cache + pos * rotary_partial;

#pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_size;
            for (int i = 0; i < rotary_partial; i++) {
                float c = cos_row[i], sn = sin_row[i];
                float q0 = q[i], q1 = q[i + rotary_partial];
                q[i] = q0 * c - q1 * sn;
                q[i + rotary_partial] = q0 * sn + q1 * c;
            }
        }
#pragma omp parallel for
        for (int h = 0; h < p->n_kv_heads; h++) {
            float *k = s->k + h * head_size;
            for (int i = 0; i < rotary_partial; i++) {
                float c = cos_row[i], sn = sin_row[i];
                float k0 = k[i], k1 = k[i + rotary_partial];
                k[i] = k0 * c - k1 * sn;
                k[i + rotary_partial] = k0 * sn + k1 * c;
            }
        }
    }

    memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
    memcpy(value_cache_row, s->v, kv_dim * sizeof(float));

    float inv_sqrt_head = 1.0f / sqrtf((float)head_size);

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
            att[t] = score * inv_sqrt_head;
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

        float *gate_ptr = s->gate + h * head_size;

#pragma omp simd
        for (int i = 0; i < head_size; i++) {
            xb[i] *= sigmoid(gate_ptr[i]);
        }
    }

    quantize_vec(&s->xq, s->xb, attn_out_dim);
    matmul_qq(s->xb2, &s->xq, &w->wo[la]);
    for (int i = 0; i < dim; i++) {
        x[i] += s->xb2[i];
    }
}

void forward_q3_5_linear_attention_layer(Q3_5 *model_q3_5, int l, int ld, int pos) {
    config_q3_5 *p = &model_q3_5->config;
    weights_q3_5 *w = &model_q3_5->weights;
    state_q3_5 *s = &model_q3_5->state;
    float *x = s->x;
    int dim = p->dim;
    float eps = p->rms_norm_eps;
    int n_k_heads = p->n_linear_k_heads;
    int n_v_heads = p->n_linear_v_heads;
    int d_k = p->d_linear_k;
    int d_v = p->d_linear_v;
    int key_dim = n_k_heads * d_k;
    int value_dim = n_v_heads * d_v;
    int conv_dim = key_dim * 2 + value_dim;
    int conv_kernel = p->linear_conv_kernel;
    float *rms_att_weight = (float *)w->rms_att_weight[l].data;
    float *in_proj_b = (float *)w->in_proj_b[ld].data;
    float *in_proj_a = (float *)w->in_proj_a[ld].data;
    float *conv1d_weight = (float *)w->conv1d_weight[ld].data;
    float *dt_bias = (float *)w->dt_bias[ld].data;
    float *A_log = (float *)w->A_log[ld].data;
    float *linear_norm = (float *)w->linear_norm[ld].data;
    float *conv_state = s->conv_state + (long long)ld * conv_dim * conv_kernel;
    float *S = s->S + (long long)ld * n_v_heads * d_k * d_v;

    if (! pos) {
        memset(conv_state, 0, (size_t)conv_dim * conv_kernel * sizeof(float));
        memset(S, 0, (size_t)n_v_heads * d_k * d_v * sizeof(float));
    }

    rmsnorm_gemma(s->xb, x, rms_att_weight, dim, eps);

    quantize_vec(&s->xq, s->xb, dim);
    matmul_qq(s->qkv, &s->xq, &w->in_proj_qkv[ld]);
    matmul_qq(s->z, &s->xq, &w->in_proj_z[ld]);

#pragma omp parallel for
    for (int i = 0; i < n_v_heads; i++) {
        s->beta[i] = sigmoid(matmul_scalar(s->xb, in_proj_b + i * dim, dim));
    }

#pragma omp parallel for
    for (int i = 0; i < n_v_heads; i++) {
        float a_val = matmul_scalar(s->xb, in_proj_a + i * dim, dim);
        float A = -expf(A_log[i]);
        s->g[i] = A * softplus(a_val + dt_bias[i]);
    }

#pragma omp parallel for
    for (int i = 0; i < conv_dim; i++) {
        for (int j = 0; j < conv_kernel - 1; j++) {
            conv_state[i * conv_kernel + j] = conv_state[i * conv_kernel + j + 1];
        }
        conv_state[i * conv_kernel + conv_kernel - 1] = s->qkv[i];
    }

    float *qkv_conv = s->qkv;
#pragma omp parallel for
    for (int i = 0; i < conv_dim; i++) {
        float val = 0.0f;
        for (int j = 0; j < conv_kernel; j++) {
            val += conv_state[i * conv_kernel + j] * conv1d_weight[i * conv_kernel + j];
        }
        qkv_conv[i] = silu(val);
    }

    float *q = qkv_conv;
    float *k = qkv_conv + key_dim;
    float *v = qkv_conv + key_dim * 2;

    float scale = 1.0f / sqrtf((float)d_k);
#pragma omp parallel for
    for (int h = 0; h < n_k_heads; h++) {
        float *k_h = k + h * d_k;
        float *q_h = q + h * d_k;
        l2norm(k_h, d_k);
        l2norm(q_h, d_k);
        for (int i = 0; i < d_k; i++) {
            q_h[i] *= scale;
        }
    }

    int r = (n_v_heads > n_k_heads) ? n_v_heads / n_k_heads : 1;

#pragma omp parallel for
    for (int h = 0; h < n_v_heads; h++) {
        float g_t = expf(s->g[h]);
        float beta_t = s->beta[h];

        float *S_h = S + h * d_k * d_v;
        float *q_h = q + (h / r) * d_k;
        float *k_h = k + (h / r) * d_k;
        float *v_h = v + h * d_v;

        for (int i = 0; i < d_k * d_v; i++) {
            S_h[i] *= g_t;
        }

        float *delta = s->delta_S + h * d_v;
        for (int j = 0; j < d_v; j++) {
            float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
            for (int i = 0; i < d_k; i++) {
                dot += S_h[i * d_v + j] * k_h[i];
            }
            delta[j] = (v_h[j] - dot) * beta_t;
        }

        for (int i = 0; i < d_k; i++) {
#pragma omp simd
            for (int j = 0; j < d_v; j++) {
                S_h[i * d_v + j] += k_h[i] * delta[j];
            }
        }

        float *out_h = s->linear_out + h * d_v;
        for (int j = 0; j < d_v; j++) {
            float val = 0.0f;
#pragma omp simd reduction(+ : val)
            for (int i = 0; i < d_k; i++) {
                val += S_h[i * d_v + j] * q_h[i];
            }
            out_h[j] = val;
        }
    }

    rmsnorm_gated(s->linear_out, s->linear_out, s->z, linear_norm, n_v_heads, d_v, eps);

    quantize_vec(&s->hq, s->linear_out, value_dim);
    matmul_qq(s->xb, &s->hq, &w->out_proj[ld]);
    for (int i = 0; i < dim; i++) {
        x[i] += s->xb[i];
    }
}

void forward_q3_5_mlp_layer(Q3_5 *model_q3_5, int l) {
    config_q3_5 *p = &model_q3_5->config;
    weights_q3_5 *w = &model_q3_5->weights;
    state_q3_5 *s = &model_q3_5->state;
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim = p->n_mlp;
    float eps = p->rms_norm_eps;
    float *rms_ffn_weight = (float *)w->rms_ffn_weight[l].data;

    rmsnorm_gemma(s->xb, x, rms_ffn_weight, dim, eps);

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
    for (int i = 0; i < dim; i++) {
        x[i] += s->xb[i];
    }
}

float *forward_q3_5(Q3_5 *model_q3_5, int token, int pos) {
    config_q3_5 *p = &model_q3_5->config;
    weights_q3_5 *w = &model_q3_5->weights;
    state_q3_5 *s = &model_q3_5->state;
    float *x = s->x;
    int dim = p->dim;

    dequantize_row(x, &w->embed_tokens_weight, token);

    for (int l = 0; l < p->n_layer; l++) {
        if (model_q3_5->layer_types[l] == 1) {
            forward_q3_5_linear_attention_layer(model_q3_5, l, model_q3_5->deltanet_layer_indices[l], pos);
        }
        else {
            forward_q3_5_attention_layer(model_q3_5, l, model_q3_5->attn_layer_indices[l], pos);
        }
        forward_q3_5_mlp_layer(model_q3_5, l);
    }

    rmsnorm_gemma(x, x, (float *)w->rms_final_weight.data, dim, p->rms_norm_eps);

    if (p->tie_word_embeddings) {
        matmul_qt(s->logits, x, &w->embed_tokens_weight);
    }
    else {
        matmul_qt(s->logits, x, &w->wcls);
    }

    return s->logits;
}

static float *forward_q3_5_wrap(void *model, int token, int pos) {
    return forward_q3_5((Q3_5 *)model, token, pos);
}

static void free_q3_5_wrap(void *model) {
    free_q3_5((Q3_5 *)model);
    free(model);
}

model_iface *init_q3_5(const char *model_path, int seq_n_max, bool think_) {
    Q3_5 *model = a_calloc(1 * sizeof(Q3_5));

    if (load_quantized_q3_5(model_path, model, seq_n_max)) {
        free_q3_5(model);
        free(model);
        return NULL;
    }

    model->tokenizer.special_tokens = SPECIAL_TOKENS_Q3_5;
    model->tokenizer.bos_token_id = 0;
    model->tokenizer.eos_token_id = 248046;
    model->tokenizer.im_end_id = 248046;

    model_iface *model_i = a_calloc(sizeof(model_iface));
    *model_i = (model_iface){
        .model = model,
        .forward = forward_q3_5_wrap,
        .free_model = free_q3_5_wrap,
        .seq_n_max = seq_n_max ? seq_n_max : model->config.seq_len,
        .chat_template = think_ ? &CHAT_TEMPLATE_THINK_Q3_5 : &CHAT_TEMPLATE_Q3_5,
        .tokenizer = &model->tokenizer,
    };
    return model_i;
}

