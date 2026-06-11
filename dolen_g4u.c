#include "dolen_g4u_common.h"

int load_quantized_gemma4u(const char *filepath, Gemma4Unified *model, int seq_n_max) {
    FILE *f = fopen(filepath, "rb");
    if (!f) { log_msg(stderr, "ERROR: Failed to open %s\n", filepath); return -1; }
    memset(model, 0, sizeof(Gemma4Unified));

    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || fread(&version, sizeof(uint32_t), 1, f) != 1) {
        log_msg(stderr, "ERROR: Failed to read header\n"); fclose(f); return -1;
    }
    if (magic != 0x55344D47) { // 'G4MU'
        log_msg(stderr, "ERROR: Invalid magic number\n"); fclose(f); return -1;
    }
    if (version != 1) { log_msg(stderr, "ERROR: Unsupported version\n"); fclose(f); return -1; }

    config_gemma4u *p = &model->config;
    if (fread(p, sizeof(config_gemma4u), 1, f) != 1) { log_msg(stderr, "ERROR: Failed to read config\n"); fclose(f); return -1; }

    if (seq_n_max != 0) p->seq_len = seq_n_max;

    model->layer_types = (int *)a_calloc((size_t)p->n_layers * sizeof(int));
    if (fread(model->layer_types, sizeof(int), (size_t)p->n_layers, f) != (size_t)p->n_layers) {
        log_msg(stderr, "ERROR: Failed to read layer_types\n"); fclose(f); return -1;
    }

    int n_full = 0, n_sliding = 0;
    for (int i = 0; i < p->n_layers; i++) {
        if (model->layer_types[i] == 1) n_full++;
        else n_sliding++;
    }

    weights_gemma4u *w = &model->weights;

    w->rms_input_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_post_attn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_pre_ffn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_post_ffn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_q_norm = (float *)a_calloc((size_t)p->n_layers * p->global_head_dim * sizeof(float));
    w->rms_k_norm = (float *)a_calloc((size_t)p->n_layers * p->global_head_dim * sizeof(float));
    w->rms_final_norm = (float *)a_calloc((size_t)p->dim * sizeof(float));

    if (!w->rms_input_layernorm || !w->rms_final_norm) { log_msg(stderr, "ERROR: Alloc failed\n"); fclose(f); return -1; }

    read_qt(f, &w->embed_tokens);
    fread(w->rms_input_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    fread(w->rms_post_attn_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    fread(w->rms_pre_ffn_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    fread(w->rms_post_ffn_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    fread(w->rms_q_norm, sizeof(float), (size_t)p->n_layers * p->global_head_dim, f);
    fread(w->rms_k_norm, sizeof(float), (size_t)p->n_layers * p->global_head_dim, f);
    fread(w->rms_final_norm, sizeof(float), (size_t)p->dim, f);

    w->q_proj = (qtensor *)a_calloc((size_t)n_full * sizeof(qtensor));
    w->k_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->v_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor)); // Added
    w->o_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->gate_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->up_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->down_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));

    if (!w->q_proj || !w->k_proj || !w->o_proj) { log_msg(stderr, "ERROR: Alloc attn failed\n"); fclose(f); return -1; }

    for (int i = 0; i < p->n_layers; i++) {
        int is_full = model->layer_types[i];
        int hd = is_full ? p->global_head_dim : p->head_dim;
        read_qt(f, &w->q_proj[is_full]);
        read_qt(f, &w->k_proj[i]);
        read_qt(f, &w->v_proj[i]);
        read_qt(f, &w->o_proj[i]);
        read_qt(f, &w->gate_proj[i]);
        read_qt(f, &w->up_proj[i]);
        read_qt(f, &w->down_proj[i]);
    }

    if (!p->tie_word_embeddings) {
        read_qt(f, &w->embed_tokens); // lm_head tied to embed_tokens in save, but we read separate if untied
    } else {
        // Tied: lm_head is same as embed_tokens. We already read embed_tokens.
        // In save format, we only write embed_tokens once.
    }

    fclose(f);
    log_msg(stderr, "INFO: Quantized Gemma4Unified loaded from %s\n", filepath);

    alloc_state_gemma4u(&model->state, p);
    return 0;
}

static void rmsnorm_gemma(float *o, float *x, float *weight, int size, float eps) {
    float ss = 0.0f;
    #pragma omp simd reduction(+:ss)
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / size + eps);
    #pragma omp simd
    for (int j = 0; j < size; j++) o[j] = weight[j] * x[j] * ss;
}

static void rmsnorm_no_scale(float *x, int size, float eps) {
    float ss = 0.0f;
    #pragma omp simd reduction(+:ss)
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / size + eps);
    #pragma omp simd
    for (int j = 0; j < size; j++) x[j] *= ss;
}

static void apply_rope(float *vec, float *cos, float *sin, int rotary_len, int pos) {
    float *cos_row = cos + pos * rotary_len;
    float *sin_row = sin + pos * rotary_len;
    for (int i = 0; i < rotary_len; i++) {
        float c = cos_row[i], sn = sin_row[i];
        float v0 = vec[i], v1 = vec[i + rotary_len];
        vec[i] = v0 * c - v1 * sn;
        vec[i + rotary_len] = v0 * sn + v1 * c;
    }
}

float *forward_gemma4u(Gemma4Unified *model, int token, int pos) {
    config_gemma4u *p = &model->config;
    weights_gemma4u *w = &model->weights;
    state_gemma4u *s = &model->state;
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    float eps = p->rms_norm_eps;
    float embed_scale = sqrtf((float)dim);

    // Embedding
    dequantize_row(x, &w->embed_tokens, token);
    #pragma omp simd
    for (int i = 0; i < dim; i++) x[i] *= embed_scale;

    for (int l = 0; l < p->n_layers; l++) {
        int is_full = model->layer_types[l];
        int current_head_dim = is_full ? p->global_head_dim : p->head_dim;
        int kv_dim = p->n_kv_heads * current_head_dim;
        int rotary_len = is_full ? (int)(p->global_head_dim * p->rope_partial_factor) : (int)(p->head_dim * p->rope_partial_factor);
        float *cos_cache = is_full ? s->cos_cache_full : s->cos_cache_sliding;
        float *sin_cache = is_full ? s->sin_cache_full : s->sin_cache_sliding;
        float *rms_input = w->rms_input_layernorm + l * dim;
        float *rms_post_attn = w->rms_post_attn_layernorm + l * dim;
        float *rms_pre_ffn = w->rms_pre_ffn_layernorm + l * dim;
        float *rms_post_ffn = w->rms_post_ffn_layernorm + l * dim;
        float *rms_q = w->rms_q_norm + l * current_head_dim;
        float *rms_k = w->rms_k_norm + l * current_head_dim;

        // Post-attention residual path
        rmsnorm_gemma(s->xb, x, rms_input, dim, eps);
        quantize_vec(&s->xq, s->xb, dim);
        matmul_qq(s->q, &s->xq, &w->q_proj[l]);
        matmul_qq(s->k, &s->xq, &w->k_proj[l]);

        // V projection (skip if k_eq_v and full attention)
        if (is_full && p->attention_k_eq_v) {
            memcpy(s->v, s->k, kv_dim * sizeof(float));
        } else {
            matmul_qq(s->v, &s->xq, &w->v_proj[l]);
        }

        // Norms & RoPE
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            rmsnorm_gemma(s->q + h * current_head_dim, s->q + h * current_head_dim, rms_q, current_head_dim, eps);
            if (rotary_len > 0 && cos_cache) apply_rope(s->q + h * current_head_dim, cos_cache, sin_cache, rotary_len, pos);
        }
        #pragma omp parallel for
        for (int h = 0; h < p->n_kv_heads; h++) {
            rmsnorm_gemma(s->k + h * current_head_dim, s->k + h * current_head_dim, rms_k, current_head_dim, eps);
            if (rotary_len > 0 && cos_cache) apply_rope(s->k + h * current_head_dim, cos_cache, sin_cache, rotary_len, pos);
            rmsnorm_no_scale(s->v + h * current_head_dim, current_head_dim, eps);
        }

        // KV Cache
        long long loff = (long long)l * p->seq_len * kv_dim;
        memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));

        // Attention
        float inv_sqrt_head = 1.0f / sqrtf((float)current_head_dim);
        int start_t = (pos >= p->sliding_window) ? (pos - p->sliding_window) : 0;

        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * current_head_dim;
            float *att = s->att + h * p->seq_len;
            int kv_head = is_full ? h / (p->n_heads / p->n_kv_heads) : h;
            float *k_base = s->key_cache + loff;
            float *v_base = s->value_cache + loff;

            for (int t = start_t; t <= pos; t++) {
                float *k = k_base + t * kv_dim + kv_head * current_head_dim;
                float score = 0.0f;
                #pragma omp simd reduction(+:score)
                for (int i = 0; i < current_head_dim; i++) score += q[i] * k[i];
                att[t] = score * inv_sqrt_head;
            }

            softmax(att, pos - start_t + 1);
            memset(s->xb, 0, current_head_dim * sizeof(float));

            for (int t = start_t; t <= pos; t++) {
                float *v = v_base + t * kv_dim + kv_head * current_head_dim;
                float a = att[t - start_t];
                #pragma omp simd
                for (int i = 0; i < current_head_dim; i++) s->xb[i] += a * v[i];
            }
        }

        // Output proj & residual
        quantize_vec(&s->xq, s->xb, p->n_heads * current_head_dim);
        matmul_qq(s->xb, &s->xq, &w->o_proj[l]);
        #pragma omp simd
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];

        // MLP
        rmsnorm_gemma(s->xb, x, rms_post_attn, dim, eps);
        quantize_vec(&s->xq, s->xb, dim);
        matmul_qq(s->hb, &s->xq, &w->gate_proj[l]);
        matmul_qq(s->hb2, &s->xq, &w->up_proj[l]);

        #pragma omp parallel for
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            float x_sq = val * val;
            val = val * (1.0f / (1.0f + expf(-val))); // silu approximation for gelu_tanh
            val *= (1.0f + 0.044715f * x_sq) * (0.797885f + 0.03567f * x_sq); // gelu_tanh approx
            s->hb[i] = val;
        }

        quantize_vec(&s->hq, s->hb, hidden_dim);
        matmul_qq(s->xb, &s->hq, &w->down_proj[l]);
        #pragma omp simd
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];

        rmsnorm_gemma(s->xb, x, rms_post_ffn, dim, eps);
        #pragma omp simd
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    // Final norm & LM head
    rmsnorm_gemma(x, x, w->rms_final_norm, dim, eps);
    matmul_qt(s->logits, x, &w->embed_tokens);

    // Logit softcapping
    float cap = p->final_logit_softcapping;
    float inv_cap = 1.0f / cap;
    #pragma omp simd
    for (int i = 0; i < p->vocab_size; i++) {
        s->logits[i] = tanhf(s->logits[i] * inv_cap) * cap;
    }

    return s->logits;
}

static float *forward_gemma4u_wrap(void *model, int token, int pos) {
    return forward_gemma4u((Gemma4Unified *)model, token, pos);
}

static void free_gemma4u_wrap(void *model) {
    free_gemma4u((Gemma4Unified *)model);
    free(model);
}

static model_iface *init_gemma4u(const char *model_path, int seq_n_max) {
    Gemma4Unified *model = a_calloc(1 * sizeof(Gemma4Unified));
    if (load_quantized_gemma4u(model_path, model, seq_n_max) != 0) {
        free_gemma4u(model); free(model);
        return NULL;
    }

    model_iface *model_i = a_calloc(sizeof(model_iface));
    *model_i = (model_iface) {
        .model = model,
        .forward = forward_gemma4u_wrap,
        .free_model = free_gemma4u_wrap,
        .seq_n_max = (seq_n_max != 0) ? seq_n_max : model->config.seq_len,
        .vocab_size = model->config.vocab_size,
        .special_tokens = NULL, // Text-only, no special tokens needed for generation loop
        .im_end_id = 1,
    };
    return model_i;
}

int main(int argc, char *argv[]) {
    return common_main(argc, argv, init_gemma4u, "dolen_g4u");
}

