#include "dolen_g4u_common.h"

int load_quantized_gemma4u(const char *filepath, Gemma4Unified *model, int seq_n_max) {
    FILE *f = fopen(filepath, "rb");
    if (!f) { log_msg(stderr, "ERROR: Failed to open %s\n", filepath); return -1; }
    memset(model, 0, sizeof(Gemma4Unified));

    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || fread(&version, sizeof(uint32_t), 1, f) != 1) {
        log_msg(stderr, "ERROR: Failed to read header\n"); fclose(f); return -1;
    }
    if (magic != 0x55344D47) { log_msg(stderr, "ERROR: Invalid magic number\n"); fclose(f); return -1; }
    if (version != 2) { log_msg(stderr, "ERROR: Unsupported version %u (expected 2). RE-RUN QUANTIZER.\n", version); fclose(f); return -1; }

    config_gemma4u *p = &model->config;
    if (fread(p, sizeof(config_gemma4u), 1, f) != 1) { log_msg(stderr, "ERROR: Failed to read config\n"); fclose(f); return -1; }
    if (seq_n_max != 0) p->seq_len = seq_n_max;

    model->layer_types = (int *)a_calloc((size_t)p->n_layers * sizeof(int));
    if (fread(model->layer_types, sizeof(int), (size_t)p->n_layers, f) != (size_t)p->n_layers) {
        log_msg(stderr, "ERROR: Failed to read layer_types\n"); fclose(f); return -1;
    }

    weights_gemma4u *w = &model->weights;
    int total_norm_dim = 0;
    w->norm_offsets = (int *)a_calloc((size_t)p->n_layers * sizeof(int));
    for (int i = 0; i < p->n_layers; i++) {
        w->norm_offsets[i] = total_norm_dim;
        total_norm_dim += (model->layer_types[i] ? p->global_head_dim : p->head_dim);
    }

    w->rms_input_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_post_attn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_pre_ffn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_post_ffn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_q_norm = (float *)a_calloc((size_t)total_norm_dim * sizeof(float));
    w->rms_k_norm = (float *)a_calloc((size_t)total_norm_dim * sizeof(float));
    w->rms_final_norm = (float *)a_calloc((size_t)p->dim * sizeof(float));

    if (!w->rms_input_layernorm || !w->rms_final_norm || !w->rms_q_norm) { log_msg(stderr, "ERROR: Alloc failed\n"); fclose(f); return -1; }

    read_qt(f, &w->embed_tokens);
    fread(w->rms_input_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    fread(w->rms_post_attn_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    fread(w->rms_pre_ffn_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    fread(w->rms_post_ffn_layernorm, sizeof(float), (size_t)p->n_layers * p->dim, f);
    fread(w->rms_q_norm, sizeof(float), total_norm_dim, f);
    fread(w->rms_k_norm, sizeof(float), total_norm_dim, f);
    fread(w->rms_final_norm, sizeof(float), (size_t)p->dim, f);

    w->q_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->k_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->v_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->o_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->gate_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->up_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->down_proj = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));

    w->layer_scalars = (float *)a_calloc((size_t)p->n_layers * sizeof(float));
    for (int i = 0; i < p->n_layers; i++) w->layer_scalars[i] = 1.0f;

    for (int i = 0; i < p->n_layers; i++) {
        read_qt(f, &w->q_proj[i]); read_qt(f, &w->k_proj[i]); read_qt(f, &w->v_proj[i]);
        read_qt(f, &w->o_proj[i]); read_qt(f, &w->gate_proj[i]); read_qt(f, &w->up_proj[i]); read_qt(f, &w->down_proj[i]);
    }
    fread(w->layer_scalars, sizeof(float), (size_t)p->n_layers, f);

    if (p->use_rope_freqs) {
        int freq_dim = p->global_head_dim / 2;
        w->rope_freqs_full = (float *)a_calloc(freq_dim * sizeof(float));
        fread(w->rope_freqs_full, sizeof(float), freq_dim, f);
    }

    fclose(f);
    log_msg(stderr, "INFO: Quantized Gemma4Unified loaded from %s\n", filepath);
    alloc_state_gemma4u(&model->state, p, w);
    return 0;
}

static void rmsnorm_gemma4u(float *o, float *x, float *weight, int size, float eps, int with_scale) {
    float ss = 0.0f;
    #pragma omp simd reduction(+:ss)
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / size + eps);
    #pragma omp simd
    for (int j = 0; j < size; j++) {
        o[j] = x[j] * ss;
        if (with_scale && weight) o[j] *= weight[j];
    }
}

static void apply_rope(float *vec, float *cos, float *sin, int rotary_dim, int vec_dim, int pos) {
    if (rotary_dim <= 0) return;
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

static inline float gemma_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

float *forward_gemma4u(Gemma4Unified *model, int token, int pos) {
    config_gemma4u *p = &model->config;
    weights_gemma4u *w = &model->weights;
    state_gemma4u *s = &model->state;
    float *x = s->x;
    int dim = p->dim;
    float eps = p->rms_norm_eps;
    float embed_scale = sqrtf((float)dim);

    int max_head_dim = p->global_head_dim > p->head_dim ? p->global_head_dim : p->head_dim;
    int max_kv_heads = p->n_global_kv_heads > p->n_kv_heads ? p->n_global_kv_heads : p->n_kv_heads;
    int max_kv_dim   = max_kv_heads * max_head_dim;

    // Embedding
    dequantize_row(x, &w->embed_tokens, token);
    for (int i = 0; i < dim; i++) x[i] *= embed_scale;

    for (int l = 0; l < p->n_layers; l++) {
        int is_full = model->layer_types[l];
        int head_dim = is_full ? p->global_head_dim : p->head_dim;
        int kv_heads = (is_full && p->attention_k_eq_v) ? p->n_global_kv_heads : p->n_kv_heads;
        int kv_dim = kv_heads * head_dim;
        int rotary_dim = is_full ? (int)(p->rope_partial_factor * p->global_head_dim) : p->head_dim;

        float *cos_cache = is_full ? s->cos_cache_full : s->cos_cache_sliding;
        float *sin_cache = is_full ? s->sin_cache_full : s->sin_cache_sliding;

        float *rms_in      = w->rms_input_layernorm + l * dim;
        float *rms_post_a  = w->rms_post_attn_layernorm + l * dim;
        float *rms_pre_f   = w->rms_pre_ffn_layernorm + l * dim;
        float *rms_post_f  = w->rms_post_ffn_layernorm + l * dim;
        float *rms_q       = w->rms_q_norm + w->norm_offsets[l];
        float *rms_k       = w->rms_k_norm + w->norm_offsets[l];

        rmsnorm_gemma4u(s->xb, x, rms_in, dim, eps, 1);
        matmul_qt(s->q, s->xb, &w->q_proj[l]);
        matmul_qt(s->k, s->xb, &w->k_proj[l]);
        if (is_full && p->attention_k_eq_v) memcpy(s->v, s->k, kv_dim * sizeof(float));
        else matmul_qt(s->v, s->xb, &w->v_proj[l]);

        for (int h = 0; h < p->n_heads; h++) {
            float *qh = s->q + h * head_dim;
            rmsnorm_gemma4u(qh, qh, rms_q, head_dim, eps, 1);
            if (rotary_dim > 0 && cos_cache) apply_rope(qh, cos_cache, sin_cache, rotary_dim, head_dim, pos);
        }

        for (int h = 0; h < kv_heads; h++) {
            float *kh = s->k + h * head_dim;
            rmsnorm_gemma4u(kh, kh, rms_k, head_dim, eps, 1);
            if (rotary_dim > 0 && cos_cache) apply_rope(kh, cos_cache, sin_cache, rotary_dim, head_dim, pos);
            rmsnorm_gemma4u(s->v + h * head_dim, s->v + h * head_dim, NULL, head_dim, eps, 0);
        }

        long long loff = (long long)l * p->seq_len * max_kv_dim;
        memcpy(s->key_cache + loff + (long long)pos * max_kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + (long long)pos * max_kv_dim, s->v, kv_dim * sizeof(float));

        int start_t = is_full ? 0 : fmax(0, pos - p->sliding_window + 1);

        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_dim;
            float *att = s->att + h * p->seq_len;
            int kv_head = h / (p->n_heads / kv_heads);
            for (int t = 0; t <= pos; t++) att[t] = -1e9f;
            for (int t = start_t; t <= pos; t++) {
                float *k = s->key_cache + loff + (long long)t * max_kv_dim + (long long)kv_head * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) score += q[i] * k[i];
                att[t] = score;
            }
            softmax(att, pos + 1);
            float *out = s->hb + h * head_dim;
            memset(out, 0, head_dim * sizeof(float));
            for (int t = start_t; t <= pos; t++) {
                float *v = s->value_cache + loff + (long long)t * max_kv_dim + (long long)kv_head * head_dim;
                float a = att[t];
                for (int i = 0; i < head_dim; i++) out[i] += a * v[i];
            }
        }

        matmul_qt(s->xb, s->hb, &w->o_proj[l]);
        rmsnorm_gemma4u(s->xb, s->xb, rms_post_a, dim, eps, 1);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];

        rmsnorm_gemma4u(s->xb, x, rms_pre_f, dim, eps, 1);
        matmul_qt(s->hb, s->xb, &w->gate_proj[l]);
        matmul_qt(s->hb2, s->xb, &w->up_proj[l]);
        int ffn_dim = w->gate_proj[l].rows;
        for (int i = 0; i < ffn_dim; i++) s->hb[i] = gemma_gelu(s->hb[i]) * s->hb2[i];
        matmul_qt(s->xb, s->hb, &w->down_proj[l]);
        rmsnorm_gemma4u(s->xb, s->xb, rms_post_f, dim, eps, 1);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];

        if (w->layer_scalars[l] != 1.0f) {
            for (int i = 0; i < dim; i++) x[i] *= w->layer_scalars[l];
        }
    }

    // Final norm (matches llama.cpp exactly)
    rmsnorm_gemma4u(x, x, w->rms_final_norm, dim, eps, 1);

    // LM HEAD (tied embedding) - NO EXTRA SCALING
    matmul_qt(s->logits, x, &w->embed_tokens);

    // Softcapping
    if (p->final_logit_softcapping > 0.0f) {
        float cap = p->final_logit_softcapping;
        float inv = 1.0f / cap;
        for (int i = 0; i < p->vocab_size; i++) {
            s->logits[i] = tanhf(s->logits[i] * inv) * cap;
        }
    }

    // Suppress known problematic special tokens to keep output clean
    static const int suppress[] = {0, 1, 2, 3, 255999, 256000, 258880, 258881, 258882, 258883, 258884};
    for (int i = 0; i < 11; i++) if (suppress[i] < p->vocab_size) s->logits[suppress[i]] = -1e9f;

    return s->logits;
}

static float *forward_gemma4u_wrap(void *model, int token, int pos) { return forward_gemma4u((Gemma4Unified *)model, token, pos); }
static void free_gemma4u_wrap(void *model) { free_gemma4u((Gemma4Unified *)model); free(model); }

static model_iface *init_gemma4u(const char *model_path, int seq_n_max) {
    Gemma4Unified *model = a_calloc(1 * sizeof(Gemma4Unified));
    if (load_quantized_gemma4u(model_path, model, seq_n_max) != 0) { free_gemma4u(model); free(model); return NULL; }
    model_iface *model_i = a_calloc(sizeof(model_iface));
    *model_i = (model_iface){ .model = model, .forward = forward_gemma4u_wrap, .free_model = free_gemma4u_wrap,
        .seq_n_max = (seq_n_max != 0) ? seq_n_max : model->config.seq_len, .vocab_size = model->config.vocab_size,
        .bos_token_id = 2, .im_end_id = 1, .special_tokens = NULL };
    return model_i;
}

int main(int argc, char *argv[]) { return common_main(argc, argv, init_gemma4u, "dolen_g4u"); }
