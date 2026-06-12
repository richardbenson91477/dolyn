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
    if (version != 1) { log_msg(stderr, "ERROR: Unsupported version\n"); fclose(f); return -1; }
    
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
        int hd = model->layer_types[i] ? p->global_head_dim : p->head_dim;
        total_norm_dim += hd;
    }
    
    w->rms_input_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_post_attn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_pre_ffn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_post_ffn_layernorm = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_q_norm = (float *)a_calloc((size_t)total_norm_dim * sizeof(float));
    w->rms_k_norm = (float *)a_calloc((size_t)total_norm_dim * sizeof(float));
    w->rms_final_norm = (float *)a_calloc((size_t)p->dim * sizeof(float));
    if (!w->rms_input_layernorm || !w->rms_final_norm || !w->rms_q_norm) {
        log_msg(stderr, "ERROR: Alloc failed\n"); fclose(f); return -1;
    }
    
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
    if (!w->q_proj || !w->k_proj || !w->o_proj) {
        log_msg(stderr, "ERROR: Alloc attn failed\n"); fclose(f); return -1;
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
    
    fclose(f);
    log_msg(stderr, "INFO: Quantized Gemma4Unified loaded from %s\n", filepath);
    alloc_state_gemma4u(&model->state, p);
    return 0;
}

static void rmsnorm_gemma4u(float *o, float *x, float *weight, int size, float eps, int with_scale) {
    float ss = 0.0f;

    #pragma omp simd reduction(+:ss)
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }

    ss = 1.0f / sqrtf(ss / size + eps);

    #pragma omp simd
    for (int j = 0; j < size; j++) {
        o[j] = x[j] * ss;
        if (with_scale && weight) {
            o[j] *= weight[j];
        }
    }
}

//static void apply_rope(float *vec, float *cos, float *sin, int rotary_dim, int vec_dim, int pos) {
//    if (rotary_dim <= 0) return;
//    int half = rotary_dim / 2;
//    float *cos_row = cos + pos * half;
//    float *sin_row = sin + pos * half;
//    for (int i = 0; i < half; i++) {
//        float c = cos_row[i], sn = sin_row[i];
//        float v0 = vec[i], v1 = vec[i + half];
//        vec[i] = v0 * c - v1 * sn;
//        vec[i + half] = v0 * sn + v1 * c;
//    }
//}
static void apply_rope(float *vec, float *cos, float *sin, int rotary_dim, int vec_dim, int pos) {
    if (rotary_dim <= 0) return;
    
    int rope_angles = rotary_dim / 2;
    int half_vec = vec_dim / 2; // FIX: Pairing distance is ALWAYS half the head dimension
    
    float *cos_row = cos + pos * rope_angles;
    float *sin_row = sin + pos * rope_angles;
    
    for (int i = 0; i < rope_angles; i++) {
        float c = cos_row[i], sn = sin_row[i];
        float v0 = vec[i], v1 = vec[i + half_vec]; // FIX: Pair vec[i] with vec[i + half_vec]
        
        vec[i] = v0 * c - v1 * sn;
        vec[i + half_vec] = v0 * sn + v1 * c;
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
    
    int max_kv_dim = (p->n_kv_heads * p->head_dim > p->n_global_kv_heads * p->global_head_dim) 
                     ? p->n_kv_heads * p->head_dim : p->n_global_kv_heads * p->global_head_dim;
    
    dequantize_row(x, &w->embed_tokens, token);
    #pragma omp simd
    for (int i = 0; i < dim; i++) x[i] *= embed_scale;
    
    for (int l = 0; l < p->n_layers; l++) {
        int is_full = model->layer_types[l];
        int current_head_dim = is_full ? p->global_head_dim : p->head_dim;
        int current_kv_heads = (is_full && p->attention_k_eq_v) ? p->n_global_kv_heads : p->n_kv_heads;
        int kv_dim = current_kv_heads * current_head_dim;
        //int rotary_dim = is_full ? p->global_head_dim : p->head_dim;
        int rotary_dim = is_full ? (int)(p->rope_partial_factor * p->global_head_dim) : p->head_dim;
        int vec_dim = current_head_dim;   // total size of q/k vector
        
        float *cos_cache = is_full ? s->cos_cache_full : s->cos_cache_sliding;
        float *sin_cache = is_full ? s->sin_cache_full : s->sin_cache_sliding;
        
        float *rms_input = w->rms_input_layernorm + l * dim;
        float *rms_post_attn = w->rms_post_attn_layernorm + l * dim;
        float *rms_pre_ffn = w->rms_pre_ffn_layernorm + l * dim;
        float *rms_post_ffn = w->rms_post_ffn_layernorm + l * dim;
        float *rms_q = w->rms_q_norm + w->norm_offsets[l];
        float *rms_k = w->rms_k_norm + w->norm_offsets[l];
        
        // Input norm + QKV
        rmsnorm_gemma4u(s->xb, x, rms_input, dim, eps, 1);
        quantize_vec(&s->xq, s->xb, dim);
        matmul_qq(s->q, &s->xq, &w->q_proj[l]);
        matmul_qq(s->k, &s->xq, &w->k_proj[l]);
        
        // V projection (K=V sharing handled here)
        if (is_full && p->attention_k_eq_v) {
            memcpy(s->v, s->k, kv_dim * sizeof(float)); // Copy raw k_proj output
        } else {
            matmul_qq(s->v, &s->xq, &w->v_proj[l]);
        }
        
        // V Norm (with_scale=False)
        for (int h = 0; h < current_kv_heads; h++) {
            rmsnorm_gemma4u(s->v + h * current_head_dim, s->v + h * current_head_dim, NULL, current_head_dim, eps, 0);
        }

        // Q/K Norm + RoPE
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            rmsnorm_gemma4u(s->q + h * current_head_dim, s->q + h * current_head_dim, rms_q, current_head_dim, eps, 1);
            if (rotary_dim > 0 && cos_cache)
                apply_rope(s->q + h * current_head_dim, cos_cache, sin_cache, rotary_dim, vec_dim, pos);
        }
        #pragma omp parallel for
        for (int h = 0; h < current_kv_heads; h++) {
            rmsnorm_gemma4u(s->k + h * current_head_dim, s->k + h * current_head_dim, rms_k, current_head_dim, eps, 1);
            if (rotary_dim > 0 && cos_cache)
                apply_rope(s->k + h * current_head_dim, cos_cache, sin_cache, rotary_dim, vec_dim, pos);
        }
        
        // V Norm (with_scale=False)
        for (int h = 0; h < current_kv_heads; h++) {
            rmsnorm_gemma4u(s->v + h * current_head_dim, s->v + h * current_head_dim, NULL, current_head_dim, eps, 0);
        }
        
        // Store K/V into cache
        long long loff = (long long)l * p->seq_len * max_kv_dim;
        memcpy(s->key_cache + loff + (long long)pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + (long long)pos * kv_dim, s->v, kv_dim * sizeof(float));
        
        // Attention
        int start_t = 0;
        if (!is_full) {
            start_t = pos - p->sliding_window + 1;
            if (start_t < 0) start_t = 0;
        }
        
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * current_head_dim;
            float *att = s->att + h * p->seq_len;
            int kv_head = h / (p->n_heads / current_kv_heads);
            float *k_base = s->key_cache + loff;
            float *v_base = s->value_cache + loff;
            
            for (int t = 0; t <= pos; t++) att[t] = -1e9f;
            
            for (int t = start_t; t <= pos; t++) {
                float *k = k_base + (long long)t * kv_dim + kv_head * current_head_dim;
                float score = 0.0f;
                #pragma omp simd reduction(+:score)
                for (int i = 0; i < current_head_dim; i++) score += q[i] * k[i];
                att[t] = score;
            }
            
            softmax(att, pos + 1);
            
            float *xb_h = s->xb + h * current_head_dim;
            memset(xb_h, 0, current_head_dim * sizeof(float));
            for (int t = start_t; t <= pos; t++) {
                float *v = v_base + (long long)t * kv_dim + kv_head * current_head_dim;
                float a = att[t];
                #pragma omp simd
                for (int i = 0; i < current_head_dim; i++) xb_h[i] += a * v[i];
            }
        }
        
        // O proj + Post-Attention Residual
        quantize_vec(&s->xq, s->xb, p->n_heads * current_head_dim);
        matmul_qq(s->xb, &s->xq, &w->o_proj[l]);
        rmsnorm_gemma4u(s->xb, s->xb, rms_post_attn, dim, eps, 1);
        #pragma omp simd
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
        
        // Pre-FFN norm + FFN
        rmsnorm_gemma4u(s->xb, x, rms_pre_ffn, dim, eps, 1);
        quantize_vec(&s->xq, s->xb, dim);
        matmul_qq(s->hb, &s->xq, &w->gate_proj[l]);
        matmul_qq(s->hb2, &s->xq, &w->up_proj[l]);
        
        #pragma omp parallel for
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            float gelu = 0.5f * val * (1.0f + tanhf(0.797885608f * (val + 0.044715f * val * val * val)));
            s->hb[i] = gelu * s->hb2[i];
        }
        
        quantize_vec(&s->hq, s->hb, hidden_dim);
        matmul_qq(s->xb, &s->hq, &w->down_proj[l]);
        
        // Post-FFN Residual
        rmsnorm_gemma4u(s->xb, s->xb, rms_post_ffn, dim, eps, 1);
        #pragma omp simd
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
        
        // Layer scalar (defaults to 1.0 in reference)
        #pragma omp simd
        for (int i = 0; i < dim; i++) x[i] *= 1.0f;
    }
    
    // 7. Final norm + logits
    rmsnorm_gemma4u(x, x, w->rms_final_norm, dim, eps, 1);
    matmul_qt(s->logits, x, &w->embed_tokens);
    
    if (p->final_logit_softcapping > 0.0f) {
        float cap = p->final_logit_softcapping;
        float inv_cap = 1.0f / cap;
        #pragma omp simd
        for (int i = 0; i < p->vocab_size; i++) {
            s->logits[i] = tanhf(s->logits[i] * inv_cap) * cap;
        }
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
        .bos_token_id = 2,
        .im_end_id = 1,
        .special_tokens = NULL,
    };
    return model_i;
}

int main(int argc, char *argv[]) {
    return common_main(argc, argv, init_gemma4u, "dolen_g4u");
}

