#ifndef DOLEN_COMMON_CMI_H
#define DOLEN_COMMON_CMI_H

#include "dolen_common_tokenizer.h"
#include "dolen_common_sampler.h"


typedef struct {
    const char *_system_s;
    const char *_main_s;
    const char *_end_turn_s;
} chat_template;

typedef struct {
    void *_model;
    float *(*forward)(void *_model, int32_t token, int32_t pos);
    void (*free_model)(void *_model);
    int32_t seq_n;
    int32_t seq_n_model_max;
    const chat_template *_chat_template;
    tokenizer *_tokenizer;
} model_iface;

typedef model_iface *(* model_init_fn)(const char *_model_path, int32_t seq_n, bool think_);


#endif // DOLEN_COMMON_CMI_H

