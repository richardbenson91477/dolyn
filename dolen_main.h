#ifndef DOLEN_MAIN_H
#define DOLEN_MAIN_H

#include "dolen_common_cmi.h"
#include "dolen_common_io.h"
#include "dolen_common_math.h"
#include "dolen_common_mem.h"
#include "dolen_common_qtensor.h"
#include "dolen_common_sampler.h"
#include "dolen_common_tokenizer.h"

#define DOLEN_MAIN_MODEL_PATH_DFLT "model.dolq"
#define DOLEN_MAIN_TEMP_DFLT (0.6)
#define DOLEN_MAIN_TOP_K_DFLT (40)
#define DOLEN_MAIN_TOP_P_DFLT (0.95)
#define DOLEN_MAIN_SEQ_N_DFLT (16384)
#define DOLEN_MAIN_PROMPT_N_MAX_DFLT (32768)
#define DOLEN_MAIN_MODE_DFLT "chat"
#define DOLEN_MAIN_SYSTEM_PROMPT_DFLT "You are a helpful assistant."
#define DOLEN_MAIN_THINK_DFLT false
#define DOLEN_MAIN_CHAT_TEMPLATE_DFLT "model"


typedef struct {
    uint64_t magic;
    const char *_name_s;
    model_init_fn init_fn;
} model_registry_entry;


extern const model_registry_entry MODEL_REGISTRY[];

extern const size_t MODEL_REGISTRY_SIZE;


model_iface *init_ms(const char *_model_path_s, int32_t seq_n, bool think_);

model_iface *init_q2(const char *_model_path_s, int32_t seq_n, bool think_);

model_iface *init_q3(const char *_model_path_s, int32_t seq_n, bool think_);

model_iface *init_q3_5(const char *_model_path_s, int32_t seq_n, bool think_);

model_iface *init_g4(const char *_model_path_s, int32_t seq_n, bool think_);

model_iface *init_ig4_1(const char *_model_path_s, int32_t seq_n, bool think_);

model_iface *init_l3(const char *_model_path_s, int32_t seq_n, bool think_);


#endif // DOLEN_MAIN_H

