#include "dolen_quantize_common.h"


int quantize_q2_to_file(const char *_model_dir_s, const char *_file_path_s,
        q_type_t embed_type, q_type_t attn_type, q_type_t mlp_type, const char *_tokenizer_path_s);
int quantize_q3_to_file(const char *_model_dir_s, const char *_file_path_s,
        q_type_t embed_type, q_type_t attn_type, q_type_t mlp_type, const char *_tokenizer_path_s);
int quantize_q3_5_to_file(const char *_model_dir_s, const char *_file_path_s,
        q_type_t embed_type, q_type_t attn_type, q_type_t mlp_type, const char *_tokenizer_path_s);
int quantize_g4_to_file(const char *_model_dir_s, const char *_file_path_s,
        q_type_t embed_type, q_type_t attn_type, q_type_t mlp_type, const char *_tokenizer_path_s);
int quantize_ig4_1_to_file(const char *_model_dir_s, const char *_file_path_s,
        q_type_t embed_type, q_type_t attn_type, q_type_t mlp_type, const char *_tokenizer_path_s);
int quantize_l3_to_file(const char *_model_dir_s, const char *_file_path_s,
        q_type_t embed_type, q_type_t attn_type, q_type_t mlp_type, const char *_tokenizer_path_s);

