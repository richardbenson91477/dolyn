#include "dolen_quantize_main.h"


int main(int argc, char *__argv[]) {
    char *_arch_s = NULL;

    if (argc < 3) {
        log_msg(stdout, "Usage: %s <model_dir> <output_file>" \
                " [--arch ARCH]" \
                " [--type Q] [--embed Q] [--attn Q] [--mlp Q]" \
                " [--tokenizer PATH]\n" \
                "Where:\n" 
                "  ARCH: [q2 | q3 | q3_5 | g4 | ig4_1 | l3]\n" \
                "  Q: [q4 | q6 | q8 | f16 | f32]\n\n", 
                __argv[0]);
        return EXIT_FAILURE;
    }

    q_type_t embed_type = Q_TYPE_Q8, attn_type = Q_TYPE_Q8, mlp_type = Q_TYPE_Q8;
    char *_tokenizer_path_s = "tokenizer.bin";
    for (int i = 3; i < argc; i++) {
        if ((! strcmp(__argv[i], "--arch")) &&
                ((i + 1) < argc)) {
            i += 1;
            _arch_s = __argv[i];
        }
        else if ((! strcmp(__argv[i], "--type")) &&
                ((i + 1) < argc)) {
            i += 1;
            q_type_t t = parse_q_type(__argv[i]);
            embed_type = attn_type = mlp_type = t;
        }
        else if ((! strcmp(__argv[i], "--embed")) &&
                ((i + 1) < argc)) {
            i += 1;
            embed_type = parse_q_type(__argv[i]);
        }
        else if ((! strcmp(__argv[i], "--attn")) &&
                ((i + 1) < argc)) {
            i += 1;
            attn_type = parse_q_type(__argv[i]);
        }
        else if ((! strcmp(__argv[i], "--mlp")) &&
                ((i + 1) < argc)) {
            i += 1;
            mlp_type = parse_q_type(__argv[i]);
        }
        else if ((! strcmp(__argv[i], "--tokenizer")) &&
                ((i + 1) < argc)) {
            i += 1;
            _tokenizer_path_s = __argv[i];
        }
    }

    if (! _arch_s) {
        log_msg(stderr, "ERROR: \"arch\" required.\n");
        exit(EXIT_FAILURE);
    }

    else if (! strcmp(_arch_s, "q2")) {
        return quantize_q2_to_file(__argv[1], __argv[2], embed_type, attn_type, mlp_type, _tokenizer_path_s)
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "q3")) {
        return quantize_q3_to_file(__argv[1], __argv[2], embed_type, attn_type, mlp_type, _tokenizer_path_s) \
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "q3_5")) {
        return quantize_q3_5_to_file(__argv[1], __argv[2], embed_type, attn_type, mlp_type, _tokenizer_path_s) \
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "g4")) {
        return quantize_g4_to_file(__argv[1], __argv[2], embed_type, attn_type, mlp_type, _tokenizer_path_s) \
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "ig4_1")) {
        return quantize_ig4_1_to_file(__argv[1], __argv[2], embed_type, attn_type, mlp_type, _tokenizer_path_s) \
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "l3")) {
        return quantize_l3_to_file(__argv[1], __argv[2], embed_type, attn_type, mlp_type, _tokenizer_path_s) \
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else {
        log_msg(stderr, "ERROR: Unknown arch \"%s\"", _arch_s);
        exit(EXIT_FAILURE);
    }
}

