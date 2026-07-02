#include "dolen_quantize_main.h"

void print_help(const char *_argv0) {
    log_msg(stdout, "Usage: %s "
            " [--model PATH] " \
            " [--out PATH]" \
            " [--arch ARCH]" \
            " [--tokenizer PATH]\n" \
            " [--default Q] [--embed Q] [--attn Q] [--mlp Q]" \
            "Where:\n" 
            "  ARCH: [ms | q2 | q3 | q3_5 | g4 | ig4_1 | l3]\n" \
            "  Q: [q4 | q6 | q8 | f16 | f32]\n\n", 
            _argv0);
}

int main(int argc, char *__argv[]) {

    if (argc < 2) {
        print_help(__argv[0]);
        return EXIT_FAILURE;
    }

    char *_model_path_s = "model";
    char *_out_path_s = "model.dolq";
    char *_arch_s = NULL;
    char *_tokenizer_path_s = "tokenizer.bin";
    q_type_t embed_type = Q_TYPE_Q8;
    q_type_t attn_type = Q_TYPE_Q8;
    q_type_t mlp_type = Q_TYPE_Q8;

    for (int32_t i = 1; i < argc; i++) {
        if (! strcmp(__argv[i], "--help")) {
            print_help(__argv[0]);
            return EXIT_FAILURE;
        }
        else if ((! strcmp(__argv[i], "--model")) &&
                ((i + 1) < argc)) {
            i += 1;
            _model_path_s = __argv[i];
        }
        else if ((! strcmp(__argv[i], "--out")) &&
                ((i + 1) < argc)) {
            i += 1;
            _out_path_s = __argv[i];
        }
        else if ((! strcmp(__argv[i], "--arch")) &&
                ((i + 1) < argc)) {
            i += 1;
            _arch_s = __argv[i];
        }
        else if ((! strcmp(__argv[i], "--tokenizer")) &&
                ((i + 1) < argc)) {
            i += 1;
            _tokenizer_path_s = __argv[i];
        }
        else if ((! strcmp(__argv[i], "--default")) &&
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
    }

    if (! _arch_s) {
        print_help(__argv[0]);
        log_msg(stderr, "ERROR: \"arch\" required.\n");
        exit(EXIT_FAILURE);
    }
    else if (! strcmp(_arch_s, "ms")) {
        return quantize_ms_to_file(_model_path_s, _out_path_s, embed_type, attn_type, mlp_type, _tokenizer_path_s)
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "q2")) {
        return quantize_q2_to_file(_model_path_s, _out_path_s, embed_type, attn_type, mlp_type, _tokenizer_path_s)
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "q3")) {
        return quantize_q3_to_file(_model_path_s, _out_path_s, embed_type, attn_type, mlp_type, _tokenizer_path_s) \
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "q3_5")) {
        return quantize_q3_5_to_file(_model_path_s, _out_path_s, embed_type, attn_type, mlp_type, _tokenizer_path_s) \
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "g4")) {
        return quantize_g4_to_file(_model_path_s, _out_path_s, embed_type, attn_type, mlp_type, _tokenizer_path_s) \
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "ig4_1")) {
        return quantize_ig4_1_to_file(_model_path_s, _out_path_s, embed_type, attn_type, mlp_type, _tokenizer_path_s) \
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "l3")) {
        return quantize_l3_to_file(_model_path_s, _out_path_s, embed_type, attn_type, mlp_type, _tokenizer_path_s) \
            ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else {
        print_help(__argv[0]);
        log_msg(stderr, "ERROR: Unknown arch \"%s\"", _arch_s);
        exit(EXIT_FAILURE);
    }
}

