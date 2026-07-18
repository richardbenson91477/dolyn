#include "dolen_quantize_main.h"


void print_help(const char *_argv0) {
    log_msg(stdout, "Usage: %s "
            " [--model PATH] "
            " [--out PATH]"
            " [--arch [ms | q2 | q3 | q3_5 | g4 | g4e | ig4_1 | l3]]"
            " [--tokenizer PATH]"
            " [--preset P]\n",
        _argv0);

    quantize_print_presets();
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
    char *_preset_s = "Q8_0";

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
        else if ((! strcmp(__argv[i], "--preset")) &&
                ((i + 1) < argc)) {
            i += 1;
            _preset_s = __argv[i];
        }
    }

    if (! _arch_s) {
        print_help(__argv[0]);
        log_msg(stderr, "ERROR: \"arch\" required.\n");
        exit(EXIT_FAILURE);
    }

    const quant_preset_t *_preset = quantize_find_preset(_preset_s);
    if (! _preset) {
        log_msg(stderr, "ERROR: Unknown preset \"%s\"\n", _preset_s);
        quantize_print_presets();
        exit(EXIT_FAILURE);
    }
    
    log_msg(stdout, "INFO: Using preset \"%s\"\n", _preset->name);

    if (! strcmp(_arch_s, "ms")) {
        return quantize_ms_to_file(_model_path_s, _out_path_s, _preset, _tokenizer_path_s) ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "q2")) {
        return quantize_q2_to_file(_model_path_s, _out_path_s, _preset, _tokenizer_path_s) ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "q3")) {
        return quantize_q3_to_file(_model_path_s, _out_path_s, _preset, _tokenizer_path_s) ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "q3_5")) {
        return quantize_q3_5_to_file(_model_path_s, _out_path_s, _preset, _tokenizer_path_s) ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "g4")) {
        return quantize_g4_to_file(_model_path_s, _out_path_s, _preset, _tokenizer_path_s) ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "g4e")) {
        return quantize_g4e_to_file(_model_path_s, _out_path_s, _preset, _tokenizer_path_s) ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "ig4_1")) {
        return quantize_ig4_1_to_file(_model_path_s, _out_path_s, _preset, _tokenizer_path_s) ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else if (! strcmp(_arch_s, "l3")) {
        return quantize_l3_to_file(_model_path_s, _out_path_s, _preset, _tokenizer_path_s) ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else {
        print_help(__argv[0]);
        log_msg(stderr, "ERROR: Unknown arch \"%s\"\n", _arch_s);
        exit(EXIT_FAILURE);
    }
}

