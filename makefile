export OMP_NUM_THREADS := 11
export OMP_PLACES := cores
export OMP_PROC_BIND := close

CC := gcc
CFLAGS = -fPIC 
CFLAGS_OPT = $(CFLAGS) -mf16c -fopenmp -O3 -pipe -march=x86-64-v4 -fomit-frame-pointer -funroll-loops -fpermissive
CFLAGS_DBG = $(CFLAGS) -mf16c -fopenmp -Og -ggdb -fkeep-inline-functions

SRC_EXT = ext/csafetensors.c ext/json.c
INC_EXT = ext/csafetensors.h ext/json.h

DOLEN_COMMON_SRC=dolen_common_io.c dolen_common_math.c dolen_common_mem.c dolen_common_qtensor.c dolen_common_sampler.c dolen_common_tokenizer.c
DOLEN_COMMON_INC=dolen_common_cmi.h dolen_common_io.h dolen_common_math.h dolen_common_mem.h dolen_common_qtensor.h dolen_common_sampler.h dolen_common_tokenizer.h

BIN_MAIN = dolen
BIN_MAIN_DBG = dolen_dbg
SRC_MAIN = dolen_main.c $(DOLEN_COMMON_SRC) dolen_ms.c dolen_ms_common.c dolen_q2.c dolen_q2_common.c dolen_q3.c dolen_q3_common.c dolen_q3_5.c dolen_q3_5_common.c dolen_g4.c dolen_g4_common.c dolen_ig4_1.c dolen_ig4_1_common.c dolen_l3.c dolen_l3_common.c
INC_MAIN = dolen_main.h $(DOLEN_COMMON_INC) dolen_ms_common.h dolen_q2_common.h dolen_q3_common.h dolen_q3_5_common.h dolen_g4_common.h dolen_ig4_1_common.h dolen_l3_common.h

BIN_Q_MAIN = dolen_quantize
BIN_Q_DBG = dolen_quantize_dbg
SRC_Q_MAIN = dolen_quantize_main.c $(DOLEN_COMMON_SRC) dolen_quantize_common.c dolen_ms_quantize.c dolen_q2_quantize.c dolen_q2_common.c dolen_q3_quantize.c dolen_q3_common.c dolen_q3_5_quantize.c dolen_q3_5_common.c dolen_g4_quantize.c dolen_g4_common.c dolen_ig4_1_quantize.c dolen_ig4_1_common.c dolen_l3_quantize.c dolen_l3_common.c
INC_Q_MAIN = dolen_quantize_main.h $(DOLEN_COMMON_INC) dolen_quantize_common.h dolen_ms_common.h dolen_q2_common.h dolen_q3_common.h dolen_q3_5_common.h dolen_g4_common.h dolen_ig4_1_common.h dolen_l3_common.h

BINS_ALL = $(BIN_MAIN) $(BIN_Q_MAIN) $(BIN_MAIN_DBG) $(BIN_Q_MAIN_DBG)
SRCS_ALL = $(SRC_MAIN) $(SRC_Q_MAIN)


all: strip

$(BIN_MAIN): $(SRC_MAIN) $(INC_MAIN)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_MAIN) -lm

$(BIN_MAIN_DBG): $(SRC_MAIN) $(INC_MAIN)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_MAIN) -lm

$(BIN_Q_MAIN): $(SRC_Q_MAIN) $(INC_Q_MAIN) $(SRC_EXT) $(INC_EXT)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_Q_MAIN) $(SRC_EXT) -lm

$(BIN_Q_DBG): $(SRC_Q_MAIN) $(INC_Q_MAIN) $(SRC_EXT) $(INC_EXT)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_Q_MAIN) $(SRC_EXT) -lm

test: $(BIN_MAIN)
	./dolen_test_dolq_chat "./test_model.dolq"

debug: $(BIN_MAIN_DBG)
	./dolen_gdb $(BIN_MAIN_DBG) $(MODEL_MAIN) $(PROMPT) $(SYS_PROMPT) $(SEQN)

strip: $(BINS_ALL)
	strip $(BINS_ALL)

format:
	clang-format -i $(SRCS_ALL)

tags_rebuild:
	./_tags_rebuild

clean:
	rm -vf $(BINS_ALL)

