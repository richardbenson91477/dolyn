export OMP_NUM_THREADS := 11
export OMP_PLACES := cores
export OMP_PROC_BIND := close

CC := gcc
CFLAGS = -fPIC 
CFLAGS_OPT = $(CFLAGS) -fopenmp -O3 -pipe -march=x86-64-v4 -fomit-frame-pointer -funroll-loops -fpermissive
CFLAGS_DBG = $(CFLAGS) -fopenmp -Og -ggdb -fkeep-inline-functions

MODEL_PATH := ${lena_rolocal_path}/models/dolen_models
MODEL_MAIN := "$(MODEL_PATH)/qwen3_5_1b_q8.dolq"
PROMPT := "Please count from 1 to 10 using a comma separated list. Spell out the numbers in English."
SYS_PROMPT := "You are a helpful language model."
SEQN := 1024

BIN_MAIN = dolen
BIN_MAIN_DBG = dolen_dbg
BIN_Q3_5_Q = dolen_q3_5_quantize
BIN_Q3_5_Q_DBG = dolen_q3_5_quantize_dbg
BIN_Q3_Q = dolen_q3_quantize
BIN_Q3_Q_DBG = dolen_q3_quantize_dbg
BIN_G4_Q = dolen_g4_quantize
BIN_G4_Q_DBG = dolen_g4_quantize_dbg
BIN_IG4_1_Q = dolen_ig4_1_quantize
BIN_IG4_1_Q_DBG = dolen_ig4_1_quantize_dbg
BIN_L3_Q = dolen_l3_quantize
BIN_L3_Q_DBG = dolen_l3_quantize_dbg

SRC_EXT = ext/csafetensors.c ext/json.c
INC_EXT = ext/csafetensors.h ext/json.h

DOLEN_COMMON_SRC=dolen_common_io.c dolen_common_math.c dolen_common_mem.c dolen_common_qtensor.c dolen_common_sampler.c dolen_common_tokenizer.c
DOLEN_COMMON_INC=dolen_common_cmi.h dolen_common_io.h dolen_common_math.h dolen_common_mem.h dolen_common_qtensor.h dolen_common_sampler.h dolen_common_tokenizer.h

SRC_MAIN = dolen_main.c $(DOLEN_COMMON_SRC) dolen_q3_5.c dolen_q3_5_common.c dolen_q3.c dolen_q3_common.c dolen_g4.c dolen_g4_common.c dolen_ig4_1.c dolen_ig4_1_common.c dolen_l3.c dolen_l3_common.c
INC_MAIN = dolen_main.h $(DOLEN_COMMON_INC) dolen_l3_common.h dolen_ig4_1_common.h dolen_g4_common.h dolen_q3_common.h dolen_q3_5_common.h

SRC_Q3_5_Q = $(DOLEN_COMMON_SRC) dolen_q3_5_quantize.c dolen_q3_5_common.c dolen_quantize_common.c
INC_Q3_5_Q = $(DOLEN_COMMON_INC) dolen_q3_5_common.h dolen_quantize_common.h

SRC_Q3_Q = $(DOLEN_COMMON_SRC) dolen_q3_quantize.c dolen_q3_common.c dolen_quantize_common.c
INC_Q3_Q = $(DOLEN_COMMON_INC) dolen_q3_common.h dolen_quantize_common.h

SRC_G4_Q = $(DOLEN_COMMON_SRC) dolen_g4_quantize.c dolen_g4_common.c dolen_quantize_common.c
INC_G4_Q = $(DOLEN_COMMON_INC) dolen_g4_common.h dolen_quantize_common.h

SRC_IG4_1_Q = $(DOLEN_COMMON_SRC) dolen_ig4_1_quantize.c dolen_ig4_1_common.c dolen_quantize_common.c
INC_IG4_1_Q = $(DOLEN_COMMON_INC) dolen_ig4_1_common.h dolen_quantize_common.h

SRC_L3_Q = $(DOLEN_COMMON_SRC) dolen_l3_quantize.c dolen_l3_common.c dolen_quantize_common.c
INC_L3_Q = $(DOLEN_COMMON_INC) dolen_l3_common.h dolen_quantize_common.h

BINS_ALL = $(BIN_MAIN) $(BIN_Q3_5_Q) $(BIN_Q3_Q) $(BIN_G4_Q) $(BIN_IG4_1_Q) $(BIN_L3_Q)
BINS_ALL_DBG = $(BIN_DBG) $(BIN_Q3_5_Q_DBG) $(BIN_Q3_Q_DBG) $(BIN_G4_Q_DBG) $(BIN_IG4_1_Q_DBG) $(BIN_L3_Q_DBG) 
SRCS_ALL = $(SRC_MAIN) $(SRC_Q3_5_Q) $(SRC_Q3_Q) $(SRC_G4_Q) $(SRC_IG4_1_Q) $(SRC_L3_Q) 


all: strip

$(BIN_MAIN): $(SRC_MAIN) $(INC_MAIN)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_MAIN) -lm

$(BIN_MAIN_DBG): $(SRC_MAIN) $(INC_MAIN)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_MAIN) -lm

test: $(BIN_MAIN)
	./$(BIN_MAIN) -m $(MODEL_MAIN) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN)

debug: $(BIN_MAIN_DBG)
	./dolen_gdb $(BIN_MAIN_DBG) $(MODEL_MAIN) $(PROMPT) $(SYS_PROMPT) $(SEQN)


$(BIN_Q3_5_Q): $(SRC_Q3_5_Q) $(SRC_EXT) $(INC_Q3_5_Q) $(INC_EXT)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_Q3_5_Q) $(SRC_EXT) -lm

$(BIN_Q3_5_Q_DBG): $(SRC_Q3_5_Q) $(SRC_EXT) $(INC_Q3_5_Q) $(INC_EXT)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_Q3_5_Q) $(SRC_EXT) -lm

$(BIN_Q3_Q): $(SRC_Q3_Q) $(SRC_EXT) $(INC_Q3_Q) $(INC_EXT)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_Q3_Q) $(SRC_EXT) -lm

$(BIN_Q3_Q_DBG): $(SRC_Q3_Q) $(SRC_EXT) $(INC_Q3_Q) $(INC_EXT)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_Q3_Q) $(SRC_EXT) -lm

$(BIN_G4_Q): $(SRC_G4_Q) $(SRC_EXT) $(INC_G4_Q) $(INC_EXT)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_G4_Q) $(SRC_EXT) -lm

$(BIN_G4_Q_DBG): $(SRC_G4_Q) $(SRC_EXT) $(INC_G4_Q) $(INC_EXT)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_G4_Q) $(SRC_EXT) -lm

$(BIN_IG4_1_Q): $(SRC_IG4_1_Q) $(SRC_EXT) $(INC_IG4_1_Q) $(INC_EXT)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_IG4_1_Q) $(SRC_EXT) -lm

$(BIN_IG4_1_Q_DBG): $(SRC_IG4_1_Q) $(SRC_EXT) $(INC_IG4_1_Q) $(INC_EXT)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_IG4_1_Q) $(SRC_EXT) -lm

$(BIN_L3_Q): $(SRC_L3_Q) $(SRC_EXT) $(INC_L3_Q) $(INC_EXT)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_L3_Q) $(SRC_EXT) -lm

$(BIN_L3_Q_DBG): $(SRC_L3_Q) $(SRC_EXT) $(INC_L3_Q) $(INC_EXT)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_L3_Q) $(SRC_EXT) -lm


strip: $(BINS_ALL)
	strip $(BINS_ALL)

format:
	clang-format -i $(SRCS_ALL)

tags_rebuild:
	./_tags_rebuild

clean:
	rm -vf $(BINS_ALL) $(BINS_ALL_DBG)

