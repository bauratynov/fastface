# FastFace Makefile
#
# On Linux / macOS:   make
# On Windows + mingw: mingw32-make
#
# The `EXE` suffix is auto-detected: `.exe` on Windows, empty on Linux/macOS.
#
# Targets:
#   all       -- build fastface_int8 + fastface_int8_batched + libfastface.a
#   test      -- run tests/run_regression.py
#   clean     -- remove built artifacts
#   calibrate -- regenerate models/op_scales_v2.bin + models/w600k_r50_ffw4.bin
#
# Tune via: CC=clang, CFLAGS_EXTRA=..., JOBS=..., PYTHON=python3

# ---- OS / exe-extension detection --------------------------------------
# mingw / MSYS / Cygwin set OS=Windows_NT; GnuMake on Linux/Mac leaves it empty.
ifeq ($(OS),Windows_NT)
  EXE := .exe
else
  EXE :=
endif

# ---- Toolchain -----------------------------------------------------------
CC          := gcc
AR          := ar
# CFLAGS_ARCH is overridable for CI / cross-compile:
#   local bare-metal build:  leave default (-march=native + VNNI)
#   CI on a non-VNNI runner: override to "-march=x86-64-v3 -mavxvnni"
#     (compiles but binary will SIGILL at runtime on non-VNNI CPUs)
#   ARM NEON port (future):  override with "-march=armv8.2-a+dotprod" etc.
CFLAGS_ARCH ?= -march=native -mavx2 -mfma -mavxvnni
CFLAGS       = -O3 $(CFLAGS_ARCH) -fopenmp $(CFLAGS_EXTRA)
DEFS         = -DFFW2_NOMAIN
PYTHON      ?= python

KERNEL_SRCS = kernels/ffw2_loader.c \
              kernels/conv2d_nhwc.c \
              kernels/gemm_int8_v2.c \
              kernels/int8_epilogue.c \
              kernels/gemm_int8_matvec.c

KERNEL_OBJS = $(KERNEL_SRCS:.c=.o)

BIN_MAIN    = fastface_int8$(EXE)
BIN_BATCHED = fastface_int8_batched$(EXE)
BIN_TESTLIB = test_libfastface$(EXE)

.PHONY: all test clean calibrate lib exes bench bench-batched bench-lfw

all: exes lib

exes: $(BIN_MAIN) $(BIN_BATCHED)

lib: libfastface.a

$(BIN_MAIN): arcface_forward_int8.c $(KERNEL_SRCS)
	$(CC) $(CFLAGS) $(DEFS) $^ -o $@

$(BIN_BATCHED): arcface_forward_int8_batched.c $(KERNEL_SRCS)
	$(CC) $(CFLAGS) $(DEFS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(DEFS) -c $< -o $@

libfastface.a: libfastface.o $(KERNEL_OBJS)
	$(AR) rcs $@ $^

$(BIN_TESTLIB): test_libfastface.c libfastface.a fastface.h
	$(CC) -O2 -fopenmp -I. test_libfastface.c libfastface.a -o $@

test: $(BIN_MAIN) tests/golden_input.bin tests/golden_int8_emb.bin
	$(PYTHON) tests/run_regression.py

calibrate:
	N_CALIB=200 PERCENTILE=99.9 WITH_PRINCESS=1 $(PYTHON) export_op_scales_v2.py
	N_CALIB=200 PERCENTILE=99.9 WITH_PRINCESS=1 $(PYTHON) prepare_weights_v3.py

bench: $(BIN_MAIN)
	$(PYTHON) bench_stable_int8_ffw4.py

bench-batched: $(BIN_BATCHED)
	$(PYTHON) bench_stable_int8_batched.py

bench-lfw: $(BIN_MAIN)
	$(PYTHON) bench_lfw_verify.py --n-pairs 1000

clean:
	rm -f fastface_int8 fastface_int8.exe \
	      fastface_int8_batched fastface_int8_batched.exe \
	      test_libfastface test_libfastface.exe \
	      libfastface.a *.o kernels/*.o
