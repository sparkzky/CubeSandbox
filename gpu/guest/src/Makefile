# Copyright (c) 2024 Tencent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Builds libcube-gpu.so, the LD_PRELOAD interposition library for the guest.
#
# -fvisibility=hidden keeps every internal helper (cube_gpu_*, rpc_*, shm_*)
# out of the dynamic symbol table. The interposed cu* entry points and dlsym
# opt back in with CUBE_GPU_EXPORT. They are exported WITHOUT an ELF version:
# driver API symbols are unversioned, and libcube_gpu_symbols.ld only defines
# the GLIBC version nodes that the dlsym interposer needs.

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra
CFLAGS  += -fPIC -fvisibility=hidden -I../../proto
LDFLAGS += -shared -Wl,-soname,libcube-gpu.so \
           -Wl,--version-script=libcube_gpu_symbols.ld \
           -Wl,--exclude-libs,ALL
LDLIBS   = -ldl -lpthread

SRC = hook.c rpc_client.c shm_ring.c
OBJ = $(SRC:.c=.o)
OUT = libcube-gpu.so

all: $(OUT)

$(OUT): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(OUT)


.PHONY: all clean
