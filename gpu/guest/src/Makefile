CC      ?= gcc
CFLAGS  ?= -Wall -Wextra -O2 -fPIC -shared
CFLAGS  += -I../../proto
LDFLAGS ?= -Wl,-soname,libcube-gpu.so \
           -Wl,--version-script=libcube_gpu_symbols.ld \
           -Wl,--exclude-libs,ALL \
           -ldl -lpthread

SRC = hook.c rpc_client.c shm_ring.c
OBJ = $(SRC:.c=.o)
OUT = libcube-gpu.so

all: $(OUT)

$(OUT): $(OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(OUT)

.PHONY: all clean
