# CubeSandbox GPU end-to-end test assets.
#
#   make kernels   -> kernel.cubin / kernel.ptx (committed fixtures)
#   make programs  -> e2e_main / e2e_link / vsock_probe
#   make           -> both
#   make clean     -> remove programs only (kernel binaries are checked in)

CC        ?= cc
NVCC      ?= nvcc
CUDA_ARCH ?= sm_86

CFLAGS    ?= -O2 -Wall -Wextra
NVCCFLAGS ?= -O3

KERNEL_BIN = kernel.cubin
KERNEL_PTX = kernel.ptx

PROGRAMS = e2e_main e2e_link vsock_probe session_hold

all: kernels programs

kernels: $(KERNEL_BIN) $(KERNEL_PTX)

$(KERNEL_BIN): kernel.cu
	$(NVCC) -cubin $(NVCCFLAGS) -arch=$(CUDA_ARCH) -o $@ $<

$(KERNEL_PTX): kernel.cu
	$(NVCC) -ptx $(NVCCFLAGS) -arch=$(CUDA_ARCH) -o $@ $<

programs: $(PROGRAMS)

e2e_main: e2e_main.c
	$(CC) $(CFLAGS) -I../proto -o $@ $< -ldl -lpthread

e2e_link: e2e_link.c
	$(CC) $(CFLAGS) -o $@ $< -lcuda -ldl

vsock_probe: vsock_probe.c
	$(CC) $(CFLAGS) -o $@ $<

session_hold: session_hold.c
	$(CC) $(CFLAGS) -o $@ $< -ldl

clean:
	rm -f $(PROGRAMS) *.o

distclean: clean
	rm -f $(KERNEL_BIN) $(KERNEL_PTX)

.PHONY: all kernels programs clean distclean
