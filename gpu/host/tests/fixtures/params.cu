// Test fixture for gpu/host/src/cubin_params.rs.
//
// Regenerate with:
//   nvcc -cubin  -arch=sm_86 -o params.cubin  params.cu
//   nvcc -fatbin -arch=sm_86 -o params.fatbin params.cu
//
// Expected parameter sizes (bytes), validated with `cuobjdump -elf`:
//   noargs -> []
//   mixed  -> [4, 8, 8, 8]
//   big    -> [32, 16]

struct Big {
    double a[4]; // 32 bytes, passed by value
};

extern "C" __global__ void noargs() {}

extern "C" __global__ void mixed(int a, double b, float* c, const char* d) {
    (void)a; (void)b; (void)c; (void)d;
}

extern "C" __global__ void big(Big s, int4 v) {
    (void)s; (void)v;
}
