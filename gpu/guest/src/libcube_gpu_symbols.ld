/*
 * Symbol version script for libcube-gpu.so.
 *
 * The interposer must provide `dlsym` at both GLIBC versions because glibc
 * 2.34 moved the symbol to a new version node; the two definitions come from
 * .symver in hook.c and therefore need their version nodes declared here.
 *
 * The cu* symbols are deliberately NOT listed (and there is no `local: *`
 * catch-all): a version script would attach a GLIBC version to them, while
 * the CUDA driver API exports them unversioned. Internal helpers are hidden
 * at compile time with -fvisibility=hidden instead, so only the
 * CUBE_GPU_EXPORT-marked entry points become dynamic symbols.
 */
GLIBC_2.2.5 {
    global:
        dlsym;
};

GLIBC_2.34 {
    global:
        dlsym;
} GLIBC_2.2.5;
