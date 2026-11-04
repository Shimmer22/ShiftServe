/* Tiny userspace memcpy / page-aligned copy microbench for Android arm64.
 * Not Hexagon DMA. Usage: memcpy_bench <bytes> <iters> [page_align=1]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <bytes> <iters> [page_align=1]\n", argv[0]);
        return 2;
    }
    size_t nbytes = (size_t)strtoull(argv[1], NULL, 10);
    int iters = atoi(argv[2]);
    int page_align = (argc >= 4) ? atoi(argv[3]) : 1;
    if (nbytes == 0 || iters <= 0) return 2;

    long psz = sysconf(_SC_PAGESIZE);
    if (psz <= 0) psz = 4096;
    size_t alloc = nbytes + (size_t)psz;
    void *raw_src = malloc(alloc);
    void *raw_dst = malloc(alloc);
    if (!raw_src || !raw_dst) {
        fprintf(stderr, "malloc failed errno=%d\n", errno);
        return 1;
    }
    unsigned char *src, *dst;
    if (page_align) {
        src = (unsigned char *)(((uintptr_t)raw_src + (uintptr_t)psz - 1) & ~((uintptr_t)psz - 1));
        dst = (unsigned char *)(((uintptr_t)raw_dst + (uintptr_t)psz - 1) & ~((uintptr_t)psz - 1));
    } else {
        src = (unsigned char *)raw_src;
        dst = (unsigned char *)raw_dst;
    }
    memset(src, 0xA5, nbytes);
    memset(dst, 0x00, nbytes);

    /* warmup */
    for (int i = 0; i < 2; i++) memcpy(dst, src, nbytes);

    double t0 = now_s();
    for (int i = 0; i < iters; i++) memcpy(dst, src, nbytes);
    double t1 = now_s();
    double elapsed_ms = (t1 - t0) * 1e3;
    double per_ms = elapsed_ms / (double)iters;
    double gib_s = (nbytes / (1024.0 * 1024.0 * 1024.0)) / ((t1 - t0) / (double)iters);
    /* touch result so optimizer keeps copy */
    unsigned char sink = dst[0] ^ dst[nbytes - 1];
    printf("bytes=%zu iters=%d page_align=%d total_ms=%.3f per_ms=%.6f gib_s=%.3f sink=%u pagesize=%ld\n",
           nbytes, iters, page_align, elapsed_ms, per_ms, gib_s, (unsigned)sink, psz);
    free(raw_src);
    free(raw_dst);
    return 0;
}
