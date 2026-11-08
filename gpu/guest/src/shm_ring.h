/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SHM ring buffer: data plane over the IVSHMEM file shared with the host
 * daemon. The wire layout is defined by gpu/proto/rpc_protocol.h (v2):
 *
 *   [0, 4096)                          cube_gpu_shm_global_t
 *   [g2h_offset, ...)                  G2H ring header (128 B) + data
 *   [h2g_offset, ...)                  H2G ring header (128 B) + data
 *
 * Both processes map the same backing file; the daemon owns the layout and
 * initializes the ring headers. The guest attaches read/write but NEVER
 * resets the cursors (another process is using them concurrently).
 *
 * The G2H ring has a single writer (guest) and single reader (daemon), and
 * vice versa. The guest serializes all ring traffic behind the RPC lock, so
 * a transaction's data is never interleaved with another transaction.
 */

#ifndef CUBESANDBOX_GPU_SHM_RING_H
#define CUBESANDBOX_GPU_SHM_RING_H

#include <stdint.h>
#include <stddef.h>

#include "rpc_protocol.h"

typedef struct cube_gpu_shm_ring {
    volatile cube_gpu_shm_ring_header_t *header;
    uint8_t *data;
    size_t   data_size;      /* == header->capacity */
} cube_gpu_shm_ring_t;

typedef struct cube_gpu_shm_channel {
    void     *base;
    size_t    total_size;
    uint64_t  g2h_offset;
    uint64_t  h2g_offset;
    cube_gpu_shm_ring_t g2h;   /* guest→host */
    cube_gpu_shm_ring_t h2g;   /* host→guest */
} cube_gpu_shm_channel_t;

/*
 * Attach to an already-initialized shared-memory region.
 *
 * Validates that both rings fit inside `total`, that both ring headers carry
 * CUBE_GPU_SHM_RING_MAGIC and that header->capacity == ring_size. Cursors
 * are only read, never written. Returns 0 on success, -1 on any mismatch.
 */
int cube_gpu_shm_channel_attach(cube_gpu_shm_channel_t *ch, void *base, size_t total,
                                uint64_t g2h_off, uint64_t h2g_off, uint64_t ring_size);

void cube_gpu_shm_channel_destroy(cube_gpu_shm_channel_t *ch);

/*
 * All-or-nothing transfers: a write/read returns the number of bytes moved
 * and returns 0 (leaving the ring untouched) when the requested length does
 * not fit. Framed messages must never be partially written.
 */
size_t cube_gpu_shm_ring_write(cube_gpu_shm_ring_t *ring, const void *data, size_t len);
size_t cube_gpu_shm_ring_read(cube_gpu_shm_ring_t *ring, void *buf, size_t len);

/* Usable bytes. One byte of the capacity is kept free to distinguish full
 * from empty without a separate counter (same rule as the daemon). */
size_t cube_gpu_shm_ring_avail_write(const cube_gpu_shm_ring_t *ring);
size_t cube_gpu_shm_ring_avail_read(const cube_gpu_shm_ring_t *ring);

#endif /* CUBESANDBOX_GPU_SHM_RING_H */
