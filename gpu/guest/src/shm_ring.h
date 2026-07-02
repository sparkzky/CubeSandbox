/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SHM ring buffer: manages data transfer over IVSHMEM between guest and host.
 *
 * Layout of the IVSHMEM region (per session):
 *
 *   [0 .. ring_size-1]                guest-to-host ring (guest writes, host reads)
 *   [ring_size .. 2*ring_size-1]      host-to-guest ring (host writes, guest reads)
 *
 * Each ring has a header at offset 0 with read/write pointers, followed
 * by the circular data area.
 */

#ifndef CUBESANDBOX_GPU_SHM_RING_H
#define CUBESANDBOX_GPU_SHM_RING_H

#include <stdint.h>
#include <stddef.h>

#define CUBE_GPU_SHM_RING_HEADER_SIZE  128

typedef struct __attribute__((__packed__)) cube_gpu_shm_ring_header {
    uint64_t write_pos;
    uint64_t read_pos;
    uint64_t capacity;
    uint32_t epoch;
    uint32_t padding[7];
} cube_gpu_shm_ring_header_t;

typedef struct cube_gpu_shm_ring {
    volatile cube_gpu_shm_ring_header_t *header;
    uint8_t *data;
    size_t data_size;
} cube_gpu_shm_ring_t;

typedef struct cube_gpu_shm_channel {
    void    *base;
    size_t   total_size;
    cube_gpu_shm_ring_t g2h;
    cube_gpu_shm_ring_t h2g;
} cube_gpu_shm_channel_t;

int  cube_gpu_shm_channel_init(cube_gpu_shm_channel_t *ch, void *shm_base, size_t shm_size);
void cube_gpu_shm_channel_destroy(cube_gpu_shm_channel_t *ch);

size_t cube_gpu_shm_ring_write(cube_gpu_shm_ring_t *ring, const void *data, size_t len);
size_t cube_gpu_shm_ring_read(cube_gpu_shm_ring_t *ring, void *buf, size_t len);
size_t cube_gpu_shm_ring_avail_write(const cube_gpu_shm_ring_t *ring);
size_t cube_gpu_shm_ring_avail_read(const cube_gpu_shm_ring_t *ring);

void cube_gpu_shm_ring_reset(cube_gpu_shm_ring_t *ring);

#endif
