/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SHM ring buffer implementation.
 */

#include "shm_ring.h"
#include <string.h>
#include <stdatomic.h>

static inline void cache_flush(volatile void *p) {
    __asm__ __volatile__("" : : "r"(p) : "memory");
}

int cube_gpu_shm_channel_init(cube_gpu_shm_channel_t *ch, void *shm_base, size_t shm_size) {
    if (!shm_base || shm_size < CUBE_GPU_SHM_RING_HEADER_SIZE * 2 + 64) return -1;

    ch->base = shm_base;
    ch->total_size = shm_size;

    size_t half = shm_size / 2;

    uint8_t *g2h_base = (uint8_t *)shm_base;
    uint8_t *h2g_base = (uint8_t *)shm_base + half;

    ch->g2h.header = (volatile cube_gpu_shm_ring_header_t *)g2h_base;
    ch->g2h.data   = g2h_base + CUBE_GPU_SHM_RING_HEADER_SIZE;
    ch->g2h.data_size = half - CUBE_GPU_SHM_RING_HEADER_SIZE;

    ch->h2g.header = (volatile cube_gpu_shm_ring_header_t *)h2g_base;
    ch->h2g.data   = h2g_base + CUBE_GPU_SHM_RING_HEADER_SIZE;
    ch->h2g.data_size = half - CUBE_GPU_SHM_RING_HEADER_SIZE;

    cube_gpu_shm_ring_header_t init_hdr = {0};
    init_hdr.capacity = ch->g2h.data_size;
    memcpy((void *)ch->g2h.header, &init_hdr, sizeof(init_hdr));

    init_hdr.capacity = ch->h2g.data_size;
    memcpy((void *)ch->h2g.header, &init_hdr, sizeof(init_hdr));

    return 0;
}

void cube_gpu_shm_channel_destroy(cube_gpu_shm_channel_t *ch) {
    ch->base = NULL;
    ch->total_size = 0;
}

size_t cube_gpu_shm_ring_avail_write(const cube_gpu_shm_ring_t *ring) {
    uint64_t w = atomic_load((_Atomic uint64_t *)&ring->header->write_pos);
    uint64_t r = atomic_load((_Atomic uint64_t *)&ring->header->read_pos);
    if (w >= r) return ring->data_size - (w - r) - 1;
    return (r - w) - 1;
}

size_t cube_gpu_shm_ring_avail_read(const cube_gpu_shm_ring_t *ring) {
    uint64_t w = atomic_load((_Atomic uint64_t *)&ring->header->write_pos);
    uint64_t r = atomic_load((_Atomic uint64_t *)&ring->header->read_pos);
    if (w >= r) return w - r;
    return ring->data_size - (r - w);
}

size_t cube_gpu_shm_ring_write(cube_gpu_shm_ring_t *ring, const void *data, size_t len) {
    if (!data || len == 0) return 0;
    size_t avail = cube_gpu_shm_ring_avail_write(ring);
    if (avail == 0) return 0;
    size_t to_write = len < avail ? len : avail;

    uint64_t w = atomic_load((_Atomic uint64_t *)&ring->header->write_pos);
    uint64_t mask = ring->data_size - 1;

    size_t first_chunk = ring->data_size - (w % ring->data_size);
    if (first_chunk > to_write) first_chunk = to_write;

    memcpy(ring->data + (w % ring->data_size), data, first_chunk);
    if (to_write > first_chunk) {
        memcpy(ring->data, (const uint8_t *)data + first_chunk, to_write - first_chunk);
    }

    atomic_store((_Atomic uint64_t *)&ring->header->write_pos, w + to_write);
    cache_flush(&ring->header->write_pos);
    return to_write;
}

size_t cube_gpu_shm_ring_read(cube_gpu_shm_ring_t *ring, void *buf, size_t len) {
    if (!buf || len == 0) return 0;
    size_t avail = cube_gpu_shm_ring_avail_read(ring);
    if (avail == 0) return 0;
    size_t to_read = len < avail ? len : avail;

    uint64_t r = atomic_load((_Atomic uint64_t *)&ring->header->read_pos);

    size_t first_chunk = ring->data_size - (r % ring->data_size);
    if (first_chunk > to_read) first_chunk = to_read;

    memcpy(buf, ring->data + (r % ring->data_size), first_chunk);
    if (to_read > first_chunk) {
        memcpy((uint8_t *)buf + first_chunk, ring->data, to_read - first_chunk);
    }

    atomic_store((_Atomic uint64_t *)&ring->header->read_pos, r + to_read);
    cache_flush(&ring->header->read_pos);
    return to_read;
}

void cube_gpu_shm_ring_reset(cube_gpu_shm_ring_t *ring) {
    atomic_store((_Atomic uint64_t *)&ring->header->write_pos, 0);
    atomic_store((_Atomic uint64_t *)&ring->header->read_pos, 0);
    cache_flush(&ring->header->write_pos);
    cache_flush(&ring->header->read_pos);
}
