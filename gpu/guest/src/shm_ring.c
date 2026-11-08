/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SHM ring buffer implementation (protocol v2).
 *
 * Cursors are 64-bit monotonic counters; positions inside the data area are
 * computed modulo the capacity (the capacity is NOT required to be a power
 * of two). The guest is a single producer on G2H and a single consumer on
 * H2G, and all ring traffic is serialized by the global RPC lock, so plain
 * acquire/release atomics are sufficient for publishing cursors.
 */

#include "shm_ring.h"

#include <string.h>

_Static_assert(sizeof(cube_gpu_shm_ring_header_t) <= CUBE_GPU_SHM_RING_HEADER_SIZE,
               "ring header must fit in its 128-byte slot");
_Static_assert(sizeof(cube_gpu_shm_global_t) <= CUBE_GPU_SHM_GLOBAL_SIZE,
               "global header must fit in its 4096-byte slot");

/* The header is shared with another process: always access the cursors with
 * explicit ordering instead of relying on compiler/CPU ordering. */
#define RING_LOAD(field) \
    __atomic_load_n((const uint64_t *)(const void *)&(field), __ATOMIC_ACQUIRE)
#define RING_STORE(field, value) \
    __atomic_store_n((uint64_t *)(void *)&(field), (value), __ATOMIC_RELEASE)

static uint64_t ring_used(const cube_gpu_shm_ring_t *ring, uint64_t *w_out, uint64_t *r_out) {
    uint64_t cap = ring->data_size;
    uint64_t w = RING_LOAD(ring->header->write_pos);
    uint64_t r = RING_LOAD(ring->header->read_pos);
    if (w_out) *w_out = w;
    if (r_out) *r_out = r;
    if (w >= r) return w - r;
    /* Corrupt/crossing cursor; treat as full to fail safe. */
    return cap;
}

size_t cube_gpu_shm_ring_avail_write(const cube_gpu_shm_ring_t *ring) {
    uint64_t cap = ring->data_size;
    uint64_t used;
    if (!ring->header || !ring->data || cap == 0) return 0;
    used = ring_used(ring, NULL, NULL);
    if (used >= cap) return 0;
    return (size_t)(cap - used - 1);
}

size_t cube_gpu_shm_ring_avail_read(const cube_gpu_shm_ring_t *ring) {
    uint64_t cap = ring->data_size;
    uint64_t used;
    if (!ring->header || !ring->data || cap == 0) return 0;
    used = ring_used(ring, NULL, NULL);
    if (used > cap) used = cap;
    return (size_t)used;
}

size_t cube_gpu_shm_ring_write(cube_gpu_shm_ring_t *ring, const void *data, size_t len) {
    uint64_t cap, w;
    size_t off, first;
    const uint8_t *src = (const uint8_t *)data;

    if (!ring || !ring->header || !ring->data || !data || len == 0) return 0;

    cap = ring->data_size;
    if (len > cube_gpu_shm_ring_avail_write(ring)) return 0; /* never split a message */

    w = RING_LOAD(ring->header->write_pos);
    off = (size_t)(w % cap);
    first = (size_t)(cap - off);
    if (first > len) first = len;

    memcpy(ring->data + off, src, first);
    if (len > first) memcpy(ring->data, src + first, len - first);

    RING_STORE(ring->header->write_pos, w + (uint64_t)len);
    return len;
}

size_t cube_gpu_shm_ring_read(cube_gpu_shm_ring_t *ring, void *buf, size_t len) {
    uint64_t cap, r;
    size_t off, first;
    uint8_t *dst = (uint8_t *)buf;

    if (!ring || !ring->header || !ring->data || !buf || len == 0) return 0;

    cap = ring->data_size;
    if (len > cube_gpu_shm_ring_avail_read(ring)) return 0;

    r = RING_LOAD(ring->header->read_pos);
    off = (size_t)(r % cap);
    first = (size_t)(cap - off);
    if (first > len) first = len;

    memcpy(dst, ring->data + off, first);
    if (len > first) memcpy(dst + first, ring->data, len - first);

    RING_STORE(ring->header->read_pos, r + (uint64_t)len);
    return len;
}

static int ring_attach_one(cube_gpu_shm_ring_t *ring, uint8_t *base, size_t total,
                           uint64_t offset, uint64_t ring_size) {
    cube_gpu_shm_ring_header_t *hdr;
    uint64_t w, r;

    if (!ring_size || offset > (uint64_t)total) return -1;
    if ((uint64_t)total - offset < CUBE_GPU_SHM_RING_HEADER_SIZE) return -1;
    if ((uint64_t)total - offset - CUBE_GPU_SHM_RING_HEADER_SIZE < ring_size) return -1;

    hdr = (cube_gpu_shm_ring_header_t *)(void *)(base + (size_t)offset);
    if (hdr->magic != CUBE_GPU_SHM_RING_MAGIC) return -1;
    if (hdr->capacity != ring_size) return -1;

    w = RING_LOAD(hdr->write_pos);
    r = RING_LOAD(hdr->read_pos);
    if (w < r || w - r > ring_size) return -1;

    ring->header = hdr;
    ring->data = base + (size_t)offset + CUBE_GPU_SHM_RING_HEADER_SIZE;
    ring->data_size = (size_t)ring_size;
    return 0;
}

int cube_gpu_shm_channel_attach(cube_gpu_shm_channel_t *ch, void *base, size_t total,
                                uint64_t g2h_off, uint64_t h2g_off, uint64_t ring_size) {
    uint8_t *bytes = (uint8_t *)base;

    if (!ch || !base || total < sizeof(cube_gpu_shm_global_t)) return -1;
    if (g2h_off < sizeof(cube_gpu_shm_global_t)) return -1;
    if (h2g_off < sizeof(cube_gpu_shm_global_t)) return -1;

    memset(ch, 0, sizeof(*ch));

    if (ring_attach_one(&ch->g2h, bytes, total, g2h_off, ring_size) < 0) return -1;
    if (ring_attach_one(&ch->h2g, bytes, total, h2g_off, ring_size) < 0) {
        cube_gpu_shm_channel_destroy(ch);
        return -1;
    }

    /* The two rings must not overlap (neither with each other nor the
     * global header, which is covered by the offset checks above). */
    if (g2h_off <= h2g_off) {
        if (g2h_off + CUBE_GPU_SHM_RING_HEADER_SIZE + ring_size > h2g_off) {
            cube_gpu_shm_channel_destroy(ch);
            return -1;
        }
    } else {
        if (h2g_off + CUBE_GPU_SHM_RING_HEADER_SIZE + ring_size > g2h_off) {
            cube_gpu_shm_channel_destroy(ch);
            return -1;
        }
    }

    ch->base = base;
    ch->total_size = total;
    ch->g2h_offset = g2h_off;
    ch->h2g_offset = h2g_off;
    return 0;
}

void cube_gpu_shm_channel_destroy(cube_gpu_shm_channel_t *ch) {
    if (!ch) return;
    memset(ch, 0, sizeof(*ch));
}
