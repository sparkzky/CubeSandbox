/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * RPC client: control channel (virtio-vsock) plus the shared-memory data
 * channel (IVSHMEM) used for bulk payloads.
 *
 * Threading contract
 * ------------------
 * The client owns ONE vsock socket. Two threads must never interleave a
 * send/recv pair on it, and a shared-memory transaction's ring writes must
 * stay ordered with the request they belong to. Therefore every request /
 * response round trip AND the ring accesses that bracket it are performed
 * while holding the single global RPC lock:
 *
 *     cube_gpu_rpc_lock();
 *     cube_gpu_shm_ring_write(&client->shm.g2h, ...);
 *     cube_gpu_rpc_call_locked(client, ...);
 *     cube_gpu_rpc_unlock();
 *
 * cube_gpu_rpc_call_locked() must only be called with the lock held.
 * No other lock may be acquired while holding the RPC lock (add
 * cube_gpu_rpc_call() for simple round trips that take the lock itself).
 */

#ifndef CUBESANDBOX_GPU_RPC_CLIENT_H
#define CUBESANDBOX_GPU_RPC_CLIENT_H

#include <stdint.h>
#include <stddef.h>

#include "rpc_protocol.h"
#include "shm_ring.h"

typedef struct cube_gpu_rpc_client {
    int      vsock_fd;
    uint64_t session_id;        /* daemon-assigned session handle */
    uint64_t next_request_id;
    int      initialized;

    void    *shm_base;          /* mmap of the IVSHMEM backing file */
    size_t   shm_total_size;
    cube_gpu_shm_channel_t shm;

    uint32_t device_count;      /* cached from session_init */
    uint32_t max_payload;       /* cached from session_init */
} cube_gpu_rpc_client_t;

/*
 * Connect to the daemon, map the IVSHMEM file (CUBE_GPU_SHM_PATH), perform
 * the v2 session handshake and attach to the rings described by the
 * response. The mmap happens before session_init is sent, as required by
 * the protocol. Returns 0 on success, -1 on any failure.
 *
 * Environment consulted: CUBE_GPU_SHM_PATH, CUBE_GPU_SHM_TOKEN,
 * CUBE_GPU_SM_PERCENT (CID/port are parameters).
 */
int  cube_gpu_rpc_client_init(cube_gpu_rpc_client_t *client, uint32_t host_cid,
                              uint32_t port, uint64_t sandbox_id,
                              uint64_t memory_quota);
void cube_gpu_rpc_client_destroy(cube_gpu_rpc_client_t *client);

/* Global RPC lock (one daemon connection per process). */
void cube_gpu_rpc_lock(void);
void cube_gpu_rpc_unlock(void);

/*
 * Send one request and receive its response while the RPC lock is held.
 * Validates magic/version/api_id/request_id. On success returns 0 and stores
 * the response header status in *out_status (may be NULL). Returns -1 on any
 * transport or protocol error.
 */
int cube_gpu_rpc_call_locked(cube_gpu_rpc_client_t *client, uint32_t api,
                             uint32_t flags, const void *req, size_t req_len,
                             void *resp, size_t resp_cap, uint32_t *out_status);

/* Convenience wrapper taking the lock for a single round trip. */
int cube_gpu_rpc_call(cube_gpu_rpc_client_t *client, uint32_t api,
                      uint32_t flags, const void *req, size_t req_len,
                      void *resp, size_t resp_cap, uint32_t *out_status);

#endif /* CUBESANDBOX_GPU_RPC_CLIENT_H */
