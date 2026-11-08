/*
 * Copyright (c) 2024 Tencent Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * RPC client implementation: vsock transport + IVSHMEM data plane setup.
 */

#define _GNU_SOURCE
#include "rpc_client.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <linux/vm_sockets.h>
/* Local integer parsing: avoid glibc's __isoc23_strtoull redirect (GLIBC_2.38)
 * so the library stays loadable on older guest userspaces (e.g. glibc 2.36). */
static unsigned long long cube_parse_u64(const char *s, int base) {
    unsigned long long v = 0;
    int neg = 0;
    if (!s) return 0;
    while (*s == ' ' || *s == '\t' || *s == '\n') s++;
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') s++;
    if (base == 0) {
        if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) { base = 16; s += 2; }
        else if (s[0] == '0') { base = 8; s++; }
        else base = 10;
    } else if (base == 16 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
    }
    for (;; s++) {
        int d;
        char c = *s;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else break;
        if (d >= base) break;
        v = v * (unsigned)base + (unsigned)d;
    }
    return neg ? (unsigned long long)(-(long long)v) : v;
}
#define strtoull(s, e, b) cube_parse_u64((s), (b))
#define strtoul(s, e, b) ((unsigned long)cube_parse_u64((s), (b)))


#define ENV_SHM_PATH     "CUBE_GPU_SHM_PATH"
#define ENV_SHM_TOKEN    "CUBE_GPU_SHM_TOKEN"
#define ENV_SM_PERCENT   "CUBE_GPU_SM_PERCENT"
#define ENV_DEBUG        "CUBE_GPU_DEBUG"

#define log_err(fmt, ...) \
    fprintf(stderr, "[cube-gpu] ERROR: " fmt "\n", ##__VA_ARGS__)
#define log_warn(fmt, ...) \
    fprintf(stderr, "[cube-gpu] WARN: " fmt "\n", ##__VA_ARGS__)
#define log_dbg(fmt, ...) \
    do { if (g_rpc_debug) fprintf(stderr, "[cube-gpu] DBG: " fmt "\n", ##__VA_ARGS__); } while (0)

static int g_rpc_debug = 0;

/* ─── Global RPC lock ─────────────────────────────────────────── */

static pthread_mutex_t g_rpc_mutex = PTHREAD_MUTEX_INITIALIZER;

void cube_gpu_rpc_lock(void) { pthread_mutex_lock(&g_rpc_mutex); }
void cube_gpu_rpc_unlock(void) { pthread_mutex_unlock(&g_rpc_mutex); }

/* ─── Socket helpers ──────────────────────────────────────────── */

static int vsock_connect(uint32_t cid, uint32_t port, int quiet) {
    int fd;
    struct sockaddr_vm addr;

    fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    if (fd < 0) {
        log_err("socket(AF_VSOCK): %s", strerror(errno));
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.svm_family = AF_VSOCK;
    addr.svm_cid = cid;
    addr.svm_port = port;

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        if (!quiet)
            log_err("connect(vsock cid=%u port=%u): %s", cid, port, strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}

static int send_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int recv_all(int fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    while (len > 0) {
        ssize_t n = read(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1; /* peer closed */
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

/* Read at most `cap` payload bytes into `buf`, discarding the remainder so
 * the stream stays aligned for the next request. */
static int recv_payload(int fd, uint32_t payload_len, void *buf, size_t cap) {
    size_t to_read = payload_len < cap ? (size_t)payload_len : cap;
    uint8_t discard[256];
    size_t left;

    if (to_read > 0) {
        if (recv_all(fd, buf, to_read) < 0) return -1;
    }
    if (cap > to_read && buf) memset((uint8_t *)buf + to_read, 0, cap - to_read);

    left = (size_t)payload_len - to_read;
    while (left > 0) {
        size_t chunk = left < sizeof(discard) ? left : sizeof(discard);
        if (recv_all(fd, discard, chunk) < 0) return -1;
        left -= chunk;
    }
    return 0;
}

/* ─── IVSHMEM mapping & token ─────────────────────────────────── */

/* CubeSandbox exposes the shared-memory window as an IVSHMEM PCI device
 * (vendor 0x1af4, device 0x1110) whose BAR2 is the backing file. Inside the
 * MicroVM there is no /dev/shm/cube-gpu-shm node, so the library locates the
 * device through sysfs and mmaps resource2 directly. */
static int read_sysfs_u32(const char *path, unsigned long *out) {
    char buf[32];
    int fd = open(path, O_RDONLY);
    ssize_t n;

    if (fd < 0) return -1;
    n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    *out = strtoul(buf, NULL, 0);
    return 0;
}

/* The sandbox VM is restored from a memory snapshot, so devices attached on
 * the host after the snapshot (IVSHMEM for GPU sharing) are not enumerated by
 * the guest kernel until a PCI rescan. */
static void trigger_pci_rescan(void) {
    int fd = open("/sys/bus/pci/rescan", O_WRONLY);
    if (fd >= 0) {
        ssize_t n = write(fd, "1\n", 2);
        (void)n;
        close(fd);
    }
}

static int scan_ivshmem_bar(void) {
    const char *base = "/sys/bus/pci/devices";
    DIR *dir = opendir(base);
    struct dirent *ent;

    if (!dir) return -1;
    while ((ent = readdir(dir)) != NULL) {
        char path[512];
        unsigned long vendor = 0, device = 0;

        if (ent->d_name[0] == '.') continue;
        snprintf(path, sizeof(path), "%s/%s/vendor", base, ent->d_name);
        if (read_sysfs_u32(path, &vendor) < 0) continue;
        if (vendor != 0x1af4) continue;
        snprintf(path, sizeof(path), "%s/%s/device", base, ent->d_name);
        if (read_sysfs_u32(path, &device) < 0) continue;
        if (device != 0x1110) continue;

        snprintf(path, sizeof(path), "%s/%s/resource2", base, ent->d_name);
        int fd = open(path, O_RDWR);
        if (fd >= 0) {
            log_dbg("using IVSHMEM BAR2 %s", path);
            closedir(dir);
            return fd;
        }
        log_warn("open(%s): %s", path, strerror(errno));
    }
    closedir(dir);
    return -1;
}

static int open_ivshmem_bar(void) {
    int fd = scan_ivshmem_bar();
    if (fd < 0) {
        log_dbg("IVSHMEM device not enumerated; triggering PCI rescan");
        trigger_pci_rescan();
        fd = scan_ivshmem_bar();
    }
    return fd;
}

static void *map_shm_file(const char *path, size_t *out_size) {
    struct stat st;
    size_t size;
    void *base;
    int fd = -1;

    *out_size = 0;

    if (path && path[0] != '\0') {
        fd = open(path, O_RDWR);
        if (fd < 0) {
            log_warn("open(%s): %s; falling back to IVSHMEM PCI BAR",
                     path, strerror(errno));
        }
    }

    if (fd < 0) {
        fd = open_ivshmem_bar();
        if (fd < 0) {
            log_warn("no shared-memory backing (env path and IVSHMEM PCI BAR unavailable); "
                     "data channel disabled");
            return NULL;
        }
    }

    if (fstat(fd, &st) == 0 && st.st_size > 0) {
        size = (size_t)st.st_size;
    } else {
        size = (size_t)CUBE_GPU_SHM_RING_SIZE * 2;
        log_warn("cannot stat SHM backing; assuming %zu bytes", size);
    }

    base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        log_err("mmap(%zu): %s", size, strerror(errno));
        close(fd);
        return NULL;
    }
    close(fd);

    log_dbg("mapped SHM backing: %zu bytes at %p", size, base);
    *out_size = size;
    return base;
}

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* The shim exports the token as 64 lowercase hex characters (32 bytes). */
static int parse_shm_token(const char *hex, uint8_t out[CUBE_GPU_SHM_TOKEN_LEN]) {
    size_t i;

    memset(out, 0, CUBE_GPU_SHM_TOKEN_LEN);
    if (!hex) return 0;
    if (hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) hex += 2;
    if (strlen(hex) != CUBE_GPU_SHM_TOKEN_LEN * 2) return 0;

    for (i = 0; i < CUBE_GPU_SHM_TOKEN_LEN; i++) {
        int hi = hex_nibble(hex[i * 2]);
        int lo = hex_nibble(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) {
            memset(out, 0, CUBE_GPU_SHM_TOKEN_LEN);
            return 0;
        }
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return 1;
}

static int token_nonzero(const uint8_t token[CUBE_GPU_SHM_TOKEN_LEN]) {
    size_t i;
    for (i = 0; i < CUBE_GPU_SHM_TOKEN_LEN; i++) {
        if (token[i] != 0) return 1;
    }
    return 0;
}

static uint64_t env_u64(const char *name, uint64_t fallback) {
    const char *v = getenv(name);
    if (!v || v[0] == '\0') return fallback;
    return strtoull(v, NULL, 10);
}

/* ─── Raw send / recv (caller holds the RPC lock) ─────────────── */

static int rpc_send(cube_gpu_rpc_client_t *client, const cube_gpu_rpc_header_t *hdr,
                    const void *payload) {
    if (send_all(client->vsock_fd, hdr, sizeof(*hdr)) < 0) return -1;
    if (hdr->payload_len > 0 && payload) {
        if (send_all(client->vsock_fd, payload, hdr->payload_len) < 0) return -1;
    }
    return 0;
}

static int rpc_recv(cube_gpu_rpc_client_t *client, cube_gpu_rpc_header_t *hdr_out,
                    void *payload_out, size_t payload_buf_size) {
    if (recv_all(client->vsock_fd, hdr_out, sizeof(*hdr_out)) < 0) return -1;
    if (hdr_out->magic != CUBE_GPU_RPC_MAGIC) return -1;
    if (hdr_out->version != CUBE_GPU_RPC_VERSION) return -1;
    if (!(hdr_out->flags & CUBE_GPU_FLAG_RESPONSE)) return -1;
    if (recv_payload(client->vsock_fd, hdr_out->payload_len,
                     payload_out, payload_buf_size) < 0) return -1;
    return 0;
}

int cube_gpu_rpc_call_locked(cube_gpu_rpc_client_t *client, uint32_t api,
                             uint32_t flags, const void *req, size_t req_len,
                             void *resp, size_t resp_cap, uint32_t *out_status) {
    cube_gpu_rpc_header_t hdr;
    cube_gpu_rpc_header_t rh;
    uint64_t request_id;

    if (!client || client->vsock_fd < 0) return -1;

    request_id = client->next_request_id++;
    cube_gpu_init_header(&hdr, (cube_gpu_api_id_t)api, request_id, client->session_id,
                         (uint32_t)req_len, flags);

    if (rpc_send(client, &hdr, req) < 0) {
        log_err("rpc_send failed (api=%u)", api);
        return -1;
    }

    if (rpc_recv(client, &rh, resp, resp_cap) < 0) {
        log_err("rpc_recv failed (api=%u)", api);
        return -1;
    }

    if (rh.api_id != api || rh.request_id != request_id) {
        log_err("rpc response mismatch (api=%u got api=%u rid=%llu want %llu)",
                api, rh.api_id, (unsigned long long)rh.request_id,
                (unsigned long long)request_id);
        return -1;
    }
    if (rh.payload_len > resp_cap) {
        log_err("rpc response too large (api=%u len=%u cap=%zu)", api, rh.payload_len, resp_cap);
        return -1;
    }

    if (out_status) *out_status = rh.status;
    return 0;
}

int cube_gpu_rpc_call(cube_gpu_rpc_client_t *client, uint32_t api,
                      uint32_t flags, const void *req, size_t req_len,
                      void *resp, size_t resp_cap, uint32_t *out_status) {
    int rc;
    cube_gpu_rpc_lock();
    rc = cube_gpu_rpc_call_locked(client, api, flags, req, req_len, resp, resp_cap, out_status);
    cube_gpu_rpc_unlock();
    return rc;
}

/* ─── Session lifecycle ──────────────────────────────────────── */

int cube_gpu_rpc_client_init(cube_gpu_rpc_client_t *client, uint32_t host_cid,
                             uint32_t port, uint64_t sandbox_id,
                             uint64_t memory_quota) {
    cube_gpu_session_init_req_t req;
    cube_gpu_session_init_resp_t resp;
    cube_gpu_rpc_header_t rh;
    cube_gpu_rpc_header_t hdr;
    uint8_t token[CUBE_GPU_SHM_TOKEN_LEN];
    uint64_t request_id;
    void *shm_base = NULL;
    size_t shm_size = 0;
    const char *dbg;

    memset(client, 0, sizeof(*client));
    client->vsock_fd = -1;
    client->next_request_id = 1;

    dbg = getenv(ENV_DEBUG);
    g_rpc_debug = (dbg && dbg[0] == '1');

    /* The daemon may still be binding the per-sandbox unix-vsock socket
     * (or starting at all), so retry the connect for a few seconds. */
    for (int attempt = 0; attempt < 100; attempt++) {
        client->vsock_fd = vsock_connect(host_cid, port, attempt != 99);
        if (client->vsock_fd >= 0) break;
        usleep(100 * 1000);
    }
    if (client->vsock_fd < 0) return -1;

    /* The mapping must exist before session_init is sent: the daemon uses
     * the reported size to lay out the rings, and both sides must map the
     * same backing file. */
    shm_base = map_shm_file(getenv(ENV_SHM_PATH), &shm_size);

    /* Zero-config bootstrap: when the sandbox was restored from a memory
     * snapshot the shim-injected environment is not visible to new
     * processes, but the shim always writes identity into the shared-memory
     * global header before the VM boots. */
    parse_shm_token(getenv(ENV_SHM_TOKEN), token);
    if (shm_base && shm_size >= sizeof(cube_gpu_shm_global_t)) {
        cube_gpu_shm_global_t gh;
        memcpy(&gh, shm_base, sizeof(gh));
        if (gh.magic == CUBE_GPU_SHM_MAGIC) {
            if (sandbox_id == 0) sandbox_id = gh.session_id;
            if (memory_quota == 0) memory_quota = gh.memory_quota;
            if (!token_nonzero(token)) memcpy(token, gh.token, sizeof(token));
            {
                uint64_t w72 = 0, w80 = 0, w88 = 0, w96 = 0;
                memcpy(&w72, (const char *)shm_base + 72, 8);
                memcpy(&w80, (const char *)shm_base + 80, 8);
                memcpy(&w88, (const char *)shm_base + 88, 8);
                memcpy(&w96, (const char *)shm_base + 96, 8);
                log_dbg("shm header raw: magic=%llx ver=%u w72=%llx w80=%llx w88=%llx w96=%llx",
                        (unsigned long long)gh.magic, gh.version,
                        (unsigned long long)w72, (unsigned long long)w80,
                        (unsigned long long)w88, (unsigned long long)w96);
            }
            log_dbg("bootstrap from SHM header: sandbox=%llu quota=%llu",
                    (unsigned long long)sandbox_id,
                    (unsigned long long)memory_quota);
        }
    }
    if (memory_quota == 0) {
        memory_quota = 4ULL * 1024 * 1024 * 1024;
        log_warn("no memory quota in env or SHM header; defaulting to 4GiB");
    }

    memset(&req, 0, sizeof(req));
    req.memory_quota = memory_quota;
    req.sandbox_id = sandbox_id;
    req.sm_percent = env_u64(ENV_SM_PERCENT, 0);
    req.shm_size_mb = (uint32_t)(shm_size >> 20);

    if (token_nonzero(token)) req.flags |= CUBE_GPU_SESSION_FLAG_TOKEN;
    else log_warn("%s missing/invalid; sending an all-zero token", ENV_SHM_TOKEN);
    memcpy(req.shm_token, token, sizeof(token));

    request_id = client->next_request_id++;
    cube_gpu_init_header(&hdr, CUBE_GPU_API_SESSION_INIT, request_id, sandbox_id,
                         (uint32_t)sizeof(req), CUBE_GPU_FLAG_NONE);

    if (rpc_send(client, &hdr, &req) < 0) {
        log_err("session_init send failed");
        goto fail;
    }

    memset(&resp, 0, sizeof(resp));
    if (rpc_recv(client, &rh, &resp, sizeof(resp)) < 0) {
        log_err("session_init receive failed");
        goto fail;
    }
    if (rh.version != CUBE_GPU_RPC_VERSION) {
        log_err("session_init: daemon protocol version %u, expected %u",
                rh.version, CUBE_GPU_RPC_VERSION);
        goto fail;
    }
    if (rh.api_id != CUBE_GPU_API_SESSION_INIT || rh.request_id != request_id) {
        log_err("session_init: unexpected response (api=%u rid=%llu)",
                rh.api_id, (unsigned long long)rh.request_id);
        goto fail;
    }
    if (rh.payload_len < sizeof(resp)) {
        log_err("session_init: short response (%u < %zu)", rh.payload_len, sizeof(resp));
        goto fail;
    }
    if (resp.status != 0) {
        log_err("session_init rejected by daemon (status=%u)", resp.status);
        goto fail;
    }

    if (shm_base) {
        if (resp.shm_total_size != 0 && resp.shm_total_size != (uint64_t)shm_size) {
            log_warn("shm size mismatch: guest mapped %zu, daemon reports %llu",
                     shm_size, (unsigned long long)resp.shm_total_size);
        }
        if (cube_gpu_shm_channel_attach(&client->shm, shm_base, shm_size,
                                        resp.g2h_offset, resp.h2g_offset,
                                        resp.ring_size) < 0) {
            log_err("SHM attach failed (total=%zu ring=%llu g2h=%llu h2g=%llu)",
                    shm_size, (unsigned long long)resp.ring_size,
                    (unsigned long long)resp.g2h_offset,
                    (unsigned long long)resp.h2g_offset);
            goto fail;
        }
        client->shm_base = shm_base;
        client->shm_total_size = shm_size;
        shm_base = NULL;
    } else if (resp.ring_size != 0) {
        log_warn("daemon expects a %llu-byte ring but no SHM file was mapped; "
                 "bulk transfers will fail", (unsigned long long)resp.ring_size);
    }

    client->session_id = resp.session_handle;
    client->device_count = resp.device_count;
    client->max_payload = resp.max_payload;
    client->initialized = 1;

    log_dbg("session %llu ready: devices=%u ring=%llu max_payload=%u",
            (unsigned long long)client->session_id, client->device_count,
            (unsigned long long)resp.ring_size, client->max_payload);
    return 0;

fail:
    if (shm_base) munmap(shm_base, shm_size);
    if (client->vsock_fd >= 0) {
        close(client->vsock_fd);
        client->vsock_fd = -1;
    }
    cube_gpu_shm_channel_destroy(&client->shm);
    client->shm_base = NULL;
    client->shm_total_size = 0;
    client->initialized = 0;
    return -1;
}

void cube_gpu_rpc_client_destroy(cube_gpu_rpc_client_t *client) {
    if (!client) return;

    if (client->vsock_fd >= 0) {
        cube_gpu_rpc_header_t hdr;
        cube_gpu_init_header(&hdr, CUBE_GPU_API_SESSION_DESTROY,
                             client->next_request_id++, client->session_id, 0,
                             CUBE_GPU_FLAG_NONE);
        (void)rpc_send(client, &hdr, NULL);
        close(client->vsock_fd);
        client->vsock_fd = -1;
    }
    if (client->shm_base) {
        munmap(client->shm_base, client->shm_total_size);
        client->shm_base = NULL;
        client->shm_total_size = 0;
    }
    cube_gpu_shm_channel_destroy(&client->shm);
    client->initialized = 0;
}
