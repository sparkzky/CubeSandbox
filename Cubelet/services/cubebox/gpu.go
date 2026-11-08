// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0
//

package cubebox

import (
	"context"

	"github.com/tencentcloud/CubeSandbox/Cubelet/api/services/errorcode/v1"
	"github.com/tencentcloud/CubeSandbox/Cubelet/pkg/config"
	"github.com/tencentcloud/CubeSandbox/Cubelet/pkg/gpu"
	"github.com/tencentcloud/CubeSandbox/Cubelet/pkg/log"
	"github.com/tencentcloud/CubeSandbox/Cubelet/pkg/ret"
)

// initGPUManager builds and publishes the process-wide GPU manager from the
// host configuration. Discovery failures are non-fatal: the manager then
// reports no GPU inventory and GPU-annotated creates fail with a clear error.
func initGPUManager() {
	cfg := gpu.Config{}
	if c := config.GetConfig(); c != nil && c.HostConf != nil {
		cfg.Enabled = c.HostConf.GPU.Enabled
		cfg.MaxSessionsPerGPU = c.HostConf.GPU.MaxSessionsPerGPU
	}
	mgr := gpu.InitDefault(cfg, gpu.DiscoverNVIDIASMI)
	log.L.Infof("gpu manager initialized: enabled=%t labels=%v", cfg.Enabled, mgr.Labels())
}

// reserveSandboxGPU parses the cube.gpu annotation and reserves node capacity
// before the sandbox is created. It reports whether a reservation was taken so
// the caller knows whether it must attach or release later.
func reserveSandboxGPU(ctx context.Context, sandboxID string, annotations map[string]string) (bool, error) {
	req, found, err := gpu.ParseRequest(annotations)
	if err != nil {
		return false, ret.Errorf(errorcode.ErrorCode_InvalidParamFormat, "%v", err)
	}
	if !found || !req.Enable {
		return false, nil
	}
	mgr := gpu.Default()
	if mgr == nil {
		return false, ret.Err(errorcode.ErrorCode_CreateContainerFailed,
			"gpu requested but the gpu manager is not initialized")
	}
	assignment, err := mgr.Reserve(sandboxID, req)
	if err != nil {
		return false, ret.Errorf(errorcode.ErrorCode_CreateContainerFailed,
			"gpu reserve for sandbox %s failed: %v", sandboxID, err)
	}
	log.G(ctx).Infof("reserved gpu %s (%s) for sandbox %s: %d MiB, sm %d%%",
		assignment.DeviceUUID, assignment.DeviceName, sandboxID,
		assignment.MemoryQuotaMB, assignment.SMPercent)
	return true, nil
}

// attachSandboxGPU marks a reservation as attached once the sandbox exists.
// An inconsistent ledger is quarantined so the capacity cannot be handed out
// again while the sandbox's real GPU usage is unknown.
func attachSandboxGPU(ctx context.Context, sandboxID string) error {
	mgr := gpu.Default()
	if mgr == nil {
		return ret.Err(errorcode.ErrorCode_CreateContainerFailed,
			"gpu requested but the gpu manager is not initialized")
	}
	if err := mgr.Attach(sandboxID); err != nil {
		mgr.Quarantine(sandboxID)
		return ret.Errorf(errorcode.ErrorCode_CreateContainerFailed,
			"gpu attach for sandbox %s failed: %v", sandboxID, err)
	}
	return nil
}

// releaseSandboxGPU is the idempotent teardown path shared by create rollback
// and destroy.
func releaseSandboxGPU(ctx context.Context, sandboxID, reason string) {
	mgr := gpu.Default()
	if mgr == nil || sandboxID == "" {
		return
	}
	if err := mgr.Release(sandboxID); err != nil {
		log.G(ctx).Warnf("release gpu for sandbox %s (%s) failed: %v", sandboxID, reason, err)
		return
	}
	log.G(ctx).Debugf("released gpu reservation for sandbox %s (%s)", sandboxID, reason)
}

// quarantineSandboxGPU pins capacity when a teardown leaves the sandbox state
// ambiguous. The reservation stays charged (and unusable for new sandboxes)
// until a later successful release or a cubelet restart.
func quarantineSandboxGPU(ctx context.Context, sandboxID string, cause error) {
	mgr := gpu.Default()
	if mgr == nil || sandboxID == "" {
		return
	}
	mgr.Quarantine(sandboxID)
	log.G(ctx).Warnf("quarantined gpu reservation for sandbox %s after failed teardown: %v", sandboxID, cause)
}
