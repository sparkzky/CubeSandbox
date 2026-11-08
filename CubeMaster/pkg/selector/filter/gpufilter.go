// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0
//

package filter

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/tencentcloud/CubeSandbox/CubeMaster/pkg/base/constants"
	"github.com/tencentcloud/CubeSandbox/CubeMaster/pkg/base/log"
	"github.com/tencentcloud/CubeSandbox/CubeMaster/pkg/base/node"
	"github.com/tencentcloud/CubeSandbox/CubeMaster/pkg/scheduler/selctx"
)

// Node labels reported by Cubelet's GPU manager. Values are strings; the
// numeric labels are parsed leniently and a missing/unparsable label can only
// make the node ineligible (never silently satisfy the constraint).
const (
	gpuLabelPresent     = "cube.gpu.present"
	gpuLabelCount       = "cube.gpu.count"
	gpuLabelMemTotalMiB = "cube.gpu.memory.total.mib"
	gpuLabelMemUsedMiB  = "cube.gpu.memory.used.mib"
	gpuLabelSessions    = "cube.gpu.sessions"
	gpuLabelMaxSessions = "cube.gpu.max.sessions"
	gpuLabelDriver      = "cube.gpu.driver"
	gpuLabelModels      = "cube.gpu.models"
)

const defaultGPUMemoryQuotaMB int64 = 4096

// gpuRequest is the structured form of the cube.gpu sandbox annotation.
type gpuRequest struct {
	Enable            bool     `json:"enable"`
	MemoryQuotaMB     int64    `json:"memory_quota_mb"`
	SMPercent         int      `json:"sm_percent"`
	RequiredDriverMin string   `json:"required_driver_min"`
	Models            []string `json:"models"`
}

// gpuFilter gates placement of GPU-backed sandboxes on Cubelet-reported GPU
// inventory labels. Non-GPU requests are never affected.
type gpuFilter struct {
}

func NewGpuFilter() *gpuFilter {
	return &gpuFilter{}
}

func (l *gpuFilter) ID() string {
	return constants.SelectorFilterID + "/" + "gpu"
}

func (l *gpuFilter) String() string {
	return l.ID()
}

func (l *gpuFilter) Select(selCtx *selctx.SelectorCtx) (node.NodeList, error) {
	inList := selCtx.Nodes()

	req, gpuEnabled, err := parseGPURequest(selCtx.GetReqRes())
	if err != nil {
		// A malformed cube.gpu annotation must reject placement rather than
		// fall through to a node without a GPU.
		log.G(selCtx.Ctx).Errorf("%v reject all nodes: invalid %s annotation: %v",
			l.ID(), constants.AnnotationsGPU, err)
		return nil, fmt.Errorf("%v: invalid %s annotation: %w", l.ID(), constants.AnnotationsGPU, err)
	}
	if !gpuEnabled {
		return inList, nil
	}

	nodes := make(node.NodeList, 0, inList.Len())
	for i := range inList {
		n := inList[i]
		if reason := gpuNodeIneligible(n, req); reason != "" {
			log.G(selCtx.Ctx).Warnf("%v select:%v reject: %s", l.ID(), n.ID(), reason)
			continue
		}
		nodes.Append(n)
	}
	if log.IsDebug() {
		log.G(selCtx.Ctx).Debugf("%v select:%v", l.ID(), nodes.String())
	} else {
		log.G(selCtx.Ctx).Infof("%v select_size:%v", l.ID(), nodes.Len())
	}
	return nodes, nil
}

// RequestNeedsGPU reports whether the request asks for GPU resources or
// carries an annotation that cannot be interpreted. The scheduler uses this to
// make sure the backoff path cannot silently place a GPU sandbox on a node
// that failed the GPU filter.
func RequestNeedsGPU(reqRes *selctx.RequestResource) bool {
	_, enabled, err := parseGPURequest(reqRes)
	return enabled || err != nil
}

func parseGPURequest(reqRes *selctx.RequestResource) (*gpuRequest, bool, error) {
	if reqRes == nil {
		return nil, false, nil
	}
	raw, ok := reqRes.Annotations[constants.AnnotationsGPU]
	if !ok || strings.TrimSpace(raw) == "" {
		return nil, false, nil
	}

	// Backward compatibility with the original boolean annotation form.
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "true":
		return &gpuRequest{Enable: true, MemoryQuotaMB: defaultGPUMemoryQuotaMB}, true, nil
	case "false":
		return nil, false, nil
	}
	req := &gpuRequest{}
	if err := json.Unmarshal([]byte(raw), req); err != nil {
		return nil, false, err
	}
	if req.MemoryQuotaMB <= 0 {
		req.MemoryQuotaMB = defaultGPUMemoryQuotaMB
	}
	if !req.Enable {
		return nil, false, nil
	}
	return req, true, nil
}

func gpuNodeIneligible(n *node.Node, req *gpuRequest) string {
	if n == nil {
		return "nil node"
	}
	labels := n.Labels()

	if !strings.EqualFold(strings.TrimSpace(labels[gpuLabelPresent]), "true") {
		return fmt.Sprintf("%s!=true", gpuLabelPresent)
	}

	count, ok := parseLabelInt(labels, gpuLabelCount)
	if !ok || count < 1 {
		return fmt.Sprintf("%s invalid (%q)", gpuLabelCount, labels[gpuLabelCount])
	}

	totalMiB, ok := parseLabelInt(labels, gpuLabelMemTotalMiB)
	if !ok {
		return fmt.Sprintf("%s invalid (%q)", gpuLabelMemTotalMiB, labels[gpuLabelMemTotalMiB])
	}
	usedMiB, ok := parseLabelInt(labels, gpuLabelMemUsedMiB)
	if !ok {
		return fmt.Sprintf("%s invalid (%q)", gpuLabelMemUsedMiB, labels[gpuLabelMemUsedMiB])
	}
	if freeMiB := totalMiB - usedMiB; freeMiB < req.MemoryQuotaMB {
		return fmt.Sprintf("free memory %dMiB < quota %dMiB", freeMiB, req.MemoryQuotaMB)
	}

	sessions, ok := parseLabelInt(labels, gpuLabelSessions)
	if !ok {
		return fmt.Sprintf("%s invalid (%q)", gpuLabelSessions, labels[gpuLabelSessions])
	}
	maxSessions, ok := parseLabelInt(labels, gpuLabelMaxSessions)
	if !ok {
		return fmt.Sprintf("%s invalid (%q)", gpuLabelMaxSessions, labels[gpuLabelMaxSessions])
	}
	if sessions >= maxSessions {
		return fmt.Sprintf("sessions %d >= max sessions %d", sessions, maxSessions)
	}

	if req.RequiredDriverMin != "" {
		driver := strings.TrimSpace(labels[gpuLabelDriver])
		if driver == "" {
			return fmt.Sprintf("%s missing, required >= %s", gpuLabelDriver, req.RequiredDriverMin)
		}
		cmp, err := compareDriverVersion(driver, req.RequiredDriverMin)
		if err != nil {
			return fmt.Sprintf("driver version %q vs required %q: %v", driver, req.RequiredDriverMin, err)
		}
		if cmp < 0 {
			return fmt.Sprintf("driver %s < required %s", driver, req.RequiredDriverMin)
		}
	}

	if len(req.Models) > 0 {
		nodeModels := splitModels(labels[gpuLabelModels])
		if len(nodeModels) == 0 {
			return fmt.Sprintf("%s missing, required one of %v", gpuLabelModels, req.Models)
		}
		if !modelsMatch(nodeModels, req.Models) {
			return fmt.Sprintf("models %v do not match required %v", nodeModels, req.Models)
		}
	}

	return ""
}

func parseLabelInt(labels map[string]string, key string) (int64, bool) {
	raw := strings.TrimSpace(labels[key])
	if raw == "" {
		return 0, false
	}
	v, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

func splitModels(raw string) []string {
	parts := strings.Split(raw, ",")
	models := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			models = append(models, p)
		}
	}
	return models
}

// modelsMatch reports whether any required model matches any node model,
// case-insensitively and allowing either side to be a substring of the other
// (e.g. "A100" vs "NVIDIA A100-SXM4").
func modelsMatch(nodeModels, required []string) bool {
	for _, want := range required {
		want = strings.ToLower(strings.TrimSpace(want))
		if want == "" {
			continue
		}
		for _, have := range nodeModels {
			have = strings.ToLower(strings.TrimSpace(have))
			if have == "" {
				continue
			}
			if strings.Contains(have, want) || strings.Contains(want, have) {
				return true
			}
		}
	}
	return false
}

// compareDriverVersion compares dot-separated numeric driver versions such as
// "575.0" and "615.71.09". Missing trailing segments count as zero.
func compareDriverVersion(a, b string) (int, error) {
	av, err := parseDriverVersion(a)
	if err != nil {
		return 0, err
	}
	bv, err := parseDriverVersion(b)
	if err != nil {
		return 0, err
	}
	n := len(av)
	if len(bv) > n {
		n = len(bv)
	}
	for i := range n {
		var x, y int64
		if i < len(av) {
			x = av[i]
		}
		if i < len(bv) {
			y = bv[i]
		}
		if x < y {
			return -1, nil
		}
		if x > y {
			return 1, nil
		}
	}
	return 0, nil
}

func parseDriverVersion(v string) ([]int64, error) {
	v = strings.TrimSpace(v)
	if v == "" {
		return nil, fmt.Errorf("empty version")
	}
	parts := strings.Split(v, ".")
	out := make([]int64, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			return nil, fmt.Errorf("malformed version %q", v)
		}
		n, err := strconv.ParseInt(p, 10, 64)
		if err != nil || n < 0 {
			return nil, fmt.Errorf("malformed version %q", v)
		}
		out = append(out, n)
	}
	return out, nil
}
