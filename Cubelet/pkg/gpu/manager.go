// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0
//

// Package gpu implements the node-level GPU inventory, placement and
// lifecycle state machine used by the cubelet.
//
// The manager owns a per-node view of the physical GPUs, tracks the memory
// quota and session count reserved by sandboxes, and publishes that view as
// node labels consumed by CubeMaster's scheduler. It never talks to the GPU
// itself: device enumeration is injected so the package can be tested without
// an actual device, and the cube shim/daemon remain the data-plane owners.
package gpu

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/klog/v2"
)

const (
	// AnnotationKey is the sandbox/template annotation carrying the GPU
	// request as a JSON object.
	AnnotationKey = "cube.gpu"

	// DefaultMemoryQuotaMB is used when the annotation omits
	// memory_quota_mb (or sets it to zero).
	DefaultMemoryQuotaMB uint64 = 4096

	// DefaultMaxSessionsPerGPU caps concurrent sandboxes sharing one card.
	DefaultMaxSessionsPerGPU = 64

	// LabelPrefix groups the node labels published by this package. The
	// heartbeat path clears every label under this prefix that is no longer
	// reported, so a disappeared GPU does not leave stale inventory behind.
	LabelPrefix = "cube.gpu."

	LabelPresent        = LabelPrefix + "present"
	LabelCount          = LabelPrefix + "count"
	LabelMemoryTotalMiB = LabelPrefix + "memory.total.mib"
	LabelMemoryUsedMiB  = LabelPrefix + "memory.used.mib"
	LabelSessions       = LabelPrefix + "sessions"
	LabelMaxSessions    = LabelPrefix + "max.sessions"
	LabelDriver         = LabelPrefix + "driver"
	LabelModels         = LabelPrefix + "models"
)

// Device is one physical GPU as reported by discovery.
type Device struct {
	UUID          string
	Name          string
	DriverVersion string
	MemoryMB      uint64
}

// Request is the parsed cube.gpu annotation.
type Request struct {
	Enable            bool     `json:"enable"`
	MemoryQuotaMB     uint64   `json:"memory_quota_mb"`
	SMPercent         int      `json:"sm_percent"`
	RequiredDriverMin string   `json:"required_driver_min"`
	Models            []string `json:"models"`
}

// Config controls the manager. MaxSessionsPerCard is accepted as an alias for
// MaxSessionsPerGPU; MaxSessionsPerGPU wins when both are set.
type Config struct {
	Enabled            bool
	MaxSessionsPerGPU  int
	MaxSessionsPerCard int
}

func (c Config) effectiveMaxSessions() int {
	if c.MaxSessionsPerGPU > 0 {
		return c.MaxSessionsPerGPU
	}
	if c.MaxSessionsPerCard > 0 {
		return c.MaxSessionsPerCard
	}
	return DefaultMaxSessionsPerGPU
}

// Assignment describes the placement handed back to the caller.
type Assignment struct {
	SandboxID     string
	DeviceUUID    string
	DeviceName    string
	DriverVersion string
	MemoryQuotaMB uint64
	SMPercent     int
}

// Sentinels callers can match with errors.Is. Capacity and compatibility
// failures additionally carry human-readable detail via fmt.Errorf wrapping.
var (
	ErrDisabled         = errors.New("gpu: support is disabled on this node")
	ErrNotRequested     = errors.New("gpu: gpu not requested")
	ErrNoGPU            = errors.New("gpu: no usable GPU device on this node")
	ErrNoCapacity       = errors.New("gpu: insufficient GPU capacity")
	ErrIncompatible     = errors.New("gpu: GPU request is incompatible with this node")
	ErrAlreadyReserved  = errors.New("gpu: sandbox already holds a different GPU reservation")
	ErrUnknownSandbox   = errors.New("gpu: sandbox has no GPU reservation")
	ErrQuarantined      = errors.New("gpu: sandbox GPU allocation is quarantined")
	ErrNotInitialized   = errors.New("gpu: manager is not initialized")
	errInvalidRequest   = errors.New("gpu: invalid request")
	errInvalidSandboxID = errors.New("gpu: sandbox id is empty")
)

// ParseRequest decodes the cube.gpu annotation. found reports whether the
// annotation was present and non-empty; defaults are applied either way.
func ParseRequest(annotations map[string]string) (Request, bool, error) {
	req := Request{MemoryQuotaMB: DefaultMemoryQuotaMB}
	raw, ok := annotations[AnnotationKey]
	if !ok {
		return req, false, nil
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return req, false, nil
	}
	// Backward compatibility with the original boolean annotation form.
	if strings.EqualFold(raw, "true") {
		req.Enable = true
		return req.normalized(), true, nil
	}
	if strings.EqualFold(raw, "false") {
		return req, false, nil
	}
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		return Request{MemoryQuotaMB: DefaultMemoryQuotaMB}, true,
			fmt.Errorf("invalid %s annotation: %w", AnnotationKey, err)
	}
	req = req.normalized()
	if err := req.validate(); err != nil {
		return req, true, fmt.Errorf("invalid %s annotation: %w", AnnotationKey, err)
	}
	return req, true, nil
}

func (r Request) normalized() Request {
	if r.MemoryQuotaMB == 0 {
		r.MemoryQuotaMB = DefaultMemoryQuotaMB
	}
	r.RequiredDriverMin = strings.TrimSpace(r.RequiredDriverMin)
	r.Models = normalizeModels(r.Models)
	return r
}

func (r Request) validate() error {
	if r.SMPercent < 0 || r.SMPercent > 100 {
		return fmt.Errorf("%w: sm_percent %d is outside [0,100]", errInvalidRequest, r.SMPercent)
	}
	if r.RequiredDriverMin != "" {
		if _, ok := parseVersion(r.RequiredDriverMin); !ok {
			return fmt.Errorf("%w: required_driver_min %q is not a version", errInvalidRequest, r.RequiredDriverMin)
		}
	}
	return nil
}

type sessionState uint8

const (
	stateReserved sessionState = iota + 1
	stateAttached
	stateReleasing
	stateQuarantined
)

func (s sessionState) String() string {
	switch s {
	case stateReserved:
		return "reserved"
	case stateAttached:
		return "attached"
	case stateReleasing:
		return "releasing"
	case stateQuarantined:
		return "quarantined"
	default:
		return "unknown"
	}
}

type gpuDevice struct {
	Device
	usedMB   uint64
	sessions int
	healthy  bool
}

func (d *gpuDevice) freeMB() uint64 {
	if d.usedMB >= d.MemoryMB {
		return 0
	}
	return d.MemoryMB - d.usedMB
}

type allocation struct {
	req        Request
	device     *gpuDevice
	state      sessionState
	assignment Assignment
}

// Manager is a concurrency-safe node-level GPU allocator.
type Manager struct {
	mu       sync.Mutex
	cfg      Config
	discover func() ([]Device, error)

	devices     []*gpuDevice
	discovered  bool
	discoverErr error
	sessions    map[string]*allocation
	quarantined map[string]struct{}
	maxSessions int
}

// NewManager constructs a manager. When discovery is nil the nvidia-smi
// scanner is used. Discovery runs eagerly (once) when the feature is enabled;
// a failure is logged and leaves the node advertising no GPU instead of
// preventing cubelet startup.
func NewManager(cfg Config, discover func() ([]Device, error)) *Manager {
	if discover == nil {
		discover = DiscoverNVIDIASMI
	}
	m := &Manager{
		cfg:         cfg,
		discover:    discover,
		sessions:    make(map[string]*allocation),
		quarantined: make(map[string]struct{}),
		maxSessions: cfg.effectiveMaxSessions(),
	}
	if cfg.Enabled {
		m.mu.Lock()
		m.loadDevicesLocked()
		m.mu.Unlock()
	}
	return m
}

func (m *Manager) loadDevicesLocked() {
	if m.discovered {
		return
	}
	m.discovered = true
	devices, err := m.discover()
	if err != nil {
		m.discoverErr = err
		klog.Warningf("gpu: device discovery failed; GPU allocation stays unavailable until cubelet restarts: %v", err)
		return
	}
	m.devices = make([]*gpuDevice, 0, len(devices))
	for _, d := range devices {
		m.devices = append(m.devices, &gpuDevice{Device: d, healthy: true})
	}
	if len(m.devices) > 0 {
		klog.Infof("gpu: discovered %d device(s), driver %s", len(m.devices), m.devices[0].DriverVersion)
	}
}

// Reserve allocates capacity for sandboxID.
//
// Selection is binpack-oriented per the design contract: the compatible card
// with the most free memory wins, provided it still has free memory for the
// request and has not reached MaxSessionsPerGPU. Reserving the same sandbox
// with the same quota/sm is idempotent.
func (m *Manager) Reserve(sandboxID string, req Request) (Assignment, error) {
	if m == nil {
		return Assignment{}, ErrNotInitialized
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.reserveLocked(sandboxID, req)
}

func (m *Manager) reserveLocked(sandboxID string, req Request) (Assignment, error) {
	if strings.TrimSpace(sandboxID) == "" {
		return Assignment{}, errInvalidSandboxID
	}
	if !m.cfg.Enabled {
		return Assignment{}, ErrDisabled
	}
	req = req.normalized()
	if !req.Enable {
		return Assignment{}, ErrNotRequested
	}
	if err := req.validate(); err != nil {
		return Assignment{}, err
	}
	if _, ok := m.quarantined[sandboxID]; ok {
		return Assignment{}, fmt.Errorf("%w: %s", ErrQuarantined, sandboxID)
	}
	if existing, ok := m.sessions[sandboxID]; ok && existing != nil {
		switch existing.state {
		case stateQuarantined:
			return Assignment{}, fmt.Errorf("%w: %s", ErrQuarantined, sandboxID)
		case stateReserved, stateAttached:
			if existing.req.MemoryQuotaMB != req.MemoryQuotaMB || existing.req.SMPercent != req.SMPercent {
				return Assignment{}, fmt.Errorf("%w: %s", ErrAlreadyReserved, sandboxID)
			}
			return existing.assignment, nil
		default:
			return Assignment{}, fmt.Errorf("%w: %s is %s", ErrAlreadyReserved, sandboxID, existing.state)
		}
	}

	m.loadDevicesLocked()
	if m.discoverErr != nil {
		return Assignment{}, fmt.Errorf("%w: %v", ErrNoGPU, m.discoverErr)
	}
	if len(m.devices) == 0 {
		return Assignment{}, ErrNoGPU
	}
	if err := m.checkCompatibilityLocked(req); err != nil {
		return Assignment{}, err
	}

	var (
		best     *gpuDevice
		bestFree uint64
		maxFree  uint64
	)
	for _, d := range m.devices {
		if !d.healthy {
			continue
		}
		if req.RequiredDriverMin != "" && !driverSatisfies(d.DriverVersion, req.RequiredDriverMin) {
			continue
		}
		if !modelMatches(d.Name, req.Models) {
			continue
		}
		free := d.freeMB()
		if free > maxFree {
			maxFree = free
		}
		if d.sessions >= m.maxSessions {
			continue
		}
		if free < req.MemoryQuotaMB {
			continue
		}
		if best == nil || free > bestFree || (free == bestFree && d.UUID < best.UUID) {
			best = d
			bestFree = free
		}
	}
	if best == nil {
		return Assignment{}, fmt.Errorf(
			"%w: request needs %d MiB and a free session slot (max %d per GPU); largest compatible free memory is %d MiB",
			ErrNoCapacity, req.MemoryQuotaMB, m.maxSessions, maxFree)
	}

	best.usedMB += req.MemoryQuotaMB
	best.sessions++
	a := &allocation{
		req:    req,
		device: best,
		state:  stateReserved,
		assignment: Assignment{
			SandboxID:     sandboxID,
			DeviceUUID:    best.UUID,
			DeviceName:    best.Name,
			DriverVersion: best.DriverVersion,
			MemoryQuotaMB: req.MemoryQuotaMB,
			SMPercent:     req.SMPercent,
		},
	}
	m.sessions[sandboxID] = a
	return a.assignment, nil
}

func (m *Manager) checkCompatibilityLocked(req Request) error {
	if req.RequiredDriverMin != "" {
		ok := false
		for _, d := range m.devices {
			if d.healthy && driverSatisfies(d.DriverVersion, req.RequiredDriverMin) {
				ok = true
				break
			}
		}
		if !ok {
			return fmt.Errorf("%w: driver %q is below required_driver_min %q",
				ErrIncompatible, m.driverSummaryLocked(), req.RequiredDriverMin)
		}
	}
	if len(req.Models) > 0 {
		ok := false
		for _, d := range m.devices {
			if d.healthy && modelMatches(d.Name, req.Models) {
				ok = true
				break
			}
		}
		if !ok {
			return fmt.Errorf("%w: requested models %v match none of the node GPUs %q",
				ErrIncompatible, req.Models, m.modelSummaryLocked())
		}
	}
	return nil
}

func (m *Manager) driverSummaryLocked() string {
	seen := make(map[string]struct{}, len(m.devices))
	out := make([]string, 0, len(m.devices))
	for _, d := range m.devices {
		if _, ok := seen[d.DriverVersion]; ok {
			continue
		}
		seen[d.DriverVersion] = struct{}{}
		out = append(out, d.DriverVersion)
	}
	return strings.Join(out, ",")
}

func (m *Manager) modelSummaryLocked() string {
	seen := make(map[string]struct{}, len(m.devices))
	out := make([]string, 0, len(m.devices))
	for _, d := range m.devices {
		if _, ok := seen[d.Name]; ok {
			continue
		}
		seen[d.Name] = struct{}{}
		out = append(out, d.Name)
	}
	return strings.Join(out, ",")
}

// Attach moves a reservation to the attached state after the sandbox has
// been created successfully. Attaching an already-attached sandbox is a
// no-op.
func (m *Manager) Attach(sandboxID string) error {
	if m == nil {
		return ErrNotInitialized
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	a, ok := m.sessions[sandboxID]
	if !ok || a == nil {
		return fmt.Errorf("%w: %s", ErrUnknownSandbox, sandboxID)
	}
	switch a.state {
	case stateAttached:
		return nil
	case stateReserved:
		a.state = stateAttached
		return nil
	case stateQuarantined:
		return fmt.Errorf("%w: %s", ErrQuarantined, sandboxID)
	default:
		return fmt.Errorf("%w: %s is %s", ErrUnknownSandbox, sandboxID, a.state)
	}
}

// Release returns the sandbox's quota to the node. It is idempotent: releasing
// an unknown sandbox succeeds.
func (m *Manager) Release(sandboxID string) error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	a, ok := m.sessions[sandboxID]
	if !ok || a == nil {
		delete(m.quarantined, sandboxID)
		return nil
	}
	a.state = stateReleasing
	if a.device != nil {
		if a.device.usedMB >= a.req.MemoryQuotaMB {
			a.device.usedMB -= a.req.MemoryQuotaMB
		} else {
			a.device.usedMB = 0
		}
		if a.device.sessions > 0 {
			a.device.sessions--
		}
	}
	delete(m.sessions, sandboxID)
	delete(m.quarantined, sandboxID)
	return nil
}

// Quarantine pins a sandbox's allocation. Quarantined capacity is never handed
// out again and the sandbox itself cannot be reserved until it is released or
// the cubelet restarts. Unknown sandbox IDs are remembered so a racing retry
// cannot allocate for them.
func (m *Manager) Quarantine(sandboxID string) {
	if m == nil || strings.TrimSpace(sandboxID) == "" {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if a, ok := m.sessions[sandboxID]; ok && a != nil {
		a.state = stateQuarantined
		return
	}
	m.quarantined[sandboxID] = struct{}{}
}

// Reconcile rebuilds the in-memory ledger from the set of sandboxes known to
// be alive. It is meant to be called once after startup, before inventory is
// reported. Existing sandboxes are treated as attached; entries that no longer
// fit are logged and skipped rather than overselling the node.
func (m *Manager) Reconcile(active map[string]Request) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.cfg.Enabled {
		return
	}
	m.loadDevicesLocked()
	for _, d := range m.devices {
		d.usedMB = 0
		d.sessions = 0
	}
	m.sessions = make(map[string]*allocation, len(active))
	m.quarantined = make(map[string]struct{})
	if m.discoverErr != nil || len(m.devices) == 0 {
		return
	}
	ids := make([]string, 0, len(active))
	for id := range active {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		req := active[id]
		if !req.Enable {
			continue
		}
		if _, err := m.reserveLocked(id, req); err != nil {
			klog.Warningf("gpu: reconcile sandbox %s: %v", id, err)
			continue
		}
		m.sessions[id].state = stateAttached
	}
	if len(m.sessions) > 0 {
		klog.Infof("gpu: reconciled %d active sandbox reservation(s)", len(m.sessions))
	}
}

// Labels returns the node-label view of the GPU inventory. It returns an empty
// map when the feature is disabled or no device could be discovered, which
// makes CubeMaster treat the node as GPU-less.
func (m *Manager) Labels() map[string]string {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.cfg.Enabled {
		return map[string]string{}
	}
	m.loadDevicesLocked()
	if m.discoverErr != nil || len(m.devices) == 0 {
		return map[string]string{}
	}
	var total, used uint64
	for _, d := range m.devices {
		total += d.MemoryMB
		used += d.usedMB
	}
	return map[string]string{
		LabelPresent:        "true",
		LabelCount:          strconv.Itoa(len(m.devices)),
		LabelMemoryTotalMiB: strconv.FormatUint(total, 10),
		LabelMemoryUsedMiB:  strconv.FormatUint(used, 10),
		LabelSessions:       strconv.Itoa(len(m.sessions)),
		LabelMaxSessions:    strconv.Itoa(m.maxSessions * len(m.devices)),
		LabelDriver:         m.devices[0].DriverVersion,
		LabelModels:         m.modelSummaryLocked(),
	}
}

// CommandRunner is the subset of os/exec used by discovery.
type CommandRunner func(ctx context.Context, name string, args ...string) ([]byte, error)

const nvidiaSMIQuery = "--query-gpu=index,uuid,name,memory.total,driver_version"

const nvidiaSMITimeout = 10 * time.Second

// DiscoverNVIDIASMI enumerates local GPUs through nvidia-smi.
func DiscoverNVIDIASMI() ([]Device, error) {
	return DiscoverNVIDIASMIWithRunner(execRunner)
}

func execRunner(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).Output()
}

// DiscoverNVIDIASMIWithRunner is the injectable form of DiscoverNVIDIASMI.
func DiscoverNVIDIASMIWithRunner(run CommandRunner) ([]Device, error) {
	if run == nil {
		return nil, errors.New("gpu: nil command runner")
	}
	ctx, cancel := context.WithTimeout(context.Background(), nvidiaSMITimeout)
	defer cancel()
	out, err := run(ctx, "nvidia-smi", nvidiaSMIQuery, "--format=csv,noheader,nounits")
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi failed: %w", err)
	}
	return ParseNVIDIASMIOutput(out)
}

// ParseNVIDIASMIOutput parses index,uuid,name,memory.total,driver_version CSV
// rows and orders them by physical index.
func ParseNVIDIASMIOutput(data []byte) ([]Device, error) {
	reader := csv.NewReader(bytes.NewReader(data))
	reader.TrimLeadingSpace = true
	reader.FieldsPerRecord = -1

	type indexedDevice struct {
		index  int
		device Device
	}
	var rows []indexedDevice
	for {
		record, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("parse nvidia-smi output: %w", err)
		}
		if len(record) == 0 {
			continue
		}
		if len(record) < 5 {
			return nil, fmt.Errorf("parse nvidia-smi output: expected 5 columns, got %d", len(record))
		}
		index, err := strconv.Atoi(strings.TrimSpace(record[0]))
		if err != nil {
			return nil, fmt.Errorf("parse nvidia-smi output: invalid index %q: %w", record[0], err)
		}
		memory, err := strconv.ParseUint(strings.TrimSpace(record[3]), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parse nvidia-smi output: invalid memory total %q: %w", record[3], err)
		}
		rows = append(rows, indexedDevice{
			index: index,
			device: Device{
				UUID:          strings.TrimSpace(record[1]),
				Name:          strings.TrimSpace(record[2]),
				MemoryMB:      memory,
				DriverVersion: strings.TrimSpace(record[4]),
			},
		})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].index < rows[j].index })
	devices := make([]Device, len(rows))
	for i := range rows {
		devices[i] = rows[i].device
	}
	return devices, nil
}

func parseVersion(version string) ([]int, bool) {
	version = strings.TrimSpace(version)
	if version == "" {
		return nil, false
	}
	parts := strings.Split(version, ".")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, false
		}
		n, err := strconv.Atoi(part)
		if err != nil || n < 0 {
			return nil, false
		}
		out = append(out, n)
	}
	return out, true
}

// driverSatisfies reports whether current >= required using dotted numeric
// comparison (missing components count as zero).
func driverSatisfies(current, required string) bool {
	cur, ok := parseVersion(current)
	if !ok {
		return false
	}
	req, ok := parseVersion(required)
	if !ok {
		return false
	}
	for i := range req {
		value := 0
		if i < len(cur) {
			value = cur[i]
		}
		if value != req[i] {
			return value > req[i]
		}
	}
	return true
}

func normalizeModels(models []string) []string {
	out := make([]string, 0, len(models))
	seen := make(map[string]struct{}, len(models))
	for _, model := range models {
		model = strings.TrimSpace(model)
		if model == "" {
			continue
		}
		key := strings.ToLower(model)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, model)
	}
	return out
}

// modelMatches implements the contract's case-insensitive substring match:
// any requested model that appears inside the GPU name is a match.
func modelMatches(name string, models []string) bool {
	if len(models) == 0 {
		return true
	}
	name = strings.ToLower(name)
	for _, model := range models {
		model = strings.ToLower(strings.TrimSpace(model))
		if model != "" && strings.Contains(name, model) {
			return true
		}
	}
	return false
}

var defaultManager atomic.Pointer[Manager]

// SetDefault publishes the process-wide manager (nil clears it, useful for
// tests).
func SetDefault(m *Manager) {
	defaultManager.Store(m)
}

// Default returns the process-wide manager or nil when none was initialised.
func Default() *Manager {
	return defaultManager.Load()
}

// InitDefault constructs and publishes the process-wide manager.
func InitDefault(cfg Config, discover func() ([]Device, error)) *Manager {
	m := NewManager(cfg, discover)
	SetDefault(m)
	return m
}
