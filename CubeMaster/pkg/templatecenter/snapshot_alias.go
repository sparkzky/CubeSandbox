// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

package templatecenter

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/tencentcloud/CubeSandbox/CubeMaster/pkg/base/constants"
	"github.com/tencentcloud/CubeSandbox/CubeMaster/pkg/base/db/models"
	"gorm.io/gorm"
)

// Snapshot aliases (issue #1522, E2B compatibility).
//
// A snapshot created with a name claims a *qualified alias key* of the form
// "alias:tag" (tag defaulted to "default" by CubeAPI before it reaches
// master). The key is stored on t_cube_snapshot.alias under a unique index.
// It is what the E2B SDK sees as `snapshotID` and must round-trip through
// sandbox-create, snapshot delete and snapshot rollback.
//
// Snapshot keys are structurally disjoint from template aliases
// (t_cube_template_definition.alias_key): a snapshot key always contains the
// ':' tag separator while template aliases forbid ':'. Each table's own
// unique index therefore suffices — no cross-table claim coordination.

var (
	// ErrSnapshotAliasConflict is returned when the requested alias key is
	// already held by another snapshot.
	ErrSnapshotAliasConflict = errors.New("snapshot name already exists")
	// ErrSnapshotAliasInvalid is returned when an alias key is not a valid
	// "alias:tag" pair.
	ErrSnapshotAliasInvalid = errors.New("invalid snapshot alias")
)

// SnapshotAliasTagDefault is appended when resolving a tag-less identifier
// on the sandbox-create path, mirroring official E2B ("my-snapshot" and
// "my-snapshot:default" name the same snapshot).
const SnapshotAliasTagDefault = "default"

// snapshotAliasKeyMaxLen matches the t_cube_snapshot.alias column width:
// alias segment ≤64 + ':' + tag segment ≤63 = 128.
const snapshotAliasKeyMaxLen = 128

// snapshotAliasSegmentRe is the same charset template aliases enforce
// (request_validation.go aliasValidationRe) applied to both the alias and
// the tag segment of a snapshot key.
var snapshotAliasSegmentRe = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,63}$`)

// ValidateSnapshotAliasKey validates a fully-qualified snapshot alias key
// ("alias:tag", exactly one ':', both segments matching the flat alias
// charset, total length ≤ snapshotAliasKeyMaxLen which is also the DB column
// width). Returns ErrSnapshotAliasInvalid wrapping the offending key.
func ValidateSnapshotAliasKey(key string) error {
	key = strings.TrimSpace(key)
	alias, tag, err := splitSnapshotAliasKey(key)
	if err != nil {
		return err
	}
	if !snapshotAliasSegmentRe.MatchString(alias) || !snapshotAliasSegmentRe.MatchString(tag) {
		return fmt.Errorf("%w: %q", ErrSnapshotAliasInvalid, key)
	}
	// Reject reserved ID prefixes on both segments, mirroring
	// validateTemplateAlias: a key like "snap-foo:default" would itself look
	// like a raw snap-* id to hasValidTemplateIDPrefix and could therefore
	// never be resolved back through the alias path.
	for _, segment := range []string{alias, tag} {
		if strings.HasPrefix(segment, "tpl-") || strings.HasPrefix(segment, "snap-") {
			return fmt.Errorf("%w: %q uses reserved prefix", ErrSnapshotAliasInvalid, key)
		}
	}
	return nil
}

func splitSnapshotAliasKey(key string) (alias, tag string, err error) {
	key = strings.TrimSpace(key)
	parts := strings.Split(key, ":")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", fmt.Errorf("%w: %q must be \"alias:tag\"", ErrSnapshotAliasInvalid, key)
	}
	if len(key) > snapshotAliasKeyMaxLen {
		return "", "", fmt.Errorf("%w: %q exceeds %d characters", ErrSnapshotAliasInvalid, key, snapshotAliasKeyMaxLen)
	}
	return parts[0], parts[1], nil
}

// stripAliasNamespace drops any "ns/" prefix from an identifier: CubeSandbox
// has a single flat alias namespace, so the E2B "team-slug/alias" form
// resolves on its last path segment (mirrors CubeAPI's alias_from_name).
func stripAliasNamespace(identifier string) string {
	id := strings.TrimSpace(identifier)
	if idx := strings.LastIndex(id, "/"); idx >= 0 {
		id = id[idx+1:]
	}
	return strings.TrimSpace(id)
}

// getSnapshotRecordByAlias loads a snapshot row by its exact alias key.
func getSnapshotRecordByAlias(ctx context.Context, aliasKey string) (*models.SnapshotRecord, error) {
	if !isReady() {
		return nil, ErrTemplateStoreNotInitialized
	}
	aliasKey = strings.TrimSpace(aliasKey)
	if aliasKey == "" {
		return nil, ErrSnapshotNotFound
	}
	rec := &models.SnapshotRecord{}
	err := store.db.WithContext(ctx).Table(constants.SnapshotTableName).
		Where("alias = ?", aliasKey).First(rec).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, ErrSnapshotNotFound
	}
	if err != nil {
		return nil, err
	}
	return rec, nil
}

// ResolveSnapshotAlias resolves a user-facing snapshot identifier (alias key,
// optionally namespace-prefixed) to the canonical snap-* id.
//
// Matching is exact against t_cube_snapshot.alias. When deriveDefault is set
// and the identifier carries no ':' tag, "<identifier>:default" is tried as
// well. deriveDefault is used ONLY by the sandbox-create resolution path
// (ResolveTemplateIdentifier): destructive endpoints (snapshot get/delete/
// rollback) must resolve exactly, so a bare alias can never shadow-delete a
// snapshot while a same-named template alias exists.
func ResolveSnapshotAlias(ctx context.Context, identifier string, deriveDefault bool) (string, error) {
	id := stripAliasNamespace(identifier)
	if id == "" {
		return "", ErrSnapshotNotFound
	}
	candidates := []string{id}
	if deriveDefault && !strings.Contains(id, ":") {
		candidates = append(candidates, id+":"+SnapshotAliasTagDefault)
	}
	for _, key := range candidates {
		rec, err := getSnapshotRecordByAlias(ctx, key)
		if err == nil && rec != nil {
			return rec.SnapshotID, nil
		}
		if err != nil && !errors.Is(err, ErrSnapshotNotFound) {
			return "", err
		}
	}
	return "", ErrSnapshotNotFound
}

// NormalizeSnapshotRef resolves a snapshot reference that may be a raw
// snap-*/tpl-* id or an exact alias key to the canonical snapshot id.
// tpl-/snap- prefixed references pass through unchanged (preserving the
// historical not-found behaviour for template ids on snapshot endpoints).
func NormalizeSnapshotRef(ctx context.Context, ref string) (string, error) {
	ref = strings.TrimSpace(ref)
	if hasValidTemplateIDPrefix(ref) {
		return ref, nil
	}
	return ResolveSnapshotAlias(ctx, ref, false)
}

// snapshotAliasFromRecord coalesces the nullable alias column to "".
func snapshotAliasFromRecord(rec *models.SnapshotRecord) string {
	if rec == nil || rec.Alias == nil {
		return ""
	}
	return strings.TrimSpace(*rec.Alias)
}
