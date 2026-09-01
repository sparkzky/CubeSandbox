-- Copyright (c) 2026 Tencent Inc.
-- SPDX-License-Identifier: Apache-2.0
--
-- Snapshot aliases (issue #1522, E2B compatibility).
--
-- t_cube_snapshot gains a first-class nullable `alias` column holding the
-- qualified snapshot name (`alias:tag`, tag defaulted to `:default` by
-- CubeAPI before it reaches master). A plain UNIQUE INDEX enforces global
-- uniqueness while allowing unlimited NULLs — every pre-existing row keeps
-- alias NULL (legacy display_name values are NOT backfilled: they were never
-- validated and may contain duplicates, so backfilling could break the
-- unique-index creation).
--
-- Snapshot alias keys always contain ':' (the tag separator) while template
-- aliases (t_cube_template_definition.alias_key) forbid ':', so the two
-- alias namespaces are structurally disjoint and need no cross-table
-- coordination.
--
-- Postgres counterpart: postgres/20260831120000_snapshot_alias.sql

-- +goose NO TRANSACTION
-- +goose Up

CALL cubemaster_acquire_migration_lock('cubemaster_migration_20260831120000_snapshot_alias', 60);

CALL cubemaster_assert_table_exists('t_cube_snapshot');

CALL cubemaster_add_column_if_missing(
  't_cube_snapshot',
  'alias',
  "varchar(128) NULL COMMENT 'qualified snapshot alias (alias:tag); NULL when the snapshot has no alias'"
);

CALL cubemaster_add_index_if_missing(
  't_cube_snapshot',
  'uniq_cube_snapshot_alias',
  "ADD UNIQUE INDEX `uniq_cube_snapshot_alias` (`alias`)"
);

SELECT RELEASE_LOCK('cubemaster_migration_20260831120000_snapshot_alias');

-- +goose Down

CALL cubemaster_acquire_migration_lock('cubemaster_migration_20260831120000_snapshot_alias', 60);

CALL cubemaster_drop_index_if_exists('t_cube_snapshot', 'uniq_cube_snapshot_alias');

CALL cubemaster_drop_column_if_exists('t_cube_snapshot', 'alias');

SELECT RELEASE_LOCK('cubemaster_migration_20260831120000_snapshot_alias');
