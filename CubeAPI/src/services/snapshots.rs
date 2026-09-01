// Copyright (c) 2026 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0
//

use uuid::Uuid;

use crate::{
    cubemaster::{
        CreateSnapshotRequest, CubeMasterClient, DeleteSnapshotRequest, ListSnapshotsRequest,
        RollbackRequest as MasterRollbackRequest, SnapshotResource,
    },
    error::{AppError, AppResult},
    models::{
        DeleteSnapshotResponse as ApiDeleteSnapshotResponse, RollbackResponse, SnapshotInfo,
        SnapshotListItem,
    },
    services::templates::is_valid_alias,
};

#[derive(Clone)]
pub struct SnapshotService {
    cubemaster: CubeMasterClient,
    instance_type: String,
}

impl SnapshotService {
    pub fn new(cubemaster: CubeMasterClient, instance_type: String) -> Self {
        Self {
            cubemaster,
            instance_type,
        }
    }

    // ── POST /sandboxes/{sandboxID}/snapshots ──────────────────────────────

    pub async fn create(
        &self,
        sandbox_id: &str,
        name: Option<String>,
        backend: Option<String>,
    ) -> AppResult<SnapshotInfo> {
        let request_id = new_request_id();
        let create_request = self
            .build_create_request_payload(sandbox_id, &request_id)
            .await?;
        // Parse the E2B name once: display_name is the namespace-stripped
        // name (what the SDK sees in `names`), alias is the qualified key
        // ("alias:tag", tag defaulted) master claims and we echo back as
        // `snapshotID`. An unparseable name degrades to an alias-less
        // snapshot (raw snap-* id) rather than failing the request,
        // mirroring the template path's silent-drop convention.
        let ParsedSnapshotName {
            display_name,
            alias,
        } = parse_snapshot_name(name.as_deref());
        let req = CreateSnapshotRequest {
            request_id: request_id.clone(),
            sandbox_id: sandbox_id.to_string(),
            display_name,
            alias,
            create_request,
            backend,
        };

        match self.cubemaster.create_snapshot(&req).await {
            Ok(resp) => {
                resp.ret.as_result().map_err(|e| {
                    if e.is_not_found() {
                        sandbox_not_found(sandbox_id)
                    } else if e.is_conflict() {
                        // Surface master's ret_msg verbatim: it distinguishes
                        // "snapshot name already exists" (alias conflict)
                        // from "active snapshot operation in progress".
                        AppError::Conflict(format!("sandbox {}: {}", sandbox_id, e))
                    } else {
                        internal_error(e)
                    }
                })?;
                let snapshot_id = resp.snapshot.snapshot_id.clone();
                if snapshot_id.trim().is_empty() {
                    return Err(AppError::Internal(anyhow::anyhow!(
                        "snapshot create response missing snapshot_id"
                    )));
                }
                ensure_snapshot_ready(&resp.snapshot)?;
                Ok(snapshot_resource_to_info(resp.snapshot))
            }
            Err(e) if e.is_not_found() => Err(sandbox_not_found(sandbox_id)),
            Err(e) => Err(internal_error(e)),
        }
    }

    // ── GET /snapshots ─────────────────────────────────────────────────────

    pub async fn list(
        &self,
        sandbox_id: Option<&str>,
        limit: Option<i32>,
        next_token: Option<&str>,
    ) -> AppResult<(Vec<SnapshotListItem>, String)> {
        let req = ListSnapshotsRequest {
            request_id: new_request_id(),
            instance_type: self.instance_type.clone(),
            sandbox_id: sandbox_id.map(str::to_string),
            name: None,
            status: None,
            limit,
            // Normalise an empty cursor (e.g. the client sent `?nextToken=`)
            // back to `None` so we don't relay a meaningless pagination token
            // to CubeMaster (Bug 3).  Whitespace-only tokens get the same
            // treatment.
            next_token: normalize_next_token(next_token),
        };

        match self.cubemaster.list_snapshots(&req).await {
            Ok(resp) => {
                resp.ret.as_result().map_err(internal_error)?;
                let items = resp
                    .items
                    .into_iter()
                    .map(snapshot_resource_to_list_item)
                    .collect();
                Ok((items, resp.next_token))
            }
            Err(e) => Err(internal_error(e)),
        }
    }

    // ── DELETE /templates/{templateID}  (when templateID is a snapshot id) ─
    //
    // Relies on CubeMaster's *synchronous* `DELETE /cube/snapshot/{id}`
    // contract: when the master returns `ret_code == 0`, the snapshot
    // (replica + metadata + cubelet-side LVM/meta) is fully gone; when it
    // errors, the operation either was rejected up-front or ran to a
    // recorded failure.  We assert this invariant via
    // `ensure_operation_ready(resp.status(), …)` — if master ever drifts
    // back to handing us Pending/Running we want the caller to see it as
    // an `Internal` error rather than silently returning success on an
    // un-finished delete.
    //
    // The 240 s router timeout (see `routes::SNAPSHOT_LONG_ROUTE_TIMEOUT`)
    // exists exactly so a slow cubelet cleanup does not get cut off by the
    // 30 s default budget.  The snapshot API is synchronous — CubeAPI waits
    // for a terminal state and does not expose a polling interface.

    pub async fn delete(&self, snapshot_id: &str) -> AppResult<ApiDeleteSnapshotResponse> {
        let snapshot_id = normalize_snapshot_identifier(snapshot_id);
        let req = DeleteSnapshotRequest {
            request_id: new_request_id(),
            instance_type: self.instance_type.clone(),
        };

        match self.cubemaster.delete_snapshot(&snapshot_id, &req).await {
            Ok(resp) => {
                let operation_id = required_operation_id(resp.operation_id(), "snapshot delete")?;
                resp.ret.as_result().map_err(|e| {
                    if e.is_not_found() {
                        snapshot_not_found(snapshot_id.as_str())
                    } else if e.is_conflict() {
                        snapshot_delete_conflict(snapshot_id.as_str())
                    } else {
                        internal_error(e)
                    }
                })?;
                let status =
                    ensure_operation_ready(resp.status(), "snapshot delete", snapshot_id.as_str())?;

                Ok(ApiDeleteSnapshotResponse {
                    template_id: snapshot_id.to_string(),
                    operation_id,
                    status,
                })
            }
            Err(e) if e.is_not_found() => Err(snapshot_not_found(&snapshot_id)),
            Err(e) if e.is_invalid_path_parameter() || e.is_params_error() => {
                Err(AppError::BadRequest(e.to_string()))
            }
            Err(e) => Err(internal_error(e)),
        }
    }

    pub async fn has_snapshot(&self, snapshot_id: &str) -> AppResult<bool> {
        let snapshot_id = normalize_snapshot_identifier(snapshot_id);
        match self.cubemaster.get_snapshot(&snapshot_id, false).await {
            Ok(resp) => {
                resp.ret.as_result().map_err(internal_error)?;
                Ok(true)
            }
            Err(e) if e.is_not_found() => Ok(false),
            Err(e) if e.is_invalid_path_parameter() || e.is_params_error() => {
                Err(AppError::BadRequest(e.to_string()))
            }
            Err(e) => Err(internal_error(e)),
        }
    }

    // ── POST /sandboxes/{sandboxID}/rollback ───────────────────────────────

    pub async fn rollback(
        &self,
        sandbox_id: &str,
        snapshot_id: &str,
        backend: Option<String>,
    ) -> AppResult<RollbackResponse> {
        let req_id = new_request_id();
        let req = MasterRollbackRequest {
            request_id: req_id.clone(),
            snapshot_id: snapshot_id.to_string(),
            instance_type: self.instance_type.clone(),
            backend,
        };

        match self.cubemaster.rollback_sandbox(sandbox_id, &req).await {
            Ok(resp) => {
                let operation_id = required_operation_id(resp.operation_id(), "snapshot rollback")?;
                resp.ret.as_result().map_err(|e| {
                    if e.is_not_found() {
                        sandbox_or_snapshot_not_found(sandbox_id, snapshot_id)
                    } else if e.is_conflict() {
                        rollback_conflict(sandbox_id, snapshot_id)
                    } else {
                        internal_error(e)
                    }
                })?;
                let status =
                    ensure_operation_ready(resp.status(), "snapshot rollback", snapshot_id)?;

                Ok(RollbackResponse {
                    sandbox_id: owned_or_fallback(resp.sandbox_id, sandbox_id),
                    snapshot_id: owned_or_fallback(resp.snapshot_id, snapshot_id),
                    operation_id,
                    status,
                })
            }
            Err(e) if e.is_not_found() => {
                Err(sandbox_or_snapshot_not_found(sandbox_id, snapshot_id))
            }
            Err(e) if e.is_invalid_path_parameter() || e.is_params_error() => {
                Err(AppError::BadRequest(e.to_string()))
            }
            Err(e) => Err(internal_error(e)),
        }
    }

    async fn build_create_request_payload(
        &self,
        sandbox_id: &str,
        request_id: &str,
    ) -> AppResult<serde_json::Value> {
        let sandbox = self.fetch_sandbox(sandbox_id).await?;
        if let Some(template_payload) = self
            .template_create_request_payload(&sandbox.template_id, request_id)
            .await?
        {
            return Ok(template_payload);
        }
        Ok(self.minimal_create_request_payload(&sandbox, request_id))
    }

    async fn fetch_sandbox(&self, sandbox_id: &str) -> AppResult<crate::cubemaster::SandboxDetail> {
        let resp = self
            .cubemaster
            .get_sandbox(sandbox_id, &self.instance_type)
            .await
            .map_err(|e| {
                if e.is_not_found() {
                    AppError::NotFound(format!("sandbox {} not found", sandbox_id))
                } else {
                    internal_error(e)
                }
            })?;
        resp.ret.as_result().map_err(|e| {
            if e.is_not_found() {
                AppError::NotFound(format!("sandbox {} not found", sandbox_id))
            } else {
                internal_error(e)
            }
        })?;
        resp.into_first_sandbox(&self.instance_type)
            .ok_or_else(|| AppError::NotFound(format!("sandbox {} not found", sandbox_id)))
    }

    async fn template_create_request_payload(
        &self,
        template_id: &str,
        request_id: &str,
    ) -> AppResult<Option<serde_json::Value>> {
        let template_id = template_id.trim();
        if template_id.is_empty() {
            return Ok(None);
        }
        let resp = match self.cubemaster.get_template(template_id).await {
            Ok(resp) => resp,
            Err(e) if e.is_not_found() => return Ok(None),
            Err(e) => return Err(internal_error(e)),
        };
        resp.ret.as_result().map_err(internal_error)?;
        let mut value = match resp.create_request {
            Some(value) => value,
            None => return Ok(None),
        };
        ensure_request_id(&mut value, request_id);
        Ok(Some(value))
    }

    fn minimal_create_request_payload(
        &self,
        sandbox: &crate::cubemaster::SandboxDetail,
        request_id: &str,
    ) -> serde_json::Value {
        let mut annotations = sandbox.annotations.clone();
        if !sandbox.template_id.trim().is_empty() {
            annotations
                .entry("cube.master.appsnapshot.template.id".to_string())
                .or_insert_with(|| sandbox.template_id.clone());
        }

        serde_json::json!({
            "request_id": request_id,
            "instance_type": self.instance_type.clone(),
            "annotations": annotations,
            "labels": sandbox.labels.clone(),
            "containers": [],
            "exposed_ports": [],
            "network_type": "tap",
        })
    }
}

// ── helpers ────────────────────────────────────────────────────────────────

fn new_request_id() -> String {
    Uuid::new_v4().to_string()
}

/// Normalise a pagination cursor coming from the request URL.  CubeMaster
/// only accepts a real cursor or no parameter at all; an empty / whitespace
/// `next_token=` query restarts pagination silently in some builds, so we
/// strip it here before reaching the client (Bug 3).
fn normalize_next_token(token: Option<&str>) -> Option<String> {
    token
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
}

fn internal_error(e: impl std::fmt::Display) -> AppError {
    AppError::Internal(anyhow::anyhow!("{}", e))
}

fn required_operation_id(operation_id: Option<&str>, operation: &str) -> AppResult<String> {
    operation_id.map(str::to_owned).ok_or_else(|| {
        AppError::Internal(anyhow::anyhow!(
            "{} response missing operation_id",
            operation
        ))
    })
}

fn sandbox_not_found(sandbox_id: &str) -> AppError {
    AppError::NotFound(format!("sandbox {} not found", sandbox_id))
}

fn snapshot_not_found(snapshot_id: &str) -> AppError {
    AppError::NotFound(format!("snapshot {} not found", snapshot_id))
}

fn sandbox_or_snapshot_not_found(sandbox_id: &str, snapshot_id: &str) -> AppError {
    AppError::NotFound(format!(
        "sandbox {} or snapshot {} not found",
        sandbox_id, snapshot_id
    ))
}

/// Tag master appends (and resolves back) when an E2B name carries no
/// explicit tag; mirrors templatecenter's SnapshotAliasTagDefault.
const SNAPSHOT_ALIAS_TAG_DEFAULT: &str = "default";

/// t_cube_snapshot.alias column width; alias ≤64 + ':' + tag ≤63.
const SNAPSHOT_ALIAS_KEY_MAX_LEN: usize = 128;

/// Parsed form of the E2B `SandboxSnapshotRequest.name`.
struct ParsedSnapshotName {
    /// Namespace-stripped name, verbatim including any user tag — this is
    /// what the SDK sees in `names` and what master stores as display_name.
    display_name: Option<String>,
    /// Qualified alias key ("alias:tag", tag defaulted to "default") that
    /// master claims; None when no valid alias can be derived.
    alias: Option<String>,
}

/// Parse an E2B snapshot name ("alias[:tag]", optionally "ns/alias[:tag]")
/// into the display name and the qualified alias key.
///
/// CubeSandbox has a single flat alias namespace, so a leading namespace is
/// dropped — mirroring `alias_from_name` on the template path. A missing tag
/// defaults to "default" so `create_snapshot(name="alias")` yields
/// `snapshotID="alias:default"`, matching official E2B.
///
/// Returns alias=None for anything that is not a valid key (missing name,
/// bad charset, reserved `tpl-`/`snap-` prefix, empty segment, more than one
/// ':' separator, over-long key): the snapshot is then created without an
/// alias and `snapshotID` falls back to the raw `snap-*` id — the SDK
/// contract's fallback shape. Invalid names degrade instead of failing the
/// request, mirroring the template path's silent-drop convention.
fn parse_snapshot_name(name: Option<&str>) -> ParsedSnapshotName {
    let Some(name) = name.map(str::trim).filter(|n| !n.is_empty()) else {
        return ParsedSnapshotName {
            display_name: None,
            alias: None,
        };
    };
    // Strip namespace: keep the last path segment. The "registry:5000/x"
    // image-ref shape needs no special case — its tag lookalike ("5000")
    // contains no '/' so the segment split happens on '/' first either way.
    let stripped = name.rsplit('/').next().unwrap_or(name).trim();
    if stripped.is_empty() {
        return ParsedSnapshotName {
            display_name: Some(name.to_string()),
            alias: None,
        };
    }
    let (alias, tag) = match stripped.split_once(':') {
        Some((alias, tag)) => (alias, tag),
        None => (stripped, SNAPSHOT_ALIAS_TAG_DEFAULT),
    };
    let valid = !alias.is_empty()
        && !tag.is_empty()
        && !alias.contains(':')
        && !tag.contains(':')
        && is_valid_alias(alias)
        && is_valid_alias(tag)
        && alias.len() + 1 + tag.len() <= SNAPSHOT_ALIAS_KEY_MAX_LEN;
    ParsedSnapshotName {
        display_name: Some(stripped.to_string()),
        alias: valid.then(|| format!("{alias}:{tag}")),
    }
}

/// Normalise a client-supplied snapshot identifier before it travels to
/// master in a URL path: trim and strip any "ns/" prefix ('/' can never be a
/// path segment character). Tag handling stays with master, the single
/// resolution authority.
fn normalize_snapshot_identifier(identifier: &str) -> String {
    let trimmed = identifier.trim();
    trimmed
        .rsplit('/')
        .next()
        .unwrap_or(trimmed)
        .trim()
        .to_string()
}

fn snapshot_delete_conflict(snapshot_id: &str) -> AppError {
    AppError::Conflict(format!(
        "snapshot {} cannot be deleted (active operation in progress)",
        snapshot_id
    ))
}
fn rollback_conflict(sandbox_id: &str, snapshot_id: &str) -> AppError {
    AppError::Conflict(format!(
        "rollback conflict: sandbox={} snapshot={}",
        sandbox_id, snapshot_id
    ))
}

fn snapshot_resource_to_info(r: SnapshotResource) -> SnapshotInfo {
    let names = snapshot_names(&r);
    let backend = snapshot_backend(&r);
    let remote_status = optional_backend(&r.remote_status);
    SnapshotInfo {
        snapshot_id: qualified_snapshot_id(&r),
        names,
        backend,
        remote_status,
    }
}

/// The identifier the E2B SDK round-trips (create → `Sandbox.create` →
/// `delete_snapshot`): the qualified alias when one was claimed, else the
/// raw `snap-*` id (the SDK contract's fallback shape).
fn qualified_snapshot_id(r: &SnapshotResource) -> String {
    let alias = r.alias.trim();
    if !alias.is_empty() {
        alias.to_string()
    } else {
        r.snapshot_id.clone()
    }
}

fn ensure_snapshot_ready(snapshot: &SnapshotResource) -> AppResult<()> {
    let status = normalized_status(Some(snapshot.status.as_str()));
    if status == "READY" {
        return Ok(());
    }
    Err(AppError::Internal(anyhow::anyhow!(
        "snapshot {} returned unexpected status {}",
        snapshot.snapshot_id,
        snapshot.status.trim()
    )))
}

fn ensure_operation_ready(
    status: Option<&str>,
    operation: &str,
    resource_id: &str,
) -> AppResult<String> {
    let status = normalized_status(status);
    if status == "READY" {
        return Ok(status);
    }
    Err(AppError::Internal(anyhow::anyhow!(
        "{} for {} returned unexpected status {}",
        operation,
        resource_id,
        status
    )))
}

fn normalized_status(status: Option<&str>) -> String {
    let status = status.unwrap_or_default().trim().to_ascii_uppercase();
    if status.is_empty() {
        "<empty>".to_string()
    } else {
        status
    }
}

fn snapshot_resource_to_list_item(r: SnapshotResource) -> SnapshotListItem {
    let names = snapshot_names(&r);
    let backend = snapshot_backend(&r);
    let remote_status = optional_backend(&r.remote_status);
    SnapshotListItem {
        snapshot_id: qualified_snapshot_id(&r),
        names,
        status: r.status,
        origin_sandbox_id: if r.origin_sandbox_id.is_empty() {
            None
        } else {
            Some(r.origin_sandbox_id)
        },
        created_at: r.created_at,
        updated_at: r.updated_at,
        backend,
        remote_status,
    }
}

fn snapshot_backend(r: &SnapshotResource) -> Option<String> {
    if let Some(b) = optional_backend(&r.backend) {
        return Some(b);
    }
    optional_backend(&r.storage_backend)
}

fn optional_backend(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn ensure_request_id(value: &mut serde_json::Value, request_id: &str) {
    if !value.is_object() {
        *value = serde_json::json!({});
    }
    let object = value.as_object_mut().expect("object just initialized");
    object.insert(
        "request_id".to_string(),
        serde_json::Value::String(request_id.to_string()),
    );
    object.insert(
        "requestID".to_string(),
        serde_json::Value::String(request_id.to_string()),
    );
    let request_field = object
        .entry("request".to_string())
        .or_insert_with(|| serde_json::json!({}));
    if !request_field.is_object() {
        *request_field = serde_json::json!({});
    }
    request_field
        .as_object_mut()
        .expect("request field just initialized")
        .insert(
            "requestID".to_string(),
            serde_json::Value::String(request_id.to_string()),
        );
}

fn snapshot_names(resource: &SnapshotResource) -> Vec<String> {
    if !resource.names.is_empty() {
        return resource.names.clone();
    }
    if !resource.display_name.trim().is_empty() {
        return vec![resource.display_name.clone()];
    }
    if !resource.alias.trim().is_empty() {
        return vec![resource.alias.clone()];
    }
    if resource.snapshot_id.trim().is_empty() {
        return Vec::new();
    }
    vec![resource.snapshot_id.clone()]
}

fn owned_or_fallback(value: String, fallback: &str) -> String {
    if value.trim().is_empty() {
        fallback.to_string()
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot(status: &str) -> SnapshotResource {
        SnapshotResource {
            snapshot_id: "snap-1".into(),
            names: vec!["snap-name".into()],
            display_name: "snap-name".into(),
            alias: String::new(),
            status: status.into(),
            origin_sandbox_id: "sb-1".into(),
            origin_node_id: "node-a".into(),
            instance_type: "cubebox".into(),
            storage_backend: String::new(),
            backend: String::new(),
            remote_status: String::new(),
            created_at: None,
            updated_at: None,
        }
    }

    #[test]
    fn normalize_next_token_drops_empty_and_whitespace_cursors() {
        assert_eq!(normalize_next_token(None), None);
        assert_eq!(normalize_next_token(Some("")), None);
        assert_eq!(normalize_next_token(Some("   ")), None);
        assert_eq!(normalize_next_token(Some("\t\n")), None);
        assert_eq!(
            normalize_next_token(Some("  cursor-42  ")),
            Some("cursor-42".to_string())
        );
        assert_eq!(
            normalize_next_token(Some("cursor-42")),
            Some("cursor-42".to_string())
        );
    }

    #[test]
    fn snapshot_ready_guard_rejects_non_ready_status() {
        let err = ensure_snapshot_ready(&sample_snapshot("CREATING"))
            .expect_err("non-ready snapshot should fail");
        match err {
            AppError::Internal(inner) => {
                assert!(inner.to_string().contains("unexpected status"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn operation_ready_guard_rejects_non_ready_status() {
        let err = ensure_operation_ready(Some("RUNNING"), "snapshot rollback", "snap-1")
            .expect_err("non-ready operation should fail");
        match err {
            AppError::Internal(inner) => {
                assert!(inner.to_string().contains("unexpected status"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn snapshot_info_uses_e2b_shape_without_operation_id() {
        let payload = serde_json::to_value(snapshot_resource_to_info(sample_snapshot("READY")))
            .expect("serialize snapshot info");

        assert_eq!(
            payload.get("snapshotID").and_then(|value| value.as_str()),
            Some("snap-1")
        );
        assert!(payload.get("operationID").is_none());
    }

    #[test]
    fn snapshot_info_falls_back_to_snapshot_id_for_names() {
        let mut snapshot = sample_snapshot("READY");
        snapshot.names.clear();
        snapshot.display_name.clear();

        let payload = serde_json::to_value(snapshot_resource_to_info(snapshot))
            .expect("serialize snapshot info");

        assert_eq!(
            payload.get("snapshotID").and_then(|value| value.as_str()),
            Some("snap-1")
        );
        assert_eq!(
            payload
                .get("names")
                .and_then(|value| value.as_array())
                .and_then(|values| values.first())
                .and_then(|value| value.as_str()),
            Some("snap-1")
        );
    }

    #[test]
    fn snapshot_list_item_uses_snapshot_id_field_shape() {
        let payload =
            serde_json::to_value(snapshot_resource_to_list_item(sample_snapshot("READY")))
                .expect("serialize snapshot list item");

        assert_eq!(
            payload.get("snapshotID").and_then(|value| value.as_str()),
            Some("snap-1")
        );
        assert_eq!(
            payload
                .get("names")
                .and_then(|value| value.as_array())
                .and_then(|values| values.first())
                .and_then(|value| value.as_str()),
            Some("snap-name")
        );
    }

    #[test]
    fn aliased_snapshot_reports_qualified_alias_as_snapshot_id() {
        let mut snapshot = sample_snapshot("READY");
        // Master never populates `names` (CubeAPI synthesizes it from
        // display_name / alias); clear the fixture's artificial value.
        snapshot.names.clear();
        snapshot.display_name = "agentscope-run".into();
        snapshot.alias = "agentscope-run:default".into();

        let info = snapshot_resource_to_info(snapshot.clone());
        assert_eq!(info.snapshot_id, "agentscope-run:default");
        assert_eq!(info.names, vec!["agentscope-run".to_string()]);

        let list_item = snapshot_resource_to_list_item(snapshot);
        assert_eq!(list_item.snapshot_id, "agentscope-run:default");
        assert_eq!(list_item.names, vec!["agentscope-run".to_string()]);
    }

    #[test]
    fn parse_snapshot_name_derives_qualified_alias_keys() {
        // Plain alias: tag defaults to "default" (official E2B shape).
        let parsed = parse_snapshot_name(Some("agentscope-run"));
        assert_eq!(parsed.display_name.as_deref(), Some("agentscope-run"));
        assert_eq!(parsed.alias.as_deref(), Some("agentscope-run:default"));

        // Explicit tag survives verbatim.
        let parsed = parse_snapshot_name(Some("foo:v2"));
        assert_eq!(parsed.display_name.as_deref(), Some("foo:v2"));
        assert_eq!(parsed.alias.as_deref(), Some("foo:v2"));

        // Namespace is stripped (flat alias namespace).
        let parsed = parse_snapshot_name(Some("team-slug/foo"));
        assert_eq!(parsed.display_name.as_deref(), Some("foo"));
        assert_eq!(parsed.alias.as_deref(), Some("foo:default"));

        let parsed = parse_snapshot_name(Some("team-slug/foo:v2"));
        assert_eq!(parsed.display_name.as_deref(), Some("foo:v2"));
        assert_eq!(parsed.alias.as_deref(), Some("foo:v2"));

        // Whitespace is trimmed; absent/empty name yields nothing.
        let parsed = parse_snapshot_name(Some("  foo  "));
        assert_eq!(parsed.alias.as_deref(), Some("foo:default"));
        let parsed = parse_snapshot_name(None);
        assert_eq!(parsed.display_name, None);
        assert_eq!(parsed.alias, None);
        let parsed = parse_snapshot_name(Some("   "));
        assert_eq!(parsed.display_name, None);
        assert_eq!(parsed.alias, None);
    }

    #[test]
    fn parse_snapshot_name_drops_invalid_aliases() {
        for name in [
            "UPPER",       // invalid charset
            "foo_bar",     // underscore not in alias charset
            "snap-foo",    // reserved id prefix
            "tpl-foo",     // reserved id prefix
            "foo:",        // empty tag
            ":v2",         // empty alias
            "foo:bar:baz", // multiple separators
            "foo:Bar",     // invalid tag charset
            "foo snap",    // whitespace inside
        ] {
            let parsed = parse_snapshot_name(Some(name));
            assert_eq!(parsed.alias, None, "alias for {name:?} should be dropped");
        }
        // Display name still records the namespace-stripped input verbatim.
        let parsed = parse_snapshot_name(Some("team/UPPER"));
        assert_eq!(parsed.display_name.as_deref(), Some("UPPER"));
        assert_eq!(parsed.alias, None);
    }

    #[test]
    fn parse_snapshot_name_enforces_key_length_budget() {
        let alias64 = "a".repeat(64);
        let tag63 = "b".repeat(63);
        // 64 + 1 + 63 = 128 fits.
        assert_eq!(
            parse_snapshot_name(Some(&format!("{alias64}:{tag63}"))).alias,
            Some(format!("{alias64}:{tag63}"))
        );
        // 64 + 1 + 64 = 129 exceeds the t_cube_snapshot.alias column width.
        let tag64 = "b".repeat(64);
        assert_eq!(
            parse_snapshot_name(Some(&format!("{alias64}:{tag64}"))).alias,
            None
        );
    }

    #[test]
    fn normalize_snapshot_identifier_strips_namespace_and_trims() {
        assert_eq!(normalize_snapshot_identifier("foo:default"), "foo:default");
        assert_eq!(
            normalize_snapshot_identifier("ns/foo:default"),
            "foo:default"
        );
        assert_eq!(normalize_snapshot_identifier("  ns/foo  "), "foo");
        assert_eq!(normalize_snapshot_identifier("snap-1"), "snap-1");
        assert_eq!(normalize_snapshot_identifier(""), "");
    }
}
