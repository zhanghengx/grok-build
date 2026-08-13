//! Model fetching, resolution, and management.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::RwLock;

use agent_client_protocol as acp;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use indexmap::IndexMap;

use crate::agent::config::{self, ModelEntry, resolve_credentials, sampling_config_for_model};
use crate::auth::{AuthManager, GrokAuth, GrokComConfig};
use crate::remote::{FetchModelsResult, fetch_models_blocking};
use crate::sampling::SamplerConfig as SamplingConfig;
use globset::{Glob, GlobSet, GlobSetBuilder};
use xai_grok_sampling_types::{ApiBackend, ReasoningEffort, ReasoningEffortOption};

// ── Auth method for model fetching ──────────────────────────────────────────

/// Credential for `/v1/models` fetching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelFetchAuth {
    Session,
    ApiKey,
    Deployment,
    CustomEndpoint,
}

impl ModelFetchAuth {
    /// custom_endpoint > session > deployment > API key.
    pub(crate) fn resolve(endpoints: &config::EndpointsConfig, has_cached_session: bool) -> Self {
        if endpoints.has_custom_endpoint() {
            Self::CustomEndpoint
        } else if has_cached_session {
            Self::Session
        } else if endpoints.deployment_key.is_some() {
            Self::Deployment
        } else {
            Self::Session
        }
    }

    fn cache_auth_method(&self) -> CacheAuthMethod {
        match self {
            Self::CustomEndpoint | Self::ApiKey => CacheAuthMethod::ApiKey,
            Self::Session => CacheAuthMethod::Session,
            Self::Deployment => CacheAuthMethod::Deployment,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CacheAuthMethod {
    Session,
    ApiKey,
    Deployment,
}

pub(crate) fn task_model_error_for_catalog(
    requested: &str,
    available: &IndexMap<String, ModelEntry>,
    is_session_auth: bool,
) -> Option<String> {
    let is_available = |entry: &ModelEntry| {
        entry.info.user_selectable && entry.info.visible_for_auth(is_session_auth)
    };
    if config::find_model_by_id(available, requested).is_some_and(&is_available) {
        return None;
    }

    let mut slugs = available
        .iter()
        .filter(|(_, entry)| is_available(entry))
        .map(|(slug, _)| slug.as_str())
        .collect::<Vec<_>>();
    slugs.sort_unstable();
    let guidance = if slugs.is_empty() {
        "No valid model slugs are currently available. Omit `model` to inherit the parent model."
            .to_string()
    } else {
        format!(
            "Valid model slugs: {}. Omit `model` to inherit the parent model.",
            slugs.join(", ")
        )
    };
    Some(format!("Unknown Task.model slug '{requested}'. {guidance}"))
}

/// Thread-safe model manager.
#[derive(Clone)]
pub struct ModelsManager {
    inner: Arc<Inner>,
}

/// Progress of the first real-catalog load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CatalogProgress {
    Pending,
    Failed,
    Ready,
}

/// Which catalog a fetch result belongs to. A model-owned endpoint catalog is
/// authoritative for the configured model, so a later global result must not
/// replace it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CatalogSource {
    Global,
    ModelEndpoint,
}
impl Default for CatalogSource {
    fn default() -> Self {
        CatalogSource::Global
    }
}

/// Catalog fields written together under one lock, so readers never see a torn mix.
#[derive(Default)]
struct CatalogState {
    prefetched: Option<IndexMap<String, ModelEntry>>,
    models: IndexMap<String, ModelEntry>,
    etag: Option<String>,
    /// Gates whether the apply path reselects the default (first real catalog)
    has_fetched_real_catalog: bool,
    /// True once the configured model's own `/models` endpoint populated the catalog.
    model_endpoint_catalog_loaded: bool,
    /// Which source populated the catalog; a global result must not replace
    /// an authoritative model endpoint catalog.
    catalog_source: CatalogSource,
    /// Configured model whose endpoint populated a model-scoped catalog.
    catalog_owner: Option<acp::ModelId>,
    /// `allowed_models` matched nothing; the prompt path blocks instead.
    allowlist_excludes_all: bool,
    /// Bumped on identity change; a fetch captured before it must not apply.
    generation: u64,
    /// Bumped when the current model's endpoint connection context changes or
    /// the identity is cleared. A model-endpoint fetch captured before it must
    /// not apply; settings-only publications leave it unchanged so an in-flight
    /// endpoint refresh can still publish.
    endpoint_generation: u64,
}

/// True when the model's endpoint connection context (base URL, credentials,
/// backend, auth scheme, request metadata) differs between two config
/// snapshots. Endpoint-derived catalog entries inherit this context, so they
/// can be reused across a config publication only while it is unchanged.
fn model_endpoint_changed(old: &config::Config, new: &config::Config, owner_key: &str) -> bool {
    let old_models = config::resolve_model_list(old, None);
    let new_models = config::resolve_model_list(new, None);
    let old_entry = old_models.get(owner_key);
    let new_entry = new_models.get(owner_key);
    match (old_entry, new_entry) {
        (Some(old_entry), Some(new_entry)) => {
            old_entry.info.base_url != new_entry.info.base_url
                || old_entry.info.api_backend != new_entry.info.api_backend
                || old_entry.info.auth_scheme != new_entry.info.auth_scheme
                || old_entry.info.extra_headers != new_entry.info.extra_headers
                || old_entry.info.query_params != new_entry.info.query_params
                || old_entry.info.env_http_headers != new_entry.info.env_http_headers
                || old_entry.api_key != new_entry.api_key
                || old_entry.env_key != new_entry.env_key
                || old_entry.auth_provider != new_entry.auth_provider
                || old_entry.api_base_url != new_entry.api_base_url
        }
        _ => true,
    }
}

/// Whether a config overlay replaced the connection context (URL, credentials,
/// backend, auth scheme, request metadata) of an endpoint-returned entry.
/// Metadata-only overlays leave the entry owned by the endpoint catalog that
/// discovered it.
fn endpoint_entry_context_differs(raw: &config::ModelEntry, resolved: &config::ModelEntry) -> bool {
    raw.info.base_url != resolved.info.base_url
        || raw.info.api_backend != resolved.info.api_backend
        || raw.info.auth_scheme != resolved.info.auth_scheme
        || raw.info.extra_headers != resolved.info.extra_headers
        || raw.info.query_params != resolved.info.query_params
        || raw.info.env_http_headers != resolved.info.env_http_headers
        || raw.api_key != resolved.api_key
        || raw.env_key != resolved.env_key
        || raw.auth_provider != resolved.auth_provider
        || raw.api_base_url != resolved.api_base_url
}

struct Inner {
    catalog: RwLock<CatalogState>,
    current_model_id: RwLock<acp::ModelId>,
    current_reasoning_effort: RwLock<Option<ReasoningEffort>>,
    // ── Owned context for self-contained refresh ────────────────
    auth_manager: Arc<AuthManager>,
    cfg: RwLock<config::Config>,
    fetch_auth: RwLock<ModelFetchAuth>,
    gateway: RwLock<Option<xai_acp_lib::AcpAgentGatewaySender>>,
    cache: ModelsCacheManager,
    endpoint: Arc<dyn ModelsEndpoint>,
    /// Guard to prevent overlapping retry loops.
    retry_in_flight: AtomicBool,
    /// Single-flight for the etag-triggered background refresh (`spawn_fetch`).
    refresh_in_flight: AtomicBool,
    fetches_in_flight: AtomicUsize,
    /// Model-switch signal: a generation counter bumped when the current model id changes.
    model_switch_watch: tokio::sync::watch::Sender<u64>,
    /// Progress of the first real-catalog load, watched by bounded waits.
    catalog_progress: tokio::sync::watch::Sender<CatalogProgress>,
    /// Set once the user explicitly picks a model (`/model`); guards the
    /// first-catalog reselect from clobbering that choice.
    user_selected_model: AtomicBool,
}

/// Clears an in-flight flag on drop so a panicking task can't wedge future refreshes.
struct RetryInFlightGuard(Arc<Inner>);
impl Drop for RetryInFlightGuard {
    fn drop(&mut self) {
        self.0.retry_in_flight.store(false, Ordering::Release);
    }
}
struct RefreshInFlightGuard(Arc<Inner>);
impl Drop for RefreshInFlightGuard {
    fn drop(&mut self) {
        self.0.refresh_in_flight.store(false, Ordering::Release);
    }
}

/// One fetch attempt (or retry sequence), counted for bounded waiters.
/// Begin before spawning the task; beginning supersedes an earlier `Failed`.
struct FetchAttemptGuard {
    inner: Arc<Inner>,
    generation: u64,
}
impl FetchAttemptGuard {
    fn begin(inner: &Arc<Inner>) -> Self {
        // Count first: a waiter that sees `Pending` must also see the attempt.
        inner.fetches_in_flight.fetch_add(1, Ordering::AcqRel);
        inner.catalog_progress.send_if_modified(|p| {
            let supersede = *p == CatalogProgress::Failed;
            if supersede {
                *p = CatalogProgress::Pending;
            }
            supersede
        });
        let generation = inner.catalog.read().generation;
        Self {
            inner: inner.clone(),
            generation,
        }
    }
}
impl Drop for FetchAttemptGuard {
    fn drop(&mut self) {
        if self.inner.fetches_in_flight.fetch_sub(1, Ordering::AcqRel) > 1 {
            return;
        }
        // Last attempt out with no outcome: latch so waiters return. The
        // lock makes the generation check atomic against `clear()`.
        let cat = self.inner.catalog.read();
        if cat.generation != self.generation
            || self.inner.fetches_in_flight.load(Ordering::Acquire) > 0
        {
            return;
        }
        self.inner.catalog_progress.send_if_modified(|p| {
            let unresolved = *p == CatalogProgress::Pending;
            if unresolved {
                *p = CatalogProgress::Failed;
            }
            unresolved
        });
    }
}

impl Default for ModelsManager {
    fn default() -> Self {
        let grok_home = crate::util::grok_home::grok_home();
        let auth_manager = Arc::new(AuthManager::new(&grok_home, GrokComConfig::default()));
        Self::new(
            None,
            IndexMap::new(),
            acp::ModelId::new("default"),
            auth_manager,
            config::Config::default(),
        )
    }
}

/// Builder for [`ModelsManager`]; transport and disk cache default to production (tests override them).
pub(crate) struct ModelsManagerBuilder {
    prefetched: Option<IndexMap<String, ModelEntry>>,
    models: IndexMap<String, ModelEntry>,
    current_model_id: acp::ModelId,
    auth_manager: Arc<AuthManager>,
    cfg: config::Config,
    endpoint: Arc<dyn ModelsEndpoint>,
    cache: ModelsCacheManager,
}

impl ModelsManagerBuilder {
    pub(crate) fn new(
        prefetched: Option<IndexMap<String, ModelEntry>>,
        models: IndexMap<String, ModelEntry>,
        current_model_id: acp::ModelId,
        auth_manager: Arc<AuthManager>,
        cfg: config::Config,
    ) -> Self {
        Self {
            prefetched,
            models,
            current_model_id,
            auth_manager,
            cfg,
            endpoint: Arc::new(HttpModelsEndpoint),
            cache: ModelsCacheManager::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn endpoint(mut self, endpoint: Arc<dyn ModelsEndpoint>) -> Self {
        self.endpoint = endpoint;
        self
    }

    #[cfg(test)]
    pub(crate) fn cache(mut self, cache: ModelsCacheManager) -> Self {
        self.cache = cache;
        self
    }

    pub(crate) fn build(self) -> ModelsManager {
        let has_session = self.auth_manager.current_or_expired().is_some();
        let fetch_auth = ModelFetchAuth::resolve(&self.cfg.endpoints, has_session);
        let current_reasoning_effort = self.cfg.models.default_reasoning_effort;
        ModelsManager {
            inner: Arc::new(Inner {
                catalog: RwLock::new(CatalogState {
                    prefetched: self.prefetched,
                    models: self.models,
                    ..Default::default()
                }),
                current_model_id: RwLock::new(self.current_model_id),
                current_reasoning_effort: RwLock::new(current_reasoning_effort),
                auth_manager: self.auth_manager,
                cfg: RwLock::new(self.cfg),
                fetch_auth: RwLock::new(fetch_auth),
                gateway: RwLock::new(None),
                cache: self.cache,
                endpoint: self.endpoint,
                retry_in_flight: AtomicBool::new(false),
                refresh_in_flight: AtomicBool::new(false),
                fetches_in_flight: AtomicUsize::new(0),
                model_switch_watch: tokio::sync::watch::channel(0u64).0,
                catalog_progress: tokio::sync::watch::channel(CatalogProgress::Pending).0,
                user_selected_model: AtomicBool::new(false),
            }),
        }
    }
}

impl ModelsManager {
    pub(crate) fn new(
        prefetched: Option<IndexMap<String, ModelEntry>>,
        models: IndexMap<String, ModelEntry>,
        current_model_id: acp::ModelId,
        auth_manager: Arc<AuthManager>,
        cfg: config::Config,
    ) -> Self {
        ModelsManagerBuilder::new(prefetched, models, current_model_id, auth_manager, cfg).build()
    }

    /// Subscribe to model-switch events. Returns a `watch::Receiver`
    pub(crate) fn subscribe_model_switch(&self) -> tokio::sync::watch::Receiver<u64> {
        self.inner.model_switch_watch.subscribe()
    }

    /// Cheap snapshot of the current model-switch generation, for the laziness-check poll loop.
    pub(crate) fn model_switch_generation(&self) -> u64 {
        *self.inner.model_switch_watch.borrow()
    }

    /// Build from a resolved config. Falls back to bundled default if no models available.
    pub(crate) fn from_config(
        cfg: &config::Config,
        prefetched_models: Option<IndexMap<String, ModelEntry>>,
        auth_manager: Arc<AuthManager>,
    ) -> Result<Self, String> {
        let has_session = auth_manager.current_or_expired().is_some();
        let is_session_auth = auth_manager
            .current_or_expired()
            .is_some_and(|a| a.is_session_auth());
        let fetch_auth = ModelFetchAuth::resolve(&cfg.endpoints, has_session);
        let mut cached_etag = None;
        let prefetched_models = prefetched_models.or_else(|| {
            let cache = ModelsCacheManager::new();
            cache
                .load_fresh(
                    &fetch_auth.cache_auth_method(),
                    &crate::remote::models_list_url(&cfg.endpoints, fetch_auth),
                )
                .map(|c| {
                    cached_etag = c.etag;
                    c.models
                })
        });
        let has_prefetched = prefetched_models.is_some();
        let catalog = resolve_model_catalog(cfg, prefetched_models.clone());

        if has_prefetched {
            validate_selectable(cfg, &catalog)?;
        }

        let (current_model_key, current_model, model_source) =
            resolve_default_model(cfg, &catalog, is_session_auth);

        tracing::info!(
            model_id = %current_model.model,
            source = %model_source,
            "default model resolved"
        );

        let current_model_id = acp::ModelId::new(Arc::from(current_model_key));

        let mgr = Self::new(
            prefetched_models,
            catalog,
            current_model_id,
            auth_manager,
            cfg.clone(),
        );
        if has_prefetched {
            let mut cat = mgr.inner.catalog.write();
            cat.has_fetched_real_catalog = true;
            // With the etag, the first check renews instead of refetching.
            cat.etag = cached_etag;
            mgr.inner
                .catalog_progress
                .send_replace(CatalogProgress::Ready);
        }
        Ok(mgr)
    }

    pub(crate) fn set_gateway(&self, gateway: xai_acp_lib::AcpAgentGatewaySender) {
        *self.inner.gateway.write() = Some(gateway);
    }

    /// Swap config, rebuild catalog, and reselect the model. Returns the
    /// rejection reason when the reload is invalid, so callers can avoid
    /// publishing a config the manager did not accept.
    pub(crate) fn apply_config(&self, new_config: config::Config) -> Result<(), String> {
        if let Err(e) = new_config.validate_model_filters() {
            tracing::error!(error = %e, "ignoring config reload: invalid model filters");
            return Err(e);
        }
        let (prefetched, has_real_catalog, catalog_source, catalog_owner) = {
            let cat = self.inner.catalog.read();
            (
                cat.prefetched.clone(),
                cat.has_fetched_real_catalog,
                cat.catalog_source,
                cat.catalog_owner.clone(),
            )
        };
        let old_config = self.inner.cfg.read().clone();
        // Model-endpoint entries carry inherited routing and credentials. Keep
        // them across a config publication while the endpoint connection
        // context is unchanged; otherwise drop them and let the caller refetch
        // so dynamically discovered models never disappear silently.
        let endpoint_catalog_active = catalog_source == CatalogSource::ModelEndpoint;
        let endpoint_catalog_invalidated = endpoint_catalog_active
            && !catalog_owner.as_ref().is_some_and(|owner| {
                !model_endpoint_changed(&old_config, &new_config, owner.0.as_ref())
            });
        // A settings-only publication must not invalidate an in-flight
        // model-endpoint fetch, but a change to the endpoint connection
        // context must. Before the endpoint catalog loads, the current model
        // id is the fetch's owner.
        let endpoint_context_changed = {
            let owner_key = catalog_owner
                .as_ref()
                .map(|owner| owner.0.as_ref().to_string())
                .unwrap_or_else(|| self.inner.current_model_id.read().0.as_ref().to_string());
            model_endpoint_changed(&old_config, &new_config, &owner_key)
        };
        let endpoint_catalog_still_valid = endpoint_catalog_active && !endpoint_catalog_invalidated;
        let retained_prefetched = if endpoint_catalog_invalidated {
            None
        } else {
            prefetched
        };
        let retained_real_catalog = has_real_catalog && !endpoint_catalog_invalidated;
        let new_catalog = resolve_model_catalog(&new_config, retained_prefetched.clone());
        if retained_real_catalog && let Err(e) = validate_selectable(&new_config, &new_catalog) {
            tracing::error!(error = %e, "ignoring config reload: allowed_models excludes all models");
            return Err(e);
        }

        let (old_preferred, old_default_is_campaign) = {
            let cfg = self.inner.cfg.read();
            (
                cfg.models.default.clone(),
                cfg.models.default_is_campaign_driven,
            )
        };
        let new_preferred = new_config.models.default.clone();
        let has_session = self.inner.auth_manager.current_or_expired().is_some();
        *self.inner.fetch_auth.write() =
            ModelFetchAuth::resolve(&new_config.endpoints, has_session);
        let mut cfg = self.inner.cfg.write();
        *cfg = new_config.clone();
        {
            let mut cat = self.inner.catalog.write();
            // A config reload can change the endpoint, credential, provider,
            // or request metadata used by any catalog fetch already in flight.
            // Publish the new config before advancing the fence so a refresh
            // that observes this generation can only build a new request.
            cat.generation += 1;
            if endpoint_context_changed {
                cat.endpoint_generation += 1;
            }
            cat.prefetched = retained_prefetched;
            cat.models = new_catalog;
            cat.has_fetched_real_catalog = retained_real_catalog;
            cat.allowlist_excludes_all = allowlist_matches_nothing(&new_config, &cat.models);
            if !retained_real_catalog {
                cat.etag = None;
                self.inner
                    .catalog_progress
                    .send_replace(CatalogProgress::Pending);
            }
            cat.model_endpoint_catalog_loaded = endpoint_catalog_still_valid;
            cat.catalog_source = if endpoint_catalog_still_valid {
                CatalogSource::ModelEndpoint
            } else {
                CatalogSource::Global
            };
            cat.catalog_owner = if endpoint_catalog_still_valid {
                catalog_owner
            } else {
                None
            };
        }
        drop(cfg);

        let preferred_changed = new_preferred != old_preferred && new_preferred.is_some();
        let mut campaign_defaults = std::collections::HashSet::new();
        if new_config.models.default_is_campaign_driven
            && let Some(d) = &new_preferred
        {
            campaign_defaults.insert(d.clone());
        }
        if old_default_is_campaign && let Some(d) = &old_preferred {
            campaign_defaults.insert(d.clone());
        }
        let campaign_only_flip =
            is_campaign_only_flip(&old_preferred, &new_preferred, &campaign_defaults);
        let current_still_ok = {
            let cat = self.inner.catalog.read();
            let models = &cat.models;
            let cur = self.inner.current_model_id.read();
            models
                .get(cur.0.as_ref())
                .is_some_and(|e| e.info.user_selectable)
        };
        if preferred_changed && !(campaign_only_flip && current_still_ok) {
            self.reselect_default_model(&new_config);
        } else {
            self.reselect_current_model_if_missing(&new_config);
        }

        self.notify_models_updated();
        Ok(())
    }

    /// [`Self::apply_config`] plus an unconditional default re-resolve, for remote-settings arrival while no session exists.
    pub(crate) fn apply_config_reselecting_default(
        &self,
        new_config: config::Config,
    ) -> Result<(), String> {
        self.apply_config(new_config.clone())?;
        self.reselect_default_model(&new_config);
        self.notify_models_updated();
        Ok(())
    }

    // ── Accessors ───────────────────────────────────────────────────

    pub fn models(&self) -> IndexMap<String, ModelEntry> {
        self.inner.catalog.read().models.clone()
    }

    pub fn endpoints(&self) -> config::EndpointsConfig {
        self.inner.cfg.read().endpoints.clone()
    }

    /// Does the current credential grant access to OAuth-only models?
    fn is_session_auth(&self) -> bool {
        self.inner
            .auth_manager
            .current_or_expired()
            .is_some_and(|a| a.is_session_auth())
    }

    /// ACP-visible (non-hidden) projection of the catalog.
    pub fn available(&self) -> IndexMap<acp::ModelId, acp::ModelInfo> {
        let snapshot = {
            let cat = self.inner.catalog.read();
            let models = &cat.models;
            models.clone()
        };

        let selectable: IndexMap<_, _> = snapshot
            .into_iter()
            .filter(|(_, e)| e.info.user_selectable)
            .collect();

        available_models(&selectable, self.is_session_auth())
    }

    pub(crate) fn task_model_error(&self, requested: &str) -> Option<String> {
        let is_session_auth = self.is_session_auth();
        let cat = self.inner.catalog.read();
        let models = &cat.models;
        task_model_error_for_catalog(requested, models, is_session_auth)
    }

    pub fn current_model_id(&self) -> acp::ModelId {
        self.inner.current_model_id.read().clone()
    }

    pub(crate) fn set_current_model_id(&self, id: acp::ModelId) {
        self.inner
            .user_selected_model
            .store(true, Ordering::Relaxed);
        self.set_current_model_id_internal(id);
    }

    fn set_current_model_id_internal(&self, id: acp::ModelId) {
        let changed = {
            let mut cur = self.inner.current_model_id.write();
            let changed = *cur != id;
            if changed {
                *cur = id.clone();
                // Publish the id and its generation while holding the same
                // lock used by endpoint-refresh snapshots.
                self.inner
                    .model_switch_watch
                    .send_modify(|generation| *generation += 1);
            }
            changed
        };
        if changed {
            let cfg = self.inner.cfg.read().clone();
            let mut cat = self.inner.catalog.write();
            if cat.catalog_source == CatalogSource::ModelEndpoint {
                // Models returned by the endpoint inherit its connection
                // context and remain owned by that catalog, including IDs that
                // carry a metadata-only `[model.<id>]` overlay. An overlay that
                // replaces the endpoint context, or a model absent from the
                // endpoint, does not: restore the config-only catalog so
                // OnlineIfUncached performs the fetch appropriate for the new
                // model. Ownership is by configured key or endpoint identity,
                // never the routing slug.
                let returned_by_endpoint = cat
                    .prefetched
                    .as_ref()
                    .is_some_and(|models| resolve_catalog_key(models, &id).is_some());
                let overlay_changes_context = match (
                    cat.prefetched
                        .as_ref()
                        .and_then(|models| resolve_catalog_key(models, &id))
                        .and_then(|key| {
                            cat.prefetched.as_ref().and_then(|m| m.get(key.0.as_ref()))
                        }),
                    resolve_catalog_key(&cat.models, &id)
                        .and_then(|key| cat.models.get(key.0.as_ref())),
                ) {
                    (Some(raw), Some(resolved)) => endpoint_entry_context_differs(raw, resolved),
                    _ => true,
                };
                let belongs_to_endpoint_catalog = cat.catalog_owner.as_ref() == Some(&id)
                    || (returned_by_endpoint && !overlay_changes_context);
                if !belongs_to_endpoint_catalog {
                    let generation = cat.generation + 1;
                    let models = resolve_model_catalog(&cfg, None);
                    let allowlist_excludes_all = allowlist_matches_nothing(&cfg, &models);
                    *cat = CatalogState {
                        models,
                        allowlist_excludes_all,
                        generation,
                        endpoint_generation: generation,
                        ..Default::default()
                    };
                    self.inner
                        .catalog_progress
                        .send_replace(CatalogProgress::Pending);
                }
            }
        }
    }

    /// Per-model Layer-3 LazinessDetector config for `model_id` (disabled default when absent).
    pub(crate) fn laziness_detector_for(
        &self,
        model_id: &str,
    ) -> config::LazinessDetectorPerModelConfig {
        self.inner
            .catalog
            .read()
            .models
            .get(model_id)
            .map(|e| e.info().laziness_detector.clone())
            .unwrap_or_default()
    }

    /// Test-only catalog poke: inserts a `ModelEntry` keyed by `id`,
    #[cfg(test)]
    pub(crate) fn insert_test_entry(&self, id: impl Into<String>, entry: ModelEntry) {
        self.inner.catalog.write().models.insert(id.into(), entry);
    }

    pub(crate) fn current_reasoning_effort(&self) -> Option<ReasoningEffort> {
        *self.inner.current_reasoning_effort.read()
    }

    pub(crate) fn set_current_reasoning_effort(&self, effort: Option<ReasoningEffort>) {
        *self.inner.current_reasoning_effort.write() = effort;
    }

    /// Whether the given model supports reasoning effort according to the catalog.
    pub(crate) fn model_supports_reasoning_effort(&self, model_id: &str) -> bool {
        self.inner
            .catalog
            .read()
            .models
            .get(model_id)
            .map(|e| e.info().supports_reasoning_effort)
            .unwrap_or(false)
    }

    pub(crate) fn model_default_reasoning_effort(&self, model_id: &str) -> Option<ReasoningEffort> {
        self.inner
            .catalog
            .read()
            .models
            .get(model_id)
            .and_then(|e| e.info().reasoning_effort)
    }

    /// The raw catalog `reasoning_efforts` list for `model_id` with no fallback,
    pub(crate) fn model_reasoning_efforts(&self, model_id: &str) -> Vec<ReasoningEffortOption> {
        self.inner
            .catalog
            .read()
            .models
            .get(model_id)
            .map(|e| e.info().reasoning_efforts.clone())
            .unwrap_or_default()
    }

    pub(crate) fn model_supports_backend_search(&self, model_id: &str) -> bool {
        self.inner
            .catalog
            .read()
            .models
            .get(model_id)
            .map(|e| e.info().supports_backend_search)
            .unwrap_or(false)
    }

    pub(crate) fn model_compactions_remaining(
        &self,
        model_id: &str,
    ) -> Option<xai_grok_sampling_types::CompactionsRemaining> {
        self.inner
            .catalog
            .read()
            .models
            .get(model_id)
            .and_then(|e| e.info().compactions_remaining)
    }

    pub(crate) fn model_compaction_at_tokens(
        &self,
        model_id: &str,
    ) -> Option<xai_grok_sampling_types::CompactionAtTokens> {
        self.inner
            .catalog
            .read()
            .models
            .get(model_id)
            .and_then(|e| e.info().compaction_at_tokens)
    }

    /// Catalog opt-in to display the served-checkpoint fingerprint for this model.
    pub(crate) fn model_show_model_fingerprint(&self, model_id: &str) -> bool {
        let cat = self.inner.catalog.read();
        let models = &cat.models;
        resolve_catalog_key(models, &acp::ModelId::new(model_id))
            .and_then(|key| models.get(key.0.as_ref()))
            .map(|e| e.info().show_model_fingerprint)
            .unwrap_or(false)
    }

    /// Resolved next-prompt-suggestion model pin from the live config
    pub(crate) fn prompt_suggest_model_pin(&self) -> crate::config::PromptSuggestModelPin {
        self.inner.cfg.read().prompt_suggest_model_pin.clone()
    }

    /// Whether `model_id` resolves in the current catalog — as a config key
    pub(crate) fn model_in_catalog(&self, model_id: &str) -> bool {
        let cat = self.inner.catalog.read();
        let models = &cat.models;
        resolve_catalog_key(models, &acp::ModelId::new(model_id)).is_some()
    }

    #[cfg(test)]
    fn prefetched(&self) -> Option<IndexMap<String, ModelEntry>> {
        self.inner.catalog.read().prefetched.clone()
    }

    #[cfg(test)]
    fn has_fetched_real_catalog(&self) -> bool {
        self.inner.catalog.read().has_fetched_real_catalog
    }

    /// Wait, bounded by one auth refresh plus one fetch, for the first
    /// fetch outcome; never triggers a fetch.
    pub(crate) async fn wait_for_first_catalog(&self) {
        self.wait_for_first_catalog_inner(crate::util::config::resolve_remote_fetch_enabled())
            .await;
    }

    async fn wait_for_first_catalog_inner(&self, remote_fetch_enabled: bool) -> bool {
        const BUDGET: std::time::Duration = crate::http::STARTUP_AUTH_REFRESH_TIMEOUT
            .saturating_add(crate::http::STARTUP_FETCH_TIMEOUT);
        let mut progress = self.inner.catalog_progress.subscribe();
        match *progress.borrow() {
            CatalogProgress::Ready => return true,
            CatalogProgress::Failed => return false,
            CatalogProgress::Pending => {}
        }
        if !remote_fetch_enabled {
            return false;
        }
        // Signed out with a session-only endpoint: no fetch is coming.
        if *self.inner.fetch_auth.read() == ModelFetchAuth::Session
            && self.inner.auth_manager.current_or_expired().is_none()
        {
            return false;
        }
        // Attempts latch `Failed` on exit, so pending plus idle means none started.
        if self.inner.fetches_in_flight.load(Ordering::Acquire) == 0 {
            return *progress.borrow() == CatalogProgress::Ready;
        }
        matches!(
            tokio::time::timeout(BUDGET, progress.wait_for(|p| *p != CatalogProgress::Pending))
                .await,
            Ok(Ok(p)) if *p == CatalogProgress::Ready
        )
    }

    // ── Mutations ───────────────────────────────────────────────────

    fn rebuild(&self, cfg: &config::Config, prefetched: Option<IndexMap<String, ModelEntry>>) {
        self.inner.catalog.write().models = resolve_model_catalog(cfg, prefetched);
    }

    /// Reset to this identity's bundled catalog and reselect a valid default.
    fn rebuild_bundled(&self, cfg: &config::Config) {
        self.rebuild(cfg, None);
        self.reselect_current_model_if_missing(cfg);
    }

    /// Refresh models when the etag changes.
    pub(crate) async fn refresh_if_new_etag(&self, etag: String) {
        let (same_etag, endpoint_owned) = {
            let cat = self.inner.catalog.read();
            (
                cat.etag.as_deref() == Some(etag.as_str()),
                cat.catalog_source == CatalogSource::ModelEndpoint,
            )
        };
        if same_etag {
            let fetch_auth = *self.inner.fetch_auth.read();
            self.inner
                .cache
                .renew_ttl(&fetch_auth.cache_auth_method(), &self.cache_origin())
                .await;
            return;
        }
        tracing::info!(etag = %etag, "models etag changed, refreshing");
        if endpoint_owned {
            let mgr = self.clone();
            tokio::task::spawn(async move {
                mgr.refresh_current_model_endpoint_inner(
                    crate::util::config::resolve_remote_fetch_enabled(),
                    Some(etag),
                )
                .await;
            });
            return;
        }
        self.spawn_fetch(Some(etag));
    }

    /// Auth identity changed: invalidate the disk cache and refresh the catalog.
    pub(crate) async fn on_auth_changed(&self) {
        let config = self.inner.cfg.read().clone();
        crate::agent::init::update_telemetry_config(&config, &self.inner.auth_manager);
        self.inner.cache.invalidate();
        // Fetches and the etag from the previous identity are stale now.
        {
            let mut cat = self.inner.catalog.write();
            cat.generation += 1;
            cat.endpoint_generation += 1;
            cat.etag = None;
        }
        let has_session = self.inner.auth_manager.current_or_expired().is_some();
        let fetch_auth = ModelFetchAuth::resolve(&config.endpoints, has_session);
        *self.inner.fetch_auth.write() = fetch_auth;
        // No session but the endpoint needs one: a fetch would 401, so skip it
        // and reset to this identity's bundled catalog.
        if !has_session && fetch_auth == ModelFetchAuth::Session {
            self.clear();
            self.rebuild_bundled(&config);
            // No fetch is coming; wake parked waiters. Lock and gate like
            // every other outcome publish.
            {
                let _cat = self.inner.catalog.read();
                self.inner.catalog_progress.send_if_modified(|p| {
                    let pending = *p == CatalogProgress::Pending;
                    if pending {
                        *p = CatalogProgress::Failed;
                    }
                    pending
                });
            }
            self.notify_models_updated();
            return;
        }

        let remote_fetch_enabled = crate::util::config::resolve_remote_fetch_enabled();
        self.fetch_and_apply_inner(remote_fetch_enabled).await;

        let needs_bundled_fallback = {
            let cat = self.inner.catalog.read();
            !cat.has_fetched_real_catalog && cat.prefetched.is_none()
        };
        if needs_bundled_fallback {
            if remote_fetch_enabled {
                xai_grok_telemetry::unified_log::warn(
                    "model catalog: falling back to bundled defaults only",
                    None,
                    Some(serde_json::json!({
                        "trigger": "on_auth_changed",
                        "had_real_catalog": false,
                    })),
                );
            } else {
                tracing::debug!("model catalog: bundled defaults in use (remote_fetch disabled)");
            }
            self.rebuild_bundled(&config);

            if remote_fetch_enabled {
                self.spawn_catalog_retry(remote_fetch_enabled);
            }
        }

        self.notify_models_updated();
    }

    fn notify_models_updated(&self) {
        let available = self.available();
        let current = self.current_model_id();
        let count = available.len();
        xai_grok_telemetry::unified_log::info(
            "model catalog: notifying clients",
            None,
            Some(serde_json::json!({
                "model_count": count,
                "current_model_id": current.0.as_ref(),
            })),
        );
        if let Some(ref gw) = *self.inner.gateway.read() {
            let model_state =
                acp::SessionModelState::new(current, available.values().cloned().collect());
            if let Ok(params) = serde_json::value::to_raw_value(&model_state) {
                gw.forward_fire_and_forget(acp::ExtNotification::new(
                    "x.ai/models/update",
                    params.into(),
                ));
            }
        }
    }

    /// Hot-reload the catalog from `~/.grok/models_cache.json` after an external write (config-watcher detected).
    pub(crate) fn reload_from_disk_cache(&self) {
        self.reload_from_cache_manager(&self.inner.cache);
    }

    /// Core of [`Self::reload_from_disk_cache`], parameterized over the cache
    fn reload_from_cache_manager(&self, cache: &ModelsCacheManager) {
        let fetch_auth = *self.inner.fetch_auth.read();
        let Some(cached) = cache.load_fresh(&fetch_auth.cache_auth_method(), &self.cache_origin())
        else {
            tracing::debug!("models cache changed on disk but is not loadable; ignoring");
            return;
        };

        let same_content = {
            let cat = self.inner.catalog.read();
            cat.prefetched.as_ref().is_some_and(|current| {
                serde_json::to_string(current).ok() == serde_json::to_string(&cached.models).ok()
            })
        };
        if same_content {
            if cached.etag.is_some() {
                self.inner.catalog.write().etag = cached.etag;
            }
            tracing::debug!("models cache changed on disk but catalog is identical; skipping");
            return;
        }

        let cfg = self.inner.cfg.read().clone();
        let count = cached.models.len();
        self.apply_catalog(&cfg, cached.models, cached.etag);
        tracing::info!(count, "model catalog hot-reloaded from disk cache");
        xai_grok_telemetry::unified_log::info(
            "model catalog: reloaded from external disk-cache write",
            None,
            Some(serde_json::json!({ "model_count": count })),
        );
        self.notify_models_updated();
    }

    /// Retry model catalog fetch in the background with exponential backoff.
    fn spawn_catalog_retry(&self, remote_fetch_enabled: bool) {
        self.spawn_catalog_retry_with_backoff(
            remote_fetch_enabled,
            crate::tools::retry::BackoffConfig::new(5, 5_000, 60_000),
        );
    }

    /// [`Self::spawn_catalog_retry`] with an injectable backoff (fast in tests).
    fn spawn_catalog_retry_with_backoff(
        &self,
        remote_fetch_enabled: bool,
        backoff: crate::tools::retry::BackoffConfig,
    ) {
        if !remote_fetch_enabled {
            return;
        }
        if self
            .inner
            .retry_in_flight
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            tracing::debug!("model catalog retry already in flight, skipping");
            return;
        }

        // The whole retry sequence is one attempt to waiters.
        let attempt = FetchAttemptGuard::begin(&self.inner);
        let mgr = self.clone();
        tokio::task::spawn(async move {
            let _attempt = attempt;
            let _retry_guard = RetryInFlightGuard(mgr.inner.clone());
            let result = crate::tools::retry::execute_with_backoff(
                &backoff,
                || {
                    let mgr = mgr.clone();
                    async move {
                        if mgr.inner.catalog.read().has_fetched_real_catalog {
                            return Ok(());
                        }

                        mgr.fetch_and_apply().await;

                        if mgr.inner.catalog.read().has_fetched_real_catalog {
                            Ok(())
                        } else {
                            Err("model catalog fetch returned no models")
                        }
                    }
                },
                |attempt, max_retries, delay| async move {
                    xai_grok_telemetry::unified_log::warn(
                        "model catalog: retry scheduled",
                        None,
                        Some(serde_json::json!({
                            "attempt": attempt,
                            "max_retries": max_retries,
                            "delay_ms": delay.as_millis() as u64,
                        })),
                    );
                },
            )
            .await;

            match result {
                Ok(()) => {
                    let count = mgr.available().len();
                    xai_grok_telemetry::unified_log::info(
                        "model catalog: retry succeeded",
                        None,
                        Some(serde_json::json!({ "model_count": count })),
                    );
                    mgr.notify_models_updated();
                }
                Err(e) => {
                    xai_grok_telemetry::unified_log::warn(
                        "model catalog: all retries exhausted",
                        None,
                        Some(serde_json::json!({ "error": e })),
                    );
                }
            }
        });
    }

    /// One-shot background catalog refresh after readiness; no-op when a fresh disk cache already loaded a real catalog.
    pub fn spawn_background_refresh(&self) {
        self.spawn_background_refresh_inner(crate::util::config::resolve_remote_fetch_enabled());
    }

    fn spawn_background_refresh_inner(&self, remote_fetch_enabled: bool) {
        if self.inner.catalog.read().has_fetched_real_catalog {
            tracing::debug!(
                "skipping startup background model refresh: fresh cache already loaded"
            );
            return;
        }
        self.spawn_catalog_retry(remote_fetch_enabled);
    }

    /// Refresh the model catalog on every auth token refresh.
    pub fn start_auth_refresh_watcher(&self, notify: Arc<tokio::sync::Notify>) {
        let mgr = self.clone();
        let had_catalog_at_start = self.inner.catalog.read().has_fetched_real_catalog;
        xai_grok_telemetry::unified_log::info(
            "model catalog: auth refresh watcher started",
            None,
            Some(serde_json::json!({
                "had_real_catalog": had_catalog_at_start,
                "model_count": self.available().len(),
            })),
        );
        tokio::spawn(async move {
            loop {
                notify.notified().await;
                if !crate::util::config::resolve_remote_fetch_enabled() {
                    tracing::debug!(
                        "model catalog: auth refresh watcher skipped (remote_fetch disabled)"
                    );
                    continue;
                }
                let had_catalog = mgr.inner.catalog.read().has_fetched_real_catalog;
                let old_count = mgr.available().len();
                xai_grok_telemetry::unified_log::info(
                    "model catalog: auth refresh watcher triggered",
                    None,
                    Some(serde_json::json!({
                        "had_real_catalog": had_catalog,
                        "model_count_before": old_count,
                    })),
                );
                mgr.fetch_and_apply().await;
                let has_catalog = mgr.inner.catalog.read().has_fetched_real_catalog;
                let new_count = mgr.available().len();
                if has_catalog {
                    if !had_catalog || new_count != old_count {
                        xai_grok_telemetry::unified_log::info(
                            "model catalog: auth refresh watcher updated catalog",
                            None,
                            Some(serde_json::json!({
                                "model_count_before": old_count,
                                "model_count_after": new_count,
                                "was_recovery": !had_catalog,
                            })),
                        );
                    }
                    mgr.notify_models_updated();
                } else {
                    xai_grok_telemetry::unified_log::warn(
                        "model catalog: auth refresh watcher fetch failed",
                        None,
                        Some(serde_json::json!({
                            "model_count": old_count,
                        })),
                    );
                }
            }
        });
    }

    /// Wipe in-memory state so a previous identity's catalog doesn't leak.
    fn clear(&self) {
        {
            let mut cat = self.inner.catalog.write();
            let generation = cat.generation + 1;
            let endpoint_generation = cat.endpoint_generation + 1;
            *cat = CatalogState::default();
            cat.generation = generation;
            cat.endpoint_generation = endpoint_generation;
            self.inner
                .catalog_progress
                .send_replace(CatalogProgress::Pending);
        }
        // A new identity starts fresh: drop the prior user's pick so its
        // first catalog reselects that identity's default.
        self.inner
            .user_selected_model
            .store(false, Ordering::Relaxed);
    }

    /// Build a `SamplingConfig` from the current model + auth state.
    pub fn sampling_config(&self) -> SamplingConfig {
        let config = self.inner.cfg.read().clone();
        let auth_manager = self.inner.auth_manager.as_ref();
        let current_model_id = self.current_model_id();
        let all_models = self.models();
        let fallback;
        let current_model = match all_models
            .get(current_model_id.0.as_ref())
            .or_else(|| all_models.values().next())
        {
            Some(m) => m,
            None => {
                tracing::warn!("no models available in catalog; defaulting to bundled model");
                let default_id = crate::models::default_model().to_string();
                fallback = ModelEntry::fallback(&default_id, &config.endpoints);
                &fallback
            }
        };

        let session_auth = auth_manager.current_or_expired();
        let credentials =
            resolve_credentials(current_model, session_auth.as_ref().map(|a| a.key.as_str()));

        sampling_config_for_model(
            current_model,
            credentials,
            config.endpoints.alpha_test_key.clone(),
            config.client_version.clone(),
            crate::managed_config::resolve_deployment_id(
                config.endpoints.deployment_key.as_deref(),
            ),
            None,
        )
    }

    /// Disk-cache origin key for this manager's current endpoints/auth shape
    fn cache_origin(&self) -> String {
        let endpoints = self.inner.cfg.read().endpoints.clone();
        let fetch_auth = *self.inner.fetch_auth.read();
        crate::remote::models_list_url(&endpoints, fetch_auth)
    }

    /// A catalog-fetch session refresh bounded by `STARTUP_AUTH_REFRESH_TIMEOUT`.
    /// A hung IdP on a cold cache degrades to a session-less fetch (the
    /// bundled/cache catalog stays and the next refresh retries) instead of
    /// stalling boot, mirroring the readiness path's no-mint auth bound.
    async fn bounded_startup_auth(auth_manager: &Arc<AuthManager>) -> Option<GrokAuth> {
        Self::bounded_auth_refresh(async { auth_manager.auth().await.ok() }).await
    }

    /// Bounds an auth-refresh future to `STARTUP_AUTH_REFRESH_TIMEOUT`, yielding
    /// `None` on timeout. Split out so the timeout contract is unit-testable
    /// without a live IdP.
    async fn bounded_auth_refresh<F>(fut: F) -> Option<GrokAuth>
    where
        F: std::future::Future<Output = Option<GrokAuth>>,
    {
        match tokio::time::timeout(crate::http::STARTUP_AUTH_REFRESH_TIMEOUT, fut).await {
            Ok(auth) => auth,
            Err(_) => {
                tracing::warn!(
                    timeout_secs = crate::http::STARTUP_AUTH_REFRESH_TIMEOUT.as_secs(),
                    "model catalog: auth refresh timed out; fetching without a fresh session"
                );
                None
            }
        }
    }

    fn spawn_fetch(&self, new_etag: Option<String>) {
        self.spawn_fetch_inner(
            new_etag,
            crate::util::config::resolve_remote_fetch_enabled(),
        );
    }

    fn spawn_fetch_inner(&self, new_etag: Option<String>, remote_fetch_enabled: bool) {
        if !remote_fetch_enabled {
            tracing::info!("model catalog refresh skipped: remote_fetch disabled");
            return;
        }
        if self
            .inner
            .refresh_in_flight
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            tracing::debug!("model catalog refresh already in flight, skipping");
            return;
        }
        // Generation first: an identity change after this point fails the
        // apply fence instead of publishing an old-credential fetch.
        let attempt = FetchAttemptGuard::begin(&self.inner);
        let generation = attempt.generation;
        let cfg = self.inner.cfg.read().clone();
        let endpoints = cfg.endpoints.clone();
        let fetch_auth = *self.inner.fetch_auth.read();
        let auth_manager = self.inner.auth_manager.clone();
        let endpoint = self.inner.endpoint.clone();
        let mgr = self.clone();

        tokio::task::spawn(async move {
            let _attempt = attempt;
            let _refresh_guard = RefreshInFlightGuard(mgr.inner.clone());
            let auth = Self::bounded_startup_auth(&auth_manager).await;
            let new_prefetched = match tokio::time::timeout(
                crate::http::STARTUP_FETCH_TIMEOUT,
                endpoint.fetch_models(endpoints, auth, fetch_auth),
            )
            .await
            {
                Ok(models) => models,
                Err(_) => {
                    tracing::warn!("etag-triggered model refresh timed out");
                    None
                }
            };
            if !mgr.apply_refresh_result_fenced(
                &cfg,
                new_prefetched,
                new_etag,
                Some(generation),
                None,
                None,
                CatalogSource::Global,
                None,
            ) {
                return;
            }
            tracing::info!("models manager refreshed");
            mgr.notify_models_updated();
        });
    }

    /// Resolve the model list: waits for or requests a catalog fetch.
    pub async fn list_models(&self, strategy: RefreshStrategy) {
        if self.current_model_has_endpoint() {
            // A model-scoped endpoint is authoritative for this configuration.
            // Do not populate the picker from the global proxy and then leave
            // the configured endpoint's model list unused.
            match strategy {
                RefreshStrategy::Offline => {
                    self.wait_for_first_catalog().await;
                }
                RefreshStrategy::OnlineIfUncached => {
                    if !self.inner.catalog.read().model_endpoint_catalog_loaded {
                        self.refresh_current_model_endpoint().await;
                    }
                }
                RefreshStrategy::Online => {
                    self.refresh_current_model_endpoint().await;
                }
            }
            return;
        }
        match strategy {
            RefreshStrategy::Offline => {
                self.wait_for_first_catalog().await;
            }
            RefreshStrategy::OnlineIfUncached => {
                if !self.inner.catalog.read().has_fetched_real_catalog {
                    self.fetch_and_apply().await;
                }
            }
            RefreshStrategy::Online => {
                self.fetch_and_apply().await;
            }
        }
    }

    fn current_model_has_endpoint(&self) -> bool {
        let current = self.current_model_id();
        let cfg = self.inner.cfg.read().clone();
        let catalog = self.inner.catalog.read();
        let Some((key, entry)) = catalog
            .models
            .get(current.0.as_ref())
            .map(|entry| (current.0.as_ref(), entry))
            .or_else(|| {
                catalog
                    .models
                    .iter()
                    .find(|(_, entry)| entry.info.model == current.0.as_ref())
                    .map(|(key, entry)| (key.as_str(), entry))
            })
        else {
            return false;
        };
        let configured_endpoint = cfg.config_models.get(key).is_some_and(|model| {
            model.base_url.is_some()
                || model.api_base_url.is_some()
                || model
                    .model_provider
                    .as_deref()
                    .and_then(|provider| cfg.model_providers.get(provider))
                    .is_some_and(|provider| {
                        provider.base_url.is_some() || provider.api_base_url.is_some()
                    })
        });
        configured_endpoint && entry.has_own_credentials()
    }

    /// Load the configured model's own `/models` catalog once at startup.
    pub(crate) async fn ensure_current_model_endpoint_catalog(&self) {
        if self.inner.catalog.read().model_endpoint_catalog_loaded {
            return;
        }
        self.refresh_current_model_endpoint().await;
    }

    /// Refresh the catalog from the current model's own OpenAI-compatible
    /// `/models` endpoint. The request is skipped unless the model has a
    /// model-owned credential, so a Grok session token cannot cross an
    /// arbitrary `base_url` boundary.
    pub(crate) async fn refresh_current_model_endpoint(&self) -> bool {
        self.refresh_current_model_endpoint_inner(
            crate::util::config::resolve_remote_fetch_enabled(),
            None,
        )
        .await
    }

    async fn refresh_current_model_endpoint_inner(
        &self,
        remote_fetch_enabled: bool,
        observed_etag: Option<String>,
    ) -> bool {
        if !remote_fetch_enabled {
            tracing::info!("model-specific catalog refresh skipped: remote_fetch disabled");
            return false;
        }
        // Capture both fences before the request: `model_endpoint_request`
        // awaits a provider refresh, during which either the current model or
        // its endpoint configuration can change. A stale result must not mark
        // the new model/configuration as loaded.
        let (catalog_owner, switch_generation) = {
            let current = self.inner.current_model_id.read();
            (current.clone(), self.model_switch_generation())
        };
        let endpoint_generation = self.inner.catalog.read().endpoint_generation;
        let Some(request) = self.model_endpoint_request().await else {
            return false;
        };
        let endpoint = self.inner.endpoint.clone();
        let (models, response_etag) = match tokio::time::timeout(
            crate::http::STARTUP_FETCH_TIMEOUT,
            endpoint.fetch_model_endpoint(request),
        )
        .await
        {
            Ok(Some((models, etag))) => (Some(models), etag),
            Ok(None) => (None, None),
            Err(_) => {
                tracing::warn!("model-specific catalog fetch timed out");
                (None, None)
            }
        };
        let new_etag = response_etag.or(observed_etag);
        let cfg = self.inner.cfg.read().clone();
        if !self.apply_refresh_result_fenced(
            &cfg,
            models,
            new_etag,
            None,
            Some(endpoint_generation),
            Some(switch_generation),
            CatalogSource::ModelEndpoint,
            Some(catalog_owner),
        ) {
            return false;
        }
        tracing::info!("model-specific catalog refreshed");
        self.notify_models_updated();
        true
    }

    async fn model_endpoint_request(&self) -> Option<ModelEndpointRequest> {
        if !self.current_model_has_endpoint() {
            return None;
        }
        let current = self.current_model_id();
        let entry = {
            let catalog = self.inner.catalog.read();
            catalog
                .models
                .get(current.0.as_ref())
                .or_else(|| {
                    catalog
                        .models
                        .values()
                        .find(|entry| entry.info.model == current.0.as_ref())
                })
                .cloned()
        }?;

        let provider = entry.effective_auth_provider().cloned();
        if let Some(provider) = &provider
            && !Self::bounded_auth_provider_refresh(provider.ensure_fresh_token(None)).await
        {
            return None;
        }

        let credentials = resolve_credentials(&entry, None);
        let api_key = credentials.api_key?;
        if !entry.has_own_credentials() {
            return None;
        }
        let configured_api_key = entry
            .api_key
            .as_deref()
            .filter(|key| !key.trim().is_empty())
            .map(str::to_owned);
        let configured_env_key = (configured_api_key.is_none())
            .then(|| entry.env_key.clone())
            .flatten();

        Some(ModelEndpointRequest {
            // `resolve_credentials` routes API-key auth to `api_base_url` when
            // a model separates session and API-key endpoints. Use that URL
            // here so the key is only sent to the operator that owns it.
            base_url: credentials.base_url,
            api_key,
            api_backend: entry.info.api_backend,
            auth_scheme: credentials.auth_scheme,
            configured_api_key,
            configured_env_key,
            auth_provider: provider,
            extra_headers: entry.info.extra_headers.clone(),
            query_params: entry.info.query_params.clone(),
            env_http_headers: entry.info.env_http_headers.clone(),
        })
    }

    /// Bounds a model auth-provider token refresh to
    /// `STARTUP_AUTH_REFRESH_TIMEOUT` so a wedged helper can't stall
    /// initialization for minutes (its own default timeout is 30s with a
    /// 600s ceiling). `false` when the provider is unusable, the mint failed,
    /// or the refresh exceeded the bound. Split out so the timeout contract is
    /// unit-testable without a live provider command.
    async fn bounded_auth_provider_refresh<F>(fut: F) -> bool
    where
        F: std::future::Future<Output = crate::auth::ProviderRefreshOutcome>,
    {
        match tokio::time::timeout(crate::http::STARTUP_AUTH_REFRESH_TIMEOUT, fut).await {
            Ok(outcome) => !matches!(
                outcome,
                crate::auth::ProviderRefreshOutcome::Unusable
                    | crate::auth::ProviderRefreshOutcome::MintFailed
            ),
            Err(_) => {
                tracing::warn!(
                    timeout_secs = crate::http::STARTUP_AUTH_REFRESH_TIMEOUT.as_secs(),
                    "model-specific catalog: auth provider refresh timed out"
                );
                false
            }
        }
    }

    async fn fetch_and_apply(&self) {
        self.fetch_and_apply_inner(crate::util::config::resolve_remote_fetch_enabled())
            .await
    }

    async fn fetch_and_apply_inner(&self, remote_fetch_enabled: bool) {
        if !remote_fetch_enabled {
            tracing::info!("model catalog refresh skipped: remote_fetch disabled");
            return;
        }
        let attempt = FetchAttemptGuard::begin(&self.inner);
        let generation = attempt.generation;
        let auth = Self::bounded_startup_auth(&self.inner.auth_manager).await;
        let has_auth = auth.is_some();
        let fetch_auth = *self.inner.fetch_auth.read();
        let cfg = self.inner.cfg.read().clone();
        xai_grok_telemetry::unified_log::info(
            "model catalog: fetching",
            None,
            Some(serde_json::json!({
                "has_auth": has_auth,
                "fetch_auth": format!("{fetch_auth:?}"),
            })),
        );
        let endpoint = self.inner.endpoint.clone();
        let new_prefetched = match tokio::time::timeout(
            crate::http::STARTUP_FETCH_TIMEOUT,
            endpoint.fetch_models(cfg.endpoints.clone(), auth, fetch_auth),
        )
        .await
        {
            Ok(res) => res,
            Err(_elapsed) => {
                tracing::warn!(
                    timeout_secs = crate::http::STARTUP_FETCH_TIMEOUT.as_secs(),
                    "model catalog fetch timed out"
                );
                None
            }
        };
        let success = self.apply_refresh_result_fenced(
            &cfg,
            new_prefetched,
            None,
            Some(generation),
            None,
            None,
            CatalogSource::Global,
            None,
        );
        if success {
            xai_grok_telemetry::unified_log::info(
                "model catalog: fetch succeeded",
                None,
                Some(serde_json::json!({
                    "model_count": self.available().len(),
                })),
            );
        }
    }

    /// Publish a resolved catalog under one atomic write, then reselect the model (default on first real catalog, else keep current if present).
    fn apply_catalog(
        &self,
        cfg: &config::Config,
        models: IndexMap<String, ModelEntry>,
        new_etag: Option<String>,
    ) {
        let _ = self.apply_catalog_fenced(
            cfg,
            models,
            new_etag,
            None,
            None,
            None,
            CatalogSource::Global,
            None,
        );
    }

    /// Discards a result captured before an identity change, or a global
    /// result that would replace an authoritative model endpoint catalog;
    /// returns whether the catalog applied.
    fn apply_catalog_fenced(
        &self,
        cfg: &config::Config,
        models: IndexMap<String, ModelEntry>,
        new_etag: Option<String>,
        generation: Option<u64>,
        endpoint_generation: Option<u64>,
        switch_generation: Option<u64>,
        source: CatalogSource,
        catalog_owner: Option<acp::ModelId>,
    ) -> bool {
        let (first_real_catalog, excludes_all, apply_cfg) = {
            let mut cat = self.inner.catalog.write();
            if source == CatalogSource::ModelEndpoint {
                if let Some(endpoint_generation) = endpoint_generation
                    && cat.endpoint_generation != endpoint_generation
                {
                    tracing::info!(
                        "model catalog result discarded: endpoint config changed during fetch"
                    );
                    return false;
                }
            } else if let Some(generation) = generation
                && cat.generation != generation
            {
                tracing::info!("model catalog result discarded: identity changed during fetch");
                return false;
            }
            if let Some(switch_generation) = switch_generation
                && self.model_switch_generation() != switch_generation
            {
                tracing::info!(
                    "model catalog result discarded: current model changed during fetch"
                );
                return false;
            }
            if source == CatalogSource::Global && cat.catalog_source == CatalogSource::ModelEndpoint
            {
                tracing::info!(
                    "global model catalog result discarded: model endpoint catalog is authoritative"
                );
                return false;
            }
            // A settings-only publication intentionally leaves the endpoint
            // fence unchanged so an in-flight fetch can still publish. Re-read
            // the current config at apply time so a stale snapshot cannot
            // overwrite the latest filters/defaults.
            let apply_cfg = if source == CatalogSource::ModelEndpoint {
                self.inner.cfg.read().clone()
            } else {
                cfg.clone()
            };
            let first_real_catalog = !cat.has_fetched_real_catalog;
            cat.has_fetched_real_catalog = true;
            cat.catalog_source = source;
            cat.catalog_owner = catalog_owner;
            cat.model_endpoint_catalog_loaded = source == CatalogSource::ModelEndpoint;
            cat.prefetched = Some(models);
            cat.models = resolve_model_catalog(&apply_cfg, cat.prefetched.clone());
            cat.etag = new_etag;
            cat.allowlist_excludes_all = allowlist_matches_nothing(&apply_cfg, &cat.models);
            // In the lock: the flag and its mirror can't desync vs `clear()`.
            self.inner
                .catalog_progress
                .send_replace(CatalogProgress::Ready);
            (first_real_catalog, cat.allowlist_excludes_all, apply_cfg)
        };
        if excludes_all {
            tracing::error!("allowed_models excludes all fetched models; prompts will be blocked");
        }

        // Respect an explicit pre-catalog `/model` pick: auto-select the
        // default on the first catalog only when the user hasn't chosen.
        // Either way a now-invalid selection is replaced.
        if first_real_catalog && !self.inner.user_selected_model.load(Ordering::Relaxed) {
            self.reselect_default_model(&apply_cfg);
        } else {
            self.reselect_current_model_if_missing(&apply_cfg);
        }
        true
    }

    /// A same-identity refresh, as the fetch paths see it.
    #[cfg(test)]
    fn apply_refresh_result(
        &self,
        config: &config::Config,
        new_prefetched: Option<IndexMap<String, ModelEntry>>,
        new_etag: Option<String>,
    ) -> bool {
        let generation = self.inner.catalog.read().generation;
        self.apply_refresh_result_fenced(
            config,
            new_prefetched,
            new_etag,
            Some(generation),
            None,
            None,
            CatalogSource::Global,
            None,
        )
    }

    fn apply_refresh_result_fenced(
        &self,
        config: &config::Config,
        new_prefetched: Option<IndexMap<String, ModelEntry>>,
        new_etag: Option<String>,
        generation: Option<u64>,
        endpoint_generation: Option<u64>,
        switch_generation: Option<u64>,
        source: CatalogSource,
        catalog_owner: Option<acp::ModelId>,
    ) -> bool {
        let Some(new_prefetched) = new_prefetched else {
            tracing::warn!("model refresh failed, leaving existing models unchanged");
            // Lock held across the send: atomic against a racing `clear()`.
            {
                let cat = self.inner.catalog.read();
                let same_identity = match generation {
                    Some(generation) => cat.generation == generation,
                    None => endpoint_generation.is_some_and(|g| cat.endpoint_generation == g),
                };
                if same_identity {
                    self.inner.catalog_progress.send_if_modified(|p| {
                        let first_failure = *p == CatalogProgress::Pending;
                        if first_failure {
                            *p = CatalogProgress::Failed;
                        }
                        first_failure
                    });
                }
            }
            xai_grok_telemetry::unified_log::warn(
                "model catalog refresh failed",
                None,
                Some(serde_json::json!({
                    "had_real_catalog": self.inner.catalog.read().has_fetched_real_catalog,
                })),
            );
            return false;
        };
        self.apply_catalog_fenced(
            config,
            new_prefetched,
            new_etag,
            generation,
            endpoint_generation,
            switch_generation,
            source,
            catalog_owner,
        )
    }

    pub fn allowlist_excludes_all(&self) -> bool {
        self.inner.catalog.read().allowlist_excludes_all
    }

    /// Re-pick the default when the current model is gone or unselectable;
    /// auth visibility never evicts an explicit user pick.
    fn reselect_current_model_if_missing(&self, config: &config::Config) {
        let current = self.inner.current_model_id.read().clone();
        let user_selected = self.inner.user_selected_model.load(Ordering::Relaxed);
        let needs_reselection = {
            let cat = self.inner.catalog.read();
            let models = &cat.models;
            match models.get(current.0.as_ref()) {
                None => true,
                Some(entry) => {
                    !entry.info.user_selectable
                        || (!user_selected && !entry.info.visible_for_auth(self.is_session_auth()))
                }
            }
        };
        if !needs_reselection {
            return;
        }
        let (key, _, source) = {
            let cat = self.inner.catalog.read();
            let models = &cat.models;
            resolve_default_model(config, models, self.is_session_auth())
        };
        let new_id = acp::ModelId::new(Arc::from(key));
        tracing::info!(
            old = %current.0, new = %new_id.0, source = %source,
            "current model not in new catalog, reselecting default"
        );
        self.set_current_model_id_internal(new_id);
    }

    /// Re-resolve the default model against the current catalog.
    fn reselect_default_model(&self, config: &config::Config) {
        let (key, _, source) = {
            let cat = self.inner.catalog.read();
            let models = &cat.models;
            resolve_default_model(config, models, self.is_session_auth())
        };
        let new_id = acp::ModelId::new(Arc::from(key));
        let current = self.inner.current_model_id.read().clone();
        if current.0.as_ref() != new_id.0.as_ref() {
            tracing::info!(
                old = %current.0, new = %new_id.0, source = %source,
                "re-resolved default model after catalog populated"
            );
            self.set_current_model_id_internal(new_id);
        }
    }
}

// ── Refresh strategy ────────────────────────────────────────────────────────

/// How to resolve the model list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshStrategy {
    /// Always fetch from network, ignore cache.
    Online,
    /// Only use cached data, never fetch.
    Offline,
    /// Use cache if fresh, otherwise fetch.
    OnlineIfUncached,
}

mod cache;
mod endpoint;
mod fetch;
mod resolution;

pub(crate) use cache::*;
pub(crate) use endpoint::*;
pub(crate) use fetch::*;
pub use fetch::{
    EarlyPrefetchHandle, EarlyPrefetchResult, start_early_prefetch,
    start_early_prefetch_settings_only, start_early_prefetch_with_auth,
};
pub(crate) use resolution::*;

#[cfg(test)]
mod tests;
