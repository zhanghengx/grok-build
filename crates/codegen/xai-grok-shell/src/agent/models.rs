//! Model fetching, resolution, and management.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

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

/// Origin of a response ETag, carried by the emitting session.
///
/// `model` is the routing slug the session sent and `base_url` is the request
/// origin the ETag came from. Both are session-local: a Leader session on the
/// bundled catalog must not be treated as endpoint-owned just because the
/// process default endpoint exposes the same slug.
///
/// `catalog_key` is the configured model key the session selected (the ACP
/// model id). Aliases can share a routing slug and URL while differing in
/// credentials, so slug+URL alone is not a unique endpoint identity.
///
/// `endpoint_owner` is the configured endpoint that produced this request,
/// captured at submit. A dynamically returned catalog id is not itself a
/// configured owner; without this field, a later Leader session can replace
/// the resident catalog and the ETag would no longer resolve to that owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EtagOrigin {
    model: acp::ModelId,
    base_url: String,
    catalog_key: Option<acp::ModelId>,
    endpoint_owner: Option<acp::ModelId>,
}

impl EtagOrigin {
    pub(crate) fn new(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            model: acp::ModelId::new(Arc::from(model.into())),
            base_url: base_url.into(),
            catalog_key: None,
            endpoint_owner: None,
        }
    }

    pub(crate) fn with_catalog_key(mut self, key: impl Into<String>) -> Self {
        let key = key.into();
        if !key.is_empty() {
            self.catalog_key = Some(acp::ModelId::new(Arc::from(key)));
        }
        self
    }

    pub(crate) fn with_endpoint_owner(mut self, owner: impl Into<String>) -> Self {
        let owner = owner.into();
        if !owner.is_empty() {
            self.endpoint_owner = Some(acp::ModelId::new(Arc::from(owner)));
        }
        self
    }

    pub(crate) fn model(&self) -> &str {
        self.model.0.as_ref()
    }

    pub(crate) fn base_url(&self) -> &str {
        &self.base_url
    }

    pub(crate) fn catalog_key(&self) -> Option<&str> {
        self.catalog_key.as_ref().map(|key| key.0.as_ref())
    }

    pub(crate) fn endpoint_owner(&self) -> Option<&str> {
        self.endpoint_owner.as_ref().map(|owner| owner.0.as_ref())
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
    /// Configured model whose endpoint populated a model-scoped catalog, or
    /// whose replacement endpoint is pending while an invalidation reloads it.
    /// This labels the still-resident prefetched models and must not change
    /// until a replacement `/models` result successfully publishes.
    catalog_owner: Option<acp::ModelId>,
    /// Endpoint a Leader ETag has targeted whose `/models` result has not
    /// yet successfully published. Distinct from `catalog_owner` so a failed
    /// or timed-out refresh cannot relabel still-resident prefetched models.
    pending_catalog_owner: Option<acp::ModelId>,
    /// `allowed_models` matched nothing; the prompt path blocks instead.
    allowlist_excludes_all: bool,
    /// Bumped on identity change; a fetch captured before it must not apply.
    generation: u64,
    /// Bumped when the effective endpoint owner changes (including a cold
    /// switch before the endpoint catalog first loads), when the current
    /// model's endpoint connection context changes, or when the identity is
    /// cleared. A model-endpoint fetch captured before it must not apply;
    /// settings-only publications leave it unchanged so an in-flight endpoint
    /// refresh can still publish.
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

/// Connection-context overlay changes across every retained prefetched key,
/// not only the process current model. Leader sessions can sample a returned
/// sibling without `set_current_model_id`.
fn any_retained_prefetched_overlay_changed_context(
    old_resolved: &IndexMap<String, ModelEntry>,
    new_resolved: &IndexMap<String, ModelEntry>,
    prefetched: &IndexMap<String, ModelEntry>,
) -> bool {
    prefetched
        .keys()
        .any(|key| match (old_resolved.get(key), new_resolved.get(key)) {
            (Some(old_entry), Some(new_entry)) => {
                endpoint_entry_context_differs(old_entry, new_entry)
            }
            _ => false,
        })
}

fn selected_entry_overlay_changed_context(
    old_resolved: &IndexMap<String, ModelEntry>,
    new_resolved: &IndexMap<String, ModelEntry>,
    prefetched: &IndexMap<String, ModelEntry>,
    current: &acp::ModelId,
) -> bool {
    if resolve_catalog_key(prefetched, current).is_none() {
        return false;
    }
    let old_entry =
        resolve_catalog_key(old_resolved, current).and_then(|key| old_resolved.get(key.0.as_ref()));
    let new_entry =
        resolve_catalog_key(new_resolved, current).and_then(|key| new_resolved.get(key.0.as_ref()));
    match (old_entry, new_entry) {
        (Some(old_entry), Some(new_entry)) => endpoint_entry_context_differs(old_entry, new_entry),
        _ => false,
    }
}

/// True when `model` / `catalog_key` was returned by the resident endpoint
/// (`cat.prefetched`) and the merged overlay still carries that endpoint's
/// connection context. Config-only `[model.*]` overlays live in `cat.models`
/// even when they were never in the `/models` response, so membership must
/// not be proven against the merged catalog.
fn returned_from_resident_endpoint(
    cat: &CatalogState,
    model: &str,
    catalog_key: &str,
    base_url: &str,
) -> bool {
    let Some(prefetched) = cat.prefetched.as_ref() else {
        return false;
    };
    let id_matches = |key: &str, entry: &config::ModelEntry| {
        key == model
            || entry.info.model == model
            || (!catalog_key.is_empty() && (key == catalog_key || entry.info.model == catalog_key))
    };
    let Some((key, _)) = prefetched
        .iter()
        .find(|(key, entry)| id_matches(key, entry))
    else {
        return false;
    };
    // Overlay context lives on the merged catalog entry; membership does not.
    cat.models.get(key).is_some_and(|entry| {
        entry.has_own_credentials() && resolve_credentials(entry, None).base_url == base_url
    })
}

/// Whether a `[model.<key>]` entry, directly or through its provider, defines
/// a model-owned endpoint URL.
fn config_model_has_endpoint(cfg: &config::Config, key: &str) -> bool {
    cfg.config_models.get(key).is_some_and(|model| {
        model.base_url.is_some()
            || model.api_base_url.is_some()
            || model
                .model_provider
                .as_deref()
                .and_then(|pid| cfg.model_providers.get(pid))
                .is_some_and(|provider| {
                    provider.base_url.is_some() || provider.api_base_url.is_some()
                })
    })
}

/// Whether a retained or pending endpoint owner is still configured with its
/// own endpoint and credential in the catalog being published.
fn pending_endpoint_owner_configured(
    cfg: &config::Config,
    catalog: &IndexMap<String, ModelEntry>,
    owner: &acp::ModelId,
) -> bool {
    let owner_key = owner.0.as_ref();
    let owner_entry = catalog
        .get(owner_key)
        .or_else(|| catalog.values().find(|entry| entry.info.model == owner_key));
    owner_entry.is_some_and(|entry| {
        entry.has_own_credentials() && config_model_has_endpoint(cfg, owner_key)
    })
}

/// Whether a pending endpoint owner should stay attached after the current
/// model is (re)selected. The owner stays while it is the current model, while
/// the selected model was returned by its endpoint, and while the current
/// model is only a temporary fallback with no selectable entry in the
/// config/global catalog. It is cleared once a selectable model from another
/// source is selected.
fn endpoint_owner_retained_for_selected_model(
    cat: &CatalogState,
    current: &acp::ModelId,
    clear_on_other_source: bool,
) -> bool {
    let Some(owner) = cat.catalog_owner.as_ref() else {
        return true;
    };
    if owner == current {
        return true;
    }
    let returned_by_endpoint = cat
        .prefetched
        .as_ref()
        .is_some_and(|models| resolve_catalog_key(models, current).is_some());
    if returned_by_endpoint {
        let overlay_changes_context = match (
            cat.prefetched
                .as_ref()
                .and_then(|models| resolve_catalog_key(models, current))
                .and_then(|key| {
                    cat.prefetched
                        .as_ref()
                        .and_then(|models| models.get(key.0.as_ref()))
                }),
            resolve_catalog_key(&cat.models, current)
                .and_then(|key| cat.models.get(key.0.as_ref())),
        ) {
            (Some(raw), Some(resolved)) => endpoint_entry_context_differs(raw, resolved),
            _ => true,
        };
        if !overlay_changes_context {
            return true;
        }
    }
    if !clear_on_other_source {
        // Automatic reselection (for example, a config-only rebuild after an
        // endpoint invalidation moves a returned slug to the bundled default)
        // is not a choice of another source. Keep the pending owner so the
        // replacement endpoint refresh still targets it.
        return true;
    }
    let selected_from_other_source = resolve_catalog_key(&cat.models, current)
        .and_then(|key| cat.models.get(key.0.as_ref()))
        .is_some_and(|entry| entry.info.user_selectable);
    !selected_from_other_source
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
    /// Serializes model-endpoint catalog refreshes so an older `/models`
    /// response cannot overwrite a newer one.
    endpoint_refresh: tokio::sync::Mutex<()>,
    /// Monotonic sequence assigned to each endpoint ETag notification before
    /// its refresh task is spawned, so out-of-order task execution cannot
    /// regress the stored endpoint ETag.
    next_endpoint_etag_seq: AtomicU64,
    /// Highest endpoint ETag notification sequence applied so far.
    applied_endpoint_etag_seq: AtomicU64,
    /// Serializes the global fetch start decision so `list_models` and the
    /// background retry task cannot both reserve the same catalog generation.
    global_fetch_start: tokio::sync::Mutex<()>,
    fetches_in_flight: AtomicUsize,
    /// A fetch attempt currently executing (network/auth work), as opposed to
    /// a retry task sleeping through its backoff between attempts.
    active_fetch: AtomicUsize,
    /// Generations of the fetch attempts currently executing network/auth
    /// work. `list_models` joins an in-flight fetch only when its generation
    /// still matches the catalog, so a config reload cannot park callers
    /// behind a stale request that the fence will discard.
    active_fetch_generations: RwLock<Vec<u64>>,
    /// Bumped every time an active fetch finishes, so callers that joined an
    /// in-flight generation can wait for its outcome even when the catalog was
    /// already `Ready` (later refreshes do not change `catalog_progress`).
    active_fetch_done: tokio::sync::watch::Sender<u64>,
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

/// One fetch attempt (or the active portion of a retry iteration) currently
/// executing. `list_models` joins on this instead of `fetches_in_flight`, so a
/// retry task sleeping through backoff does not block a fresh catalog request.
struct ActiveFetchGuard {
    inner: Arc<Inner>,
    generation: u64,
}
impl ActiveFetchGuard {
    fn begin(inner: &Arc<Inner>, generation: u64) -> Self {
        inner.active_fetch.fetch_add(1, Ordering::AcqRel);
        inner.active_fetch_generations.write().push(generation);
        Self {
            inner: inner.clone(),
            generation,
        }
    }
}
impl Drop for ActiveFetchGuard {
    fn drop(&mut self) {
        self.inner.active_fetch.fetch_sub(1, Ordering::AcqRel);
        let mut generations = self.inner.active_fetch_generations.write();
        if let Some(index) = generations
            .iter()
            .rposition(|generation| *generation == self.generation)
        {
            generations.remove(index);
        }
        drop(generations);
        let next_version = *self.inner.active_fetch_done.borrow() + 1;
        self.inner.active_fetch_done.send_replace(next_version);
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
        let generation = inner.catalog.read().generation;
        Self::begin_with_generation(inner, generation)
    }

    fn begin_with_generation(inner: &Arc<Inner>, generation: u64) -> Self {
        // Count first: a waiter that sees `Pending` must also see the attempt.
        inner.fetches_in_flight.fetch_add(1, Ordering::AcqRel);
        inner.catalog_progress.send_if_modified(|p| {
            let supersede = *p == CatalogProgress::Failed;
            if supersede {
                *p = CatalogProgress::Pending;
            }
            supersede
        });
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
                endpoint_refresh: tokio::sync::Mutex::new(()),
                next_endpoint_etag_seq: AtomicU64::new(0),
                applied_endpoint_etag_seq: AtomicU64::new(0),
                global_fetch_start: tokio::sync::Mutex::new(()),
                fetches_in_flight: AtomicUsize::new(0),
                active_fetch: AtomicUsize::new(0),
                active_fetch_generations: RwLock::new(Vec::new()),
                active_fetch_done: tokio::sync::watch::channel(0u64).0,
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
        let new_preferred = new_config.models.default.clone();
        let has_session = self.inner.auth_manager.current_or_expired().is_some();
        let new_fetch_auth = ModelFetchAuth::resolve(&new_config.endpoints, has_session);
        // Take the lock before the catalog snapshot so a rejected reload cannot
        // publish the proposed auth mode, while keeping the same serialization
        // point used by the concurrent-refresh test.
        let mut fetch_auth = self.inner.fetch_auth.write();
        // Keep one lock order everywhere: catalog, then cfg. The apply path
        // (`apply_catalog_fenced`) reads cfg while holding catalog, so taking
        // cfg first here would invert the order and can deadlock a reload
        // against an in-flight endpoint refresh. Re-read the live catalog
        // state under the write lock: an endpoint refresh can commit between
        // an unlocked snapshot and this publication, and applying the stale
        // retained entries while keeping the fresh etag would latch the old
        // models as loaded.
        let mut cat = self.inner.catalog.write();
        // Snapshot the current model under the catalog lock: a model switch can
        // otherwise land between this read and the endpoint-context check and
        // let a B refresh pass an unchanged-A fence.
        let current_model_key = self.inner.current_model_id.read().0.as_ref().to_string();
        let (old_config, old_preferred, old_default_is_campaign) = {
            let cfg = self.inner.cfg.read();
            (
                cfg.clone(),
                cfg.models.default.clone(),
                cfg.models.default_is_campaign_driven,
            )
        };
        let (prefetched, has_real_catalog, catalog_source, catalog_owner) = (
            cat.prefetched.clone(),
            cat.has_fetched_real_catalog,
            cat.catalog_source,
            cat.catalog_owner.clone(),
        );
        // Model-endpoint entries carry inherited routing and credentials. Keep
        // them across a config publication while the endpoint connection
        // context is unchanged; otherwise drop them and let the caller refetch
        // so dynamically discovered models never disappear silently.
        let endpoint_catalog_active = catalog_source == CatalogSource::ModelEndpoint;
        // A returned slug that gains its own endpoint or credentials through a
        // config overlay no longer belongs to the endpoint catalog: sampling
        // routes through the overlay, so retaining the old owner would keep
        // refreshing the previous endpoint. Compare every retained prefetched
        // entry — Leader sessions can sample a sibling without updating the
        // process current model.
        let (selected_overlay_changed_context, overlay_changed_context) = if endpoint_catalog_active
        {
            prefetched.as_ref().map_or((false, false), |prefetched| {
                let current = acp::ModelId::new(Arc::from(current_model_key.clone()));
                let old_resolved = resolve_model_catalog(&old_config, Some(prefetched.clone()));
                let new_resolved = resolve_model_catalog(&new_config, Some(prefetched.clone()));
                (
                    selected_entry_overlay_changed_context(
                        &old_resolved,
                        &new_resolved,
                        prefetched,
                        &current,
                    ),
                    any_retained_prefetched_overlay_changed_context(
                        &old_resolved,
                        &new_resolved,
                        prefetched,
                    ),
                )
            })
        } else {
            (false, false)
        };
        let endpoint_catalog_invalidated = endpoint_catalog_active
            && (overlay_changed_context
                || !catalog_owner.as_ref().is_some_and(|owner| {
                    !model_endpoint_changed(&old_config, &new_config, owner.0.as_ref())
                }));
        // A settings-only publication must not invalidate an in-flight
        // model-endpoint fetch, but a change to the endpoint connection
        // context must. Before the endpoint catalog loads, the current model
        // id is the fetch's owner.
        let endpoint_context_changed = overlay_changed_context || {
            let owner_key = catalog_owner
                .as_ref()
                .map(|owner| owner.0.as_ref().to_string())
                .unwrap_or_else(|| current_model_key.clone());
            model_endpoint_changed(&old_config, &new_config, &owner_key)
        };
        let retained_prefetched = if endpoint_catalog_invalidated {
            None
        } else {
            prefetched
        };
        let new_catalog = resolve_model_catalog(&new_config, retained_prefetched.clone());
        // Filters can remove the endpoint owner from a retained catalog (for
        // example `disabled_models` matching the owner key). The owner can no
        // longer resolve its own credentials or refresh the endpoint, and
        // keeping the endpoint source would fence global results forever, so
        // invalidate the catalog and rebuild from config only.
        let owner_removed_by_filters = endpoint_catalog_active
            && !endpoint_catalog_invalidated
            && catalog_owner
                .as_ref()
                .is_some_and(|owner| resolve_catalog_key(&new_catalog, owner).is_none());
        let endpoint_catalog_invalidated = endpoint_catalog_invalidated || owner_removed_by_filters;
        let endpoint_catalog_still_valid = endpoint_catalog_active && !endpoint_catalog_invalidated;
        let retained_prefetched = if endpoint_catalog_invalidated {
            None
        } else {
            retained_prefetched
        };
        let retained_real_catalog = has_real_catalog && !endpoint_catalog_invalidated;
        let new_catalog = if owner_removed_by_filters {
            resolve_model_catalog(&new_config, None)
        } else {
            new_catalog
        };
        let endpoint_context_changed = endpoint_context_changed || owner_removed_by_filters;
        // Keep the owner as a pending refresh target when the endpoint
        // context changes (or a replacement is already pending) and the new
        // config still configures that endpoint. A stale owner whose endpoint
        // was removed must be cleared, otherwise every global result stays
        // fenced and the catalog can never recover. When the selected returned
        // slug gains its own endpoint, point the pending refresh at that model
        // instead of the previous owner.
        let pending_endpoint_owner = if endpoint_catalog_active && !endpoint_catalog_invalidated {
            catalog_owner
        } else if selected_overlay_changed_context {
            let current = acp::ModelId::new(Arc::from(current_model_key.clone()));
            if pending_endpoint_owner_configured(&new_config, &new_catalog, &current) {
                Some(current)
            } else {
                catalog_owner
                    .as_ref()
                    .filter(|owner| {
                        pending_endpoint_owner_configured(&new_config, &new_catalog, owner)
                    })
                    .cloned()
            }
        } else {
            catalog_owner
                .as_ref()
                .filter(|owner| pending_endpoint_owner_configured(&new_config, &new_catalog, owner))
                .cloned()
        };
        if retained_real_catalog && let Err(e) = validate_selectable(&new_config, &new_catalog) {
            tracing::error!(error = %e, "ignoring config reload: allowed_models excludes all models");
            return Err(e);
        }

        let mut cfg = self.inner.cfg.write();
        *cfg = new_config.clone();
        let old_fetch_auth = *fetch_auth;
        *fetch_auth = new_fetch_auth;
        // A config reload can change the endpoint or auth used by a global
        // catalog fetch already in flight. Only then must the catalog fence
        // advance: a settings-only publication should not discard an in-flight
        // refresh, or the retained real catalog keeps `OnlineIfUncached` from
        // recovering and the picker stays stale until the next ETag. Publish
        // the new config before advancing the fence so a refresh that observes
        // this generation can only build a new request.
        let global_fetch_shape_changed =
            old_config.endpoints != new_config.endpoints || new_fetch_auth != old_fetch_auth;
        if global_fetch_shape_changed {
            cat.generation += 1;
        }
        if endpoint_context_changed {
            cat.endpoint_generation += 1;
            // In-flight Leader retargets are bound to the previous fence.
            cat.pending_catalog_owner = None;
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
        // Keep the endpoint owner as a pending refresh target while its
        // replacement catalog loads, even after the current model is reselected
        // away from a provider-returned slug.
        cat.catalog_owner = pending_endpoint_owner;
        drop(cat);
        drop(cfg);
        drop(fetch_auth);

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
        let selection_moved = preferred_changed && !(campaign_only_flip && current_still_ok);
        if selection_moved {
            self.reselect_default_model(&new_config, true);
        } else {
            self.reselect_current_model_if_missing(&new_config, false);
        }

        self.revalidate_pending_owner_for_selected_model(selection_moved);
        self.notify_models_updated();
        Ok(())
    }

    /// [`Self::apply_config`] plus an unconditional default re-resolve, for remote-settings arrival while no session exists.
    pub(crate) fn apply_config_reselecting_default(
        &self,
        new_config: config::Config,
    ) -> Result<(), String> {
        self.apply_config(new_config.clone())?;
        self.reselect_default_model(&new_config, true);
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
        self.set_current_model_id_internal(id, true);
    }

    fn set_current_model_id_internal(&self, id: acp::ModelId, clear_pending_owner: bool) {
        let (changed, previous_model_id) = {
            let mut cur = self.inner.current_model_id.write();
            let previous_model_id = cur.clone();
            let changed = *cur != id;
            if changed {
                *cur = id.clone();
                // Publish the id and its generation while holding the same
                // lock used by endpoint-refresh snapshots.
                self.inner
                    .model_switch_watch
                    .send_modify(|generation| *generation += 1);
            }
            (changed, previous_model_id)
        };
        if changed {
            let mut cat = self.inner.catalog.write();
            // Snapshot config only after taking the catalog lock: `apply_config`
            // uses catalog-then-config order, and a switch that waited behind
            // the publication must rebuild against the config it committed.
            // The effective endpoint owner can change before the endpoint
            // catalog first loads. Capture the previous owner under the same
            // lock so a cold switch still advances the endpoint fence and
            // rejects ETag/refresh work captured for the old origin.
            let previous_endpoint_owner = cat.catalog_owner.clone().unwrap_or(previous_model_id);
            let cfg = self.inner.cfg.read().clone();
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
                    let endpoint_generation = cat.endpoint_generation + 1;
                    let models = resolve_model_catalog(&cfg, None);
                    let allowlist_excludes_all = allowlist_matches_nothing(&cfg, &models);
                    *cat = CatalogState {
                        models,
                        allowlist_excludes_all,
                        generation,
                        endpoint_generation,
                        ..Default::default()
                    };
                    self.inner
                        .catalog_progress
                        .send_replace(CatalogProgress::Pending);
                }
            }
            // A pending owner survives an endpoint invalidation. Once the
            // selected model belongs to another source, drop it so refreshes
            // target that source instead of the stale endpoint.
            if !endpoint_owner_retained_for_selected_model(&cat, &id, clear_pending_owner) {
                tracing::info!(model = %id.0, "clearing pending endpoint owner after model switch");
                cat.catalog_owner = None;
            }
            let effective_endpoint_owner = cat.catalog_owner.clone().unwrap_or_else(|| id.clone());
            if previous_endpoint_owner != effective_endpoint_owner {
                tracing::info!(
                    model = %id.0,
                    "advancing endpoint fence after endpoint owner change"
                );
                cat.endpoint_generation += 1;
            }
        } else if clear_pending_owner {
            // Explicitly reselecting the already-current model is still a
            // selection from another source (for example, after an automatic
            // fallback retained the pending endpoint owner). Revalidate it
            // even though the id did not change.
            self.revalidate_pending_owner_for_selected_model(true);
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

    /// Test-only: publish an endpoint-owned catalog as if `/models` just
    /// returned `prefetched` for `owner`. Session tests use this to replace
    /// the resident catalog without touching private `CatalogState` fields.
    #[cfg(test)]
    pub(crate) fn install_test_endpoint_catalog(
        &self,
        owner: &str,
        prefetched: IndexMap<String, ModelEntry>,
        etag: &str,
    ) {
        let owner_id = acp::ModelId::new(Arc::from(owner));
        let endpoint_generation = {
            let mut cat = self.inner.catalog.write();
            if cat.catalog_owner.as_ref() != Some(&owner_id) {
                if cat.catalog_owner.is_some() {
                    cat.endpoint_generation = cat.endpoint_generation.saturating_add(1);
                    cat.etag = None;
                }
                cat.catalog_owner = Some(owner_id.clone());
            }
            cat.endpoint_generation
        };
        let applied = self.apply_refresh_result_fenced(
            None,
            Some(prefetched),
            Some(etag.to_string()),
            None,
            Some(endpoint_generation),
            None,
            CatalogSource::ModelEndpoint,
            Some(owner_id),
        );
        debug_assert!(applied, "test endpoint catalog must apply");
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
        self.reselect_current_model_if_missing(cfg, false);
    }

    /// Refresh models when the etag changes.
    ///
    /// `emitting` identifies the session whose response carried the ETag.
    /// Session model switches (Leader mode) do not update `current_model_id`,
    /// so an ETag from a global-model session must not be routed to the
    /// tracked endpoint catalog just because the process default owns one.
    pub(crate) async fn refresh_if_new_etag(&self, etag: String, emitting: Option<EtagOrigin>) {
        let current_has_endpoint = self.current_model_has_endpoint();
        let (
            same_etag,
            endpoint_owned,
            endpoint_owner,
            catalog_is_endpoint_owned,
            resident_owner,
            observed_endpoint_generation,
        ) = {
            let cat = self.inner.catalog.read();
            let catalog_is_endpoint_owned = cat.catalog_source == CatalogSource::ModelEndpoint;
            let endpoint_owner = emitting.as_ref().and_then(|origin| {
                // The session carries the request origin (routing slug plus
                // base URL). Only a session that actually hit the configured
                // endpoint resource may route into its catalog; a process
                // default endpoint with a colliding slug does not make a
                // global session's ETag endpoint-scoped.
                self.session_origin_endpoint_owner(&cat, origin)
            });
            let endpoint_owned = endpoint_owner.is_some()
                || (emitting.is_none()
                    && (catalog_is_endpoint_owned
                        || (!cat.model_endpoint_catalog_loaded
                            && (cat.catalog_owner.is_some()
                                || cat.pending_catalog_owner.is_some()
                                || current_has_endpoint))));
            (
                cat.etag.as_deref() == Some(etag.as_str()),
                endpoint_owned,
                endpoint_owner,
                catalog_is_endpoint_owned,
                cat.catalog_owner.clone(),
                cat.endpoint_generation,
            )
        };
        if endpoint_owned {
            // ETags are scoped to their resource/origin. A matching opaque
            // value from a different emitting owner must not skip that
            // owner's `/models` refresh just because some other endpoint
            // catalog is already resident.
            let same_resident_owner = endpoint_owner
                .as_ref()
                .is_none_or(|owner| resident_owner.as_ref() == Some(owner));
            if catalog_is_endpoint_owned && same_etag && same_resident_owner {
                return;
            }
            tracing::info!(etag = %etag, "models etag changed, refreshing endpoint catalog");
            // Leader session switches do not update the process current
            // model. Fence this emitting owner as pending so its result can
            // publish and any in-flight previous owner is rejected. Do not
            // relabel `catalog_owner` until that result successfully
            // publishes: a failed/timed-out B refresh must leave A's
            // still-resident prefetched models owned by A.
            let observed_endpoint_generation = if let Some(owner) = endpoint_owner.as_ref() {
                let mut cat = self.inner.catalog.write();
                let already_resident = cat.catalog_owner.as_ref() == Some(owner);
                if already_resident {
                    cat.pending_catalog_owner = None;
                } else if cat.pending_catalog_owner.as_ref() != Some(owner) {
                    if cat.catalog_owner.is_some() || cat.pending_catalog_owner.is_some() {
                        cat.endpoint_generation = cat.endpoint_generation.saturating_add(1);
                        // The resident ETag belongs to the previous origin.
                        // Keep it from suppressing the newly targeted owner.
                        cat.etag = None;
                    }
                    cat.pending_catalog_owner = Some(owner.clone());
                }
                cat.endpoint_generation
            } else {
                observed_endpoint_generation
            };
            // Assign the notification sequence before spawning: spawned tasks
            // can acquire `endpoint_refresh` out of notification order, and
            // this value lets the newer task's committed ETag win.
            let seq = self
                .inner
                .next_endpoint_etag_seq
                .fetch_add(1, Ordering::AcqRel)
                + 1;
            let mgr = self.clone();
            tokio::task::spawn(async move {
                mgr.refresh_current_model_endpoint_inner_with_origin(
                    crate::util::config::resolve_remote_fetch_enabled(),
                    Some(etag),
                    Some(seq),
                    Some(observed_endpoint_generation),
                    endpoint_owner,
                )
                .await;
            });
            return;
        }
        // The stored ETag only suppresses a global refresh when the resident
        // catalog is itself the global resource. An endpoint catalog's ETag is
        // scoped to the endpoint, so equality must not renew the global cache.
        if same_etag && !catalog_is_endpoint_owned {
            let fetch_auth = *self.inner.fetch_auth.read();
            self.inner
                .cache
                .renew_ttl(&fetch_auth.cache_auth_method(), &self.cache_origin())
                .await;
            return;
        }
        tracing::info!(etag = %etag, "models etag changed, refreshing");
        // Keep the serial sampling-event drainer off the refresh path: an
        // ETag change can arrive while a startup/global fetch owns the
        // generation, and joining it can block for the full auth+fetch bounds.
        // Run the join/replay in a background task, matching the endpoint path.
        let mgr = self.clone();
        tokio::task::spawn(async move {
            mgr.spawn_fetch(Some(etag)).await;
        });
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
        let mut endpoint_configured = self.current_model_has_endpoint();
        // No session but the endpoint needs one: a fetch would 401, so skip it
        // and reset to this identity's bundled catalog. A model-owned catalog
        // with its own API key is independent of the Grok identity and must
        // not be dropped by the session-only fallback.
        if !has_session && fetch_auth == ModelFetchAuth::Session {
            // Preserve a resident catalog only when it is actually
            // endpoint-owned. A prior identity's global catalog must be
            // cleared before the BYOK refresh so a failed refresh cannot
            // leave it behind as the "real" catalog.
            let catalog_is_endpoint_owned =
                self.inner.catalog.read().catalog_source == CatalogSource::ModelEndpoint;
            if !endpoint_configured || !catalog_is_endpoint_owned {
                self.clear();
                self.rebuild_bundled(&config);
                endpoint_configured = self.current_model_has_endpoint();
            }
            if !endpoint_configured {
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
        }

        let remote_fetch_enabled = crate::util::config::resolve_remote_fetch_enabled();
        if endpoint_configured {
            self.refresh_current_model_endpoint_inner(remote_fetch_enabled, None, None)
                .await;
        } else {
            self.fetch_and_apply_inner(remote_fetch_enabled).await;
        }

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

        let count = cached.models.len();
        self.apply_catalog(cached.models, cached.etag);
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

    async fn spawn_fetch(&self, new_etag: Option<String>) {
        self.spawn_fetch_inner(
            new_etag,
            crate::util::config::resolve_remote_fetch_enabled(),
        )
        .await;
    }

    async fn spawn_fetch_inner(&self, new_etag: Option<String>, remote_fetch_enabled: bool) {
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
        // Reserve the current generation under the same global single-flight
        // used by `fetch_and_apply_inner`. If `models/list` or a background
        // fetch already owns it, join that attempt instead of racing a second
        // same-generation request whose older response could land last.
        let (started, joined_generation) = self.reserve_global_fetch().await;
        let Some((attempt, active)) = started else {
            self.inner.refresh_in_flight.store(false, Ordering::Release);
            // Wait for the fetch we joined to finish, not just for the first
            // catalog to become ready. With a real catalog already loaded,
            // `catalog_progress` stays `Ready` during later refreshes, so the
            // old wait returned immediately and replaying the etag here would
            // recurse while the same active fetch was still registered.
            self.wait_for_active_generation(joined_generation).await;
            // The joined fetch started before this etag change and applies no
            // etag, so it can publish the old catalog. Re-issue the refresh
            // when the catalog still does not carry the observed etag instead
            // of letting the change signal disappear. Never replay the etag
            // across an identity/config generation change: `on_auth_changed`
            // can advance the catalog generation while we wait, and the new
            // catalog's etag is scoped to a different identity/resource.
            if self.inner.catalog.read().generation == joined_generation
                && new_etag
                    .as_deref()
                    .is_some_and(|etag| self.inner.catalog.read().etag.as_deref() != Some(etag))
            {
                Box::pin(self.spawn_fetch_inner(new_etag, remote_fetch_enabled)).await;
            }
            return;
        };
        let generation = attempt.generation;
        let cfg = self.inner.cfg.read().clone();
        let endpoints = cfg.endpoints.clone();
        let fetch_auth = *self.inner.fetch_auth.read();
        let auth_manager = self.inner.auth_manager.clone();
        let endpoint = self.inner.endpoint.clone();
        let mgr = self.clone();

        tokio::task::spawn(async move {
            let _attempt = attempt;
            let _active = active;
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
                None,
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
        // `x.ai/models/list` is a one-shot consumer, so the source decision
        // made at entry must not go stale. A model switch can land while a
        // catalog fetch is awaiting; recheck the source after each operation
        // and run the path for the current model instead of serving a catalog
        // from the branch that was valid when the request started.
        for _ in 0..3 {
            if self.current_model_has_endpoint() {
                // A model-scoped endpoint is authoritative for this
                // configuration. Do not populate the picker from the global
                // proxy and then leave the configured endpoint's model list
                // unused.
                let owner_before = self.current_endpoint_owner();
                match strategy {
                    RefreshStrategy::Offline => {
                        self.wait_for_first_catalog().await;
                    }
                    RefreshStrategy::OnlineIfUncached => {
                        self.refresh_current_model_endpoint_if_uncached().await;
                    }
                    RefreshStrategy::Online => {
                        self.refresh_current_model_endpoint().await;
                    }
                }
                if !self.current_model_has_endpoint() {
                    // The model changed away from the endpoint source while
                    // the refresh was awaiting; run the global path for it.
                    continue;
                }
                if self.current_model_endpoint_catalog_loaded() {
                    return;
                }
                // The refresh did not leave a loaded catalog for the current
                // model. This can happen when the current model switched from
                // one endpoint owner to another while the first `/models`
                // request was pending; run the loop again so the new owner's
                // catalog is fetched instead of serving the stale/config-only
                // catalog. A same-owner failure is left alone: retrying the
                // same endpoint here would turn one failed `models/list` into
                // three network requests.
                if self.current_endpoint_owner() != owner_before {
                    continue;
                }
                return;
            } else {
                match strategy {
                    RefreshStrategy::Offline => {
                        self.wait_for_first_catalog().await;
                        return;
                    }
                    RefreshStrategy::OnlineIfUncached => {
                        if !self.inner.catalog.read().has_fetched_real_catalog {
                            let remote_fetch_enabled =
                                crate::util::config::resolve_remote_fetch_enabled();
                            if !remote_fetch_enabled {
                                return;
                            }
                            // The generation lookup, active-fetch check, and
                            // fetch start must be one synchronized decision: a
                            // config reload can otherwise advance the
                            // generation between the two reads and park this
                            // request behind a stale fetch that can never
                            // publish for the current config.
                            let (started, joined_generation) = self.reserve_global_fetch().await;
                            if let Some(started) = started {
                                self.fetch_and_apply_reserved(started).await;
                                // A config reload can advance the catalog
                                // generation while the reserved request is in
                                // flight, so its result is discarded by the
                                // fence. Re-run the fetch for the current
                                // generation instead of returning without a
                                // real catalog.
                                if !self.current_model_has_endpoint()
                                    && !self.inner.catalog.read().has_fetched_real_catalog
                                {
                                    self.fetch_and_apply().await;
                                }
                            } else {
                                // Wait for the joined active generation to
                                // finish instead of `catalog_progress`: a
                                // config reload can advance the generation
                                // while the joined fetch is in flight, and the
                                // stale attempt deliberately does not publish
                                // `Failed`, so the progress wait would sit out
                                // the full startup budget before retrying.
                                self.wait_for_active_generation(joined_generation).await;
                                // The joined fetch may have been captured under
                                // an older generation (a reload advanced it
                                // while we waited). If it could not publish,
                                // start a fresh fetch for the current config
                                // instead of leaving the request without a
                                // catalog.
                                if !self.inner.catalog.read().has_fetched_real_catalog {
                                    self.fetch_and_apply().await;
                                }
                            }
                        }
                    }
                    RefreshStrategy::Online => {
                        self.fetch_and_apply().await;
                    }
                }
                if self.current_model_has_endpoint() {
                    // The model changed to an endpoint source while the global
                    // path was awaiting; run the endpoint path for it.
                    continue;
                }
                return;
            }
        }
    }

    fn current_model_has_endpoint(&self) -> bool {
        self.model_has_endpoint(&self.current_model_id())
    }

    /// Whether the currently selected model's configured endpoint catalog is
    /// the loaded one. The catalog belongs to the current model only when it is
    /// the owner or was returned by that owner's `/models` response; a switch
    /// to a different endpoint owner must not be satisfied by the previous
    /// owner's catalog.
    fn current_model_endpoint_catalog_loaded(&self) -> bool {
        let current = self.current_model_id();
        let cat = self.inner.catalog.read();
        if cat.catalog_source != CatalogSource::ModelEndpoint || !cat.model_endpoint_catalog_loaded
        {
            return false;
        }
        cat.catalog_owner.as_ref().is_some_and(|owner| {
            owner == &current
                || cat
                    .prefetched
                    .as_ref()
                    .is_some_and(|models| resolve_catalog_key(models, &current).is_some())
        })
    }

    /// The model whose endpoint owns the catalog (or would own a pending
    /// refresh), falling back to the current model id when no owner is tracked.
    fn current_endpoint_owner(&self) -> acp::ModelId {
        let current = self.current_model_id();
        self.inner
            .catalog
            .read()
            .catalog_owner
            .clone()
            .unwrap_or(current)
    }

    /// Configured endpoint owner that should be stamped onto a request
    /// origin at submit. Uses the session catalog key when that key is a
    /// configured endpoint; otherwise the currently resident/pending owner
    /// if the request is sampling a dynamically returned model from it.
    /// Never searches the live catalog at ETag time for this value.
    pub(crate) fn configured_endpoint_owner_for_origin(
        &self,
        model: &str,
        base_url: &str,
        catalog_key: &str,
    ) -> Option<String> {
        if base_url.is_empty() {
            return None;
        }
        let cat = self.inner.catalog.read();
        let cfg = self.inner.cfg.read();
        let configured = config::resolve_model_list(&cfg, None);
        if !catalog_key.is_empty()
            && let Some(entry) = configured.get(catalog_key)
            && entry.has_own_credentials()
            && config_model_has_endpoint(&cfg, catalog_key)
            && resolve_credentials(entry, None).base_url == base_url
        {
            return Some(catalog_key.to_string());
        }
        // Dynamic / returned slug: persist the resident configured owner
        // that produced this catalog, not the returned id.
        if cat.catalog_source == CatalogSource::ModelEndpoint
            && let Some(owner) = cat.catalog_owner.as_ref()
        {
            let owner_key = owner.0.as_ref();
            let owner_matches_url = configured.get(owner_key).is_some_and(|entry| {
                entry.has_own_credentials()
                    && config_model_has_endpoint(&cfg, owner_key)
                    && resolve_credentials(entry, None).base_url == base_url
            });
            if owner_matches_url
                && returned_from_resident_endpoint(&cat, model, catalog_key, base_url)
            {
                return Some(owner_key.to_string());
            }
        }
        None
    }

    /// Resolve the configured endpoint owner for a response ETag's
    /// session-local origin. Reads catalog before cfg to preserve the lock
    /// order used by the catalog apply path. It never reacquires the catalog
    /// lock, so a queued writer cannot deadlock behind this read guard.
    fn session_origin_endpoint_owner(
        &self,
        cat: &CatalogState,
        origin: &EtagOrigin,
    ) -> Option<acp::ModelId> {
        if origin.base_url.is_empty() {
            return None;
        }
        let cfg = self.inner.cfg.read();
        let configured = config::resolve_model_list(&cfg, None);
        // Prefer the owner captured at submit: a dynamically returned
        // catalog id is not a configured key, and the resident catalog
        // may have been replaced by another Leader session since then.
        if let Some(owner) = origin.endpoint_owner.as_ref() {
            let owner_key = owner.0.as_ref();
            if let Some(entry) = configured.get(owner_key)
                && entry.has_own_credentials()
                && config_model_has_endpoint(&cfg, owner_key)
                && resolve_credentials(entry, None).base_url == origin.base_url
            {
                return Some(owner.clone());
            }
        }
        // Prefer the session catalog key: two aliases can share a routing
        // slug and base URL while using different credentials/headers.
        if let Some(key) = origin.catalog_key.as_ref() {
            let key_str = key.0.as_ref();
            if let Some(entry) = configured.get(key_str)
                && entry.has_own_credentials()
                && config_model_has_endpoint(&cfg, key_str)
                && resolve_credentials(entry, None).base_url == origin.base_url
            {
                return Some(key.clone());
            }
        }
        let slug_matches: Vec<acp::ModelId> = configured
            .iter()
            .filter(|(key, entry)| {
                entry.info.model == origin.model.0.as_ref()
                    && entry.has_own_credentials()
                    && config_model_has_endpoint(&cfg, key)
                    && resolve_credentials(entry, None).base_url == origin.base_url
            })
            .map(|(key, _)| acp::ModelId::new(Arc::from(key.clone())))
            .collect();
        match slug_matches.len() {
            1 => return slug_matches.into_iter().next(),
            n if n > 1 => {
                // Duplicate routing slugs are supported; insertion-order
                // first-match would refresh the wrong credential context.
                return None;
            }
            _ => {}
        }
        if cat.catalog_source != CatalogSource::ModelEndpoint {
            return None;
        }
        // A dynamic model returned by a configured endpoint stays owned by
        // that catalog owner. Using the returned key as the owner would
        // fail the Leader fence and then stop later ETag refreshes.
        let catalog_key = origin
            .catalog_key
            .as_ref()
            .map(|key| key.0.as_ref())
            .unwrap_or("");
        if returned_from_resident_endpoint(
            cat,
            origin.model.0.as_ref(),
            catalog_key,
            &origin.base_url,
        ) {
            return cat.catalog_owner.clone();
        }
        None
    }

    /// Whether `model_id` has a configured model-owned endpoint plus a
    /// credential for it. Reads catalog before cfg to preserve the lock order
    /// used by the catalog apply path.
    fn model_has_endpoint(&self, model_id: &acp::ModelId) -> bool {
        let (configured_endpoint, has_own_credentials) = {
            let catalog = self.inner.catalog.read();
            // A loaded or pending endpoint catalog is keyed to its configured
            // owner. Resolve both the endpoint config and the credential
            // against that owner even when the selected model is a returned
            // slug or a bundled fallback reselected after invalidation.
            let lookup_id = catalog.catalog_owner.as_ref().unwrap_or(model_id);
            let Some((key, entry)) = catalog
                .models
                .get(lookup_id.0.as_ref())
                .map(|entry| (lookup_id.0.as_ref(), entry))
                .or_else(|| {
                    catalog
                        .models
                        .iter()
                        .find(|(_, entry)| entry.info.model == lookup_id.0.as_ref())
                        .map(|(key, entry)| (key.as_str(), entry))
                })
            else {
                return false;
            };
            let cfg = self.inner.cfg.read();
            let configured_endpoint = config_model_has_endpoint(&cfg, key);
            (configured_endpoint, entry.has_own_credentials())
        };
        configured_endpoint && has_own_credentials
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
            None,
        )
        .await
    }

    async fn refresh_current_model_endpoint_inner(
        &self,
        remote_fetch_enabled: bool,
        observed_etag: Option<String>,
        observed_seq: Option<u64>,
    ) -> bool {
        self.refresh_current_model_endpoint_inner_with_origin(
            remote_fetch_enabled,
            observed_etag,
            observed_seq,
            None,
            None,
        )
        .await
    }

    async fn refresh_current_model_endpoint_inner_with_origin(
        &self,
        remote_fetch_enabled: bool,
        observed_etag: Option<String>,
        observed_seq: Option<u64>,
        observed_endpoint_generation: Option<u64>,
        origin_owner: Option<acp::ModelId>,
    ) -> bool {
        if !remote_fetch_enabled {
            tracing::info!("model-specific catalog refresh skipped: remote_fetch disabled");
            return false;
        }
        // Endpoint refreshes are serialized: overlapping `/models` calls all
        // share the same generation fences, so without a queue an older
        // response could land last and overwrite a newer catalog/etag.
        let _endpoint_refresh = self.inner.endpoint_refresh.lock().await;
        // Spawned ETag watchers can run out of notification order. Once a
        // newer notification has committed, an older watcher's result must
        // not overwrite it, even when the server omits its own ETag and this
        // watcher would otherwise fall back to its observed value.
        if let Some(seq) = observed_seq
            && seq < self.inner.applied_endpoint_etag_seq.load(Ordering::Acquire)
        {
            return false;
        }
        // A queued ETag watcher may have observed the same change as a refresh
        // that already committed while it waited for the lock. Recheck before
        // spending another auth and HTTP round trip, but only against an
        // endpoint-owned catalog's etag: a global etag is scoped to a
        // different resource and must not suppress the endpoint fetch.
        if let Some(observed_etag) = observed_etag.as_deref() {
            let cat = self.inner.catalog.read();
            let same_resident_owner = origin_owner
                .as_ref()
                .is_none_or(|owner| cat.catalog_owner.as_ref() == Some(owner));
            if cat.catalog_source == CatalogSource::ModelEndpoint
                && cat.etag.as_deref() == Some(observed_etag)
                && same_resident_owner
                && observed_endpoint_generation.map_or(true, |g| cat.endpoint_generation == g)
            {
                if let Some(seq) = observed_seq {
                    self.inner
                        .applied_endpoint_etag_seq
                        .store(seq, Ordering::Release);
                }
                return true;
            }
        }
        self.refresh_current_model_endpoint_locked(
            observed_etag,
            observed_seq,
            observed_endpoint_generation,
            origin_owner,
        )
        .await
    }

    /// `OnlineIfUncached`: fetch the configured endpoint only when the catalog
    /// is not already loaded. The recheck happens after the serialization lock
    /// is acquired, so a burst of list requests joins the first fetch instead
    /// of queuing one full auth/HTTP round trip per caller.
    async fn refresh_current_model_endpoint_if_uncached(&self) -> bool {
        let remote_fetch_enabled = crate::util::config::resolve_remote_fetch_enabled();
        if !remote_fetch_enabled {
            tracing::info!("model-specific catalog refresh skipped: remote_fetch disabled");
            return false;
        }
        let _endpoint_refresh = self.inner.endpoint_refresh.lock().await;
        if self.inner.catalog.read().model_endpoint_catalog_loaded {
            return true;
        }
        self.refresh_current_model_endpoint_locked(None, None, None, None)
            .await
    }

    async fn refresh_current_model_endpoint_locked(
        &self,
        observed_etag: Option<String>,
        observed_seq: Option<u64>,
        observed_endpoint_generation: Option<u64>,
        origin_owner: Option<acp::ModelId>,
    ) -> bool {
        // Capture the endpoint identity and its config fence before the
        // request: `model_endpoint_request` awaits a provider refresh, during
        // which either the current model or its endpoint configuration can
        // change. A stale result must not mark the new model/configuration as
        // loaded. The endpoint fence, not the model-switch generation, is the
        // correct bound for this result: switching between models returned by
        // the same endpoint must not discard the refresh, while switching to a
        // different endpoint (or clearing the identity) bumps
        // `endpoint_generation` and rejects it.
        let (catalog_owner, endpoint_generation) = {
            // Catalog before current_model_id: same lock order as
            // `apply_catalog_fenced`.
            let mut cat = self.inner.catalog.write();
            let current = self.inner.current_model_id.read();
            // ETag refreshes must use the configured owner of an
            // endpoint-owned catalog, not the currently selected returned
            // slug or metadata-only overlay.
            let catalog_owner = origin_owner
                .as_ref()
                .cloned()
                .or_else(|| cat.pending_catalog_owner.clone())
                .or_else(|| cat.catalog_owner.clone())
                .unwrap_or_else(|| current.clone());
            // An ETag watcher can target an endpoint configured for a session
            // model while the process current model is global. Record that
            // owner as pending so a result from the emitting session can
            // publish even when it is not the current model. Do not steal a
            // newer pending owner back to this request's origin — that
            // retarget happens in `refresh_if_new_etag`. Do not label the
            // still-resident catalog until this fetch publishes.
            if cat.catalog_owner.as_ref() != Some(&catalog_owner)
                && cat.pending_catalog_owner.is_none()
            {
                if let Some(origin) = origin_owner.clone() {
                    cat.pending_catalog_owner = Some(origin);
                }
            }
            (catalog_owner, cat.endpoint_generation)
        };
        let Some(request) = self.model_endpoint_request(&catalog_owner).await else {
            // The retarget never left the process. Drop the unsuccessful
            // pending owner so still-resident models keep the previous
            // successful publisher as their ETag fence.
            self.clear_failed_pending_catalog_owner(&catalog_owner);
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
        // The observed ETag belongs to the endpoint that produced the
        // notification. If the endpoint changed while the watcher waited for
        // the refresh lock, do not stamp the old ETag onto the new endpoint's
        // catalog when `/models` omits its own ETag.
        let observed_etag =
            if observed_endpoint_generation.map_or(true, |g| g == endpoint_generation) {
                observed_etag
            } else {
                None
            };
        let new_etag = response_etag.or(observed_etag);
        if !self.apply_refresh_result_fenced(
            None,
            models,
            new_etag,
            None,
            Some(endpoint_generation),
            None,
            CatalogSource::ModelEndpoint,
            Some(catalog_owner),
        ) {
            return false;
        }
        if let Some(seq) = observed_seq {
            self.inner
                .applied_endpoint_etag_seq
                .store(seq, Ordering::Release);
        }
        tracing::info!("model-specific catalog refreshed");
        self.notify_models_updated();
        true
    }

    async fn model_endpoint_request(&self, owner: &acp::ModelId) -> Option<ModelEndpointRequest> {
        if !self.model_has_endpoint(owner) {
            return None;
        }
        let entry = {
            let catalog = self.inner.catalog.read();
            catalog
                .models
                .get(owner.0.as_ref())
                .or_else(|| {
                    catalog
                        .models
                        .values()
                        .find(|entry| entry.info.model == owner.0.as_ref())
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

    /// Reserve the current catalog generation for a global fetch. Returns the
    /// reservation plus the generation the decision was made under. `None`
    /// means an attempt already owns that generation, so callers join it with
    /// `wait_for_active_generation` instead of racing a second same-generation
    /// request whose older response could land last.
    async fn reserve_global_fetch(&self) -> (Option<(FetchAttemptGuard, ActiveFetchGuard)>, u64) {
        let _start = self.inner.global_fetch_start.lock().await;
        let cat = self.inner.catalog.read();
        let generation = cat.generation;
        if self
            .inner
            .active_fetch_generations
            .read()
            .contains(&generation)
        {
            return (None, generation);
        }
        let attempt = FetchAttemptGuard::begin_with_generation(&self.inner, generation);
        let active = ActiveFetchGuard::begin(&self.inner, generation);
        (Some((attempt, active)), generation)
    }

    /// Wait until the active fetch that owns `generation` has finished. Uses a
    /// versioned watch so a completion racing the initial check cannot be lost,
    /// unlike a bare `Notify`.
    async fn wait_for_active_generation(&self, generation: u64) {
        let mut done = self.inner.active_fetch_done.subscribe();
        loop {
            {
                let active = self.inner.active_fetch_generations.read();
                if !active.contains(&generation) {
                    return;
                }
            }
            if done.changed().await.is_err() {
                return;
            }
        }
    }

    async fn fetch_and_apply_inner(&self, remote_fetch_enabled: bool) {
        if !remote_fetch_enabled {
            tracing::info!("model catalog refresh skipped: remote_fetch disabled");
            return;
        }
        let (started, joined_generation) = self.reserve_global_fetch().await;
        let Some(started) = started else {
            // Wait for the joined active generation to finish, not just for the
            // first catalog to become ready: once a real catalog is loaded,
            // `catalog_progress` stays `Ready` during later refreshes, so the
            // old wait returned immediately and this caller could complete and
            // notify with the stale catalog while the shared fetch still ran.
            self.wait_for_active_generation(joined_generation).await;
            return;
        };
        self.fetch_and_apply_reserved(started).await;
    }

    /// Run a global catalog fetch for an already-reserved generation.
    async fn fetch_and_apply_reserved(&self, started: (FetchAttemptGuard, ActiveFetchGuard)) {
        let (attempt, active) = started;
        let generation = attempt.generation;
        let _active = active;
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
            None,
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
    fn apply_catalog(&self, models: IndexMap<String, ModelEntry>, new_etag: Option<String>) {
        let _ = self.apply_catalog_fenced(
            None,
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
        cfg: Option<&config::Config>,
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
                if let Some(expected_owner) = catalog_owner.as_ref() {
                    let current = self.inner.current_model_id.read();
                    // Leader session switches do not update the process
                    // current model. A pending ETag target may publish even
                    // while `catalog_owner` still labels the resident
                    // catalog. A stale result whose owner is merely the
                    // process current model must not win.
                    let owner_matches = if let Some(pending) = cat.pending_catalog_owner.as_ref() {
                        expected_owner == pending
                    } else if let Some(resident) = cat.catalog_owner.as_ref() {
                        expected_owner == resident
                    } else {
                        expected_owner == &*current
                            || (cat.catalog_source == CatalogSource::ModelEndpoint
                                && cat.prefetched.as_ref().is_some_and(|models| {
                                    resolve_catalog_key(models, &current).is_some()
                                }))
                    };
                    if !owner_matches {
                        tracing::info!(
                            "model catalog result discarded: current model no longer belongs to the endpoint owner"
                        );
                        return false;
                    }
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
            let endpoint_authoritative = cat.catalog_source == CatalogSource::ModelEndpoint
                || (!cat.model_endpoint_catalog_loaded
                    && (cat.catalog_owner.is_some() || cat.pending_catalog_owner.is_some()));
            if source == CatalogSource::Global && endpoint_authoritative {
                tracing::info!(
                    "global model catalog result discarded: model endpoint catalog is authoritative"
                );
                return false;
            }
            // A settings-only publication intentionally leaves the catalog
            // fence unchanged so an in-flight fetch can still publish. Re-read
            // the current config under the catalog lock so a stale snapshot
            // cannot overwrite the latest filters/defaults. Production callers
            // pass `None`; tests may pin an explicit snapshot.
            let apply_cfg = match cfg {
                Some(cfg) => cfg.clone(),
                None => self.inner.cfg.read().clone(),
            };
            let first_real_catalog = !cat.has_fetched_real_catalog;
            cat.has_fetched_real_catalog = true;
            cat.catalog_source = source;
            cat.catalog_owner = catalog_owner;
            cat.pending_catalog_owner = None;
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
            self.reselect_default_model(&apply_cfg, true);
        } else {
            self.reselect_current_model_if_missing(&apply_cfg, true);
        }
        true
    }

    /// A same-identity refresh, as the fetch paths see it.
    #[cfg(test)]
    fn apply_endpoint_result_for_owner(
        &self,
        models: IndexMap<String, ModelEntry>,
        new_etag: Option<String>,
        catalog_owner: acp::ModelId,
    ) -> bool {
        let endpoint_generation = self.inner.catalog.read().endpoint_generation;
        self.apply_catalog_fenced(
            None,
            models,
            new_etag,
            None,
            Some(endpoint_generation),
            None,
            CatalogSource::ModelEndpoint,
            Some(catalog_owner),
        )
    }

    /// A same-identity refresh, as the fetch paths see it.
    #[cfg(test)]
    fn apply_refresh_result(
        &self,
        config: &config::Config,
        new_prefetched: Option<IndexMap<String, ModelEntry>>,
        new_etag: Option<String>,
    ) -> bool {
        // Resolve under the explicit snapshot; production callers pass `None`
        // and read the config published under the catalog lock.
        let generation = self.inner.catalog.read().generation;
        self.apply_refresh_result_fenced(
            Some(config),
            new_prefetched,
            new_etag,
            Some(generation),
            None,
            None,
            CatalogSource::Global,
            None,
        )
    }

    /// Drop a Leader retarget that never published so later ETags still
    /// route through the resident successful owner.
    fn clear_failed_pending_catalog_owner(&self, failed_owner: &acp::ModelId) {
        let mut cat = self.inner.catalog.write();
        if cat.pending_catalog_owner.as_ref() == Some(failed_owner) {
            cat.pending_catalog_owner = None;
        }
    }

    fn apply_refresh_result_fenced(
        &self,
        config: Option<&config::Config>,
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
                let mut cat = self.inner.catalog.write();
                let same_identity = match generation {
                    Some(generation) => cat.generation == generation,
                    None => endpoint_generation.is_some_and(|g| cat.endpoint_generation == g),
                };
                if same_identity {
                    // A failed or timed-out Leader retarget must not leave the
                    // unsuccessful owner as the publish fence. The resident
                    // catalog_owner stays the previous successful publisher so
                    // a later A-returned ETag still refreshes A, not B.
                    if source == CatalogSource::ModelEndpoint
                        && catalog_owner
                            .as_ref()
                            .is_some_and(|owner| cat.pending_catalog_owner.as_ref() == Some(owner))
                    {
                        cat.pending_catalog_owner = None;
                    }
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
    fn reselect_current_model_if_missing(
        &self,
        config: &config::Config,
        clear_pending_owner: bool,
    ) {
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
        self.set_current_model_id_internal(new_id, clear_pending_owner);
    }

    /// Drop a pending endpoint owner once the selected model belongs to another
    /// source (or a temporary fallback no longer needs the pending refresh).
    fn revalidate_pending_owner_for_selected_model(&self, clear_pending_owner: bool) {
        let mut cat = self.inner.catalog.write();
        let current = self.inner.current_model_id.read().clone();
        let previous_endpoint_owner = cat.catalog_owner.clone().unwrap_or_else(|| current.clone());
        if !endpoint_owner_retained_for_selected_model(&cat, &current, clear_pending_owner) {
            tracing::info!(
                model = %current.0,
                "clearing pending endpoint owner after model selection"
            );
            cat.catalog_owner = None;
        }
        let effective_endpoint_owner = cat.catalog_owner.clone().unwrap_or_else(|| current.clone());
        if previous_endpoint_owner != effective_endpoint_owner {
            tracing::info!(
                model = %current.0,
                "advancing endpoint fence after endpoint owner change"
            );
            cat.endpoint_generation += 1;
        }
    }

    /// Re-resolve the default model against the current catalog.
    fn reselect_default_model(&self, config: &config::Config, clear_pending_owner: bool) {
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
            self.set_current_model_id_internal(new_id, clear_pending_owner);
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
