use super::support::*;
use super::*;
use crate::agent::models::resolve_model_catalog;
use crate::agent::models::{
    ModelEndpointFetchFuture, ModelEndpointRequest, ModelFetchAuth, ModelsEndpoint,
    ModelsFetchFuture, ModelsManagerBuilder, build_prefetched_map_with_model_context,
};
use crate::auth::{AuthManager, GrokAuth, GrokComConfig};
use indexmap::IndexMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;

fn endpoint_cfg() -> crate::agent::config::Config {
    crate::agent::config::Config::new_from_toml_cfg(
        &toml::from_str(
            r#"
            [model.alias-a]
            model = "slug-a"
            base_url = "https://provider-a.example/v1"
            api_key = "a-key"

            [model.alias-b]
            model = "slug-b"
            base_url = "https://provider-b.example/v1"
            api_key = "b-key"
            "#,
        )
        .expect("toml"),
    )
    .expect("config")
}

struct OriginCapture {
    calls: Arc<AtomicUsize>,
    last_url: Arc<std::sync::Mutex<Option<String>>>,
    last_key: Arc<std::sync::Mutex<Option<String>>>,
}
impl ModelsEndpoint for OriginCapture {
    fn fetch_models(
        &self,
        _endpoints: crate::agent::config::EndpointsConfig,
        _auth: Option<GrokAuth>,
        _fetch_auth: ModelFetchAuth,
    ) -> ModelsFetchFuture {
        Box::pin(async { None })
    }

    fn fetch_model_endpoint(&self, request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.last_url.lock().unwrap() = Some(request.base_url.clone());
        *self.last_key.lock().unwrap() = Some(request.api_key.clone());
        Box::pin(async { Some((IndexMap::new(), Some("etag-from-a".to_string()))) })
    }
}

fn colliding_alias_cfg() -> crate::agent::config::Config {
    crate::agent::config::Config::new_from_toml_cfg(
        &toml::from_str(
            r#"
            [model.alias-a]
            model = "shared-slug"
            base_url = "https://shared.example/v1"
            api_key = "a-key"
            extra_headers = { X-Tenant = "one" }

            [model.alias-b]
            model = "shared-slug"
            base_url = "https://shared.example/v1"
            api_key = "b-key"
            extra_headers = { X-Tenant = "two" }
            "#,
        )
        .expect("toml"),
    )
    .expect("config")
}

fn sampling(model: &str, url: &str) -> xai_grok_sampler::SamplerConfig {
    xai_grok_sampler::SamplerConfig {
        model: model.to_string(),
        base_url: url.to_string(),
        api_key: Some("k".to_string()),
        context_window: 256_000,
        ..Default::default()
    }
}

#[tokio::test(flavor = "current_thread")]
async fn inflight_etag_keeps_origin_after_set_session_model() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) =
                mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel::<PersistenceMsg>();
            let mut actor = create_test_actor(0, 256_000, 85, gateway_tx, persistence_tx).await;

            let cfg = endpoint_cfg();
            let tmp = tempfile::TempDir::new().unwrap();
            let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
            let calls = Arc::new(AtomicUsize::new(0));
            let last_url = Arc::new(std::sync::Mutex::new(None));
            actor.models_manager = ModelsManagerBuilder::new(
                None,
                resolve_model_catalog(&cfg, None),
                agent_client_protocol::ModelId::new("alias-a"),
                auth_manager,
                cfg,
            )
            .endpoint(Arc::new(OriginCapture {
                calls: calls.clone(),
                last_url: last_url.clone(),
                last_key: Arc::new(std::sync::Mutex::new(None)),
            }))
            .build();

            actor.chat_state_handle.update_sampling_config(
                xai_grok_sampling_types::SamplingConfig {
                    model: "slug-a".to_string(),
                    base_url: "https://provider-a.example/v1".to_string(),
                    max_completion_tokens: None,
                    temperature: None,
                    top_p: None,
                    api_backend: Default::default(),
                    extra_headers: Default::default(),
                    query_params: Default::default(),
                    env_http_headers: Default::default(),
                    context_window: std::num::NonZeroU64::new(256_000).unwrap(),
                    reasoning_effort: None,
                    stream_tool_calls: None,
                },
            );
            *actor.session_catalog_key.lock() = "alias-a".to_string();

            let origin = actor
                .capture_request_etag_origin()
                .await
                .expect("origin at submit");
            actor.bind_request_etag_origin("req-a", origin);

            let _ = actor
                .handle_set_session_model(
                    sampling("slug-b", "https://provider-b.example/v1"),
                    "alias-b".to_string(),
                    false,
                    false,
                    true,
                    85,
                )
                .await;
            let live = actor.chat_state_handle.get_sampling_config().await.unwrap();
            assert_eq!(live.model, "slug-b");
            assert_eq!(actor.session_catalog_key.lock().as_str(), "alias-b");

            actor
                .handle_model_metadata_update_for_request(
                    Some("req-a"),
                    crate::sampling::ResponseModelMetadata {
                        context_window: None,
                        max_completion_tokens: None,
                        models_etag: Some("etag-from-a".to_string()),
                    },
                )
                .await;

            for _ in 0..200 {
                if calls.load(Ordering::SeqCst) > 0 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                last_url.lock().unwrap().as_deref(),
                Some("https://provider-a.example/v1"),
                "A's in-flight ETag must keep A's origin after the session switched to B"
            );
        })
        .await;
}

fn wait_for_fetch(calls: &AtomicUsize, min: usize) -> impl Future<Output = ()> + '_ {
    async move {
        for _ in 0..200 {
            if calls.load(Ordering::SeqCst) >= min {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }
}

/// After `prepare_sampler_for_turn` has already snapshotted model A, a
/// concurrent `SetSessionModel` to B must not rebind that request's origin
/// from a live chat-state reread. Bind from the submitted `SamplerConfig`.
#[tokio::test(flavor = "current_thread")]
async fn submitted_sampler_config_origin_survives_set_session_model() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) =
                mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel::<PersistenceMsg>();
            let mut actor = create_test_actor(0, 256_000, 85, gateway_tx, persistence_tx).await;

            let cfg = endpoint_cfg();
            let tmp = tempfile::TempDir::new().unwrap();
            let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
            let calls = Arc::new(AtomicUsize::new(0));
            let last_url = Arc::new(std::sync::Mutex::new(None));
            actor.models_manager = ModelsManagerBuilder::new(
                None,
                resolve_model_catalog(&cfg, None),
                agent_client_protocol::ModelId::new("alias-a"),
                auth_manager,
                cfg,
            )
            .endpoint(Arc::new(OriginCapture {
                calls: calls.clone(),
                last_url: last_url.clone(),
                last_key: Arc::new(std::sync::Mutex::new(None)),
            }))
            .build();

            actor.chat_state_handle.update_sampling_config(
                xai_grok_sampling_types::SamplingConfig {
                    model: "slug-a".to_string(),
                    base_url: "https://provider-a.example/v1".to_string(),
                    max_completion_tokens: None,
                    temperature: None,
                    top_p: None,
                    api_backend: Default::default(),
                    extra_headers: Default::default(),
                    query_params: Default::default(),
                    env_http_headers: Default::default(),
                    context_window: std::num::NonZeroU64::new(256_000).unwrap(),
                    reasoning_effort: None,
                    stream_tool_calls: None,
                },
            );
            *actor.session_catalog_key.lock() = "alias-a".to_string();

            let (submitted, catalog_key, persisted_owner) =
                actor.prepare_sampler_for_turn_with_origin_key().await;
            assert_eq!(submitted.model, "slug-a");
            assert_eq!(submitted.base_url, "https://provider-a.example/v1");
            assert_eq!(catalog_key, "alias-a");

            let _ = actor
                .handle_set_session_model(
                    sampling("slug-b", "https://provider-b.example/v1"),
                    "alias-b".to_string(),
                    false,
                    false,
                    true,
                    85,
                )
                .await;
            let live = actor.chat_state_handle.get_sampling_config().await.unwrap();
            assert_eq!(live.model, "slug-b");
            assert_eq!(actor.session_catalog_key.lock().as_str(), "alias-b");

            let live_origin = actor
                .capture_request_etag_origin()
                .await
                .expect("live origin after switch");
            assert_eq!(live_origin.model(), "slug-b");
            assert_eq!(live_origin.base_url(), "https://provider-b.example/v1");

            let origin = actor
                .etag_origin_from_submitted_config(&submitted, catalog_key, persisted_owner)
                .expect("origin from submitted config");
            assert_eq!(origin.model(), "slug-a");
            assert_eq!(origin.base_url(), "https://provider-a.example/v1");
            assert_eq!(origin.catalog_key(), Some("alias-a"));
            assert_ne!(
                origin, live_origin,
                "submitted A origin must not equal the post-switch live B origin"
            );

            actor.bind_request_etag_origin("req-submit-a", origin);
            let bound = actor
                .bound_request_etag_origin("req-submit-a")
                .expect("bound after submit");
            assert_eq!(bound.model(), "slug-a");
            assert_eq!(bound.base_url(), "https://provider-a.example/v1");

            actor
                .handle_model_metadata_update_for_request(
                    Some("req-submit-a"),
                    crate::sampling::ResponseModelMetadata {
                        context_window: None,
                        max_completion_tokens: None,
                        models_etag: Some("etag-from-submitted-a".to_string()),
                    },
                )
                .await;

            wait_for_fetch(&calls, 1).await;
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                last_url.lock().unwrap().as_deref(),
                Some("https://provider-a.example/v1"),
                "metadata after A-submit then switch-to-B must refresh A's origin"
            );
            let still_bound = actor
                .bound_request_etag_origin("req-submit-a")
                .expect("first metadata must retain the origin");
            assert_eq!(still_bound.model(), "slug-a");
            assert_eq!(still_bound.base_url(), "https://provider-a.example/v1");
        })
        .await;
}

/// Sampler retries reuse one `RequestId` and emit another `ModelMetadata`.
/// The first event must not drop the origin, or a later attempt falls back
/// to live catalog/session state after the owner changes.
#[tokio::test(flavor = "current_thread")]
async fn retry_metadata_keeps_bound_origin_after_owner_change() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) =
                mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel::<PersistenceMsg>();
            let mut actor = create_test_actor(0, 256_000, 85, gateway_tx, persistence_tx).await;

            let cfg = endpoint_cfg();
            let tmp = tempfile::TempDir::new().unwrap();
            let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
            let calls = Arc::new(AtomicUsize::new(0));
            let last_url = Arc::new(std::sync::Mutex::new(None));
            actor.models_manager = ModelsManagerBuilder::new(
                None,
                resolve_model_catalog(&cfg, None),
                agent_client_protocol::ModelId::new("alias-a"),
                auth_manager,
                cfg,
            )
            .endpoint(Arc::new(OriginCapture {
                calls: calls.clone(),
                last_url: last_url.clone(),
                last_key: Arc::new(std::sync::Mutex::new(None)),
            }))
            .build();

            actor.chat_state_handle.update_sampling_config(
                xai_grok_sampling_types::SamplingConfig {
                    model: "slug-a".to_string(),
                    base_url: "https://provider-a.example/v1".to_string(),
                    max_completion_tokens: None,
                    temperature: None,
                    top_p: None,
                    api_backend: Default::default(),
                    extra_headers: Default::default(),
                    query_params: Default::default(),
                    env_http_headers: Default::default(),
                    context_window: std::num::NonZeroU64::new(256_000).unwrap(),
                    reasoning_effort: None,
                    stream_tool_calls: None,
                },
            );
            *actor.session_catalog_key.lock() = "alias-a".to_string();

            let origin = actor
                .capture_request_etag_origin()
                .await
                .expect("origin at first attempt");
            actor.bind_request_etag_origin("req-retry", origin);

            actor
                .handle_model_metadata_update_for_request(
                    Some("req-retry"),
                    crate::sampling::ResponseModelMetadata {
                        context_window: None,
                        max_completion_tokens: None,
                        models_etag: Some("etag-attempt-1".to_string()),
                    },
                )
                .await;
            wait_for_fetch(&calls, 1).await;
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert!(
                actor.bound_request_etag_origin("req-retry").is_some(),
                "first ModelMetadata must clone, not remove, the request origin"
            );

            let _ = actor
                .handle_set_session_model(
                    sampling("slug-b", "https://provider-b.example/v1"),
                    "alias-b".to_string(),
                    false,
                    false,
                    true,
                    85,
                )
                .await;
            assert_eq!(
                actor
                    .chat_state_handle
                    .get_sampling_config()
                    .await
                    .unwrap()
                    .model,
                "slug-b"
            );

            actor
                .handle_model_metadata_update_for_request(
                    Some("req-retry"),
                    crate::sampling::ResponseModelMetadata {
                        context_window: None,
                        max_completion_tokens: None,
                        models_etag: Some("etag-attempt-2".to_string()),
                    },
                )
                .await;
            wait_for_fetch(&calls, 2).await;
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            assert_eq!(
                last_url.lock().unwrap().as_deref(),
                Some("https://provider-a.example/v1"),
                "retry ModelMetadata on the same RequestId must keep A's bound origin"
            );
            let still = actor
                .bound_request_etag_origin("req-retry")
                .expect("origin lives until Completed/Failed");
            assert_eq!(still.model(), "slug-a");
            assert_eq!(still.base_url(), "https://provider-a.example/v1");
        })
        .await;
}

/// `get_sampling_config` can enqueue while A is active; after yield,
/// `SetSessionModel` writes B's catalog key before reconstruct resumes.
/// The shipped reconstruct must still bind A's key, not B's.
#[tokio::test(flavor = "current_thread")]
async fn reconstruct_catalog_key_survives_set_session_model_during_config_query() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) =
                mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel::<PersistenceMsg>();
            let mut actor = create_test_actor(0, 256_000, 85, gateway_tx, persistence_tx).await;

            let cfg = colliding_alias_cfg();
            let tmp = tempfile::TempDir::new().unwrap();
            let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
            let calls = Arc::new(AtomicUsize::new(0));
            let last_url = Arc::new(std::sync::Mutex::new(None));
            let last_key = Arc::new(std::sync::Mutex::new(None));
            actor.models_manager = ModelsManagerBuilder::new(
                None,
                resolve_model_catalog(&cfg, None),
                agent_client_protocol::ModelId::new("alias-a"),
                auth_manager,
                cfg,
            )
            .endpoint(Arc::new(OriginCapture {
                calls: calls.clone(),
                last_url: last_url.clone(),
                last_key: last_key.clone(),
            }))
            .build();

            actor.chat_state_handle.update_sampling_config(
                xai_grok_sampling_types::SamplingConfig {
                    model: "shared-slug".to_string(),
                    base_url: "https://shared.example/v1".to_string(),
                    max_completion_tokens: None,
                    temperature: None,
                    top_p: None,
                    api_backend: Default::default(),
                    extra_headers: Default::default(),
                    query_params: Default::default(),
                    env_http_headers: Default::default(),
                    context_window: std::num::NonZeroU64::new(256_000).unwrap(),
                    reasoning_effort: None,
                    stream_tool_calls: None,
                },
            );
            *actor.session_catalog_key.lock() = "alias-a".to_string();
            // Flush A's config through the chat-state actor before the race.
            let primed = actor.chat_state_handle.get_sampling_config().await.unwrap();
            assert_eq!(primed.model, "shared-slug");
            assert_eq!(primed.base_url, "https://shared.example/v1");

            // Fill the chat-state queue so GetSamplingConfig stays pending
            // while SetSessionModel writes B's catalog key (the interleaving
            // the review describes).
            for _ in 0..64 {
                actor.chat_state_handle.record_token_usage(0);
            }

            let actor = std::sync::Arc::new(actor);
            let reconstruct = tokio::task::spawn_local({
                let actor = actor.clone();
                async move { actor.reconstruct_full_config_with_catalog_key().await }
            });
            // Yield so reconstruct can snapshot A's key and enqueue
            // GetSamplingConfig, then switch the session to B.
            tokio::task::yield_now().await;
            let _ = actor
                .handle_set_session_model(
                    sampling("shared-slug", "https://shared.example/v1"),
                    "alias-b".to_string(),
                    false,
                    false,
                    true,
                    85,
                )
                .await;
            assert_eq!(actor.session_catalog_key.lock().as_str(), "alias-b");

            let (submitted, catalog_key, persisted_owner) =
                reconstruct.await.expect("reconstruct joined");
            assert_eq!(submitted.model, "shared-slug");
            assert_eq!(submitted.base_url, "https://shared.example/v1");
            assert_eq!(
                catalog_key, "alias-a",
                "catalog key must be captured before the config query, not after SetSessionModel"
            );

            let origin = actor
                .etag_origin_from_submitted_config(&submitted, catalog_key, persisted_owner)
                .expect("origin from reconstruct snapshot");
            assert_eq!(origin.model(), "shared-slug");
            assert_eq!(origin.base_url(), "https://shared.example/v1");
            assert_eq!(origin.catalog_key(), Some("alias-a"));
            assert_eq!(
                origin.endpoint_owner(),
                Some("alias-a"),
                "A's submitted origin must keep A's configured owner"
            );

            actor.bind_request_etag_origin("req-race-a", origin);
            actor
                .handle_model_metadata_update_for_request(
                    Some("req-race-a"),
                    crate::sampling::ResponseModelMetadata {
                        context_window: None,
                        max_completion_tokens: None,
                        models_etag: Some("etag-from-shared-a".to_string()),
                    },
                )
                .await;
            wait_for_fetch(&calls, 1).await;
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                last_url.lock().unwrap().as_deref(),
                Some("https://shared.example/v1")
            );
            assert_eq!(
                last_key.lock().unwrap().as_deref(),
                Some("a-key"),
                "A's ETag must refresh A's credentials, not B's colliding alias"
            );
        })
        .await;
}

fn inherit_request(url: &str, key: &str) -> ModelEndpointRequest {
    ModelEndpointRequest {
        base_url: url.to_string(),
        api_key: key.to_string(),
        api_backend: Default::default(),
        auth_scheme: Default::default(),
        configured_api_key: Some(key.to_string()),
        configured_env_key: None,
        auth_provider: None,
        extra_headers: IndexMap::new(),
        query_params: IndexMap::new(),
        env_http_headers: IndexMap::new(),
    }
}

fn returned_entry(id: &str, url: &str) -> crate::agent::config::ModelEntryConfig {
    crate::agent::config::ModelEntryConfig {
        id: Some(id.to_string()),
        model: id.to_string(),
        base_url: url.to_string(),
        name: None,
        description: None,
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        api_key: None,
        env_key: None,
        extra_headers: IndexMap::new(),
        api_backend: None,
        context_window: std::num::NonZeroU64::new(256_000).unwrap(),
        auto_compact_threshold_percent: None,
        system_prompt_label: None,
        api_base_url: None,
        use_concise: false,
        agent_type: crate::agent::config::default_agent_type(),
        inference_idle_timeout_secs: None,
        max_retries: None,
        hidden: false,
        supported_in_api: true,
        auth_scheme: None,
        reasoning_effort: None,
        supports_reasoning_effort: false,
        reasoning_efforts: Vec::new(),
        supports_backend_search: false,
        compactions_remaining: None,
        compaction_at_tokens: None,
        show_model_fingerprint: false,
        stream_tool_calls: None,
        laziness_detector: crate::agent::config::LazinessDetectorPerModelConfig::default(),
    }
}

/// A later request from a Leader session using a dynamically returned model
/// must keep the configured owner persisted with the session catalog key.
/// Re-deriving from the shared resident catalog would classify the ETag as
/// global after another session replaces that catalog.
#[tokio::test(flavor = "current_thread")]
async fn persisted_session_owner_survives_catalog_replacement_on_later_request() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) =
                mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel::<PersistenceMsg>();
            let mut actor = create_test_actor(0, 256_000, 85, gateway_tx, persistence_tx).await;

            let cfg = endpoint_cfg();
            let tmp = tempfile::TempDir::new().unwrap();
            let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
            let calls = Arc::new(AtomicUsize::new(0));
            let last_url = Arc::new(std::sync::Mutex::new(None));
            let last_key = Arc::new(std::sync::Mutex::new(None));
            actor.models_manager = ModelsManagerBuilder::new(
                None,
                resolve_model_catalog(&cfg, None),
                agent_client_protocol::ModelId::new("alias-a"),
                auth_manager,
                cfg,
            )
            .endpoint(Arc::new(OriginCapture {
                calls: calls.clone(),
                last_url: last_url.clone(),
                last_key: last_key.clone(),
            }))
            .build();

            let inherit_a = inherit_request("https://provider-a.example/v1", "a-key");
            let returned_a = build_prefetched_map_with_model_context(
                vec![returned_entry(
                    "dynamic-from-a",
                    "https://provider-a.example/v1",
                )],
                &inherit_a,
            );
            actor
                .models_manager
                .install_test_endpoint_catalog("alias-a", returned_a, "etag-a-loaded");
            assert!(
                actor
                    .models_manager
                    .model_in_catalog("dynamic-from-a"),
                "A's dynamic model must be resident before the first request"
            );

            actor.chat_state_handle.update_sampling_config(
                xai_grok_sampling_types::SamplingConfig {
                    model: "dynamic-from-a".to_string(),
                    base_url: "https://provider-a.example/v1".to_string(),
                    max_completion_tokens: None,
                    temperature: None,
                    top_p: None,
                    api_backend: Default::default(),
                    extra_headers: Default::default(),
                    query_params: Default::default(),
                    env_http_headers: Default::default(),
                    context_window: std::num::NonZeroU64::new(256_000).unwrap(),
                    reasoning_effort: None,
                    stream_tool_calls: None,
                },
            );
            *actor.session_catalog_key.lock() = "dynamic-from-a".to_string();
            *actor.session_endpoint_owner.lock() = None;

            let first = actor
                .capture_request_etag_origin()
                .await
                .expect("first request origin while A's catalog is resident");
            assert_eq!(first.model(), "dynamic-from-a");
            assert_eq!(first.catalog_key(), Some("dynamic-from-a"));
            assert_eq!(
                first.endpoint_owner(),
                Some("alias-a"),
                "first request must capture A's configured owner"
            );
            assert_eq!(
                actor.session_endpoint_owner.lock().as_deref(),
                Some("alias-a"),
                "the session must persist the owner with the selected catalog key"
            );

            let inherit_b = inherit_request("https://provider-b.example/v1", "b-key");
            let returned_b = build_prefetched_map_with_model_context(
                vec![returned_entry(
                    "dynamic-from-b",
                    "https://provider-b.example/v1",
                )],
                &inherit_b,
            );
            actor
                .models_manager
                .install_test_endpoint_catalog("alias-b", returned_b, "etag-b-loaded");
            assert!(
                !actor
                    .models_manager
                    .model_in_catalog("dynamic-from-a"),
                "B's catalog must have replaced A's dynamic entry"
            );
            let live_owner = actor.models_manager.configured_endpoint_owner_for_origin(
                "dynamic-from-a",
                "https://provider-a.example/v1",
                "dynamic-from-a",
            );
            assert_eq!(
                live_owner, None,
                "live catalog lookup must fail after B replaces the resident catalog"
            );

            let later = actor
                .capture_request_etag_origin()
                .await
                .expect("later request origin after catalog replacement");
            assert_eq!(later.model(), "dynamic-from-a");
            assert_eq!(later.catalog_key(), Some("dynamic-from-a"));
            assert_eq!(
                later.endpoint_owner(),
                Some("alias-a"),
                "later request must stamp the session-held owner, not re-derive from the live catalog"
            );

            actor.bind_request_etag_origin("req-later-a", later);
            actor
                .handle_model_metadata_update_for_request(
                    Some("req-later-a"),
                    crate::sampling::ResponseModelMetadata {
                        context_window: None,
                        max_completion_tokens: None,
                        models_etag: Some("etag-a-next".to_string()),
                    },
                )
                .await;
            wait_for_fetch(&calls, 1).await;
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                last_url.lock().unwrap().as_deref(),
                Some("https://provider-a.example/v1"),
                "A's later ETag must refresh A's endpoint, not B's"
            );
            assert_eq!(
                last_key.lock().unwrap().as_deref(),
                Some("a-key"),
                "A's later ETag must use A's credentials, not B's"
            );
        })
        .await;
}
