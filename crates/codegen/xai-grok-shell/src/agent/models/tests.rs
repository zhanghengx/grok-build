use super::*;

fn test_manager() -> ModelsManager {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let tmp = std::env::temp_dir().join("grok-test-models-manager");
    let auth_manager = Arc::new(AuthManager::new(&tmp, GrokComConfig::default()));
    ModelsManagerBuilder::new(
        None,
        IndexMap::new(),
        acp::ModelId::new("default"),
        auth_manager,
        config::Config::default(),
    )
    .cache(test_cache_manager(&tmp))
    .build()
}

/// Cold manager (no prefetch, isolated cache and auth) over `endpoint`.
fn cold_manager(cfg: config::Config, endpoint: Arc<dyn ModelsEndpoint>) -> ModelsManager {
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    ModelsManagerBuilder::new(
        None,
        IndexMap::new(),
        acp::ModelId::new("default"),
        auth_manager,
        cfg,
    )
    .endpoint(endpoint)
    .cache(test_cache_manager(tmp.path()))
    .build()
}

/// Never resolves.
struct HangingEndpoint;
impl ModelsEndpoint for HangingEndpoint {
    fn fetch_models(
        &self,
        _endpoints: config::EndpointsConfig,
        _auth: Option<GrokAuth>,
        _fetch_auth: ModelFetchAuth,
    ) -> ModelsFetchFuture {
        Box::pin(std::future::pending())
    }
}

/// Fails every fetch immediately.
struct FailingEndpoint;
impl ModelsEndpoint for FailingEndpoint {
    fn fetch_models(
        &self,
        _endpoints: config::EndpointsConfig,
        _auth: Option<GrokAuth>,
        _fetch_auth: ModelFetchAuth,
    ) -> ModelsFetchFuture {
        Box::pin(async { None })
    }
}

/// Serves `catalog` after `delay`.
struct SlowEndpoint {
    catalog: IndexMap<String, ModelEntry>,
    delay: std::time::Duration,
}
impl ModelsEndpoint for SlowEndpoint {
    fn fetch_models(
        &self,
        _endpoints: config::EndpointsConfig,
        _auth: Option<GrokAuth>,
        _fetch_auth: ModelFetchAuth,
    ) -> ModelsFetchFuture {
        let catalog = self.catalog.clone();
        let delay = self.delay;
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            Some(catalog)
        })
    }
}

#[tokio::test]
async fn catalog_retry_recovers_after_endpoint_returns() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RecoveringEndpoint {
        calls: Arc<AtomicUsize>,
        catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for RecoveringEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let out = if n == 0 {
                None
            } else {
                Some(self.catalog.clone())
            };
            Box::pin(async move { out })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let tmp = std::env::temp_dir().join("grok-test-catalog-retry");
    let auth_manager = Arc::new(AuthManager::new(&tmp, GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        IndexMap::new(),
        acp::ModelId::new("default"),
        auth_manager,
        config::Config::default(),
    )
    .endpoint(Arc::new(RecoveringEndpoint {
        calls: calls.clone(),
        catalog: make_prefetched(&["grok-4"]),
    }))
    .build();
    assert!(!mgr.has_fetched_real_catalog());

    mgr.spawn_catalog_retry_with_backoff(
        /*remote_fetch_enabled*/ true,
        crate::tools::retry::BackoffConfig::new(5, 1, 10),
    );

    let mut recovered = false;
    for _ in 0..200 {
        if mgr.has_fetched_real_catalog() {
            recovered = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(
        recovered,
        "catalog retry did not recover after the endpoint returned"
    );
    assert!(mgr.models().contains_key("grok-4"));
    assert!(
        calls.load(Ordering::SeqCst) >= 2,
        "expected a failed attempt then a success",
    );
}

#[tokio::test]
async fn disk_cache_reload_applies_without_fetching() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingEndpoint {
        calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for CountingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { None })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        IndexMap::new(),
        acp::ModelId::new("default"),
        auth_manager,
        config_from_toml("[models]\ndefault = \"grok-4.5\""),
    )
    .endpoint(Arc::new(CountingEndpoint {
        calls: calls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    let seeder = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    seeder.persist(
        &make_prefetched(&["grok-4.5"]),
        Some("etag-x"),
        auth_method,
        &mgr.cache_origin(),
    );

    mgr.reload_from_disk_cache();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the disk cache load must never hit the transport",
    );
    assert!(mgr.models().contains_key("grok-4.5"));
    assert!(mgr.has_fetched_real_catalog());
    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4.5",
        "first real catalog from the disk cache must resolve the configured default",
    );
}

#[tokio::test]
async fn auth_refresh_watcher_refetches_on_notify() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct NotifyEndpoint {
        calls: Arc<AtomicUsize>,
        catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for NotifyEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let catalog = self.catalog.clone();
            Box::pin(async move { Some(catalog) })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let tmp = std::env::temp_dir().join("grok-test-auth-refresh-watcher");
    let auth_manager = Arc::new(AuthManager::new(&tmp, GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        IndexMap::new(),
        acp::ModelId::new("default"),
        auth_manager,
        config::Config::default(),
    )
    .endpoint(Arc::new(NotifyEndpoint {
        calls: calls.clone(),
        catalog: make_prefetched(&["grok-4"]),
    }))
    .build();
    assert!(!mgr.has_fetched_real_catalog());

    let notify = Arc::new(tokio::sync::Notify::new());
    mgr.start_auth_refresh_watcher(notify.clone());
    notify.notify_one();

    let mut updated = false;
    for _ in 0..200 {
        if mgr.has_fetched_real_catalog() {
            updated = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(updated, "watcher did not re-fetch the catalog on notify");
    assert!(mgr.models().contains_key("grok-4"));
    assert!(calls.load(Ordering::SeqCst) >= 1);
}

#[tokio::test(start_paused = true)]
async fn hanging_fetch_does_not_block_refresh() {
    let mgr = cold_manager(config::Config::default(), Arc::new(HangingEndpoint));

    tokio::time::timeout(
        crate::http::STARTUP_FETCH_TIMEOUT * 10,
        mgr.fetch_and_apply_inner(/*remote_fetch_enabled*/ true),
    )
    .await
    .expect("fetch_and_apply_inner must return despite a hanging endpoint");

    assert!(
        !mgr.has_fetched_real_catalog(),
        "a timed-out fetch must not mark a real catalog",
    );
}

#[tokio::test(start_paused = true)]
async fn slow_fetch_within_timeout_still_applies() {
    // "Slow but succeeds": a fetch that returns just under STARTUP_FETCH_TIMEOUT
    // must still be applied, not degraded to offline.
    let mgr = cold_manager(
        config::Config::default(),
        Arc::new(SlowEndpoint {
            catalog: make_prefetched(&["grok-4"]),
            delay: crate::http::STARTUP_FETCH_TIMEOUT / 2,
        }),
    );

    mgr.fetch_and_apply_inner(/*remote_fetch_enabled*/ true)
        .await;
    assert!(
        mgr.has_fetched_real_catalog(),
        "a fetch within the timeout must apply, not degrade",
    );
    assert!(mgr.models().contains_key("grok-4"));
}

#[tokio::test(start_paused = true)]
async fn etag_refresh_is_bounded_and_single_flighted() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingHangEndpoint {
        calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for CountingHangEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(std::future::pending())
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        IndexMap::new(),
        acp::ModelId::new("default"),
        auth_manager,
        config::Config::default(),
    )
    .endpoint(Arc::new(CountingHangEndpoint {
        calls: calls.clone(),
    }))
    .build();

    // First etag change spawns a bounded fetch; let the task register in-flight.
    mgr.spawn_fetch_inner(Some("etag-1".into()), /*remote_fetch_enabled*/ true)
        .await;
    tokio::task::yield_now().await;
    // Single-flight: a second spawn while one is in flight must not fetch again.
    mgr.spawn_fetch_inner(Some("etag-2".into()), /*remote_fetch_enabled*/ true)
        .await;
    tokio::task::yield_now().await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "single-flight: only one etag fetch in flight at a time",
    );

    // Advance past the bound so the hung fetch is abandoned and the guard clears.
    tokio::time::sleep(crate::http::STARTUP_FETCH_TIMEOUT * 2).await;
    tokio::task::yield_now().await;

    // Guard released → a later etag change fetches again.
    mgr.spawn_fetch_inner(Some("etag-3".into()), /*remote_fetch_enabled*/ true)
        .await;
    tokio::task::yield_now().await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "after the timeout cleared the in-flight guard, a new etag fetch proceeds",
    );

    // remote_fetch disabled is a no-op: no additional fetch.
    mgr.spawn_fetch_inner(Some("etag-4".into()), /*remote_fetch_enabled*/ false)
        .await;
    tokio::task::yield_now().await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "disabled gate must not fetch"
    );
}

#[tokio::test(start_paused = true)]
async fn first_catalog_wait_unblocks_on_fetch_and_skips_dead_dwell() {
    // Deployment auth: a fetch can succeed without a session, so the wait
    // dwells regardless of ambient API-key env.
    let mgr = cold_manager(
        config_from_toml("[endpoints]\ndeployment_key = \"deploy-key\""),
        Arc::new(SlowEndpoint {
            catalog: make_prefetched(&["grok-4"]),
            delay: crate::http::STARTUP_FETCH_TIMEOUT / 2,
        }),
    );

    // Cold cache, remote fetch disabled: no fetch is coming, so no dwell.
    let start = tokio::time::Instant::now();
    assert!(
        !mgr.wait_for_first_catalog_inner(/*remote_fetch_enabled*/ false)
            .await
    );
    assert_eq!(start.elapsed(), std::time::Duration::ZERO);

    // Cold cache, no attempt spawned: nothing to wait for, so no dwell.
    let start = tokio::time::Instant::now();
    assert!(
        !mgr.wait_for_first_catalog_inner(/*remote_fetch_enabled*/ true)
            .await
    );
    assert_eq!(start.elapsed(), std::time::Duration::ZERO);

    // Cold cache, fetch in flight: the wait unblocks when the fetch lands.
    mgr.spawn_fetch_inner(None, /*remote_fetch_enabled*/ true)
        .await;
    assert!(
        mgr.wait_for_first_catalog_inner(/*remote_fetch_enabled*/ true)
            .await,
        "the wait must observe the completed fetch",
    );
    assert!(mgr.models().contains_key("grok-4"));

    // Warm: an already-loaded catalog returns immediately.
    let start = tokio::time::Instant::now();
    assert!(
        mgr.wait_for_first_catalog_inner(/*remote_fetch_enabled*/ true)
            .await
    );
    assert_eq!(start.elapsed(), std::time::Duration::ZERO);
}

#[tokio::test(start_paused = true)]
async fn first_catalog_wait_unblocks_on_failed_fetch() {
    let mgr = cold_manager(
        config_from_toml("[endpoints]\ndeployment_key = \"deploy-key\""),
        Arc::new(FailingEndpoint),
    );
    let budget = crate::http::STARTUP_AUTH_REFRESH_TIMEOUT + crate::http::STARTUP_FETCH_TIMEOUT;
    let start = tokio::time::Instant::now();
    mgr.spawn_fetch_inner(None, /*remote_fetch_enabled*/ true)
        .await;
    assert!(
        !mgr.wait_for_first_catalog_inner(/*remote_fetch_enabled*/ true)
            .await
    );
    assert!(start.elapsed() < budget, "failure must beat the budget");
}

#[tokio::test(start_paused = true)]
async fn first_catalog_wait_is_bounded() {
    let mgr = cold_manager(
        config_from_toml("[endpoints]\ndeployment_key = \"deploy-key\""),
        Arc::new(HangingEndpoint),
    );
    let budget = crate::http::STARTUP_AUTH_REFRESH_TIMEOUT + crate::http::STARTUP_FETCH_TIMEOUT;
    let _attempt = FetchAttemptGuard::begin(&mgr.inner);
    let start = tokio::time::Instant::now();
    assert!(
        !mgr.wait_for_first_catalog_inner(/*remote_fetch_enabled*/ true)
            .await
    );
    assert_eq!(start.elapsed(), budget, "only the budget ends this wait");
}

#[tokio::test(start_paused = true)]
#[serial]
async fn first_catalog_wait_skips_doomed_signed_out_fetch() {
    let _no_key = EnvGuard::unset("XAI_API_KEY");
    let _no_legacy_key = EnvGuard::unset("GROK_CODE_XAI_API_KEY");
    let mgr = cold_manager(config::Config::default(), Arc::new(HangingEndpoint));
    let start = tokio::time::Instant::now();
    mgr.spawn_fetch_inner(None, /*remote_fetch_enabled*/ true)
        .await;
    assert!(
        !mgr.wait_for_first_catalog_inner(/*remote_fetch_enabled*/ true)
            .await
    );
    assert_eq!(start.elapsed(), std::time::Duration::ZERO);
}

#[tokio::test(start_paused = true)]
async fn first_catalog_wait_observes_inline_fetch() {
    let mgr = cold_manager(
        config_from_toml("[endpoints]\ndeployment_key = \"deploy-key\""),
        Arc::new(SlowEndpoint {
            catalog: make_prefetched(&["grok-4"]),
            delay: crate::http::STARTUP_FETCH_TIMEOUT / 2,
        }),
    );
    // Fetch first in the join, so its attempt registers on first poll.
    let ((), ready) = tokio::join!(
        mgr.fetch_and_apply_inner(/*remote_fetch_enabled*/ true),
        mgr.wait_for_first_catalog_inner(/*remote_fetch_enabled*/ true),
    );
    assert!(ready, "the wait must observe the inline fetch's outcome");
}

#[tokio::test(start_paused = true)]
async fn new_fetch_attempt_supersedes_failed_latch() {
    let mgr = cold_manager(
        config_from_toml("[endpoints]\ndeployment_key = \"deploy-key\""),
        Arc::new(FailingEndpoint),
    );
    mgr.fetch_and_apply_inner(/*remote_fetch_enabled*/ true)
        .await;
    assert_eq!(
        *mgr.inner.catalog_progress.borrow(),
        CatalogProgress::Failed
    );

    let attempt = FetchAttemptGuard::begin(&mgr.inner);
    assert_eq!(
        *mgr.inner.catalog_progress.borrow(),
        CatalogProgress::Pending,
        "a new attempt must supersede the stale failure",
    );
    drop(attempt);
    assert_eq!(
        *mgr.inner.catalog_progress.borrow(),
        CatalogProgress::Failed,
        "the last attempt out without an outcome must latch",
    );

    let start = tokio::time::Instant::now();
    assert!(
        !mgr.wait_for_first_catalog_inner(/*remote_fetch_enabled*/ true)
            .await
    );
    assert_eq!(start.elapsed(), std::time::Duration::ZERO);
}

#[test]
fn stale_fetch_result_is_discarded_after_identity_change() {
    let mgr = test_manager();
    let cfg = config::Config::default();
    let stale_generation = mgr.inner.catalog.read().generation;
    mgr.clear();

    assert!(!mgr.apply_refresh_result_fenced(
        &cfg,
        Some(make_prefetched(&["stale-model"])),
        None,
        Some(stale_generation),
        None,
        None,
        CatalogSource::Global,
        None,
    ));
    assert!(!mgr.models().contains_key("stale-model"));
    assert!(!mgr.has_fetched_real_catalog());

    assert!(!mgr.apply_refresh_result_fenced(
        &cfg,
        None,
        None,
        Some(stale_generation),
        None,
        None,
        CatalogSource::Global,
        None,
    ));
    assert_eq!(
        *mgr.inner.catalog_progress.borrow(),
        CatalogProgress::Pending,
        "a stale failure must not latch",
    );

    assert!(mgr.apply_refresh_result(&cfg, Some(make_prefetched(&["new-model"])), None));
    assert!(mgr.models().contains_key("new-model"));
}

#[test]
fn global_refresh_cannot_replace_model_endpoint_catalog() {
    let mgr = test_manager();
    let cfg = config::Config::default();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.model_endpoint_catalog_loaded = true;
        cat.prefetched = Some(make_prefetched(&["endpoint-model"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
    }

    let generation = mgr.inner.catalog.read().generation;
    assert!(!mgr.apply_refresh_result_fenced(
        &cfg,
        Some(make_prefetched(&["global-model"])),
        None,
        Some(generation),
        None,
        None,
        CatalogSource::Global,
        None,
    ));
    assert!(
        mgr.models().contains_key("endpoint-model"),
        "the endpoint catalog must remain authoritative",
    );
    assert!(
        !mgr.models().contains_key("global-model"),
        "a later global result must not replace the endpoint catalog",
    );
    assert!(mgr.inner.catalog.read().model_endpoint_catalog_loaded);
}

#[tokio::test]
async fn stale_endpoint_refresh_is_discarded_after_model_switch() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SlowModelEndpoint {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for SlowModelEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }
        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["old-endpoint-model"]);
            self.calls.fetch_add(1, Ordering::SeqCst);
            started.notify_one();
            Box::pin(async move {
                release.notified().await;
                Some((catalog, None))
            })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg,
    )
    .endpoint(Arc::new(SlowModelEndpoint {
        started: started.clone(),
        release: release.clone(),
        calls: calls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    let mgr_ref = mgr.clone();
    let task = tokio::spawn(async move {
        mgr_ref
            .refresh_current_model_endpoint_inner(true, None, None)
            .await
    });
    started.notified().await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the endpoint fetch must start"
    );

    // Switch models while the fetch is in flight, then let it return.
    mgr.set_current_model_id(acp::ModelId::new("other-model"));
    release.notify_one();

    assert!(
        !task.await.unwrap(),
        "a stale endpoint result must be discarded after the model switch",
    );
    assert!(
        !mgr.inner.catalog.read().model_endpoint_catalog_loaded,
        "a stale endpoint result must not mark the new model loaded",
    );
    assert!(!mgr.models().contains_key("old-endpoint-model"));
}

#[tokio::test]
async fn stale_endpoint_refresh_is_discarded_after_endpoint_config_change() {
    use std::sync::Mutex;

    struct ConfigChangingEndpoint {
        old_started: Arc<tokio::sync::Notify>,
        release_old: Arc<tokio::sync::Notify>,
        requests: Arc<Mutex<Vec<(String, String)>>>,
    }
    impl ModelsEndpoint for ConfigChangingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.requests
                .lock()
                .unwrap()
                .push((request.base_url.clone(), request.api_key.clone()));
            if request.base_url == "https://old-provider.example/v1" {
                let started = self.old_started.clone();
                let release = self.release_old.clone();
                Box::pin(async move {
                    started.notify_one();
                    release.notified().await;
                    Some((make_prefetched(&["old-provider-model"]), None))
                })
            } else {
                Box::pin(async { Some((make_prefetched(&["new-provider-model"]), None)) })
            }
        }
    }

    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://old-provider.example/v1"
            api_key = "old-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let old_started = Arc::new(tokio::sync::Notify::new());
    let release_old = Arc::new(tokio::sync::Notify::new());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg,
    )
    .endpoint(Arc::new(ConfigChangingEndpoint {
        old_started: old_started.clone(),
        release_old: release_old.clone(),
        requests: requests.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    let mgr_ref = mgr.clone();
    let stale = tokio::spawn(async move {
        mgr_ref
            .refresh_current_model_endpoint_inner(true, None, None)
            .await
    });
    old_started.notified().await;

    let new_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://new-provider.example/v1"
            api_key = "new-api-key"
            "#,
    );
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");
    release_old.notify_one();

    assert!(
        !stale.await.unwrap(),
        "a result requested with the old endpoint config must be discarded",
    );
    assert!(!mgr.models().contains_key("old-provider-model"));
    assert!(!mgr.inner.catalog.read().model_endpoint_catalog_loaded);

    assert!(
        mgr.refresh_current_model_endpoint_inner(true, None, None)
            .await
    );
    assert!(mgr.models().contains_key("new-provider-model"));
    assert_eq!(
        requests.lock().unwrap().as_slice(),
        [
            (
                "https://old-provider.example/v1".to_string(),
                "old-api-key".to_string(),
            ),
            (
                "https://new-provider.example/v1".to_string(),
                "new-api-key".to_string(),
            ),
        ],
    );
}

#[tokio::test]
async fn endpoint_refresh_survives_settings_only_config_publication() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SlowModelEndpoint {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for SlowModelEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }
        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["provider-model"]);
            self.calls.fetch_add(1, Ordering::SeqCst);
            started.notify_one();
            Box::pin(async move {
                release.notified().await;
                Some((catalog, None))
            })
        }
    }

    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg,
    )
    .endpoint(Arc::new(SlowModelEndpoint {
        started: started.clone(),
        release: release.clone(),
        calls: calls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    let mgr_ref = mgr.clone();
    let task = tokio::spawn(async move {
        mgr_ref
            .refresh_current_model_endpoint_inner(true, None, None)
            .await
    });
    started.notified().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Settings-only publication: the endpoint connection context is unchanged,
    // so the in-flight fetch must still be allowed to publish.
    let settings_cfg = config_from_toml(
        r#"
            [models]
            default = "endpoint-model"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    mgr.apply_config(settings_cfg)
        .expect("settings-only reload should apply");
    release.notify_one();

    assert!(
        task.await.unwrap(),
        "an endpoint fetch in flight during a settings-only publication must still apply",
    );
    let cat = mgr.inner.catalog.read();
    assert!(cat.model_endpoint_catalog_loaded);
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert!(cat.models.contains_key("provider-model"));
    drop(cat);
}

#[tokio::test]
async fn endpoint_refresh_applies_latest_settings_after_settings_only_publication() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SlowModelEndpoint {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for SlowModelEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }
        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["endpoint-model", "provider-model"]);
            self.calls.fetch_add(1, Ordering::SeqCst);
            started.notify_one();
            Box::pin(async move {
                release.notified().await;
                Some((catalog, None))
            })
        }
    }

    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg,
    )
    .endpoint(Arc::new(SlowModelEndpoint {
        started: started.clone(),
        release: release.clone(),
        calls: calls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    let mgr_ref = mgr.clone();
    let task = tokio::spawn(async move {
        mgr_ref
            .refresh_current_model_endpoint_inner(true, None, None)
            .await
    });
    started.notified().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // The endpoint connection context is unchanged, so the in-flight fetch is
    // still allowed to publish, but it must be resolved against the settings
    // published after the fetch snapshot.
    let settings_cfg = config_from_toml(
        r#"
            [models]
            allowed_models = ["endpoint-model"]
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    mgr.apply_config(settings_cfg)
        .expect("settings-only reload should apply");
    release.notify_one();

    assert!(
        task.await.unwrap(),
        "an endpoint fetch in flight during a settings-only publication must still apply",
    );
    let cat = mgr.inner.catalog.read();
    assert!(cat.model_endpoint_catalog_loaded);
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert!(cat.models.contains_key("provider-model"));
    assert!(
        !cat.models["provider-model"].info.user_selectable,
        "the endpoint result must be resolved against the latest settings",
    );
    assert!(cat.models["endpoint-model"].info.user_selectable);
    drop(cat);
}

#[test]
fn current_model_has_endpoint_recognizes_api_base_url_only() {
    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            api_base_url = "https://api-key.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg,
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    assert!(
        mgr.current_model_has_endpoint(),
        "a model with only api_base_url must still count as endpoint-configured",
    );
}

#[test]
fn current_model_has_endpoint_recognizes_provider_api_base_url() {
    let cfg = config_from_toml(
        r#"
            [model_providers.provider]
            api_base_url = "https://api-key.example/v1"
            api_key = "provider-api-key"

            [model.endpoint-model]
            model_provider = "provider"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg,
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    assert!(
        mgr.current_model_has_endpoint(),
        "a provider api_base_url must count as endpoint-configured",
    );
}

#[test]
fn current_model_has_endpoint_resolves_returned_slug_through_catalog_owner() {
    let cfg = config_from_toml(
        r#"
            [model.alias]
            model = "provider-model"
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("alias"),
        auth_manager,
        cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        let mut prefetched = make_prefetched(&["provider-model"]);
        for entry in prefetched.values_mut() {
            entry.api_key = Some("model-api-key".to_string());
        }
        cat.prefetched = Some(prefetched.clone());
        cat.models = resolve_model_catalog(&cfg, Some(prefetched));
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("alias"));
    }

    mgr.set_current_model_id(acp::ModelId::new("provider-model"));
    assert!(
        mgr.current_model_has_endpoint(),
        "a provider-returned slug must resolve endpoint ownership through the catalog owner",
    );
}

#[test]
fn current_model_has_endpoint_resolves_pending_owner_credentials_for_bundled_model() {
    let cfg = config_from_toml(
        r#"
            [model.alias]
            model = "provider-model"
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("alias"),
        auth_manager,
        cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    {
        let mut cat = mgr.inner.catalog.write();
        cat.models = resolve_model_catalog(&cfg, None);
        cat.has_fetched_real_catalog = false;
        cat.model_endpoint_catalog_loaded = false;
        cat.catalog_source = CatalogSource::Global;
        cat.catalog_owner = Some(acp::ModelId::new("alias"));
    }

    assert!(
        mgr.current_model_has_endpoint(),
        "a pending endpoint owner must provide the endpoint and credential even after reselection moves to a bundled model",
    );
}

fn config_from_toml(toml: &str) -> config::Config {
    config::Config::new_from_toml_cfg(&toml::from_str(toml).unwrap()).unwrap()
}

#[test]
fn model_show_model_fingerprint_reads_catalog_flag() {
    let mgr = test_manager();

    let mut flagged = ModelEntry {
        info: config::ModelInfo::fallback("fp-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    flagged.info.show_model_fingerprint = true;
    mgr.insert_test_entry("fp-model", flagged);

    mgr.insert_test_entry(
        "plain-model",
        ModelEntry {
            info: config::ModelInfo::fallback("plain-model"),
            api_key: None,
            env_key: None,
            auth_provider: None,
            api_base_url: None,
        },
    );

    let mut custom = ModelEntry {
        info: config::ModelInfo::fallback("enterprise-slug"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    custom.info.show_model_fingerprint = true;
    mgr.insert_test_entry("enterprise-key", custom);

    assert!(mgr.model_show_model_fingerprint("fp-model"));
    assert!(!mgr.model_show_model_fingerprint("plain-model"));
    assert!(!mgr.model_show_model_fingerprint("missing-model"));
    assert!(
        mgr.model_show_model_fingerprint("enterprise-slug"),
        "slug lookup must resolve to the catalog key and read the flag",
    );
    assert!(mgr.model_show_model_fingerprint("enterprise-key"));
}

#[test]
fn default_model_honors_allowlist_when_no_default_set() {
    let cfg = config_from_toml(
        r#"
            [models]
            allowed_models = ["keep-*"]
            [model.zzz-first]
            model = "zzz-first"
            base_url = "https://api.x.ai/v1"
            context_window = 256000
            [model.keep-one]
            model = "keep-one"
            base_url = "https://api.x.ai/v1"
            context_window = 256000
            "#,
    );
    let catalog = resolve_model_catalog(&cfg, None);
    let (_key, entry, _src) = resolve_default_model(&cfg, &catalog, true);
    assert!(
        entry.info.user_selectable,
        "picked non-selectable {}",
        entry.model
    );
}

#[test]
fn validate_selectable_rejects_bad_allowlists() {
    let excluded = config_from_toml(
        r#"
            [models]
            default = "grok-3"
            allowed_models = ["grok-4*"]
            [model.grok-3]
            model = "grok-3"
            base_url = "https://api.x.ai/v1"
            context_window = 256000
            [model.grok-4]
            model = "grok-4"
            base_url = "https://api.x.ai/v1"
            context_window = 256000
            "#,
    );
    let catalog = resolve_model_catalog(&excluded, None);
    assert!(
        validate_selectable(&excluded, &catalog)
            .unwrap_err()
            .contains("grok-3")
    );

    let zero = config_from_toml(
        r#"
            [models]
            allowed_models = ["nomatch-*"]
            [model.grok-4]
            model = "grok-4"
            base_url = "https://api.x.ai/v1"
            context_window = 256000
            "#,
    );
    let catalog = resolve_model_catalog(&zero, None);
    assert!(validate_selectable(&zero, &catalog).is_err());
}

#[tokio::test]
async fn refresh_if_new_etag_skips_when_same() {
    let mgr = test_manager();
    mgr.inner.catalog.write().etag = Some("\"abc123\"".to_string());

    mgr.refresh_if_new_etag("\"abc123\"".to_string()).await;
    assert_eq!(
        mgr.inner.catalog.read().etag.as_deref(),
        Some("\"abc123\""),
        "etag should remain unchanged when same"
    );
}

#[tokio::test]
async fn etag_refresh_routes_to_endpoint_catalog_owner() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EndpointEtagFetcher {
        global_calls: Arc<AtomicUsize>,
        endpoint_calls: Arc<AtomicUsize>,
        catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for EndpointEtagFetcher {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.global_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Some(make_prefetched(&["global-model"])) })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.endpoint_calls.fetch_add(1, Ordering::SeqCst);
            let catalog = self.catalog.clone();
            Box::pin(async move { Some((catalog, Some("\"etag-new\"".to_string()))) })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let global_calls = Arc::new(AtomicUsize::new(0));
    let endpoint_calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .endpoint(Arc::new(EndpointEtagFetcher {
        global_calls: global_calls.clone(),
        endpoint_calls: endpoint_calls.clone(),
        catalog: make_prefetched(&["provider-model"]),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.etag = Some("\"etag-old\"".to_string());
    }

    mgr.refresh_if_new_etag("\"etag-new\"".to_string()).await;

    for _ in 0..100 {
        if mgr.inner.catalog.read().etag.as_deref() == Some("\"etag-new\"") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    assert_eq!(
        endpoint_calls.load(Ordering::SeqCst),
        1,
        "the new etag must refresh through the endpoint catalog owner",
    );
    assert_eq!(
        global_calls.load(Ordering::SeqCst),
        0,
        "an endpoint-owned catalog must never trigger a global etag fetch",
    );
    assert_eq!(
        mgr.inner.catalog.read().etag.as_deref(),
        Some("\"etag-new\""),
        "the endpoint refresh must store the response etag",
    );
    assert!(mgr.inner.catalog.read().model_endpoint_catalog_loaded);
    assert!(mgr.models().contains_key("provider-model"));

    mgr.refresh_if_new_etag("\"etag-new\"".to_string()).await;
    assert_eq!(
        endpoint_calls.load(Ordering::SeqCst),
        1,
        "a matching etag must skip the endpoint refresh",
    );
}

#[tokio::test]
async fn etag_refresh_routes_cold_configured_endpoint_to_endpoint_fetch() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EndpointEtagFetcher {
        global_calls: Arc<AtomicUsize>,
        endpoint_calls: Arc<AtomicUsize>,
        catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for EndpointEtagFetcher {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.global_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Some(make_prefetched(&["global-model"])) })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.endpoint_calls.fetch_add(1, Ordering::SeqCst);
            let catalog = self.catalog.clone();
            Box::pin(async move { Some((catalog, Some("\"etag-new\"".to_string()))) })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let global_calls = Arc::new(AtomicUsize::new(0));
    let endpoint_calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg,
    )
    .endpoint(Arc::new(EndpointEtagFetcher {
        global_calls: global_calls.clone(),
        endpoint_calls: endpoint_calls.clone(),
        catalog: make_prefetched(&["provider-model"]),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    mgr.refresh_if_new_etag("\"etag-new\"".to_string()).await;

    for _ in 0..100 {
        if mgr.inner.catalog.read().etag.as_deref() == Some("\"etag-new\"") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    assert_eq!(
        endpoint_calls.load(Ordering::SeqCst),
        1,
        "a cold configured endpoint must refresh through its own /models endpoint",
    );
    assert_eq!(
        global_calls.load(Ordering::SeqCst),
        0,
        "a configured endpoint etag must not launch a global fetch",
    );
    let cat = mgr.inner.catalog.read();
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert!(cat.model_endpoint_catalog_loaded);
    assert!(cat.models.contains_key("provider-model"));
    assert_eq!(cat.etag.as_deref(), Some("\"etag-new\""));
}

#[tokio::test]
async fn etag_refresh_cold_endpoint_fetches_even_when_global_etag_matches() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ColdEndpointEtagFetcher {
        global_calls: Arc<AtomicUsize>,
        endpoint_calls: Arc<AtomicUsize>,
        catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for ColdEndpointEtagFetcher {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.global_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Some(make_prefetched(&["global-model"])) })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.endpoint_calls.fetch_add(1, Ordering::SeqCst);
            let catalog = self.catalog.clone();
            Box::pin(async move { Some((catalog, Some("\"same\"".to_string()))) })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let global_calls = Arc::new(AtomicUsize::new(0));
    let endpoint_calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .endpoint(Arc::new(ColdEndpointEtagFetcher {
        global_calls: global_calls.clone(),
        endpoint_calls: endpoint_calls.clone(),
        catalog: make_prefetched(&["provider-model"]),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["global-model"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.catalog_source = CatalogSource::Global;
        cat.etag = Some("\"same\"".to_string());
    }

    mgr.refresh_if_new_etag("\"same\"".to_string()).await;

    for _ in 0..100 {
        if mgr.inner.catalog.read().catalog_source == CatalogSource::ModelEndpoint {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    assert_eq!(
        endpoint_calls.load(Ordering::SeqCst),
        1,
        "an endpoint etag that equals the stale global etag must still refresh the endpoint",
    );
    assert_eq!(
        global_calls.load(Ordering::SeqCst),
        0,
        "a cold configured endpoint must not launch a global fetch",
    );
    let cat = mgr.inner.catalog.read();
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert!(cat.model_endpoint_catalog_loaded);
    assert!(cat.models.contains_key("provider-model"));
    assert_eq!(cat.etag.as_deref(), Some("\"same\""));
}

#[tokio::test]
async fn etag_refresh_uses_catalog_owner_when_current_is_returned_slug() {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EndpointEtagFetcher {
        global_calls: Arc<AtomicUsize>,
        endpoint_calls: Arc<AtomicUsize>,
        request: Arc<Mutex<Option<ModelEndpointRequest>>>,
        catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for EndpointEtagFetcher {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.global_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.endpoint_calls.fetch_add(1, Ordering::SeqCst);
            *self.request.lock().unwrap() = Some(request);
            let catalog = self.catalog.clone();
            Box::pin(async move { Some((catalog, Some("\"etag-new\"".to_string()))) })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let global_calls = Arc::new(AtomicUsize::new(0));
    let endpoint_calls = Arc::new(AtomicUsize::new(0));
    let request = Arc::new(Mutex::new(None));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .endpoint(Arc::new(EndpointEtagFetcher {
        global_calls: global_calls.clone(),
        endpoint_calls: endpoint_calls.clone(),
        request: request.clone(),
        catalog: make_prefetched(&["provider-model"]),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.etag = Some("\"etag-old\"".to_string());
    }

    mgr.set_current_model_id(acp::ModelId::new("provider-model"));
    assert_eq!(
        mgr.inner
            .catalog
            .read()
            .catalog_owner
            .as_ref()
            .map(|id| id.0.as_ref()),
        Some("endpoint-model"),
        "the endpoint-returned slug must remain owned by the configured endpoint",
    );

    mgr.refresh_if_new_etag("\"etag-new\"".to_string()).await;

    for _ in 0..100 {
        if mgr.inner.catalog.read().etag.as_deref() == Some("\"etag-new\"") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    assert_eq!(
        endpoint_calls.load(Ordering::SeqCst),
        1,
        "the endpoint etag refresh must run even when the current model is a returned slug",
    );
    assert_eq!(
        global_calls.load(Ordering::SeqCst),
        0,
        "an endpoint-owned catalog must never trigger a global etag fetch",
    );
    let request = request.lock().unwrap().take().expect("request captured");
    assert_eq!(request.base_url, "https://provider.example/v1");
    assert_eq!(request.api_key, "model-api-key");
    assert_eq!(
        mgr.inner.catalog.read().etag.as_deref(),
        Some("\"etag-new\""),
        "the endpoint refresh must store the response etag",
    );
    assert!(mgr.models().contains_key("provider-model"));
}

#[tokio::test]
async fn endpoint_owned_same_etag_does_not_renew_global_cache() {
    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let mgr = cold_manager(cfg, Arc::new(FailingEndpoint));
    {
        let mut cat = mgr.inner.catalog.write();
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.etag = Some("\"same\"".to_string());
    }
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    mgr.inner.cache.persist(
        &make_prefetched(&["global-model"]),
        Some("\"same\""),
        auth_method,
        &mgr.cache_origin(),
    );
    let before = std::fs::read(&mgr.inner.cache.path).unwrap();

    mgr.refresh_if_new_etag("\"same\"".to_string()).await;

    let after = std::fs::read(&mgr.inner.cache.path).unwrap();
    assert_eq!(
        before, after,
        "an endpoint-owned matching etag must not renew the global models cache"
    );
}

#[tokio::test]
async fn endpoint_etag_refreshes_are_serialized() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SerialEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for SerialEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let catalog = if n == 0 {
                make_prefetched(&["provider-model-v1"])
            } else {
                make_prefetched(&["provider-model-v2"])
            };
            let etag = if n == 0 {
                "\"etag-1\"".to_string()
            } else {
                "\"etag-2\"".to_string()
            };
            let started = self.started.clone();
            let release = self.release.clone();
            Box::pin(async move {
                if n == 0 {
                    started.notify_one();
                    release.notified().await;
                }
                Some((catalog, Some(etag)))
            })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .endpoint(Arc::new(SerialEndpoint {
        calls: calls.clone(),
        started: started.clone(),
        release: release.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model-v1"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.etag = Some("\"etag-old\"".to_string());
    }

    let first = tokio::spawn({
        let mgr = mgr.clone();
        async move {
            mgr.refresh_current_model_endpoint_inner(true, Some("etag-1".into()), None)
                .await
        }
    });
    started.notified().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let second = tokio::spawn({
        let mgr = mgr.clone();
        async move {
            mgr.refresh_current_model_endpoint_inner(true, Some("etag-2".into()), None)
                .await
        }
    });
    tokio::task::yield_now().await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the second endpoint refresh must wait for the first",
    );

    release.notify_one();
    assert!(first.await.unwrap());
    assert!(second.await.unwrap());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        mgr.inner.catalog.read().etag.as_deref(),
        Some("\"etag-2\""),
        "the newest endpoint refresh must win",
    );
    assert!(mgr.models().contains_key("provider-model-v2"));
}

#[tokio::test]
async fn endpoint_etag_older_notification_cannot_regress_newer_commit() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct OutOfOrderEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for OutOfOrderEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["provider-model-v2"]);
            // `/models` omits its own ETag, so the refresh falls back to the
            // notification's observed value exactly as the review describes.
            Box::pin(async move {
                started.notify_one();
                release.notified().await;
                Some((catalog, None))
            })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .endpoint(Arc::new(OutOfOrderEndpoint {
        calls: calls.clone(),
        started: started.clone(),
        release: release.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model-v1"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.etag = Some("etag-old".to_string());
    }

    // The newer notification (seq 2) acquires the refresh lock and commits
    // before the older notification (seq 1) runs. The older watcher must not
    // fall back to its observed ETag and overwrite the newer commit.
    let newer = tokio::spawn({
        let mgr = mgr.clone();
        async move {
            mgr.refresh_current_model_endpoint_inner(true, Some("etag-newer".into()), Some(2))
                .await
        }
    });
    started.notified().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let older = tokio::spawn({
        let mgr = mgr.clone();
        async move {
            mgr.refresh_current_model_endpoint_inner(true, Some("etag-older".into()), Some(1))
                .await
        }
    });
    tokio::task::yield_now().await;

    release.notify_one();
    assert!(newer.await.unwrap());
    assert!(
        !older.await.unwrap(),
        "an older endpoint ETag notification must be rejected after a newer commit",
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the older watcher must not issue another endpoint request",
    );
    assert_eq!(
        mgr.inner.catalog.read().etag.as_deref(),
        Some("etag-newer"),
        "the stored endpoint ETag must not regress to the older notification",
    );
    assert!(mgr.models().contains_key("provider-model-v2"));
}

#[tokio::test]
async fn endpoint_etag_refresh_rechecks_after_lock() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SameEtagEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for SameEtagEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["provider-model"]);
            Box::pin(async move {
                started.notify_one();
                release.notified().await;
                Some((catalog, Some("\"etag-new\"".to_string())))
            })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .endpoint(Arc::new(SameEtagEndpoint {
        calls: calls.clone(),
        started: started.clone(),
        release: release.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.etag = Some("\"etag-old\"".to_string());
    }

    let first = tokio::spawn({
        let mgr = mgr.clone();
        async move {
            mgr.refresh_current_model_endpoint_inner(true, Some("\"etag-new\"".into()), None)
                .await
        }
    });
    started.notified().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let second = tokio::spawn({
        let mgr = mgr.clone();
        async move {
            mgr.refresh_current_model_endpoint_inner(true, Some("\"etag-new\"".into()), None)
                .await
        }
    });
    tokio::task::yield_now().await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the second refresh must wait for the first to commit",
    );

    release.notify_one();
    assert!(first.await.unwrap());
    assert!(second.await.unwrap());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a queued refresh must recheck the committed etag instead of fetching again",
    );
    assert_eq!(
        mgr.inner.catalog.read().etag.as_deref(),
        Some("\"etag-new\""),
    );
}

#[tokio::test]
async fn endpoint_etag_observed_fallback_discarded_after_endpoint_switch() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SwitchEndpoint {
        calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for SwitchEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            // `/models` omits its own ETag, so without the origin fence the
            // stale endpoint-A ETag would be stored on the endpoint-B catalog.
            Box::pin(async { Some((make_prefetched(&["endpoint-model"]), None)) })
        }
    }

    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider-a.example/v1"
            api_key = "a-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .endpoint(Arc::new(SwitchEndpoint {
        calls: calls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    let observed_endpoint_generation = {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["endpoint-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.etag = Some("\"etag-a\"".to_string());
        cat.endpoint_generation
    };

    // Queue the endpoint-A ETag watcher behind the refresh lock, then switch
    // the endpoint before the watcher can run.
    let _endpoint_refresh = mgr.inner.endpoint_refresh.lock().await;
    let stale_watcher = tokio::spawn({
        let mgr = mgr.clone();
        async move {
            mgr.refresh_current_model_endpoint_inner_with_origin(
                true,
                Some("etag-a".to_string()),
                Some(1),
                Some(observed_endpoint_generation),
            )
            .await
        }
    });
    let new_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider-b.example/v1"
            api_key = "b-key"
            "#,
    );
    mgr.apply_config(new_cfg)
        .expect("endpoint switch should apply");
    drop(_endpoint_refresh);
    assert!(
        stale_watcher.await.unwrap(),
        "the queued endpoint watcher should still refresh the new endpoint",
    );

    {
        let cat = mgr.inner.catalog.read();
        assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
        assert!(cat.model_endpoint_catalog_loaded);
        assert_eq!(
            cat.etag.as_deref(),
            None,
            "the endpoint-A ETag must not be stored on the endpoint-B catalog",
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // A later endpoint-B notification carrying the same string must still
    // trigger a fetch instead of being suppressed as unchanged.
    let notify_mgr = mgr.clone();
    tokio::spawn(async move {
        notify_mgr.refresh_if_new_etag("etag-a".to_string()).await;
    });
    for _ in 0..200 {
        if calls.load(Ordering::SeqCst) >= 2 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "a same-valued endpoint-B ETag notification must not be suppressed by the stale endpoint-A ETag",
    );
    assert_eq!(mgr.inner.catalog.read().etag.as_deref(), Some("etag-a"));
}

#[tokio::test]
async fn set_current_model_id_change_fires_watch_to_all_subscribers() {
    let mgr = test_manager();
    let mut rx_a = mgr.subscribe_model_switch();
    let mut rx_b = mgr.subscribe_model_switch();
    let initial_a = *rx_a.borrow_and_update();
    let initial_b = *rx_b.borrow_and_update();
    assert_eq!(initial_a, initial_b);

    mgr.set_current_model_id(acp::ModelId::new("default"));
    let same_id_ticked = tokio::time::timeout(std::time::Duration::from_millis(25), rx_a.changed())
        .await
        .is_ok();
    assert!(
        !same_id_ticked,
        "set_current_model_id(same id) must NOT bump the watch generation",
    );

    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    tokio::time::timeout(std::time::Duration::from_millis(100), rx_a.changed())
        .await
        .expect("rx_a saw the switch")
        .expect("watch channel still open");
    tokio::time::timeout(std::time::Duration::from_millis(100), rx_b.changed())
        .await
        .expect("rx_b saw the switch")
        .expect("watch channel still open");
    assert_ne!(*rx_a.borrow(), initial_a);
    assert_eq!(*rx_a.borrow(), *rx_b.borrow());
    assert!(mgr.model_switch_generation() > initial_a);
}

#[tokio::test]
async fn model_switch_generation_snapshot_reflects_current_state() {
    let mgr = test_manager();
    let start = mgr.model_switch_generation();
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    assert_eq!(mgr.model_switch_generation(), start + 1);
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    assert_eq!(mgr.model_switch_generation(), start + 1);
    mgr.set_current_model_id(acp::ModelId::new("grok-3"));
    assert_eq!(mgr.model_switch_generation(), start + 2);
}

#[test]
fn first_catalog_reselect_bumps_model_switch_watch() {
    let mgr = test_manager();
    let start = mgr.model_switch_generation();
    let cfg = config_from_toml("[models]\ndefault = \"grok-4.5\"");
    mgr.apply_refresh_result(&cfg, Some(make_prefetched(&["grok-4.5", "grok-4"])), None);
    assert_eq!(mgr.current_model_id().0.as_ref(), "grok-4.5");
    assert!(
        mgr.model_switch_generation() > start,
        "background reselection must fire the model-switch watch",
    );
}

#[test]
fn reselect_missing_current_model_bumps_watch() {
    let mgr = test_manager();
    let cfg = config::Config::default();
    mgr.apply_refresh_result(&cfg, Some(make_prefetched(&["grok-4", "grok-3"])), None);
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    let start = mgr.model_switch_generation();
    // A later catalog drops the current model → reselect_current_model_if_missing.
    mgr.apply_refresh_result(&cfg, Some(make_prefetched(&["grok-3"])), None);
    assert_ne!(mgr.current_model_id().0.as_ref(), "grok-4");
    assert!(
        mgr.model_switch_generation() > start,
        "reselecting away from a removed current model must fire the watch",
    );
}

#[test]
fn rebuild_updates_models_and_available() {
    let mgr = test_manager();
    assert!(mgr.models().is_empty());
    assert!(mgr.available().is_empty());

    let cfg = config::Config::default();
    let mut prefetched = IndexMap::new();
    prefetched.insert(
        "test-model".to_string(),
        ModelEntry {
            info: config::ModelInfo::fallback("test-model"),
            api_key: None,
            env_key: None,
            auth_provider: None,
            api_base_url: None,
        },
    );

    mgr.rebuild(&cfg, Some(prefetched));

    assert!(
        !mgr.models().is_empty(),
        "models should be populated after rebuild"
    );
}

#[test]
fn current_reasoning_effort_round_trip() {
    let mgr = test_manager();
    assert_eq!(mgr.current_reasoning_effort(), None);

    mgr.set_current_reasoning_effort(Some(ReasoningEffort::High));
    assert_eq!(mgr.current_reasoning_effort(), Some(ReasoningEffort::High));

    mgr.set_current_reasoning_effort(None);
    assert_eq!(mgr.current_reasoning_effort(), None);
}

#[test]
fn current_reasoning_effort_seeded_from_config() {
    let tmp = std::env::temp_dir().join("grok-test-models-manager-seed");
    let auth_manager = Arc::new(AuthManager::new(&tmp, GrokComConfig::default()));
    let mut cfg = config::Config::default();
    cfg.models.default_reasoning_effort = Some(ReasoningEffort::Xhigh);
    let mgr = ModelsManager::new(
        None,
        IndexMap::new(),
        acp::ModelId::new("default"),
        auth_manager,
        cfg,
    );
    assert_eq!(mgr.current_reasoning_effort(), Some(ReasoningEffort::Xhigh),);
}

#[test]
fn default_reasoning_effort_only_stamps_supporting_model() {
    use indexmap::IndexMap;

    let mut cfg = config::Config::default();
    cfg.models.default = Some("reasoning-model".to_string());
    cfg.models.default_reasoning_effort = Some(ReasoningEffort::High);

    let mut prefetched = IndexMap::new();
    let mut reasoning_entry = ModelEntry {
        info: config::ModelInfo::fallback("reasoning-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    reasoning_entry.info.supports_reasoning_effort = true;
    prefetched.insert("reasoning-model".to_string(), reasoning_entry);

    let catalog = resolve_model_catalog(&cfg, Some(prefetched));
    assert_eq!(
        catalog["reasoning-model"].info.reasoning_effort,
        Some(ReasoningEffort::High),
        "reasoning-supporting default model should be stamped",
    );

    let mut cfg = config::Config::default();
    cfg.models.default = Some("plain-model".to_string());
    cfg.models.default_reasoning_effort = Some(ReasoningEffort::High);

    let mut prefetched = IndexMap::new();
    let plain_entry = ModelEntry {
        info: config::ModelInfo::fallback("plain-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    prefetched.insert("plain-model".to_string(), plain_entry);

    let catalog = resolve_model_catalog(&cfg, Some(prefetched));
    assert_eq!(
        catalog["plain-model"].info.reasoning_effort, None,
        "non-reasoning default model must NOT be stamped with persisted effort",
    );
}

#[test]
fn reasoning_effort_override_skips_models_that_do_not_offer_level() {
    use indexmap::IndexMap;
    use xai_grok_sampling_types::ReasoningEffortOption;

    let cfg = config::Config {
        reasoning_effort_override: Some(ReasoningEffort::None),
        ..Default::default()
    };

    let mut prefetched = IndexMap::new();
    let mut no_none = ModelEntry {
        info: config::ModelInfo::fallback("grok-4.5"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    no_none.info.supports_reasoning_effort = true;
    no_none.info.reasoning_efforts = vec![ReasoningEffortOption {
        id: "high".into(),
        value: ReasoningEffort::High,
        label: "High".into(),
        description: None,
        default: true,
    }];
    no_none.info.reasoning_effort = Some(ReasoningEffort::High);
    prefetched.insert("grok-4.5".to_string(), no_none);

    let mut with_none = ModelEntry {
        info: config::ModelInfo::fallback("legacy-none"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    with_none.info.supports_reasoning_effort = true;
    with_none.info.reasoning_efforts = vec![ReasoningEffortOption {
        id: "none".into(),
        value: ReasoningEffort::None,
        label: "None".into(),
        description: None,
        default: true,
    }];
    prefetched.insert("legacy-none".to_string(), with_none);

    let catalog = resolve_model_catalog(&cfg, Some(prefetched));
    assert_eq!(
        catalog["grok-4.5"].info.reasoning_effort,
        Some(ReasoningEffort::High),
        "--effort none must not stamp onto models that do not offer none"
    );
    assert_eq!(
        catalog["legacy-none"].info.reasoning_effort,
        Some(ReasoningEffort::None),
        "models that list none should still accept the override"
    );
}

#[test]
fn config_menu_only_model_derives_support_and_default() {
    let mut cfg = config::Config::default();
    cfg.config_models.insert(
        "menu-only".to_string(),
        config::ConfigModelOverride {
            reasoning_efforts: vec![
                ReasoningEffortOption {
                    id: "balanced".to_string(),
                    value: ReasoningEffort::Medium,
                    label: "Balanced".to_string(),
                    description: None,
                    default: false,
                },
                ReasoningEffortOption {
                    id: "deep".to_string(),
                    value: ReasoningEffort::Xhigh,
                    label: "Deep".to_string(),
                    description: None,
                    default: true,
                },
            ],
            ..Default::default()
        },
    );
    cfg.config_models
        .insert("plain".to_string(), config::ConfigModelOverride::default());

    let catalog = resolve_model_catalog(&cfg, None);
    let info = &catalog["menu-only"].info;
    assert!(
        info.supports_reasoning_effort,
        "menu-only model must derive support"
    );
    assert_eq!(
        info.reasoning_effort,
        Some(ReasoningEffort::Xhigh),
        "derived default = marked-default option value"
    );
    assert!(!catalog["plain"].info.supports_reasoning_effort);
    assert_eq!(catalog["plain"].info.reasoning_effort, None);

    let tmp = std::env::temp_dir().join("grok-test-models-manager-menu-only");
    let auth_manager = Arc::new(AuthManager::new(&tmp, GrokComConfig::default()));
    let mgr = ModelsManager::new(
        None,
        catalog,
        acp::ModelId::new("menu-only"),
        auth_manager,
        cfg,
    );
    assert!(mgr.model_supports_reasoning_effort("menu-only"));
    assert_eq!(
        mgr.model_default_reasoning_effort("menu-only"),
        Some(ReasoningEffort::Xhigh)
    );
    assert_eq!(mgr.model_reasoning_efforts("menu-only").len(), 2);
    assert!(!mgr.model_supports_reasoning_effort("plain"));
    assert_eq!(mgr.model_default_reasoning_effort("plain"), None);
}

#[test]
fn cli_reasoning_effort_override_only_stamps_supporting_models() {
    use indexmap::IndexMap;

    let cfg = config::Config {
        reasoning_effort_override: Some(ReasoningEffort::High),
        ..config::Config::default()
    };

    let mut prefetched = IndexMap::new();
    let mut reasoning_entry = ModelEntry {
        info: config::ModelInfo::fallback("reasoning-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    reasoning_entry.info.supports_reasoning_effort = true;
    prefetched.insert("reasoning-model".to_string(), reasoning_entry);

    let plain_entry = ModelEntry {
        info: config::ModelInfo::fallback("plain-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    prefetched.insert("plain-model".to_string(), plain_entry);

    let catalog = resolve_model_catalog(&cfg, Some(prefetched));
    assert_eq!(
        catalog["reasoning-model"].info.reasoning_effort,
        Some(ReasoningEffort::High),
        "reasoning-supporting model should be stamped",
    );
    assert_eq!(
        catalog["plain-model"].info.reasoning_effort, None,
        "non-reasoning model must NOT be stamped",
    );
}

#[test]
fn apply_refresh_result_only_updates_etag_on_success() {
    let mgr = test_manager();
    let cfg = config::Config::default();
    mgr.inner.catalog.write().etag = Some("\"old\"".to_string());

    assert!(
        !mgr.apply_refresh_result(&cfg, None, Some("\"new\"".to_string())),
        "failed refresh should report no update"
    );
    assert_eq!(
        mgr.inner.catalog.read().etag.as_deref(),
        Some("\"old\""),
        "etag should remain unchanged when refresh fails"
    );
    assert!(
        mgr.prefetched().is_none(),
        "prefetched models should stay unchanged"
    );
}

fn make_model_entry(model_id: &str) -> ModelEntry {
    ModelEntry {
        info: config::ModelInfo::fallback(model_id),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    }
}

fn make_prefetched(ids: &[&str]) -> IndexMap<String, ModelEntry> {
    ids.iter()
        .map(|id| (id.to_string(), make_model_entry(id)))
        .collect()
}

// ── startup background refresh ─────────────────────────────────────

#[test]
fn spawn_background_refresh_is_noop_when_real_catalog_present() {
    let mgr = test_manager();
    mgr.inner.catalog.write().has_fetched_real_catalog = true;
    mgr.spawn_background_refresh_inner(/*remote_fetch_enabled*/ true); // must not panic (no tokio::spawn taken)
    assert!(mgr.has_fetched_real_catalog());
}

// Guards the readiness-never-blocks invariant in CI; the e2e proofs are `#[ignore]`.
// current_thread: the post-spawn `!polled` check relies on the task not being
// polled until this test awaits.
#[tokio::test(flavor = "current_thread")]
async fn spawn_background_refresh_never_blocks_on_a_hanging_endpoint() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::sync::Notify;

    // Never resolves; signals the instant the detached task first polls it.
    struct NeverResolvingEndpoint {
        polled: Arc<AtomicBool>,
        dispatched: Arc<Notify>,
    }
    impl ModelsEndpoint for NeverResolvingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            let polled = self.polled.clone();
            let dispatched = self.dispatched.clone();
            Box::pin(async move {
                polled.store(true, Ordering::SeqCst);
                dispatched.notify_one();
                std::future::pending().await
            })
        }
    }

    let polled = Arc::new(AtomicBool::new(false));
    let dispatched = Arc::new(Notify::new());
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        make_prefetched(&["grok-4", "grok-4.5"]),
        acp::ModelId::new("grok-4.5"),
        auth_manager,
        config_from_toml("[models]\ndefault = \"grok-4.5\""),
    )
    .endpoint(Arc::new(NeverResolvingEndpoint {
        polled: polled.clone(),
        dispatched: dispatched.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    mgr.spawn_background_refresh_inner(/*remote_fetch_enabled*/ true);
    assert!(
        !polled.load(Ordering::SeqCst),
        "fetch ran inline on the readiness path; it must be spawned",
    );

    // Generous failure bound: the dispatch may sit behind a full 5s auth dwell.
    tokio::time::timeout(std::time::Duration::from_secs(30), dispatched.notified())
        .await
        .expect("background refresh was never dispatched");
}

#[tokio::test]
#[serial]
async fn sign_out_clears_catalog_rebuilds_bundled_without_fetching() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct BoomEndpoint {
        calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for BoomEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { None })
        }
    }

    // Unset keys so fetch_auth resolves to Session (the sign-out branch).
    let _no_key = EnvGuard::unset("XAI_API_KEY");
    let _no_legacy_key = EnvGuard::unset("GROK_CODE_XAI_API_KEY");
    let calls = Arc::new(AtomicUsize::new(0));
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        make_prefetched(&["grok-4", "grok-4.5"]),
        acp::ModelId::new("grok-4.5"),
        auth_manager,
        config_from_toml("[models]\ndefault = \"grok-4.5\""),
    )
    .endpoint(Arc::new(BoomEndpoint {
        calls: calls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    mgr.inner.catalog.write().has_fetched_real_catalog = true;
    mgr.inner.user_selected_model.store(true, Ordering::Relaxed);

    mgr.on_auth_changed().await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "sign-out must skip the doomed Session-auth fetch",
    );
    assert!(
        !mgr.has_fetched_real_catalog(),
        "sign-out must drop the prior identity's real catalog",
    );
    assert!(
        !mgr.inner.user_selected_model.load(Ordering::Relaxed),
        "sign-out must reset the user-pick latch",
    );
    assert!(
        !mgr.models().is_empty(),
        "sign-out must rebuild the bundled default catalog",
    );
    assert_eq!(
        *mgr.inner.catalog_progress.borrow(),
        CatalogProgress::Failed,
        "sign-out publishes an outcome so parked waiters wake",
    );
}

#[tokio::test]
#[serial]
async fn byok_endpoint_catalog_survives_sign_out() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EndpointAndGlobalCounter {
        global_calls: Arc<AtomicUsize>,
        endpoint_calls: Arc<AtomicUsize>,
        catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for EndpointAndGlobalCounter {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.global_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.endpoint_calls.fetch_add(1, Ordering::SeqCst);
            let catalog = self.catalog.clone();
            Box::pin(async move { Some((catalog, Some("\"etag-new\"".to_string()))) })
        }
    }

    let _no_key = EnvGuard::unset("XAI_API_KEY");
    let _no_legacy_key = EnvGuard::unset("GROK_CODE_XAI_API_KEY");
    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let global_calls = Arc::new(AtomicUsize::new(0));
    let endpoint_calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .endpoint(Arc::new(EndpointAndGlobalCounter {
        global_calls: global_calls.clone(),
        endpoint_calls: endpoint_calls.clone(),
        catalog: make_prefetched(&["provider-model"]),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.etag = Some("\"etag-old\"".to_string());
    }
    mgr.inner.user_selected_model.store(true, Ordering::Relaxed);

    mgr.on_auth_changed().await;

    assert_eq!(
        global_calls.load(Ordering::SeqCst),
        0,
        "sign-out must not fall back to the session/global catalog for a BYOK endpoint",
    );
    assert_eq!(
        endpoint_calls.load(Ordering::SeqCst),
        1,
        "sign-out must reload the model-owned endpoint catalog",
    );
    let cat = mgr.inner.catalog.read();
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert!(cat.model_endpoint_catalog_loaded);
    assert!(cat.models.contains_key("provider-model"));
    assert!(mgr.inner.user_selected_model.load(Ordering::Relaxed));
}

#[tokio::test]
#[serial]
async fn byok_logout_clears_stale_global_catalog_when_endpoint_refresh_fails() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FailingEndpointRefresh {
        global_calls: Arc<AtomicUsize>,
        endpoint_calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for FailingEndpointRefresh {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.global_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.endpoint_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { None })
        }
    }

    let _no_key = EnvGuard::unset("XAI_API_KEY");
    let _no_legacy_key = EnvGuard::unset("GROK_CODE_XAI_API_KEY");
    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let global_calls = Arc::new(AtomicUsize::new(0));
    let endpoint_calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .endpoint(Arc::new(FailingEndpointRefresh {
        global_calls: global_calls.clone(),
        endpoint_calls: endpoint_calls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    // A prior identity's global catalog is resident and marked real, but the
    // BYOK endpoint catalog has never loaded.
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["grok-4"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.catalog_source = CatalogSource::Global;
        cat.etag = Some("\"global-etag\"".to_string());
    }
    mgr.inner.user_selected_model.store(true, Ordering::Relaxed);

    mgr.on_auth_changed().await;

    assert_eq!(
        endpoint_calls.load(Ordering::SeqCst),
        1,
        "sign-out must still attempt the BYOK endpoint refresh",
    );
    assert_eq!(
        global_calls.load(Ordering::SeqCst),
        0,
        "sign-out must not fetch the prior identity's global catalog",
    );
    let cat = mgr.inner.catalog.read();
    assert!(
        !cat.has_fetched_real_catalog,
        "a failed BYOK refresh must not leave the prior identity's global catalog resident",
    );
    assert!(
        cat.prefetched.is_none(),
        "the prior identity's prefetched models must be dropped",
    );
    assert!(
        !cat.models.contains_key("grok-4"),
        "the prior identity's models must not survive sign-out",
    );
    assert!(
        cat.models.contains_key("endpoint-model"),
        "the bundled/config catalog must remain usable after sign-out",
    );
}

#[test]
fn from_config_without_prefetch_produces_usable_catalog() {
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let cfg = config::Config::default();

    let mgr = ModelsManager::from_config(&cfg, None, auth_manager).unwrap();

    let cat = mgr.inner.catalog.read();
    let catalog = &cat.models;
    assert!(
        !catalog.is_empty(),
        "zero-network boot must produce at least one model in the internal catalog"
    );
    let default = mgr.current_model_id();
    assert!(
        catalog.contains_key(default.0.as_ref()),
        "default model {:?} not in internal catalog: {:?}",
        default,
        catalog.keys().collect::<Vec<_>>()
    );
    drop(cat);
    assert!(
        !mgr.has_fetched_real_catalog(),
        "cold-cache boot must not claim a real catalog"
    );
}

#[tokio::test]
async fn model_endpoint_refresh_uses_model_key_and_updates_catalog() {
    use std::sync::Mutex;

    struct CapturingEndpoint {
        request: Arc<Mutex<Option<ModelEndpointRequest>>>,
        catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for CapturingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            *self.request.lock().unwrap() = Some(request);
            let catalog = self.catalog.clone();
            Box::pin(async move { Some((catalog, None)) })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            api_backend = "responses"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    auth_manager.hot_swap(GrokAuth::test_default());
    let request = Arc::new(Mutex::new(None));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg,
    )
    .endpoint(Arc::new(CapturingEndpoint {
        request: request.clone(),
        catalog: make_prefetched(&["provider-model"]),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    assert!(mgr.refresh_current_model_endpoint().await);
    let request = request.lock().unwrap().take().expect("request captured");
    assert_eq!(request.base_url, "https://provider.example/v1");
    assert_eq!(request.api_key, "model-api-key");
    assert_eq!(request.api_backend, ApiBackend::Responses);
    assert_eq!(request.auth_provider, None);

    let inherited = build_prefetched_map_with_model_context(
        vec![make_entry_config("provider-model", None)],
        &request,
    );
    assert_eq!(
        inherited["provider-model"].info.api_backend,
        ApiBackend::Responses
    );

    let mut explicit = make_entry_config("provider-model", None);
    explicit.api_backend = Some(ApiBackend::Messages);
    let preserved = build_prefetched_map_with_model_context(vec![explicit], &request);
    assert_eq!(
        preserved["provider-model"].info.api_backend,
        ApiBackend::Messages
    );

    let mut explicit_default = make_entry_config("provider-model", None);
    explicit_default.api_backend = Some(ApiBackend::ChatCompletions);
    let preserved = build_prefetched_map_with_model_context(vec![explicit_default], &request);
    assert_eq!(
        preserved["provider-model"].info.api_backend,
        ApiBackend::ChatCompletions
    );
    assert!(
        mgr.available()
            .contains_key(&acp::ModelId::new("provider-model"))
    );
    assert!(mgr.models().contains_key("provider-model"));
}

#[tokio::test]
async fn model_endpoint_refresh_uses_api_key_base_url_when_separated() {
    use std::sync::Mutex;

    struct ApiKeyCapturingEndpoint {
        request: Arc<Mutex<Option<ModelEndpointRequest>>>,
        catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for ApiKeyCapturingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            *self.request.lock().unwrap() = Some(request);
            let catalog = self.catalog.clone();
            Box::pin(async move { Some((catalog, None)) })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://session.example/v1"
            api_base_url = "https://api-key.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let request = Arc::new(Mutex::new(None));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg,
    )
    .endpoint(Arc::new(ApiKeyCapturingEndpoint {
        request: request.clone(),
        catalog: make_prefetched(&["provider-model"]),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    assert!(mgr.refresh_current_model_endpoint().await);
    let request = request.lock().unwrap().take().expect("request captured");
    assert_eq!(request.base_url, "https://api-key.example/v1");
    assert_eq!(request.api_key, "model-api-key");
}

#[tokio::test]
async fn list_models_online_if_uncached_fetches_configured_model_endpoint_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingEndpoint {
        calls: Arc<AtomicUsize>,
        catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for CountingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let catalog = self.catalog.clone();
            Box::pin(async move { Some((catalog, None)) })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg,
    )
    .endpoint(Arc::new(CountingEndpoint {
        calls: calls.clone(),
        catalog: make_prefetched(&["endpoint-model", "provider-model"]),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    mgr.list_models(RefreshStrategy::OnlineIfUncached).await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a cold configured endpoint must be fetched for the model picker"
    );
    assert!(
        mgr.models().contains_key("provider-model"),
        "the configured endpoint's catalog must replace the bundled default"
    );

    mgr.list_models(RefreshStrategy::OnlineIfUncached).await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "an already-fetched model endpoint must not be fetched again"
    );
}

#[tokio::test]
async fn list_models_online_if_uncached_joins_in_flight_global_fetch() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SlowGlobalEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for SlowGlobalEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            let _n = self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let catalog = make_prefetched(&["grok-4"]);
            Box::pin(async move {
                started.notify_one();
                // Keep the fetch in flight long enough for the list path to
                // observe and join it, but short enough to stay under the
                // startup fetch timeout.
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                Some(catalog)
            })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            Box::pin(async { None })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let mgr = cold_manager(
        config_from_toml(
            "[endpoints]
deployment_key = \"deploy-key\"",
        ),
        Arc::new(SlowGlobalEndpoint {
            calls: calls.clone(),
            started: started.clone(),
        }),
    );

    // Simulate the cold-start background refresh already being in flight.
    let mgr_ref = mgr.clone();
    let fetch_task = tokio::spawn(async move {
        mgr_ref
            .fetch_and_apply_inner(/*remote_fetch_enabled*/ true)
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(15), started.notified())
        .await
        .expect("the startup fetch never reached the transport");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // `x.ai/models/list` must join the active attempt instead of racing a
    // second request with the same generation.
    mgr.list_models(RefreshStrategy::OnlineIfUncached).await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the list handler must join the in-flight startup fetch, not issue a second one",
    );

    fetch_task.await.unwrap();
    assert!(mgr.has_fetched_real_catalog());
    assert!(mgr.models().contains_key("grok-4"));
}

#[tokio::test(start_paused = true)]
async fn list_models_online_if_uncached_reserves_scheduled_retry_start() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct BlockingGlobalEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for BlockingGlobalEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["grok-4"]);
            Box::pin(async move {
                started.notify_one();
                release.notified().await;
                Some(catalog)
            })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            Box::pin(async { None })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = cold_manager(
        config_from_toml(
            "[endpoints]
deployment_key = \"deploy-key\"",
        ),
        Arc::new(BlockingGlobalEndpoint {
            calls: calls.clone(),
            started: started.clone(),
            release: release.clone(),
        }),
    );
    // Schedule the background retry but do not poll it yet: the cold-start
    // `models/list` request must reserve the generation before the retry task
    // runs its own attempt.
    mgr.spawn_catalog_retry_with_backoff(
        /*remote_fetch_enabled*/ true,
        crate::tools::retry::BackoffConfig::new(2, 0, 0),
    );

    let list_mgr = mgr.clone();
    let list_task = tokio::spawn(async move {
        list_mgr
            .list_models(RefreshStrategy::OnlineIfUncached)
            .await
    });
    started.notified().await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "models/list must own the first fetch",
    );

    // Let the scheduled retry task run while the fetch is still in flight. It
    // must join the reserved attempt rather than starting a second one.
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the retry task must join the active fetch, not start another",
    );

    release.notify_one();
    list_task.await.unwrap();
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(mgr.has_fetched_real_catalog());
    assert!(mgr.models().contains_key("grok-4"));
}

#[tokio::test]
async fn etag_refresh_joins_in_flight_global_fetch() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FirstBlockingGlobalEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for FirstBlockingGlobalEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                let started = self.started.clone();
                let release = self.release.clone();
                let catalog = make_prefetched(&["grok-4"]);
                Box::pin(async move {
                    started.notify_one();
                    release.notified().await;
                    Some(catalog)
                })
            } else {
                Box::pin(async { Some(make_prefetched(&["grok-4"])) })
            }
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            Box::pin(async { None })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = cold_manager(
        config_from_toml("[endpoints]\ndeployment_key = \"deploy-key\""),
        Arc::new(FirstBlockingGlobalEndpoint {
            calls: calls.clone(),
            started: started.clone(),
            release: release.clone(),
        }),
    );

    let fetch_mgr = mgr.clone();
    let fetch_task = tokio::spawn(async move {
        fetch_mgr
            .fetch_and_apply_inner(/*remote_fetch_enabled*/ true)
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the global fetch never reached the transport");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let etag_mgr = mgr.clone();
    let etag_task = tokio::spawn(async move {
        etag_mgr
            .refresh_if_new_etag("\"etag-new\"".to_string())
            .await;
    });
    tokio::task::yield_now().await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "an etag refresh must join the in-flight global fetch, not race it",
    );

    release.notify_one();
    fetch_task.await.unwrap();
    etag_task.await.unwrap();

    // The joined fetch predated the etag change and applied no etag, so the
    // etag refresh must be replayed rather than lost.
    let mut replayed = false;
    for _ in 0..200 {
        if mgr.inner.catalog.read().etag.as_deref() == Some("\"etag-new\"") {
            replayed = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(
        replayed,
        "the etag change must be replayed after the older fetch completes",
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "replaying the etag change must issue a fresh fetch",
    );
    assert!(mgr.has_fetched_real_catalog());
    assert!(mgr.models().contains_key("grok-4"));
}

#[tokio::test(start_paused = true)]
async fn etag_replay_is_dropped_after_identity_generation_change() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FirstBlockingGlobalEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for FirstBlockingGlobalEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["grok-4"]);
            Box::pin(async move {
                if n == 0 {
                    started.notify_one();
                    release.notified().await;
                }
                Some(catalog)
            })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            Box::pin(async { None })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = cold_manager(
        config_from_toml("[endpoints]\ndeployment_key = \"deploy-key\""),
        Arc::new(FirstBlockingGlobalEndpoint {
            calls: calls.clone(),
            started: started.clone(),
            release: release.clone(),
        }),
    );

    let fetch_mgr = mgr.clone();
    let fetch_task = tokio::spawn(async move {
        fetch_mgr
            .fetch_and_apply_inner(/*remote_fetch_enabled*/ true)
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the global fetch never reached the transport");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let join_mgr = mgr.clone();
    let join_task = tokio::spawn(async move {
        join_mgr
            .spawn_fetch_inner(
                Some("\"etag-new\"".to_string()),
                /*remote_fetch_enabled*/ true,
            )
            .await;
    });
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the etag refresh must join the in-flight fetch, not race it",
    );

    // `on_auth_changed` advances the catalog generation while the etag refresh
    // waits, so the pending etag is scoped to the previous identity.
    mgr.inner.catalog.write().generation += 1;
    release.notify_one();
    fetch_task.await.unwrap();
    join_task.await.unwrap();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "an etag from a previous identity must not be replayed after the generation changes",
    );
    assert_ne!(
        mgr.inner.catalog.read().etag.as_deref(),
        Some("\"etag-new\""),
        "the stale identity's etag must not be committed",
    );
}

#[tokio::test]
async fn list_models_online_if_uncached_starts_fresh_fetch_when_active_fetch_is_stale() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct StaleGenerationEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for StaleGenerationEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["grok-4"]);
            Box::pin(async move {
                if n == 0 {
                    started.notify_one();
                    release.notified().await;
                    None
                } else {
                    Some(catalog)
                }
            })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            Box::pin(async { None })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = cold_manager(
        config_from_toml(
            "[endpoints]
deployment_key = \"deploy-key\"",
        ),
        Arc::new(StaleGenerationEndpoint {
            calls: calls.clone(),
            started: started.clone(),
            release: release.clone(),
        }),
    );

    let stale_mgr = mgr.clone();
    let stale_task = tokio::spawn(async move {
        stale_mgr
            .fetch_and_apply_inner(/*remote_fetch_enabled*/ true)
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the stale fetch never reached the transport");

    // A config reload advances the generation while the old fetch is in flight.
    mgr.clear();
    mgr.list_models(RefreshStrategy::OnlineIfUncached).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "a fetch from an obsolete generation must not park models/list behind it",
    );
    assert!(mgr.has_fetched_real_catalog());
    assert!(mgr.models().contains_key("grok-4"));

    release.notify_one();
    stale_task.await.unwrap();
    assert!(
        mgr.models().contains_key("grok-4"),
        "the stale fetch result must not replace the fresh catalog",
    );
}

#[tokio::test]
async fn list_models_online_if_uncached_deduplicates_concurrent_endpoint_fetches() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SlowEndpointFetch {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for SlowEndpointFetch {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let catalog = self.catalog.clone();
            Box::pin(async move {
                started.notify_one();
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                Some((catalog, None))
            })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg,
    )
    .endpoint(Arc::new(SlowEndpointFetch {
        calls: calls.clone(),
        started: started.clone(),
        catalog: make_prefetched(&["provider-model"]),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    let first_mgr = mgr.clone();
    let first = tokio::spawn(async move {
        first_mgr
            .list_models(RefreshStrategy::OnlineIfUncached)
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the endpoint fetch never reached the transport");

    let second_mgr = mgr.clone();
    let second = tokio::spawn(async move {
        second_mgr
            .list_models(RefreshStrategy::OnlineIfUncached)
            .await
    });

    first.await.unwrap();
    second.await.unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a concurrent models/list burst must join the in-flight endpoint fetch",
    );
    assert!(mgr.inner.catalog.read().model_endpoint_catalog_loaded);
}

#[tokio::test(start_paused = true)]
async fn list_models_online_if_uncached_refetches_when_joined_fetch_generation_goes_stale() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FirstBlockingGlobalEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for FirstBlockingGlobalEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["grok-4"]);
            Box::pin(async move {
                if n == 0 {
                    started.notify_one();
                    release.notified().await;
                }
                Some(catalog)
            })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            Box::pin(async { None })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = cold_manager(
        config_from_toml("[endpoints]\ndeployment_key = \"deploy-key\""),
        Arc::new(FirstBlockingGlobalEndpoint {
            calls: calls.clone(),
            started: started.clone(),
            release: release.clone(),
        }),
    );

    let fetch_mgr = mgr.clone();
    let fetch_task = tokio::spawn(async move {
        fetch_mgr
            .fetch_and_apply_inner(/*remote_fetch_enabled*/ true)
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the global fetch never reached the transport");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let list_mgr = mgr.clone();
    let list_task = tokio::spawn(async move {
        list_mgr
            .list_models(RefreshStrategy::OnlineIfUncached)
            .await;
    });
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }

    // A config reload advances the generation while models/list joins the
    // older in-flight fetch, which then cannot publish for the new config.
    mgr.inner.catalog.write().generation += 1;
    release.notify_one();
    fetch_task.await.unwrap();
    list_task.await.unwrap();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "models/list must start a fresh fetch when the joined fetch belongs to an older generation",
    );
    assert!(mgr.has_fetched_real_catalog());
    assert!(mgr.models().contains_key("grok-4"));
}

#[tokio::test]
async fn list_models_rechecks_source_after_endpoint_model_switches_to_regular() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EndpointThenGlobal {
        endpoint_calls: Arc<AtomicUsize>,
        global_calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        endpoint_catalog: IndexMap<String, ModelEntry>,
        global_catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for EndpointThenGlobal {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.global_calls.fetch_add(1, Ordering::SeqCst);
            let catalog = self.global_catalog.clone();
            Box::pin(async move { Some(catalog) })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.endpoint_calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = self.endpoint_catalog.clone();
            Box::pin(async move {
                started.notify_one();
                release.notified().await;
                Some((catalog, None))
            })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"

            [model.regular-model]
            model = "regular-model"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let endpoint_calls = Arc::new(AtomicUsize::new(0));
    let global_calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg,
    )
    .endpoint(Arc::new(EndpointThenGlobal {
        endpoint_calls: endpoint_calls.clone(),
        global_calls: global_calls.clone(),
        started: started.clone(),
        release: release.clone(),
        endpoint_catalog: make_prefetched(&["provider-model"]),
        global_catalog: make_prefetched(&["regular-model"]),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    let list_mgr = mgr.clone();
    let list_task = tokio::spawn(async move {
        list_mgr
            .list_models(RefreshStrategy::OnlineIfUncached)
            .await;
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the endpoint fetch never reached the transport");
    assert_eq!(endpoint_calls.load(Ordering::SeqCst), 1);

    // Switch to a regular model while the endpoint fetch is in flight. The
    // endpoint result is discarded and the one-shot list must run the global
    // path for the newly selected model.
    mgr.set_current_model_id(acp::ModelId::new("regular-model"));
    release.notify_one();
    list_task.await.unwrap();

    assert_eq!(endpoint_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        global_calls.load(Ordering::SeqCst),
        1,
        "after the endpoint model is replaced by a regular model, list_models must fetch the global catalog",
    );
    assert!(mgr.has_fetched_real_catalog());
    assert!(mgr.models().contains_key("regular-model"));
}

#[tokio::test]
async fn list_models_rechecks_source_after_regular_model_switches_to_endpoint() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct GlobalThenEndpoint {
        endpoint_calls: Arc<AtomicUsize>,
        global_calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        endpoint_catalog: IndexMap<String, ModelEntry>,
        global_catalog: IndexMap<String, ModelEntry>,
    }
    impl ModelsEndpoint for GlobalThenEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.global_calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = self.global_catalog.clone();
            Box::pin(async move {
                started.notify_one();
                release.notified().await;
                Some(catalog)
            })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.endpoint_calls.fetch_add(1, Ordering::SeqCst);
            let catalog = self.endpoint_catalog.clone();
            Box::pin(async move { Some((catalog, None)) })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.regular-model]
            model = "regular-model"

            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let endpoint_calls = Arc::new(AtomicUsize::new(0));
    let global_calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("regular-model"),
        auth_manager,
        cfg,
    )
    .endpoint(Arc::new(GlobalThenEndpoint {
        endpoint_calls: endpoint_calls.clone(),
        global_calls: global_calls.clone(),
        started: started.clone(),
        release: release.clone(),
        endpoint_catalog: make_prefetched(&["provider-model"]),
        global_catalog: make_prefetched(&["regular-model", "endpoint-model"]),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    let list_mgr = mgr.clone();
    let list_task = tokio::spawn(async move {
        list_mgr
            .list_models(RefreshStrategy::OnlineIfUncached)
            .await;
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the global fetch never reached the transport");
    assert_eq!(global_calls.load(Ordering::SeqCst), 1);

    // Switch to the endpoint model while the global fetch is in flight. The
    // one-shot list must serve the endpoint catalog, not the global one.
    mgr.set_current_model_id(acp::ModelId::new("endpoint-model"));
    release.notify_one();
    list_task.await.unwrap();

    assert_eq!(global_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        endpoint_calls.load(Ordering::SeqCst),
        1,
        "after switching to an endpoint model, list_models must fetch its endpoint catalog",
    );
    let cat = mgr.inner.catalog.read();
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert!(cat.model_endpoint_catalog_loaded);
    assert!(cat.models.contains_key("provider-model"));
}

#[tokio::test]
async fn model_endpoint_without_model_credential_is_not_requested() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingEndpoint {
        calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for CountingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { None })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    auth_manager.hot_swap(GrokAuth::test_default());
    let calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg,
    )
    .endpoint(Arc::new(CountingEndpoint {
        calls: calls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    assert!(!mgr.refresh_current_model_endpoint().await);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn model_endpoint_refresh_respects_remote_fetch_gate() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingEndpoint {
        calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for CountingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { None })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg,
    )
    .endpoint(Arc::new(CountingEndpoint {
        calls: calls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    assert!(
        !mgr.refresh_current_model_endpoint_inner(false, None, None)
            .await,
        "a model-owned catalog must not refresh when remote_fetch is disabled"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no model-endpoint fetch may be issued under the remote_fetch gate"
    );
}

#[tokio::test(start_paused = true)]
async fn bounded_auth_provider_refresh_times_out() {
    use crate::auth::ProviderRefreshOutcome;

    let started = tokio::time::Instant::now();
    let result = ModelsManager::bounded_auth_provider_refresh(std::future::pending::<
        ProviderRefreshOutcome,
    >())
    .await;
    assert!(
        !result,
        "a hung provider refresh must degrade to no request instead of blocking"
    );
    assert!(
        started.elapsed() >= crate::http::STARTUP_AUTH_REFRESH_TIMEOUT,
        "must wait the full bound before giving up",
    );
}

#[test]
fn switching_current_model_invalidates_endpoint_catalog_cache() {
    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"

            [model.regular-model]
            model = "regular-model"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        // `regular-model` is configured but not returned by the endpoint, so
        // selecting it must invalidate the endpoint-owned catalog.
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.etag = Some("endpoint-etag".to_string());
    }
    let stale_generation = mgr.inner.catalog.read().generation;

    mgr.set_current_model_id(acp::ModelId::new("regular-model"));

    let cat = mgr.inner.catalog.read();
    assert!(!cat.model_endpoint_catalog_loaded);
    assert_eq!(cat.catalog_source, CatalogSource::Global);
    assert!(!cat.has_fetched_real_catalog);
    assert!(cat.prefetched.is_none());
    assert!(cat.etag.is_none());
    assert!(!cat.models.contains_key("provider-model"));
    assert!(cat.models.contains_key("regular-model"));
    assert!(cat.generation > stale_generation);
    drop(cat);

    assert!(
        !mgr.apply_refresh_result_fenced(
            &cfg,
            Some(make_prefetched(&["stale-global-model"])),
            None,
            Some(stale_generation),
            None,
            None,
            CatalogSource::Global,
            None,
        ),
        "a global fetch started before the model transition must stay fenced",
    );
    assert!(!mgr.models().contains_key("stale-global-model"));
}

#[tokio::test]
async fn switching_from_endpoint_catalog_refetches_global_catalog() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct GlobalEndpoint {
        calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for GlobalEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Some(make_prefetched(&["global-model"])) })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"

            [model.regular-model]
            model = "regular-model"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .endpoint(Arc::new(GlobalEndpoint {
        calls: calls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
    }

    mgr.set_current_model_id(acp::ModelId::new("regular-model"));
    mgr.list_models(RefreshStrategy::OnlineIfUncached).await;

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(mgr.models().contains_key("global-model"));
    assert!(!mgr.models().contains_key("provider-model"));
}

#[test]
fn switching_to_endpoint_returned_slug_keeps_endpoint_catalog() {
    let cfg = config_from_toml(
        r#"
            [model.alias]
            model = "provider-model"
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("alias"),
        auth_manager,
        cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("alias"));
    }

    mgr.set_current_model_id(acp::ModelId::new("provider-model"));

    let cat = mgr.inner.catalog.read();
    assert!(
        cat.model_endpoint_catalog_loaded,
        "a provider-returned slug that shares the owner's routing slug must stay endpoint-owned",
    );
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert_eq!(
        cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
        Some("alias")
    );
    assert!(cat.models.contains_key("provider-model"));
    drop(cat);
}

#[test]
fn apply_config_preserves_unchanged_endpoint_catalog() {
    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }

    let new_cfg = config_from_toml(
        r#"
            [models]
            default = "endpoint-model"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");

    let cat = mgr.inner.catalog.read();
    assert!(
        cat.model_endpoint_catalog_loaded,
        "a config publication with an unchanged endpoint must keep the endpoint catalog",
    );
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert_eq!(
        cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
        Some("endpoint-model")
    );
    assert!(cat.models.contains_key("provider-model"));
    drop(cat);
}

#[tokio::test]
async fn apply_config_switches_owner_to_selected_returned_slug_overlay_endpoint() {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RecordingEndpoint {
        calls: Arc<AtomicUsize>,
        base_urls: Arc<Mutex<Vec<String>>>,
    }
    impl ModelsEndpoint for RecordingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.base_urls
                .lock()
                .unwrap()
                .push(request.base_url.clone());
            Box::pin(async { Some((make_prefetched(&["provider-model"]), None)) })
        }
    }

    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let base_urls = Arc::new(Mutex::new(Vec::new()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .endpoint(Arc::new(RecordingEndpoint {
        calls: calls.clone(),
        base_urls: base_urls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    let mut prefetched = make_prefetched(&["provider-model", "provider-sibling"]);
    for entry in prefetched.values_mut() {
        entry.api_key = Some("endpoint-api-key".to_string());
    }
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(prefetched.clone());
        cat.models = resolve_model_catalog(&old_cfg, Some(prefetched));
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }
    mgr.set_current_model_id(acp::ModelId::new("provider-model"));

    let new_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"

            [model.provider-model]
            base_url = "https://own.example/v1"
            api_key = "own-api-key"
            "#,
    );
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");

    {
        let cat = mgr.inner.catalog.read();
        assert_eq!(cat.catalog_source, CatalogSource::Global);
        assert!(!cat.model_endpoint_catalog_loaded);
        assert_eq!(
            cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
            Some("provider-model"),
            "the pending owner must switch to the selected slug's own endpoint",
        );
        assert!(
            !cat.models.contains_key("provider-sibling"),
            "the old endpoint catalog must not be retained when the selected overlay changes context",
        );
        assert_eq!(
            cat.models["provider-model"].api_key.as_deref(),
            Some("own-api-key"),
        );
    }

    assert!(
        mgr.current_model_has_endpoint(),
        "the selected slug's own endpoint must remain the refresh target",
    );
    assert!(
        mgr.refresh_current_model_endpoint_inner(true, None, None)
            .await,
        "the replacement fetch must target the selected slug's own endpoint",
    );
    assert_eq!(
        base_urls.lock().unwrap().as_slice(),
        ["https://own.example/v1"],
        "the replacement must query the overlay endpoint, not the previous owner",
    );
    assert!(mgr.models().contains_key("provider-model"));
}

#[test]
fn apply_config_rejects_invalid_allowlist_for_retained_endpoint_catalog() {
    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }

    let new_cfg = config_from_toml(
        r#"
            [models]
            allowed_models = ["nomatch-*"]
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    assert!(
        mgr.apply_config(new_cfg).is_err(),
        "an allowlist that excludes every retained endpoint model must be rejected",
    );

    let cat = mgr.inner.catalog.read();
    assert!(
        cat.model_endpoint_catalog_loaded,
        "an allowlist that excludes every retained endpoint model must be rejected",
    );
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert!(!cat.allowlist_excludes_all);
    assert!(cat.models.contains_key("provider-model"));
    assert!(
        mgr.inner.cfg.read().models.allowed_models.is_none(),
        "the invalid allowlist reload must not be published",
    );
    drop(cat);
}

#[test]
fn switching_to_endpoint_returned_configured_overlay_keeps_endpoint_catalog() {
    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"

            [model.provider-model]
            name = "Configured Overlay"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    let mut prefetched = make_prefetched(&["provider-model", "provider-sibling"]);
    for entry in prefetched.values_mut() {
        entry.api_key = Some("endpoint-api-key".to_string());
    }
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(prefetched.clone());
        cat.models = resolve_model_catalog(&cfg, Some(prefetched));
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }

    mgr.set_current_model_id(acp::ModelId::new("provider-model"));

    let cat = mgr.inner.catalog.read();
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert!(cat.model_endpoint_catalog_loaded);
    assert_eq!(
        cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
        Some("endpoint-model")
    );
    assert!(
        cat.models.contains_key("provider-sibling"),
        "switching to an endpoint-returned overlay must not drop endpoint-discovered siblings",
    );
    assert_eq!(
        cat.models["provider-model"].api_key.as_deref(),
        Some("endpoint-api-key"),
        "a metadata-only config overlay must keep the endpoint catalog's inherited credential",
    );
    drop(cat);
    assert_eq!(mgr.current_model_id().0.as_ref(), "provider-model");
}

#[test]
fn switching_to_endpoint_returned_overlay_with_own_endpoint_resets_catalog() {
    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"

            [model.provider-model]
            base_url = "https://own.example/v1"
            api_key = "own-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    let mut prefetched = make_prefetched(&["provider-model", "provider-sibling"]);
    for entry in prefetched.values_mut() {
        entry.api_key = Some("endpoint-api-key".to_string());
    }
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(prefetched.clone());
        cat.models = resolve_model_catalog(&cfg, Some(prefetched));
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }

    mgr.set_current_model_id(acp::ModelId::new("provider-model"));

    let cat = mgr.inner.catalog.read();
    assert_eq!(cat.catalog_source, CatalogSource::Global);
    assert!(!cat.model_endpoint_catalog_loaded);
    assert!(
        !cat.models.contains_key("provider-sibling"),
        "an overlay with its own endpoint context must leave the parent endpoint catalog",
    );
    assert_eq!(
        cat.models["provider-model"].api_key.as_deref(),
        Some("own-api-key"),
    );
    drop(cat);
}

#[test]
fn apply_config_reselecting_default_stops_after_rejection() {
    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }

    let new_cfg = config_from_toml(
        r#"
            [models]
            default = "other-model"
            allowed_models = ["nomatch-*"]
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    assert!(
        mgr.apply_config_reselecting_default(new_cfg).is_err(),
        "a rejected reload must propagate to the reselecting-default caller",
    );

    assert_eq!(mgr.current_model_id().0.as_ref(), "endpoint-model");
    let cat = mgr.inner.catalog.read();
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert!(cat.model_endpoint_catalog_loaded);
    assert!(cat.models.contains_key("provider-model"));
    assert!(!cat.allowlist_excludes_all);
    assert!(
        mgr.inner.cfg.read().models.allowed_models.is_none(),
        "the rejected reload must not be published",
    );
}

#[test]
fn apply_config_revalidates_endpoint_snapshot_after_concurrent_refresh() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["old-provider-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.etag = Some("\"etag-old\"".to_string());
    }

    let settings_cfg = config_from_toml(
        r#"
            [models]
            default = "endpoint-model"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );

    // Hold the fetch_auth write lock: apply_config blocks on it just before
    // its final catalog write, so this pauses the publication in the window
    // where an endpoint refresh can commit.
    let _fetch_auth = mgr.inner.fetch_auth.write();

    let entered = Arc::new(AtomicBool::new(false));
    let entered_for_thread = entered.clone();
    let mgr2 = mgr.clone();
    let apply_thread = std::thread::spawn(move || {
        entered_for_thread.store(true, Ordering::SeqCst);
        mgr2.apply_config(settings_cfg)
    });
    while !entered.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }
    // Let the publication thread reach the fetch_auth block before the
    // refresh commits.
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Simulate an endpoint ETag refresh committing in that window: fresh
    // prefetched entries plus a new response etag.
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["new-provider-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.etag = Some("\"etag-new\"".to_string());
        cat.has_fetched_real_catalog = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.model_endpoint_catalog_loaded = true;
    }
    drop(_fetch_auth);
    apply_thread
        .join()
        .expect("apply_config thread panicked")
        .expect("settings-only reload should apply");

    let cat = mgr.inner.catalog.read();
    assert!(cat.model_endpoint_catalog_loaded);
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert!(
        cat.models.contains_key("new-provider-model"),
        "the settings publication must not overwrite the refreshed endpoint models",
    );
    assert!(
        !cat.models.contains_key("old-provider-model"),
        "stale endpoint entries must not be re-latched",
    );
    assert_eq!(
        cat.etag.as_deref(),
        Some("\"etag-new\""),
        "the fresh etag must be preserved so a same-etag response skips the refresh",
    );
}

#[test]
fn apply_config_recomputes_allowlist_gate_for_pending_catalog() {
    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["keep-1"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }

    let new_cfg = config_from_toml(
        r#"
            [models]
            allowed_models = ["keep-*"]
            "#,
    );
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");

    assert!(
        mgr.allowlist_excludes_all(),
        "the pending config-only catalog must stay fail-closed while the endpoint catalog is invalidated",
    );
    assert!(!mgr.inner.catalog.read().model_endpoint_catalog_loaded);
}

#[tokio::test]
async fn apply_config_retains_pending_endpoint_owner_for_returned_slug() {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RecordingEndpoint {
        calls: Arc<AtomicUsize>,
        base_urls: Arc<Mutex<Vec<String>>>,
    }
    impl ModelsEndpoint for RecordingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.base_urls
                .lock()
                .unwrap()
                .push(request.base_url.clone());
            Box::pin(async { Some((make_prefetched(&["provider-model"]), None)) })
        }
    }

    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://old-provider.example/v1"
            api_key = "old-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let base_urls = Arc::new(Mutex::new(Vec::new()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .endpoint(Arc::new(RecordingEndpoint {
        calls: calls.clone(),
        base_urls: base_urls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }
    mgr.set_current_model_id(acp::ModelId::new("provider-model"));

    let new_cfg = config_from_toml(
        r#"
            [models]
            allowed_models = ["provider-model"]
            [model.endpoint-model]
            base_url = "https://new-provider.example/v1"
            api_key = "new-api-key"
            "#,
    );
    mgr.apply_config(new_cfg)
        .expect("endpoint context reload should apply");

    {
        let cat = mgr.inner.catalog.read();
        assert_eq!(
            cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
            Some("endpoint-model"),
            "the endpoint owner must be retained as a pending refresh target",
        );
        assert!(!cat.model_endpoint_catalog_loaded);
        assert_eq!(cat.catalog_source, CatalogSource::Global);
        assert!(
            !cat.models.contains_key("provider-model"),
            "the returned slug is gone from the config-only catalog",
        );
    }

    assert!(
        mgr.refresh_current_model_endpoint_inner(true, None, None)
            .await,
        "the replacement fetch must still target the retained endpoint owner",
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        base_urls.lock().unwrap().as_slice(),
        ["https://new-provider.example/v1"],
        "the replacement must use the new endpoint, not the reselected model",
    );
    assert!(mgr.models().contains_key("provider-model"));
    assert!(mgr.inner.catalog.read().model_endpoint_catalog_loaded);
}

#[test]
fn apply_config_revalidates_pending_endpoint_owner() {
    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.models = resolve_model_catalog(&old_cfg, None);
        cat.has_fetched_real_catalog = false;
        cat.model_endpoint_catalog_loaded = false;
        cat.catalog_source = CatalogSource::Global;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }

    let new_cfg = config_from_toml("");
    mgr.apply_config(new_cfg.clone())
        .expect("removing the endpoint must apply");
    {
        let cat = mgr.inner.catalog.read();
        assert_eq!(
            cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
            None,
            "a pending owner whose endpoint was removed must be cleared",
        );
        assert_eq!(cat.catalog_source, CatalogSource::Global);
    }

    let generation = mgr.inner.catalog.read().generation;
    assert!(
        mgr.apply_refresh_result_fenced(
            &new_cfg,
            Some(make_prefetched(&["global-model"])),
            None,
            Some(generation),
            None,
            None,
            CatalogSource::Global,
            None,
        ),
        "a global refresh must no longer be rejected once the stale owner is gone",
    );
    assert!(mgr.models().contains_key("global-model"));
}

#[test]
fn apply_config_clears_pending_owner_when_selection_moves_to_another_source() {
    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"

            [model.regular-model]
            model = "regular-model"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }
    mgr.set_current_model_id(acp::ModelId::new("provider-model"));

    let new_cfg = config_from_toml(
        r#"
            [models]
            default = "regular-model"
            [model.endpoint-model]
            base_url = "https://new-provider.example/v1"
            api_key = "new-api-key"

            [model.regular-model]
            model = "regular-model"
            "#,
    );
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");

    let cat = mgr.inner.catalog.read();
    assert_eq!(
        cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
        None,
        "a pending endpoint owner must be cleared when the reload selects another source",
    );
    assert_eq!(cat.catalog_source, CatalogSource::Global);
    assert!(!cat.model_endpoint_catalog_loaded);
    drop(cat);
    assert_eq!(mgr.current_model_id().0.as_ref(), "regular-model");
    assert!(
        !mgr.current_model_has_endpoint(),
        "refreshes must resolve through the selected global model, not the stale endpoint",
    );
}

#[test]
fn apply_config_snapshots_current_model_with_catalog_lock() {
    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-a]
            base_url = "https://a.example/v1"
            api_key = "a-key"

            [model.endpoint-b]
            base_url = "https://b-old.example/v1"
            api_key = "b-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-a"),
        auth_manager,
        old_cfg,
    )
    .cache(test_cache_manager(tmp.path()))
    .build();

    let start_generation = mgr.inner.catalog.read().endpoint_generation;
    let cat_guard = mgr.inner.catalog.write();
    let apply_mgr = mgr.clone();
    let new_cfg = config_from_toml(
        r#"
            [model.endpoint-a]
            base_url = "https://a.example/v1"
            api_key = "a-key"

            [model.endpoint-b]
            base_url = "https://b-new.example/v1"
            api_key = "b-key"
            "#,
    );
    let apply_thread = std::thread::spawn(move || apply_mgr.apply_config(new_cfg));
    // Let apply_config take the fetch_auth lock and (in the old implementation)
    // snapshot the stale current model before it blocks on the catalog lock.
    std::thread::sleep(std::time::Duration::from_millis(100));
    {
        let mut current = mgr.inner.current_model_id.write();
        *current = acp::ModelId::new("endpoint-b");
        mgr.inner
            .model_switch_watch
            .send_modify(|generation| *generation += 1);
    }
    drop(cat_guard);
    apply_thread
        .join()
        .expect("apply_config thread panicked")
        .expect("config reload should apply");

    assert_eq!(mgr.current_model_id().0.as_ref(), "endpoint-b");
    assert!(
        mgr.inner.catalog.read().endpoint_generation > start_generation,
        "the endpoint fence must be bumped for the model that is current at publication time",
    );
}

#[test]
fn set_current_model_id_snapshots_config_under_catalog_lock() {
    let old_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"

            [model.other-model]
            model = "other-model"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }

    let cat_guard = mgr.inner.catalog.write();
    let switch_mgr = mgr.clone();
    let switch_thread = std::thread::spawn(move || {
        switch_mgr.set_current_model_id(acp::ModelId::new("other-model"));
    });
    while mgr.current_model_id().0.as_ref() != "other-model" {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let new_cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"

            [model.other-model]
            model = "other-model"

            [model.newly-added-model]
            model = "newly-added-model"
            "#,
    );
    *mgr.inner.cfg.write() = new_cfg;
    drop(cat_guard);
    switch_thread.join().expect("model switch thread panicked");

    assert_eq!(mgr.current_model_id().0.as_ref(), "other-model");
    assert!(
        mgr.models().contains_key("newly-added-model"),
        "a model switch must rebuild against the config committed while it waited for the catalog lock",
    );
}

#[tokio::test(start_paused = true)]
async fn list_models_online_if_uncached_fetches_during_retry_backoff() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FirstFailEndpoint {
        calls: Arc<AtomicUsize>,
    }
    impl ModelsEndpoint for FirstFailEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let catalog = make_prefetched(&["grok-4"]);
            Box::pin(async move { if n == 0 { None } else { Some(catalog) } })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            Box::pin(async { None })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let mgr = cold_manager(
        config_from_toml(
            "[endpoints]
deployment_key = \"deploy-key\"",
        ),
        Arc::new(FirstFailEndpoint {
            calls: calls.clone(),
        }),
    );
    mgr.spawn_catalog_retry_with_backoff(
        /*remote_fetch_enabled*/ true,
        crate::tools::retry::BackoffConfig::new(2, 60_000, 60_000),
    );

    // Let the first attempt fail and the retry task enter its backoff sleep.
    for _ in 0..1000 {
        if calls.load(Ordering::SeqCst) >= 1 && mgr.inner.active_fetch.load(Ordering::Acquire) == 0
        {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(mgr.inner.active_fetch.load(Ordering::Acquire), 0);

    mgr.list_models(RefreshStrategy::OnlineIfUncached).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "a sleeping retry task must not block a fresh models/list fetch",
    );
    assert!(mgr.has_fetched_real_catalog());
    assert!(mgr.models().contains_key("grok-4"));
}

#[test]
fn apply_config_rejected_reload_does_not_publish_fetch_auth() {
    let old_cfg = config_from_toml(
        r#"
            [endpoints]
            deployment_key = "deploy-key"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }
    assert_eq!(
        *mgr.inner.fetch_auth.read(),
        ModelFetchAuth::Deployment,
        "setup: the old config should resolve to deployment auth",
    );

    let new_cfg = config_from_toml(
        r#"
            [models]
            allowed_models = ["nomatch-*"]
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    assert!(
        mgr.apply_config(new_cfg).is_err(),
        "an allowlist that excludes every retained endpoint model must be rejected",
    );
    assert_eq!(
        *mgr.inner.fetch_auth.read(),
        ModelFetchAuth::Deployment,
        "a rejected reload must not publish its proposed fetch auth",
    );
    assert_eq!(
        mgr.inner.cfg.read().endpoints.deployment_key.as_deref(),
        Some("deploy-key"),
        "the rejected reload must leave manager config unchanged",
    );
}

// ── auth-change refresh: has_fetched_real_catalog flag ─────────────

#[test]
fn first_apply_refresh_reselects_default_model() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    assert!(!mgr.has_fetched_real_catalog());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    assert!(mgr.has_fetched_real_catalog());
    assert_eq!(mgr.current_model_id().0.as_ref(), "grok-3");
}

#[test]
fn subsequent_apply_refresh_preserves_user_model() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    mgr.inner.catalog.write().prefetched = None;
    mgr.inner.catalog.write().etag = None;

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4",
        "user's model selection must survive auth-change refresh"
    );
}

#[test]
fn subsequent_refresh_reselects_when_model_removed() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    let prefetched = make_prefetched(&["grok-3", "grok-4.5"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-3",
        "should fall back to config default when current is removed"
    );
}

#[test]
fn failed_refresh_does_not_set_has_fetched_real_catalog() {
    let mgr = test_manager();
    let cfg = config::Config::default();

    mgr.apply_refresh_result(&cfg, None, None);

    assert!(
        !mgr.has_fetched_real_catalog(),
        "failed refresh must not flip has_fetched_real_catalog"
    );
}

// ── apply_config: honor changed preferred model from config ────────

#[test]
fn apply_config_honors_new_preferred_model() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    let mut stale_cfg = config::Config::default();
    stale_cfg.models.default = None;
    *mgr.inner.cfg.write() = stale_cfg;

    let mut new_cfg = config::Config::default();
    new_cfg.models.default = Some("grok-3".to_string());
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-3",
        "apply_config must honor updated preferred model from config"
    );
}

#[test]
fn apply_config_preserves_current_when_preferred_unchanged() {
    let mgr = test_manager();
    let cfg = config::Config::default();

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    let new_cfg = config::Config::default();
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4",
        "apply_config must not reset model when preferred hasn't changed"
    );
}

#[test]
fn apply_config_falls_back_when_preferred_not_in_catalog() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    let mut new_cfg = config::Config::default();
    new_cfg.models.default = Some("grok-nonexistent".to_string());
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");

    let current = mgr.current_model_id();
    let first_available = mgr.available().keys().next().unwrap().clone();
    assert_eq!(
        current.0.as_ref(),
        first_available.0.as_ref(),
        "should fall back to first visible model when preferred not in catalog"
    );
}

#[test]
fn apply_config_both_none_preferred_preserves_current() {
    let mgr = test_manager();
    let cfg = config::Config::default();
    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    let new_cfg = config::Config::default();
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4",
        "both-None preferred must preserve user's runtime model"
    );
}

#[test]
fn apply_config_old_some_new_none_preserves_current() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    assert_eq!(mgr.current_model_id().0.as_ref(), "grok-3");

    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    let new_cfg = config::Config::default();
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4",
        "old=Some new=None must not reset model (is_some guard)"
    );
}

// ── end-to-end: auth refresh + config reload compose correctly ───

#[test]
fn auth_refresh_then_config_reload_preserves_user_model() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);

    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    mgr.inner.catalog.write().prefetched = None;
    mgr.inner.catalog.write().etag = None;

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    assert_eq!(mgr.current_model_id().0.as_ref(), "grok-4");

    let mut new_cfg = config::Config::default();
    new_cfg.models.default = Some("grok-4".to_string());
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");
    assert_eq!(mgr.current_model_id().0.as_ref(), "grok-4");
}

// ── disk-cache hot-reload (external models_cache.json writes) ────

fn test_cache_manager(dir: &std::path::Path) -> ModelsCacheManager {
    ModelsCacheManager {
        path: dir.join(MODELS_CACHE_FILE),
        ttl: CACHE_TTL,
    }
}

#[test]
fn reload_from_disk_cache_applies_external_catalog() {
    let mgr = test_manager();
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());

    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    cache.persist(
        &make_prefetched(&["grok-4.5", "grok-4.3"]),
        Some("etag-ext"),
        auth_method,
        &mgr.cache_origin(),
    );

    mgr.reload_from_cache_manager(&cache);

    assert!(mgr.has_fetched_real_catalog());
    assert!(mgr.models().contains_key("grok-4.5"));
    assert!(mgr.models().contains_key("grok-4.3"));
    assert_eq!(mgr.inner.catalog.read().etag.as_deref(), Some("etag-ext"));
}

#[test]
fn reload_from_disk_cache_recomputes_allowlist_excludes_all() {
    let mgr = test_manager();
    let cfg = config_from_toml("[models]\nallowed_models = [\"keep-*\"]");

    mgr.apply_refresh_result(&cfg, Some(make_prefetched(&["other-1"])), None);
    assert!(
        mgr.allowlist_excludes_all(),
        "setup: allowlist should exclude the entire catalog"
    );
    *mgr.inner.cfg.write() = cfg.clone();

    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    cache.persist(
        &make_prefetched(&["keep-1"]),
        Some("etag-keep"),
        auth_method,
        &mgr.cache_origin(),
    );

    mgr.reload_from_cache_manager(&cache);

    assert!(mgr.models().contains_key("keep-1"));
    assert!(
        !mgr.allowlist_excludes_all(),
        "corrective external cache write must unlatch the prompt block"
    );
}

#[test]
fn reload_from_disk_cache_resolves_default_on_first_catalog() {
    let mgr = test_manager();
    assert!(!mgr.has_fetched_real_catalog());
    let cfg = config_from_toml("[models]\ndefault = \"keep-1\"");
    *mgr.inner.cfg.write() = cfg.clone();

    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    cache.persist(
        &make_prefetched(&["keep-1", "other-1"]),
        Some("etag-first"),
        auth_method,
        &mgr.cache_origin(),
    );

    mgr.reload_from_cache_manager(&cache);

    assert!(mgr.has_fetched_real_catalog());
    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "keep-1",
        "first real catalog must resolve the configured default"
    );
}

#[test]
fn reload_from_disk_cache_skips_identical_catalog_and_adopts_etag() {
    let mgr = test_manager();
    let cfg = config::Config::default();
    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched.clone()), Some("etag-a".into()));
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    cache.persist(
        &prefetched,
        Some("etag-b"),
        auth_method,
        &mgr.cache_origin(),
    );

    mgr.reload_from_cache_manager(&cache);

    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4",
        "identical catalog must not disturb the user's model"
    );
    assert_eq!(
        mgr.inner.catalog.read().etag.as_deref(),
        Some("etag-b"),
        "etag should be adopted so refresh_if_new_etag stays accurate"
    );
}

#[test]
fn reload_from_disk_cache_ignores_stale_cache() {
    let mgr = test_manager();
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    let stale = ModelsCache {
        fetched_at: Utc::now() - ChronoDuration::seconds(3600),
        grok_version: Some(xai_grok_version::VERSION.to_string()),
        auth_method: Some(auth_method),
        origin: Some(mgr.cache_origin()),
        etag: Some("etag-stale".into()),
        models: make_prefetched(&["grok-stale"]),
    };
    cache.atomic_write(&stale);

    mgr.reload_from_cache_manager(&cache);

    assert!(!mgr.models().contains_key("grok-stale"));
    assert!(mgr.inner.catalog.read().etag.is_none());
}

#[test]
fn reload_from_disk_cache_ignores_auth_method_mismatch() {
    let mgr = test_manager();
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let current = mgr.inner.fetch_auth.read().cache_auth_method();
    let other = if current == CacheAuthMethod::Session {
        CacheAuthMethod::ApiKey
    } else {
        CacheAuthMethod::Session
    };
    cache.persist(
        &make_prefetched(&["grok-other-auth"]),
        Some("etag-x"),
        other,
        &mgr.cache_origin(),
    );

    mgr.reload_from_cache_manager(&cache);

    assert!(!mgr.models().contains_key("grok-other-auth"));
}

#[test]
fn reload_from_disk_cache_ignores_origin_mismatch() {
    let mgr = test_manager();
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    cache.persist(
        &make_prefetched(&["grok-other-origin"]),
        Some("etag-y"),
        auth_method,
        "http://127.0.0.1:49953/v1/models",
    );

    mgr.reload_from_cache_manager(&cache);

    assert!(!mgr.models().contains_key("grok-other-origin"));
    assert!(mgr.inner.catalog.read().etag.is_none());
}

#[test]
fn reload_from_disk_cache_ignores_legacy_cache_without_origin() {
    let mgr = test_manager();
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = test_cache_manager(tmp.path());
    let auth_method = mgr.inner.fetch_auth.read().cache_auth_method();
    let legacy = ModelsCache {
        fetched_at: Utc::now(),
        grok_version: Some(xai_grok_version::VERSION.to_string()),
        auth_method: Some(auth_method),
        origin: None,
        etag: Some("etag-legacy".into()),
        models: make_prefetched(&["grok-legacy"]),
    };
    cache.atomic_write(&legacy);

    mgr.reload_from_cache_manager(&cache);

    assert!(!mgr.models().contains_key("grok-legacy"));
}

// ── clear() resets has_fetched_real_catalog ──────────────────────

#[test]
fn clear_resets_has_fetched_real_catalog() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-3".to_string());

    let prefetched = make_prefetched(&["grok-3", "grok-4"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    assert!(mgr.has_fetched_real_catalog());

    mgr.clear();
    assert!(!mgr.has_fetched_real_catalog());

    let prefetched = make_prefetched(&["grok-4.5", "grok-4.3"]);
    mgr.apply_refresh_result(&cfg, Some(prefetched), None);
    let first_available = mgr.available().keys().next().unwrap().clone();
    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        first_available.0.as_ref()
    );
}

#[test]
fn is_campaign_only_flip_detects_campaign_driven_changes() {
    let camp: std::collections::HashSet<String> = ["beta".into()].into_iter().collect();
    assert!(is_campaign_only_flip(
        &Some("alpha".into()),
        &Some("beta".into()),
        &camp
    ));
    assert!(is_campaign_only_flip(
        &Some("beta".into()),
        &Some("alpha".into()),
        &camp
    ));
    assert!(!is_campaign_only_flip(
        &Some("alpha".into()),
        &Some("gamma".into()),
        &camp
    ));
    assert!(!is_campaign_only_flip(
        &Some("beta".into()),
        &Some("beta".into()),
        &camp
    ));
    assert!(!is_campaign_only_flip(&Some("beta".into()), &None, &camp));
    assert!(!is_campaign_only_flip(
        &Some("alpha".into()),
        &Some("beta".into()),
        &std::collections::HashSet::new()
    ));
}

#[test]
fn campaign_only_flip_does_not_reselect_live_session() {
    let mgr = test_manager();
    let mut cfg = config::Config::default();
    cfg.models.default = Some("alpha".to_string());
    mgr.apply_refresh_result(&cfg, Some(make_prefetched(&["alpha", "beta"])), None);
    *mgr.inner.cfg.write() = cfg.clone(); // old_preferred = "alpha"
    assert_eq!(mgr.current_model_id().0.as_ref(), "alpha");

    let mut new_cfg = config::Config::default();
    new_cfg.models.default = Some("beta".to_string());
    new_cfg.models.default_is_campaign_driven = true; // campaign overriding
    mgr.apply_config(new_cfg)
        .expect("config reload should apply");
    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "alpha",
        "campaign-only flip must not yank a still-selectable live session"
    );

    let mgr2 = test_manager();
    let mut cfg2 = config::Config::default();
    cfg2.models.default = Some("alpha".to_string());
    mgr2.apply_refresh_result(&cfg2, Some(make_prefetched(&["alpha", "beta"])), None);
    *mgr2.inner.cfg.write() = cfg2.clone();
    let mut new_cfg2 = config::Config::default();
    new_cfg2.models.default = Some("beta".to_string());
    mgr2.apply_config(new_cfg2)
        .expect("config reload should apply");
    assert_eq!(
        mgr2.current_model_id().0.as_ref(),
        "beta",
        "a non-campaign preferred change must reselect"
    );
}

#[test]
fn unavailable_campaign_default_falls_back_to_config_default() {
    let catalog = make_prefetched(&["real-model", "other-model"]);

    let mut cfg = config::Config::default();
    cfg.models.default = Some("missing-model".to_string());
    cfg.models.default_is_campaign_driven = true;
    cfg.models.pre_campaign_default = Some("real-model".to_string());
    let (key, _, _) = resolve_default_model(&cfg, &catalog, true);
    assert_eq!(
        key, "real-model",
        "must fall back to the pre-campaign default"
    );

    let mut cfg2 = config::Config::default();
    cfg2.models.default = Some("missing-model".to_string());
    cfg2.models.default_is_campaign_driven = true;
    cfg2.models.pre_campaign_default = Some("also-missing".to_string());
    let (key2, _, _) = resolve_default_model(&cfg2, &catalog, true);
    assert_eq!(&key2, catalog.keys().next().unwrap());

    let mut cfg3 = config::Config::default();
    cfg3.models.default = Some("missing-model".to_string());
    cfg3.models.pre_campaign_default = Some("real-model".to_string());
    let (key3, _, _) = resolve_default_model(&cfg3, &catalog, true);
    assert_eq!(
        &key3,
        catalog.keys().next().unwrap(),
        "non-campaign catalog miss must not recover via campaign state"
    );

    let mut cfg4 = config::Config {
        default_model_override: Some("missing-cli-model".to_string()),
        ..Default::default()
    };
    cfg4.models.default = Some("campaign-model".to_string());
    cfg4.models.default_is_campaign_driven = true;
    cfg4.models.pre_campaign_default = Some("real-model".to_string());
    let (key4, _, _) = resolve_default_model(&cfg4, &catalog, true);
    assert_eq!(
        &key4,
        catalog.keys().next().unwrap(),
        "a CLI pref miss must not detour through pre_campaign_default"
    );
}

// ── ModelFetchAuth::resolve priority tests ──────────────────────

use serial_test::serial;
use xai_grok_test_support::EnvGuard;

#[test]
#[serial]
fn resolve_custom_endpoint_always_wins() {
    let _key = EnvGuard::set("XAI_API_KEY", "test-key");
    let endpoints = config::EndpointsConfig {
        models_base_url: Some("https://custom.example.com".to_owned()),
        ..config::EndpointsConfig::default()
    };
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, true),
        ModelFetchAuth::CustomEndpoint,
    );
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, false),
        ModelFetchAuth::CustomEndpoint,
    );
}

#[test]
#[serial]
fn resolve_cached_session_wins_over_api_key() {
    let _key = EnvGuard::set("XAI_API_KEY", "test-key");
    let endpoints = config::EndpointsConfig::default();
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, true),
        ModelFetchAuth::Session,
        "cached session should take priority over API key",
    );
}

#[test]
#[serial]
fn resolve_api_key_used_when_no_session() {
    let _key = EnvGuard::set("XAI_API_KEY", "test-key");
    let endpoints = config::EndpointsConfig::default();
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, false),
        ModelFetchAuth::Session,
        "ambient XAI_API_KEY alone must not redirect the catalog endpoint; \
         model-owned api_key/env_key config controls model catalog auth",
    );
}

#[test]
#[serial]
fn resolve_falls_back_to_session_when_nothing_set() {
    let _unset = EnvGuard::unset("XAI_API_KEY");
    let _unset_legacy = EnvGuard::unset("GROK_CODE_XAI_API_KEY");
    let endpoints = config::EndpointsConfig::default();
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, false),
        ModelFetchAuth::Session,
        "should fall back to Session when nothing else is configured",
    );
}

#[test]
#[serial]
fn resolve_deployment_key_when_no_session_or_api_key() {
    let _unset = EnvGuard::unset("XAI_API_KEY");
    let _unset_legacy = EnvGuard::unset("GROK_CODE_XAI_API_KEY");
    let endpoints = config::EndpointsConfig {
        deployment_key: Some("deploy-key".to_owned()),
        ..config::EndpointsConfig::default()
    };
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, false),
        ModelFetchAuth::Deployment,
    );
}

#[test]
#[serial]
fn resolve_deployment_key_outranks_ambient_api_key() {
    let _key = EnvGuard::set("XAI_API_KEY", "stray-env-key");
    let endpoints = config::EndpointsConfig {
        deployment_key: Some("deploy-key".to_owned()),
        ..config::EndpointsConfig::default()
    };
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, false),
        ModelFetchAuth::Deployment,
        "managed deployment_key should outrank an ambient XAI_API_KEY",
    );
    assert_eq!(
        ModelFetchAuth::resolve(&endpoints, true),
        ModelFetchAuth::Session,
        "an active session should still win over a managed deployment",
    );
}

// ── remote_fetch gate: resolve_prefetch_env_from_parts ───────────

#[test]
#[serial]
fn prefetch_env_none_when_remote_fetch_disabled_despite_credentials() {
    let _key = EnvGuard::set("XAI_API_KEY", "stray-env-key");
    let endpoints = config::EndpointsConfig {
        deployment_key: Some("deploy-key".to_owned()),
        models_base_url: Some("https://custom.example.com".to_owned()),
        ..config::EndpointsConfig::default()
    };
    assert!(
        resolve_prefetch_env_from_parts(Some(GrokAuth::test_default()), endpoints.clone(), false,)
            .is_none(),
        "session auth must not re-arm the prefetch when remote_fetch is off",
    );
    assert!(
        resolve_prefetch_env_from_parts(None, endpoints, false).is_none(),
        "API key / deployment key / custom endpoint must not re-arm it either",
    );
}

#[test]
#[serial]
fn prefetch_env_resolves_when_remote_fetch_enabled() {
    let _unset = EnvGuard::unset("XAI_API_KEY");
    let _unset_legacy = EnvGuard::unset("GROK_CODE_XAI_API_KEY");
    let endpoints = config::EndpointsConfig {
        deployment_key: Some("deploy-key".to_owned()),
        ..config::EndpointsConfig::default()
    };
    assert!(resolve_prefetch_env_from_parts(None, endpoints, true).is_some());
    assert!(
        resolve_prefetch_env_from_parts(None, config::EndpointsConfig::default(), true).is_none(),
        "no credentials and no custom endpoint must stay a no-prefetch launch",
    );
}

#[tokio::test]
async fn fetch_and_apply_degrades_offline_when_remote_fetch_disabled() {
    let mgr = test_manager();
    mgr.insert_test_entry(
        "static-one",
        ModelEntry {
            info: config::ModelInfo::fallback("static-one"),
            api_key: None,
            env_key: None,
            auth_provider: None,
            api_base_url: None,
        },
    );

    mgr.fetch_and_apply_inner(false).await;

    assert!(
        !mgr.has_fetched_real_catalog(),
        "no catalog fetch may be recorded when remote_fetch is disabled",
    );
    assert!(
        mgr.models().contains_key("static-one"),
        "the static catalog must keep resolving",
    );
}

// ── supported_in_api tests ──────────────────────────────────────

#[test]
fn default_model_skips_oauth_only_for_api_key_users() {
    let cfg = config::Config::default();
    let mut catalog = IndexMap::new();

    let mut oauth_only = ModelEntry {
        info: config::ModelInfo::fallback("oauth-only"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    oauth_only.info.supported_in_api = false;
    catalog.insert("oauth-only".to_string(), oauth_only);

    let public = ModelEntry {
        info: config::ModelInfo::fallback("public-model"),
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    };
    catalog.insert("public-model".to_string(), public);

    let (key, _, _) = resolve_default_model(&cfg, &catalog, false);
    assert_ne!(
        key, "oauth-only",
        "API-key default must not be an OAuth-only model"
    );
    assert_eq!(key, "public-model");

    let (key, _, _) = resolve_default_model(&cfg, &catalog, true);
    assert!(
        key == "oauth-only" || key == "public-model",
        "OAuth user should be able to use either model as default"
    );
}

#[test]
fn visible_for_auth_logic() {
    let mut info = config::ModelInfo::fallback("test");

    assert!(info.visible_for_auth(true));
    assert!(info.visible_for_auth(false));

    info.hidden = true;
    assert!(!info.visible_for_auth(true));
    assert!(!info.visible_for_auth(false));

    info.hidden = false;
    info.supported_in_api = false;
    assert!(info.visible_for_auth(true));
    assert!(!info.visible_for_auth(false));
}

// ── duplicate model slug re-keying (A/B experiment "auto" alias) ──

fn make_entry_config(model: &str, name: Option<&str>) -> config::ModelEntryConfig {
    make_entry_config_with_id(None, model, name)
}

fn make_entry_config_with_id(
    id: Option<&str>,
    model: &str,
    name: Option<&str>,
) -> config::ModelEntryConfig {
    config::ModelEntryConfig {
        id: id.map(|s| s.to_owned()),
        model: model.to_owned(),
        base_url: "https://test.api/v1".to_owned(),
        name: name.map(|n| n.to_owned()),
        description: None,
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        api_key: None,
        env_key: None,
        api_backend: None,
        context_window: std::num::NonZeroU64::new(200_000).unwrap(),
        auto_compact_threshold_percent: None,
        system_prompt_label: None,
        extra_headers: IndexMap::new(),
        api_base_url: None,
        use_concise: false,
        agent_type: config::default_agent_type(),
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
        laziness_detector: config::LazinessDetectorPerModelConfig::default(),
    }
}

#[test]
fn build_prefetched_map_distinct_ids_same_slug() {
    let entries = vec![
        make_entry_config_with_id(Some("auto"), "grok-build", Some("Auto")),
        make_entry_config_with_id(Some("grok-build"), "grok-build", Some("Grok Build")),
        make_entry_config_with_id(
            Some("experimental-fast"),
            "experimental-fast",
            Some("Grok Fast"),
        ),
    ];
    let map = build_prefetched_map(entries, None);

    assert_eq!(map.len(), 3, "all three entries should survive");
    assert!(map.contains_key("auto"));
    assert!(map.contains_key("grok-build"));
    assert!(map.contains_key("experimental-fast"));
    assert_eq!(
        map["auto"].info.model, "grok-build",
        "auto entry should still route to grok-build"
    );
    assert_eq!(map["grok-build"].info.model, "grok-build");
}

#[test]
fn build_prefetched_map_no_id_falls_back_to_slug() {
    let entries = vec![
        make_entry_config("model-a", Some("Model A")),
        make_entry_config("model-b", Some("Model B")),
    ];
    let map = build_prefetched_map(entries, None);

    assert_eq!(map.len(), 2);
    assert!(map.contains_key("model-a"));
    assert!(map.contains_key("model-b"));
}

#[test]
fn build_prefetched_map_duplicate_id_overwrites() {
    let entries = vec![
        make_entry_config_with_id(Some("grok-build"), "grok-build", Some("First")),
        make_entry_config_with_id(Some("grok-build"), "grok-build", Some("Second")),
    ];
    let map = build_prefetched_map(entries, None);

    assert_eq!(map.len(), 1, "duplicate id: second overwrites first");
    assert_eq!(map["grok-build"].info.name.as_deref(), Some("Second"));
}

#[test]
fn resolve_default_model_prefers_id_over_model_slug() {
    let mut catalog: IndexMap<String, ModelEntry> = IndexMap::new();
    catalog.insert(
        "auto-grok-build".to_string(),
        make_model_entry("grok-build"),
    );
    catalog.insert("grok-build".to_string(), make_model_entry("grok-build"));

    let mut cfg = config::Config::default();
    cfg.models.default = Some("grok-build".to_string());

    let (key, _, _) = resolve_default_model(&cfg, &catalog, true);
    assert_eq!(key, "grok-build", "must match id, not first slug hit");
}

#[test]
fn build_prefetched_map_none_id_falls_back_to_slug() {
    let entries = vec![make_entry_config_with_id(
        None,
        "grok-build",
        Some("Grok Build"),
    )];
    let map = build_prefetched_map(entries, None);

    assert_eq!(map.len(), 1);
    assert!(map.contains_key("grok-build"));
}

// ── persisted model id → catalog key (session resume) ─────────────

#[test]
fn resolve_catalog_key_maps_routing_slug_to_config_key() {
    let mut models = IndexMap::new();
    models.insert(
        "enterprise-grok-build".to_string(),
        make_model_entry("grok-4.5"),
    );
    models.insert("grok-4.3".to_string(), make_model_entry("grok-4.3"));

    let persisted = acp::ModelId::new("grok-4.5");
    let key = resolve_catalog_key(&models, &persisted).expect("slug must resolve");
    assert_eq!(key.0.as_ref(), "enterprise-grok-build");
}

#[test]
fn resolve_catalog_key_prefers_exact_key_match() {
    let mut models = IndexMap::new();
    models.insert("grok-4.5".to_string(), make_model_entry("grok-4.5"));

    let persisted = acp::ModelId::new("grok-4.5");
    let key = resolve_catalog_key(&models, &persisted).expect("exact key must resolve");
    assert_eq!(key.0.as_ref(), "grok-4.5");
}

#[test]
fn resolve_catalog_key_last_slug_match_wins() {
    let mut models = IndexMap::new();
    models.insert(
        "default-grok-build".to_string(),
        make_model_entry("grok-4.5"),
    );
    models.insert("user-grok-build".to_string(), make_model_entry("grok-4.5"));

    let persisted = acp::ModelId::new("grok-4.5");
    let key = resolve_catalog_key(&models, &persisted).expect("slug must resolve");
    assert_eq!(key.0.as_ref(), "user-grok-build");
}

#[test]
fn selectable_catalog_key_for_persisted_none_when_resolved_not_available() {
    let mut models = IndexMap::new();
    models.insert(
        "enterprise-grok-build".to_string(),
        make_model_entry("grok-4.5"),
    );

    let available: IndexMap<_, _> = IndexMap::new();
    let persisted = acp::ModelId::new("grok-4.5");
    assert!(selectable_catalog_key_for_persisted(&models, &available, &persisted).is_none());
}

#[test]
fn selectable_prefers_available_identity_over_non_selectable_exact_key() {
    let mut models = IndexMap::new();
    models.insert("grok-build".to_string(), make_model_entry("grok-build"));
    models.insert(
        "enterprise-grok-build".to_string(),
        make_model_entry("grok-build"),
    );
    models.insert("grok-4.3".to_string(), make_model_entry("grok-4.3"));

    let available = test_available_keys(&["enterprise-grok-build", "grok-4.3"]);

    let persisted = acp::ModelId::new("grok-build");
    assert_eq!(
        resolve_catalog_key(&models, &persisted)
            .expect("exact key exists")
            .0
            .as_ref(),
        "grok-build"
    );
    let key = selectable_catalog_key_for_persisted(&models, &available, &persisted)
        .expect("must resolve to selectable section");
    assert_eq!(key.0.as_ref(), "enterprise-grok-build");
}

#[test]
fn selectable_matches_routing_slug_when_no_exact_key() {
    let mut models = IndexMap::new();
    models.insert(
        "enterprise-grok-build".to_string(),
        make_model_entry("grok-build"),
    );
    models.insert("grok-4.3".to_string(), make_model_entry("grok-4.3"));

    let available = test_available_keys(&["enterprise-grok-build", "grok-4.3"]);

    let persisted = acp::ModelId::new("grok-build");
    let key = selectable_catalog_key_for_persisted(&models, &available, &persisted)
        .expect("slug must resolve to selectable key");
    assert_eq!(key.0.as_ref(), "enterprise-grok-build");
}

#[test]
fn selectable_prefers_exact_key_over_later_slug_match() {
    let mut models = IndexMap::new();
    models.insert("grok-build".to_string(), make_model_entry("grok-4.5"));
    models.insert("other".to_string(), make_model_entry("grok-build"));

    let available = test_available_keys(&["grok-build", "other"]);

    let persisted = acp::ModelId::new("grok-build");
    let key = selectable_catalog_key_for_persisted(&models, &available, &persisted)
        .expect("exact selectable key must win");
    assert_eq!(key.0.as_ref(), "grok-build");
}

fn test_available_keys(keys: &[&str]) -> IndexMap<acp::ModelId, acp::ModelInfo> {
    keys.iter()
        .map(|k| {
            let id = acp::ModelId::new(*k);
            (id.clone(), acp::ModelInfo::new(id, (*k).to_string()))
        })
        .collect()
}

#[tokio::test(start_paused = true)]
async fn bounded_auth_refresh_times_out_to_none() {
    // A hung IdP (never-ready auth future) must degrade to None within the
    // bound so a cold-cache boot fetch can't stall on it.
    let started = tokio::time::Instant::now();
    let result =
        ModelsManager::bounded_auth_refresh(std::future::pending::<Option<GrokAuth>>()).await;
    assert!(result.is_none(), "a hung auth refresh must yield None");
    assert!(
        started.elapsed() >= crate::http::STARTUP_AUTH_REFRESH_TIMEOUT,
        "must wait the full bound before giving up",
    );
}

#[tokio::test]
async fn bounded_auth_refresh_passes_through_ready_value() {
    let result =
        ModelsManager::bounded_auth_refresh(async { Some(GrokAuth::test_default()) }).await;
    assert!(
        result.is_some(),
        "a ready session must pass through unchanged"
    );
}

#[tokio::test]
async fn explicit_model_pick_survives_first_real_catalog() {
    // Non-blocking boot lets the user pick a model before the first real
    // catalog lands; that pick must not be clobbered by default reselection.
    let mgr = test_manager();
    let cfg = config_from_toml("[models]\ndefault = \"grok-4.5\"");
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    mgr.apply_refresh_result(&cfg, Some(make_prefetched(&["grok-4.5", "grok-4"])), None);
    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4",
        "an explicit /model pick must survive the first real catalog",
    );
}

#[tokio::test]
async fn identity_switch_clears_user_pick_latch() {
    // After an identity change (`clear()`), the new identity's first catalog must
    // reselect its own default rather than inherit the prior user's pick.
    let mgr = test_manager();
    let cfg = config_from_toml("[models]\ndefault = \"grok-4.5\"");
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    mgr.clear();
    mgr.apply_refresh_result(&cfg, Some(make_prefetched(&["grok-4.5", "grok-4"])), None);
    assert_eq!(
        mgr.current_model_id().0.as_ref(),
        "grok-4.5",
        "a new identity's first catalog must reselect the default after clear()",
    );
}

#[tokio::test]
async fn etag_refresh_waits_for_active_fetch_when_catalog_already_ready() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FirstBlockingGlobalEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for FirstBlockingGlobalEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["grok-4"]);
            Box::pin(async move {
                if n == 0 {
                    started.notify_one();
                    release.notified().await;
                }
                Some(catalog)
            })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            Box::pin(async { None })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = cold_manager(
        config_from_toml("[endpoints]\ndeployment_key = \"deploy-key\""),
        Arc::new(FirstBlockingGlobalEndpoint {
            calls: calls.clone(),
            started: started.clone(),
            release: release.clone(),
        }),
    );
    let cfg = config::Config::default();
    assert!(
        mgr.apply_refresh_result(
            &cfg,
            Some(make_prefetched(&["grok-4"])),
            Some("\"etag-old\"".into())
        ),
        "seeding a real catalog should succeed"
    );
    assert!(mgr.has_fetched_real_catalog());

    let fetch_mgr = mgr.clone();
    let fetch_task = tokio::spawn(async move {
        fetch_mgr.fetch_and_apply_inner(true).await;
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the global fetch never reached the transport");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let etag_mgr = mgr.clone();
    let etag_task = tokio::spawn(async move {
        etag_mgr
            .refresh_if_new_etag("\"etag-new\"".to_string())
            .await;
    });
    tokio::task::yield_now().await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the etag refresh must wait for the active fetch instead of recursing while it is registered",
    );

    release.notify_one();
    fetch_task.await.unwrap();
    etag_task.await.unwrap();

    // The join/replay now runs in a background task, so wait for the replayed
    // fetch instead of assuming it completed with `refresh_if_new_etag`.
    let mut replayed = false;
    for _ in 0..200 {
        if mgr.inner.catalog.read().etag.as_deref() == Some("\"etag-new\"") {
            replayed = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(
        replayed,
        "after the active fetch finishes, the etag change must be replayed",
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "replaying the etag change must issue a fresh fetch",
    );
    assert_eq!(
        mgr.inner.catalog.read().etag.as_deref(),
        Some("\"etag-new\""),
    );
    assert!(mgr.has_fetched_real_catalog());
    assert!(mgr.models().contains_key("grok-4"));
}

#[tokio::test]
async fn etag_refresh_does_not_block_sampling_path_behind_active_fetch() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FirstBlockingGlobalEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for FirstBlockingGlobalEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["grok-4"]);
            Box::pin(async move {
                if n == 0 {
                    started.notify_one();
                    release.notified().await;
                }
                Some(catalog)
            })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            Box::pin(async { None })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = cold_manager(
        config_from_toml("[endpoints]\ndeployment_key = \"deploy-key\""),
        Arc::new(FirstBlockingGlobalEndpoint {
            calls: calls.clone(),
            started: started.clone(),
            release: release.clone(),
        }),
    );
    let cfg = config::Config::default();
    assert!(
        mgr.apply_refresh_result(
            &cfg,
            Some(make_prefetched(&["grok-4"])),
            Some("\"etag-old\"".into())
        ),
        "seeding a real catalog should succeed"
    );
    assert!(mgr.has_fetched_real_catalog());

    let fetch_mgr = mgr.clone();
    let fetch_task = tokio::spawn(async move {
        fetch_mgr.fetch_and_apply_inner(true).await;
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the global fetch never reached the transport");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // The sampling-event drainer calls `refresh_if_new_etag` inline; it must
    // return promptly while the active fetch is still registered instead of
    // waiting out the full auth+fetch startup bounds.
    tokio::time::timeout(
        std::time::Duration::from_millis(100),
        mgr.refresh_if_new_etag("\"etag-new\"".to_string()),
    )
    .await
    .expect("etag refresh must not block the sampling path behind an active fetch");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the etag refresh must join, not race, the active fetch",
    );

    release.notify_one();
    fetch_task.await.unwrap();

    let mut replayed = false;
    for _ in 0..200 {
        if mgr.inner.catalog.read().etag.as_deref() == Some("\"etag-new\"") {
            replayed = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(
        replayed,
        "the background replay must complete after the active fetch",
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "replaying the etag change must issue a fresh fetch",
    );
    assert_eq!(
        mgr.inner.catalog.read().etag.as_deref(),
        Some("\"etag-new\""),
    );
    assert!(mgr.models().contains_key("grok-4"));
}

#[tokio::test]
async fn list_models_refetches_after_switching_between_endpoint_owners() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EndpointAThenB {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for EndpointAThenB {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if request.base_url == "https://a.example/v1" {
                let started = self.started.clone();
                let release = self.release.clone();
                let catalog = make_prefetched(&["provider-a"]);
                Box::pin(async move {
                    started.notify_one();
                    release.notified().await;
                    Some((catalog, None))
                })
            } else {
                let _ = n;
                Box::pin(async { Some((make_prefetched(&["provider-b"]), None)) })
            }
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-a]
            base_url = "https://a.example/v1"
            api_key = "a-key"

            [model.endpoint-b]
            base_url = "https://b.example/v1"
            api_key = "b-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-a"),
        auth_manager,
        cfg,
    )
    .endpoint(Arc::new(EndpointAThenB {
        calls: calls.clone(),
        started: started.clone(),
        release: release.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();

    let list_mgr = mgr.clone();
    let list_task = tokio::spawn(async move {
        list_mgr
            .list_models(RefreshStrategy::OnlineIfUncached)
            .await;
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the endpoint-a fetch never reached the transport");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    mgr.set_current_model_id(acp::ModelId::new("endpoint-b"));
    release.notify_one();
    list_task.await.unwrap();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "after switching to another endpoint owner, models/list must fetch the new owner's catalog",
    );
    let cat = mgr.inner.catalog.read();
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert!(cat.model_endpoint_catalog_loaded);
    assert_eq!(
        cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
        Some("endpoint-b"),
    );
    assert!(cat.models.contains_key("provider-b"));
}

#[tokio::test]
async fn list_models_refetches_when_reserved_fetch_generation_goes_stale() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FirstBlockingGlobalEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for FirstBlockingGlobalEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["grok-4"]);
            Box::pin(async move {
                if n == 0 {
                    started.notify_one();
                    release.notified().await;
                    None
                } else {
                    Some(catalog)
                }
            })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            Box::pin(async { None })
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = cold_manager(
        config_from_toml("[endpoints]\ndeployment_key = \"deploy-key\""),
        Arc::new(FirstBlockingGlobalEndpoint {
            calls: calls.clone(),
            started: started.clone(),
            release: release.clone(),
        }),
    );

    let list_mgr = mgr.clone();
    let list_task = tokio::spawn(async move {
        list_mgr
            .list_models(RefreshStrategy::OnlineIfUncached)
            .await;
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the reserved fetch never reached the transport");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // A config reload advances the generation while the reserved request is in
    // flight, so its result cannot publish for the current config.
    mgr.clear();
    release.notify_one();
    list_task.await.unwrap();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "models/list must start a fresh fetch when the reserved fetch could not publish",
    );
    assert!(mgr.has_fetched_real_catalog());
    assert!(mgr.models().contains_key("grok-4"));
}

#[tokio::test]
async fn endpoint_refresh_survives_switch_to_sibling_model() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SlowEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for SlowEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["endpoint-model", "provider-sibling"]);
            Box::pin(async move {
                started.notify_one();
                release.notified().await;
                Some((catalog, Some("\"etag-new\"".to_string())))
            })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .endpoint(Arc::new(SlowEndpoint {
        calls: calls.clone(),
        started: started.clone(),
        release: release.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["endpoint-model", "provider-sibling"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
        cat.etag = Some("\"etag-old\"".to_string());
    }

    let refresh_mgr = mgr.clone();
    let refresh_task = tokio::spawn(async move {
        refresh_mgr
            .refresh_current_model_endpoint_inner(true, Some("\"etag-new\"".into()), None)
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the endpoint fetch never reached the transport");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Selecting another model returned by the same endpoint keeps the catalog
    // owner, so the in-flight refresh must still be able to publish.
    mgr.set_current_model_id(acp::ModelId::new("provider-sibling"));
    release.notify_one();
    assert!(refresh_task.await.unwrap());

    let cat = mgr.inner.catalog.read();
    assert_eq!(cat.catalog_source, CatalogSource::ModelEndpoint);
    assert_eq!(
        cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
        Some("endpoint-model"),
    );
    assert_eq!(cat.etag.as_deref(), Some("\"etag-new\""));
    assert!(cat.models.contains_key("provider-sibling"));
}

#[tokio::test]
async fn apply_config_keeps_pending_endpoint_owner_across_automatic_fallback() {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RecordingEndpoint {
        calls: Arc<AtomicUsize>,
        base_urls: Arc<Mutex<Vec<String>>>,
    }
    impl ModelsEndpoint for RecordingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.base_urls
                .lock()
                .unwrap()
                .push(request.base_url.clone());
            Box::pin(async { Some((make_prefetched(&["provider-model"]), None)) })
        }
    }

    let old_cfg = config_from_toml(
        r#"
            [models]
            default = "grok-4"

            [model.endpoint-model]
            model = "provider-model"
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"

            [model.grok-4]
            model = "grok-4"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let base_urls = Arc::new(Mutex::new(Vec::new()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .endpoint(Arc::new(RecordingEndpoint {
        calls: calls.clone(),
        base_urls: base_urls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }
    mgr.set_current_model_id(acp::ModelId::new("provider-model"));

    let new_cfg = config_from_toml(
        r#"
            [models]
            default = "grok-4"

            [model.endpoint-model]
            model = "provider-model"
            base_url = "https://new-provider.example/v1"
            api_key = "new-api-key"

            [model.grok-4]
            model = "grok-4"
            "#,
    );
    mgr.apply_config(new_cfg)
        .expect("endpoint context reload should apply");

    {
        let cat = mgr.inner.catalog.read();
        assert_eq!(mgr.current_model_id().0.as_ref(), "grok-4");
        assert_eq!(
            cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
            Some("endpoint-model"),
            "an automatic fallback after endpoint invalidation must keep the pending owner",
        );
        assert!(!cat.model_endpoint_catalog_loaded);
        assert_eq!(cat.catalog_source, CatalogSource::Global);
    }

    assert!(
        mgr.current_model_has_endpoint(),
        "the pending owner must keep the replacement refresh on the endpoint",
    );
    assert!(
        mgr.refresh_current_model_endpoint_inner(true, None, None)
            .await,
        "the replacement fetch must still target the retained endpoint owner",
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        base_urls.lock().unwrap().as_slice(),
        ["https://new-provider.example/v1"],
        "the replacement must use the new endpoint, not the reselected fallback model",
    );
    assert!(mgr.models().contains_key("provider-model"));
    assert!(mgr.inner.catalog.read().model_endpoint_catalog_loaded);
}

#[tokio::test]
async fn endpoint_refresh_discarded_through_global_catalog_siblings() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct BlockingEndpoint {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl ModelsEndpoint for BlockingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, _request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let started = self.started.clone();
            let release = self.release.clone();
            let catalog = make_prefetched(&["provider-model"]);
            Box::pin(async move {
                started.notify_one();
                release.notified().await;
                Some((catalog, None))
            })
        }
    }

    let cfg = config_from_toml(
        r#"
            [model.endpoint-model]
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        cfg.clone(),
    )
    .endpoint(Arc::new(BlockingEndpoint {
        calls: calls.clone(),
        started: started.clone(),
        release: release.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    // A prior global catalog load (for example, after an initial endpoint
    // failure) populated `prefetched` with global models while the endpoint
    // refresh is still in flight.
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["grok-4"]));
        cat.models = resolve_model_catalog(&cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = false;
        cat.catalog_source = CatalogSource::Global;
        cat.catalog_owner = None;
    }

    let refresh_mgr = mgr.clone();
    let refresh_task = tokio::spawn(async move {
        refresh_mgr
            .refresh_current_model_endpoint_inner(true, None, None)
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .expect("the endpoint fetch never reached the transport");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Switching to a model that happens to be in the global catalog must not
    // make the in-flight endpoint response a "sibling" of the endpoint owner.
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));
    release.notify_one();
    assert!(
        !refresh_task.await.unwrap(),
        "an endpoint result must not apply through a global catalog's prefetched entries",
    );

    let cat = mgr.inner.catalog.read();
    assert_eq!(cat.catalog_source, CatalogSource::Global);
    assert!(!cat.model_endpoint_catalog_loaded);
    assert_eq!(
        cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
        None,
        "no endpoint owner may be latched from the discarded response",
    );
    assert!(cat.models.contains_key("grok-4"));
    assert!(
        !cat.models.contains_key("provider-model"),
        "the endpoint response must not replace the global catalog",
    );
    drop(cat);
}

#[tokio::test]
async fn explicit_reselection_clears_pending_owner_after_automatic_fallback() {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RecordingEndpoint {
        calls: Arc<AtomicUsize>,
        base_urls: Arc<Mutex<Vec<String>>>,
    }
    impl ModelsEndpoint for RecordingEndpoint {
        fn fetch_models(
            &self,
            _endpoints: config::EndpointsConfig,
            _auth: Option<GrokAuth>,
            _fetch_auth: ModelFetchAuth,
        ) -> ModelsFetchFuture {
            Box::pin(async { None })
        }

        fn fetch_model_endpoint(&self, request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.base_urls
                .lock()
                .unwrap()
                .push(request.base_url.clone());
            Box::pin(async { Some((make_prefetched(&["provider-model"]), None)) })
        }
    }

    let old_cfg = config_from_toml(
        r#"
            [models]
            default = "grok-4"

            [model.endpoint-model]
            model = "provider-model"
            base_url = "https://provider.example/v1"
            api_key = "model-api-key"

            [model.grok-4]
            model = "grok-4"
            "#,
    );
    let tmp = tempfile::TempDir::new().unwrap();
    let auth_manager = Arc::new(AuthManager::new(tmp.path(), GrokComConfig::default()));
    let calls = Arc::new(AtomicUsize::new(0));
    let base_urls = Arc::new(Mutex::new(Vec::new()));
    let mgr = ModelsManagerBuilder::new(
        None,
        resolve_model_catalog(&old_cfg, None),
        acp::ModelId::new("endpoint-model"),
        auth_manager,
        old_cfg.clone(),
    )
    .endpoint(Arc::new(RecordingEndpoint {
        calls: calls.clone(),
        base_urls: base_urls.clone(),
    }))
    .cache(test_cache_manager(tmp.path()))
    .build();
    {
        let mut cat = mgr.inner.catalog.write();
        cat.prefetched = Some(make_prefetched(&["provider-model"]));
        cat.models = resolve_model_catalog(&old_cfg, cat.prefetched.clone());
        cat.has_fetched_real_catalog = true;
        cat.model_endpoint_catalog_loaded = true;
        cat.catalog_source = CatalogSource::ModelEndpoint;
        cat.catalog_owner = Some(acp::ModelId::new("endpoint-model"));
    }
    mgr.set_current_model_id(acp::ModelId::new("provider-model"));

    let new_cfg = config_from_toml(
        r#"
            [models]
            default = "grok-4"

            [model.endpoint-model]
            model = "provider-model"
            base_url = "https://new-provider.example/v1"
            api_key = "new-api-key"

            [model.grok-4]
            model = "grok-4"
            "#,
    );
    mgr.apply_config(new_cfg)
        .expect("endpoint context reload should apply");

    {
        let cat = mgr.inner.catalog.read();
        assert_eq!(mgr.current_model_id().0.as_ref(), "grok-4");
        assert_eq!(
            cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
            Some("endpoint-model"),
            "the automatic fallback must still retain the pending owner before an explicit pick",
        );
    }

    // Explicitly re-picking the already-current fallback is a selection from
    // another source: it must drop the stale pending endpoint owner so
    // refreshes target the global catalog instead.
    mgr.set_current_model_id(acp::ModelId::new("grok-4"));

    {
        let cat = mgr.inner.catalog.read();
        assert_eq!(
            cat.catalog_owner.as_ref().map(|o| o.0.as_ref()),
            None,
            "an explicit pick of the fallback must clear the retained pending owner",
        );
        assert_eq!(cat.catalog_source, CatalogSource::Global);
    }
    assert!(
        !mgr.current_model_has_endpoint(),
        "without the pending owner the current model must not route through the old endpoint",
    );
}
