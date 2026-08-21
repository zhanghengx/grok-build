use super::*;

/// Boxed future returned by [`ModelsEndpoint::fetch_models`].
pub(crate) type ModelsFetchFuture =
    Pin<Box<dyn Future<Output = Option<IndexMap<String, ModelEntry>>> + Send>>;

/// Boxed future returned by [`ModelsEndpoint::fetch_model_endpoint`]. The
/// resolved prefetched map is paired with the response `ETag`, if any.
pub(crate) type ModelEndpointFetchFuture =
    Pin<Box<dyn Future<Output = Option<(IndexMap<String, ModelEntry>, Option<String>)>> + Send>>;

/// Request context for a model-owned `/models` endpoint.
///
/// The credential is intentionally explicit so this request cannot
/// accidentally inherit the Grok session token used by the global catalog.
#[derive(Clone)]
pub(crate) struct ModelEndpointRequest {
    pub(crate) base_url: String,
    /// The resolved credential used for this one request. This is not copied
    /// into the returned catalog when the source is an auth provider/env var.
    pub(crate) api_key: String,
    pub(crate) api_backend: ApiBackend,
    pub(crate) auth_scheme: xai_grok_sampler::AuthScheme,
    pub(crate) configured_api_key: Option<String>,
    pub(crate) configured_env_key: Option<config::EnvKeys>,
    pub(crate) auth_provider: Option<crate::auth::AuthProviderRef>,
    pub(crate) extra_headers: indexmap::IndexMap<String, String>,
    pub(crate) query_params: indexmap::IndexMap<String, String>,
    pub(crate) env_http_headers: indexmap::IndexMap<String, String>,
}

/// Injectable `/v1/models` transport; tests inject a fake.
pub(crate) trait ModelsEndpoint: Send + Sync {
    fn fetch_models(
        &self,
        endpoints: config::EndpointsConfig,
        auth: Option<GrokAuth>,
        fetch_auth: ModelFetchAuth,
    ) -> ModelsFetchFuture;

    /// Fetch the catalog from a model-specific endpoint. The default keeps
    /// existing test transports source-compatible; production uses HTTP.
    fn fetch_model_endpoint(&self, request: ModelEndpointRequest) -> ModelEndpointFetchFuture {
        Box::pin(fetch_model_endpoint_async(request))
    }
}

/// Default transport: the real `/v1/models` fetch.
pub(crate) struct HttpModelsEndpoint;

impl ModelsEndpoint for HttpModelsEndpoint {
    fn fetch_models(
        &self,
        endpoints: config::EndpointsConfig,
        auth: Option<GrokAuth>,
        fetch_auth: ModelFetchAuth,
    ) -> ModelsFetchFuture {
        Box::pin(fetch_models_async(endpoints, auth, fetch_auth))
    }
}

pub(crate) async fn fetch_models_async(
    endpoints: config::EndpointsConfig,
    auth: Option<GrokAuth>,
    fetch_auth: ModelFetchAuth,
) -> Option<IndexMap<String, ModelEntry>> {
    tokio::task::spawn_blocking(move || {
        prefetch_models_blocking(&endpoints, auth.as_ref(), fetch_auth)
    })
    .await
    .unwrap_or(None)
}

pub(crate) async fn fetch_model_endpoint_async(
    request: ModelEndpointRequest,
) -> Option<(IndexMap<String, ModelEntry>, Option<String>)> {
    tokio::task::spawn_blocking(move || {
        let result = crate::remote::client::fetch_model_models_blocking(
            &request.base_url,
            &request.api_key,
            request.auth_scheme,
            &request.extra_headers,
            &request.query_params,
            &request.env_http_headers,
        )
        .ok()?;
        if result.models.is_empty() {
            tracing::warn!("Model-specific models endpoint returned an empty list");
            return None;
        }
        Some((
            build_prefetched_map_with_model_context(result.models, &request),
            result.etag,
        ))
    })
    .await
    .unwrap_or(None)
}

pub(crate) fn build_prefetched_map_with_model_context(
    models: Vec<config::ModelEntryConfig>,
    request: &ModelEndpointRequest,
) -> IndexMap<String, ModelEntry> {
    let mut map = IndexMap::with_capacity(models.len());
    for model in models {
        let key = model.id.clone().unwrap_or_else(|| model.model.clone());
        let mut info = config::ModelInfo::from_config(&model);
        info.auth_scheme = request.auth_scheme;
        if let Some(api_backend) = model.api_backend.as_ref() {
            info.api_backend = api_backend.clone();
        } else {
            info.api_backend = request.api_backend.clone();
        }

        // The configured endpoint headers are part of the model's connection
        // context. Config wins over metadata returned by the endpoint.
        for (name, value) in &request.extra_headers {
            if let Some(existing) = info
                .extra_headers
                .keys()
                .find(|existing| existing.eq_ignore_ascii_case(name))
                .cloned()
            {
                info.extra_headers.shift_remove(&existing);
            }
            info.extra_headers.insert(name.clone(), value.clone());
        }
        info.query_params = request.query_params.clone();
        info.env_http_headers = request.env_http_headers.clone();

        map.insert(
            key,
            ModelEntry {
                info,
                api_key: request.configured_api_key.clone(),
                env_key: request.configured_env_key.clone(),
                auth_provider: request.auth_provider.clone(),
                api_base_url: model.api_base_url,
            },
        );
    }
    map
}
