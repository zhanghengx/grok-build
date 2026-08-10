# Authentication

Grok authenticates using a manually configured API key and base URL. There is
no interactive login flow — set your credentials via environment variables or
in `~/.grok/config.toml`.

---

## API Key

Set your API key from [console.x.ai](https://console.x.ai) as an environment
variable:

```bash
Configure api_key in config.toml
grok
```

Or configure it in `~/.grok/config.toml`:

```toml
[model.grok-4]
api_key = "xai-..."
```

### Per-model API keys

You can set a different API key per model. This is useful when routing
requests to different providers or accounts:

```toml
[model.grok-4]
api_key = "xai-..."

[model.my-custom-model]
api_key = "sk-..."
base_url = "https://api.example.com/v1"
```

You can also reference an environment variable instead of hardcoding the key:

```toml
[model.my-custom-model]
env_key = "MY_API_KEY"
base_url = "https://api.example.com/v1"
```

### Auth precedence

Grok resolves credentials for each request in this order, highest to lowest:

1. **Per-model `api_key` or `env_key`** — set under `[model.<name>]` in `config.toml`. Wins whenever present.
2. **`api_key (config.toml)`** — fallback when no per-model key is configured.

---

## Custom Base URL

Point Grok at your own API endpoint by setting `base_url`:

```bash
export GROK_CLI_CHAT_PROXY_BASE_URL="https://grok-proxy.example.com/v1"
grok
```

Or in `~/.grok/config.toml`:

```toml
[endpoints]
xai_api_base_url = "https://api.example.com/v1"
```

When `base_url` is set, Grok uses API key auth (`Authorization: Bearer`)
instead of session auth.

---

## Related settings

| Setting | How to set it |
|---------|---------------|
| `[features] telemetry` | `config.toml` or `GROK_TELEMETRY_ENABLED` |
| `[telemetry] trace_upload` | `config.toml` or `GROK_TELEMETRY_TRACE_UPLOAD` |
| External OpenTelemetry | `GROK_EXTERNAL_OTEL` / `[telemetry] otel_*`. See [Monitoring Usage](24-monitoring-usage.md). |

See [Monitoring Usage](24-monitoring-usage.md#related-settings) and [Configuration](05-configuration.md#telemetry).

---

## Troubleshooting

### Debug logging

Set `RUST_LOG` to control the verbosity of the file log and headless stderr output. In the TUI, set `GROK_LOG_FILE` to an absolute path to write logs to that file:

```bash
GROK_LOG_FILE=/tmp/grok.log RUST_LOG=debug grok
tail -f /tmp/grok.log
```

In headless mode, logs go to stderr:

```bash
RUST_LOG=debug grok -p "hello" 2> /tmp/grok.log
```

### Common fixes

- **"No credentials found"** — Set `api_key (config.toml)` or configure `api_key` in `~/.grok/config.toml`.
- **401 Unauthorized** — Check that your API key is valid and not expired.
- **Wrong endpoint** — Verify `base_url` / `xai_api_base_url` points to the correct API endpoint.
