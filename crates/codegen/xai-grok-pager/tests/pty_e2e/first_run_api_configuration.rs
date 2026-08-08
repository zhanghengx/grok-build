// Per-test-case module for the `pty_e2e_config_ui` integration test crate.
#[allow(unused_imports)]
use super::common::*;

/// The first-run API form must leave the pager with a usable session after the
/// config file is written and reloaded. Before the reload auth fallback was
/// installed, the first session/new failed with "no auth method id provided"
/// and the prompt appeared to flash while the user tried to type.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn first_run_api_configuration_accepts_input_after_reload() {
    let content = ContentController::start().await.expect("start content");
    content.set_response("FIRST_CONFIG_RESPONSE from the mock server.");

    let binary = pager_binary().expect("resolve pager binary");
    let mut harness = PtyHarness::spawn_with_content_env_ops_in_dir(
        &binary,
        DEFAULT_ROWS,
        DEFAULT_COLS,
        &content,
        &[],
        &[EnvOp::remove("XAI_API_KEY")],
        Some(content.home()),
    )
    .expect("spawn pager without startup API key");

    harness
        .wait_for_text("Configure Grok API", WELCOME_TIMEOUT)
        .expect("first-run API configuration screen");

    // The base URL field is prefilled with xAI's default. The configuration
    // form intentionally ignores control chords, so clear it with the
    // ordinary Home/Delete editing keys before entering the mock endpoint.
    harness
        .inject_keys(b"\x1b[H")
        .expect("move to base URL start");
    for _ in 0.."https://api.x.ai/v1".len() {
        harness
            .inject_keys(b"\x1b[3~")
            .expect("delete default base URL character");
    }
    harness
        .inject_keys(content.url().as_bytes())
        .expect("enter mock base URL");
    harness
        .inject_keys(b"\txai-test-key\r")
        .expect("enter API key and submit");

    harness
        .wait_for_text("Logged in with API key", Duration::from_secs(30))
        .expect("API configuration reload completes");

    const PROMPT_AFTER_CONFIG: &str = "FIRST_CONFIG_INPUT";
    harness
        .inject_keys(format!("{PROMPT_AFTER_CONFIG}\r").as_bytes())
        .expect("type and submit after first configuration");
    if let Err(error) = harness.wait_for_text("FIRST_CONFIG_RESPONSE", Duration::from_secs(30)) {
        let config_path = content.home().join(".grok/config.toml");
        let config = std::fs::read_to_string(&config_path)
            .unwrap_or_else(|read_error| format!("<read failed: {read_error}>"));
        panic!(
            "{error}\nmock requests: {requests:#?}\nmock request bodies: {bodies:#?}\nconfig: {config}\nscreen:\n{screen}",
            requests = content.requests(),
            bodies = content.request_bodies(),
            screen = harness.screen_contents(),
        );
    }

    let screen = harness.screen_contents();
    assert!(
        !screen.contains("Session creation failed"),
        "first configured session failed and input may flash:\n{screen}"
    );
    assert!(
        !screen.contains("no auth method id provided"),
        "first configured session had no auth method:\n{screen}"
    );

    harness.quit().expect("clean quit");
}
