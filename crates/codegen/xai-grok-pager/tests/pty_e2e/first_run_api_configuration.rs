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
    harness.update(Duration::from_millis(100));
    for _ in 0.."https://api.x.ai/v1".len() {
        harness
            .inject_keys(b"\x1b[3~")
            .expect("delete default base URL character");
        harness.update(Duration::from_millis(50));
    }
    inject_keys_paced(&mut harness, content.url().as_bytes());
    harness.inject_keys(b"\r").expect("focus API key");
    harness.update(Duration::from_millis(100));
    harness.inject_keys(b"xai-test-key").expect("enter API key");
    harness.update(Duration::from_millis(100));
    harness.inject_keys(b"\r").expect("focus backend");
    harness.update(Duration::from_millis(100));
    harness
        .wait_for_text("Chat Completions", WELCOME_TIMEOUT)
        .expect("backend field is visible after API key entry");
    harness
        .inject_keys(keys::RIGHT)
        .expect("select Responses backend");
    harness
        .wait_for_text("Responses", WELCOME_TIMEOUT)
        .expect("Responses backend is selected");
    harness.inject_keys(b"\r").expect("submit selected backend");

    harness
        .wait_for_text("Logged in with API key", Duration::from_secs(30))
        .expect("API configuration reload completes");

    let config_path = content.home().join(".grok/config.toml");
    let config = std::fs::read_to_string(&config_path).expect("read persisted API configuration");
    assert!(
        config.contains("api_backend = \"responses\""),
        "first-run configuration should persist the selected Responses backend:\n{config}"
    );

    const PROMPT_AFTER_CONFIG: &str = "FIRST_CONFIG_INPUT";
    harness
        .inject_keys(format!("{PROMPT_AFTER_CONFIG}\r").as_bytes())
        .expect("type and submit after first configuration");
    if let Err(error) = harness.wait_for_text("FIRST_CONFIG_RESPONSE", Duration::from_secs(30)) {
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

/// The selector must remain usable in a normal small terminal and through the
/// field-navigation path users are most likely to take.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn first_run_api_configuration_tab_flow_uses_selected_backend_on_small_terminal() {
    let content = ContentController::start().await.expect("start content");
    content.set_response("SMALL_CONFIG_RESPONSE from the mock server.");

    let binary = pager_binary().expect("resolve pager binary");
    let mut harness = PtyHarness::spawn_with_content_env_ops_in_dir(
        &binary,
        24,
        80,
        &content,
        &[],
        &[EnvOp::remove("XAI_API_KEY")],
        Some(content.home()),
    )
    .expect("spawn pager without startup API key");

    harness
        .wait_for_text("Configure Grok API", WELCOME_TIMEOUT)
        .expect("first-run API configuration screen");

    harness
        .inject_keys(b"\x1b[H")
        .expect("move to base URL start");
    harness.update(Duration::from_millis(100));
    for _ in 0.."https://api.x.ai/v1".len() {
        harness
            .inject_keys(b"\x1b[3~")
            .expect("delete default base URL character");
        harness.update(Duration::from_millis(50));
    }
    inject_keys_paced(&mut harness, content.url().as_bytes());

    harness.inject_keys(b"\t").expect("focus API key with Tab");
    harness.update(Duration::from_millis(100));
    harness.inject_keys(b"xai-test-key").expect("enter API key");
    harness.inject_keys(b"\t").expect("focus backend with Tab");
    harness.update(Duration::from_millis(100));
    harness
        .wait_for_text("Chat Completions", WELCOME_TIMEOUT)
        .expect("backend field is visible on a small terminal");

    harness
        .inject_keys(keys::RIGHT)
        .expect("select Responses backend");
    harness
        .wait_for_text("Responses", WELCOME_TIMEOUT)
        .expect("Responses backend is selected");
    harness.inject_keys(keys::ENTER).expect("submit backend");

    harness
        .wait_for_text("Logged in with API key", Duration::from_secs(30))
        .expect("API configuration reload completes");
    harness
        .inject_keys(b"SMALL_CONFIG_INPUT\r")
        .expect("submit prompt after first configuration");
    harness
        .wait_for_text("SMALL_CONFIG_RESPONSE", Duration::from_secs(30))
        .expect("selected backend serves the first configured prompt");

    let requests = content.requests();
    assert!(
        requests
            .iter()
            .any(|request| request.path == "/v1/responses"),
        "selected Responses backend should receive the configured prompt; requests: {requests:#?}"
    );

    harness.quit().expect("clean quit");
}

/// The backend selector must still be painted when the terminal is shorter
/// than the normal first-run form. This catches fixed-layout overflow that can
/// make the selector look absent even though keyboard navigation reaches it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn first_run_api_configuration_backend_is_visible_on_short_terminals() {
    let content = ContentController::start().await.expect("start content");
    let binary = pager_binary().expect("resolve pager binary");

    for rows in [20, 16] {
        let mut harness = PtyHarness::spawn_with_content_env_ops_in_dir(
            &binary,
            rows,
            80,
            &content,
            &[],
            &[EnvOp::remove("XAI_API_KEY")],
            Some(content.home()),
        )
        .expect("spawn pager without startup API key");

        harness
            .wait_for_text("Configure Grok API", WELCOME_TIMEOUT)
            .unwrap_or_else(|error| {
                panic!(
                    "{error} at {rows} rows\nscreen:\n{}",
                    harness.screen_contents()
                )
            });
        harness
            .wait_for_text("Chat Completions", WELCOME_TIMEOUT)
            .unwrap_or_else(|error| {
                panic!(
                    "{error} at {rows} rows: backend selector is not visible\nscreen:\n{}",
                    harness.screen_contents()
                )
            });
        harness
            .inject_keys(keys::DOWN)
            .expect("focus API key on short terminal");
        harness
            .inject_keys(keys::DOWN)
            .expect("focus backend on short terminal");
        harness
            .inject_keys(keys::RIGHT)
            .expect("select Responses on short terminal");
        harness
            .wait_for_text("Responses", WELCOME_TIMEOUT)
            .unwrap_or_else(|error| {
                panic!(
                    "{error} at {rows} rows: backend selector did not accept input\nscreen:\n{}",
                    harness.screen_contents()
                )
            });

        harness.quit().expect("clean quit");
    }
}

/// Real terminals commonly deliver a bracketed paste and the user's follow-up
/// Tab/Enter in one PTY write. The configuration form must preserve those
/// navigation keys instead of inserting them into the pasted field.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn first_run_api_configuration_accepts_bracketed_pastes_with_immediate_navigation() {
    let content = ContentController::start().await.expect("start content");
    content.set_response("BRACKETED_CONFIG_RESPONSE from the mock server.");

    let binary = pager_binary().expect("resolve pager binary");
    let mut harness = PtyHarness::spawn_with_content_env_ops_in_dir(
        &binary,
        24,
        80,
        &content,
        &[],
        &[EnvOp::remove("XAI_API_KEY")],
        Some(content.home()),
    )
    .expect("spawn pager without startup API key");

    harness
        .wait_for_text("Configure Grok API", WELCOME_TIMEOUT)
        .expect("first-run API configuration screen");

    // Replace the prefilled default endpoint, then paste the replacement and
    // confirm it in the same PTY write.
    harness
        .inject_keys(b"\x1b[H")
        .expect("move to base URL start");
    harness.update(Duration::from_millis(100));
    for _ in 0.."https://api.x.ai/v1".len() {
        harness
            .inject_keys(b"\x1b[3~")
            .expect("delete default base URL character");
        harness.update(Duration::from_millis(25));
    }
    harness
        .inject_keys(format!("\x1b[200~{}\x1b[201~\r", content.url()).as_bytes())
        .expect("bracketed-paste mock URL and immediate Enter");

    // Paste the key and immediately Tab to the backend selector. A lost Tab
    // leaves focus in the masked field, so the following Right cannot select
    // Responses and this test fails with a useful screen dump.
    harness
        .inject_keys(b"\x1b[200~xai-bracketed-key\x1b[201~\t")
        .expect("bracketed-paste API key and immediate Tab");
    harness
        .inject_keys(keys::RIGHT)
        .expect("select Responses after pasted configuration");
    harness
        .wait_for_text("Responses", WELCOME_TIMEOUT)
        .expect("Responses backend selected after immediate paste navigation");
    harness
        .inject_keys(keys::ENTER)
        .expect("submit Responses backend");

    harness
        .wait_for_text("Logged in with API key", Duration::from_secs(30))
        .expect("API configuration reload completes");
    harness
        .inject_keys(b"BRACKETED_CONFIG_INPUT\r")
        .expect("submit prompt after pasted configuration");
    harness
        .wait_for_text("BRACKETED_CONFIG_RESPONSE", Duration::from_secs(30))
        .expect("selected backend serves the first configured prompt");

    let config_path = content.home().join(".grok/config.toml");
    let config = std::fs::read_to_string(&config_path).expect("read persisted API configuration");
    assert!(
        config.contains("api_backend = \"responses\""),
        "immediate paste navigation should persist Responses backend:\n{config}"
    );
    assert!(
        content
            .requests()
            .iter()
            .any(|request| request.path == "/v1/responses"),
        "the selected Responses backend should receive the first prompt; requests: {:#?}",
        content.requests()
    );

    harness.quit().expect("clean quit");
}

/// Terminals without bracketed-paste support can deliver a whole first-run
/// form interaction as one burst. Enter and Tab must remain navigation events
/// instead of being folded into a synthetic multiline paste.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn first_run_api_configuration_accepts_unbracketed_rapid_navigation() {
    let content = ContentController::start().await.expect("start content");
    content.set_response("RAPID_CONFIG_RESPONSE from the mock server.");

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

    harness
        .inject_keys(b"\x1b[H")
        .expect("move to base URL start");
    harness.update(Duration::from_millis(100));
    for _ in 0.."https://api.x.ai/v1".len() {
        harness
            .inject_keys(b"\x1b[3~")
            .expect("delete default base URL character");
        harness.update(Duration::from_millis(25));
    }

    let mut rapid_input = Vec::new();
    rapid_input.extend_from_slice(content.url().as_bytes());
    rapid_input.extend_from_slice(b"\r");
    rapid_input.extend_from_slice(b"xai-rapid-key");
    rapid_input.extend_from_slice(b"\t");
    rapid_input.extend_from_slice(keys::RIGHT);
    rapid_input.extend_from_slice(keys::ENTER);
    harness
        .inject_keys(&rapid_input)
        .expect("send unbracketed URL, key, navigation, and submit burst");

    harness
        .wait_for_text("Logged in with API key", Duration::from_secs(30))
        .expect("rapid API configuration reload completes");

    let config_path = content.home().join(".grok/config.toml");
    let config = std::fs::read_to_string(&config_path).expect("read persisted API configuration");
    assert!(
        config.contains("api_backend = \"responses\""),
        "rapid navigation should persist the selected Responses backend:\n{config}"
    );

    harness
        .inject_keys(b"RAPID_CONFIG_INPUT\r")
        .expect("submit prompt after rapid configuration");
    harness
        .wait_for_text("RAPID_CONFIG_RESPONSE", Duration::from_secs(30))
        .expect("selected backend serves the first rapid configured prompt");

    assert!(
        content
            .requests()
            .iter()
            .any(|request| request.path == "/v1/responses"),
        "selected Responses backend should receive the rapid configured prompt; requests: {:#?}",
        content.requests()
    );
    harness.quit().expect("clean quit");
}

/// The selector is a real interactive control in the compact layout, not just
/// a keyboard-only display. Clicking its value must focus it so the next
/// horizontal navigation changes the selected backend.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn first_run_api_configuration_compact_selector_accepts_mouse_focus() {
    let content = ContentController::start().await.expect("start content");
    let binary = pager_binary().expect("resolve pager binary");
    let mut harness = PtyHarness::spawn_with_content_env_ops_in_dir(
        &binary,
        16,
        80,
        &content,
        &[],
        &[EnvOp::remove("XAI_API_KEY")],
        Some(content.home()),
    )
    .expect("spawn pager without startup API key");

    harness
        .wait_for_text("Chat Completions", WELCOME_TIMEOUT)
        .expect("compact backend selector");
    let (row, col) = locate_screen_text(&harness.screen_contents(), "Chat Completions")
        .expect("locate compact backend selector value");
    harness
        .inject_keys(sgr_mouse(0, row, col, 'M').as_bytes())
        .expect("click compact backend selector");
    harness
        .inject_keys(keys::RIGHT)
        .expect("advance backend after mouse focus");
    harness
        .wait_for_text("Responses", WELCOME_TIMEOUT)
        .expect("mouse click should focus compact backend selector");

    harness.quit().expect("clean quit");
}
