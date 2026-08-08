//! Load `config.toml` as a [`toml_edit::DocumentMut`] for in-place edits.
//! A non-empty file that does not parse is left untouched (`None`).

use std::path::Path;

#[must_use]
pub(crate) fn read_config_document_for_edit(path: &Path) -> Option<toml_edit::DocumentMut> {
    #[allow(clippy::manual_unwrap_or_default)]
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => String::new(),
    };
    match content.parse() {
        Ok(d) => Some(d),
        Err(e) => {
            if content.is_empty() {
                return Some(toml_edit::DocumentMut::new());
            }
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "config.toml is not valid TOML; refusing to overwrite"
            );
            None
        }
    }
}

/// Set `[hints].<key>` to `value` in `~/.grok/config.toml`, preserving every
/// other key and table. Creates the file and parent dir when missing, and
/// no-ops when the existing file is non-empty but unparseable (so a malformed
/// config is never clobbered). Performs blocking I/O.
pub(crate) fn set_hint(key: &str, value: impl Into<toml_edit::Value>) -> std::io::Result<()> {
    let path = xai_grok_tools::util::grok_home::grok_home().join("config.toml");
    set_hint_at(&path, key, value)
}

/// Persist the API model connection settings under the bundled default model.
///
/// This deliberately edits the existing document instead of serializing a
/// partial config struct: model entries commonly live beside unrelated user
/// settings that must survive an onboarding write.
pub(crate) fn set_model_configuration(base_url: &str, api_key: &str) -> std::io::Result<()> {
    let path = xai_grok_tools::util::grok_home::grok_home().join("config.toml");
    set_model_configuration_at(&path, base_url, api_key)
}

/// Path-injectable core of [`set_model_configuration`].
pub(crate) fn set_model_configuration_at(
    path: &Path,
    base_url: &str,
    api_key: &str,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let Some(mut doc) = read_config_document_for_edit(path) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "config.toml is not valid TOML",
        ));
    };

    let model = doc.entry("model").or_insert(toml_edit::table());
    let Some(model) = model.as_table_mut() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "config.toml [model] must be a table",
        ));
    };
    let default_model = xai_grok_shell::models::default_model();
    let model_entry = model.entry(default_model).or_insert(toml_edit::table());
    let Some(model_entry) = model_entry.as_table_mut() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("config.toml [model.{default_model}] must be a table"),
        ));
    };
    model_entry["base_url"] = toml_edit::value(base_url);
    model_entry["api_key"] = toml_edit::value(api_key);
    // The onboarding endpoint is user-supplied and is expected to expose the
    // OpenAI-compatible chat-completions route, regardless of the bundled
    // model's first-party backend.
    model_entry["api_backend"] = toml_edit::value("chat_completions");
    std::fs::write(path, doc.to_string())
}

/// Path-injectable core of [`set_hint`].
fn set_hint_at(path: &Path, key: &str, value: impl Into<toml_edit::Value>) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let Some(mut doc) = read_config_document_for_edit(path) else {
        return Ok(());
    };
    doc["hints"][key] = toml_edit::value(value);
    std::fs::write(path, doc.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn merge_round_trip_preserves_sibling_tables() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            "[ui]\ncompact_mode = false\n\n[mcpServers]\nx = \"y\"\n",
        )
        .unwrap();

        let mut doc = read_config_document_for_edit(&path).expect("parse");
        doc["ui"]["show_timestamps"] = toml_edit::value(false);
        fs::write(&path, doc.to_string()).unwrap();

        let body = fs::read_to_string(&path).unwrap();
        assert!(
            body.contains("show_timestamps") && body.contains("mcpServers"),
            "expected merged TOML, got:\n{body}"
        );
    }

    #[test]
    fn nonempty_unparseable_returns_none_and_leaves_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let bad = "this is [not valid toml\n";
        fs::write(&path, bad).unwrap();

        assert!(read_config_document_for_edit(&path).is_none());
        assert_eq!(fs::read_to_string(&path).unwrap(), bad);
    }

    #[test]
    fn missing_file_is_editable_empty_doc() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("absent.toml");
        let doc = read_config_document_for_edit(&path).expect("editable");
        assert!(!doc.contains_key("ui"));
    }

    #[test]
    fn set_hint_at_round_trips_and_preserves_siblings() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "[ui]\ncompact_mode = false\n").unwrap();

        set_hint_at(&path, "memory_modal_fullscreen", true).unwrap();

        let doc = read_config_document_for_edit(&path).expect("reparse");
        assert_eq!(
            doc.get("hints")
                .and_then(|h| h.get("memory_modal_fullscreen"))
                .and_then(|v| v.as_bool()),
            Some(true),
        );
        assert!(
            fs::read_to_string(&path).unwrap().contains("compact_mode"),
            "sibling [ui] should be preserved"
        );
    }

    #[test]
    fn set_hint_at_creates_missing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/config.toml");
        set_hint_at(&path, "memory_modal_fullscreen", true).unwrap();
        assert!(
            path.exists(),
            "missing file and parent dir should be created"
        );
    }

    #[test]
    fn set_hint_write_then_read_back_round_trips() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "[ui]\ntheme = \"dark\"\n").unwrap();

        set_hint_at(&path, "memory_modal_fullscreen", true).unwrap();

        let doc = read_config_document_for_edit(&path).expect("reparse");
        let disabled = doc
            .get("hints")
            .and_then(|h| h.get("memory_modal_fullscreen"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        assert!(disabled, "should read back true after set_hint write");
    }

    #[test]
    fn set_hint_at_leaves_unparseable_file_untouched() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let bad = "this is [not valid toml\n";
        fs::write(&path, bad).unwrap();

        // No-op (no write, no clobber) when the existing file cannot be parsed.
        set_hint_at(&path, "memory_modal_fullscreen", true).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), bad);
    }

    #[test]
    fn set_model_configuration_at_preserves_existing_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            "[ui]\ntheme = \"dark\"\n\n[model.other]\nbase_url = \"https://other.example\"\n",
        )
        .unwrap();

        set_model_configuration_at(&path, "https://api.example/v1", "xai-secret").unwrap();

        let doc = read_config_document_for_edit(&path).expect("reparse");
        assert_eq!(doc["ui"]["theme"].as_str(), Some("dark"));
        assert_eq!(
            doc["model"]["other"]["base_url"].as_str(),
            Some("https://other.example")
        );
        let default_model = xai_grok_shell::models::default_model();
        assert_eq!(
            doc["model"][default_model]["base_url"].as_str(),
            Some("https://api.example/v1")
        );
        assert_eq!(
            doc["model"][default_model]["api_key"].as_str(),
            Some("xai-secret")
        );
        assert_eq!(
            doc["model"][default_model]["api_backend"].as_str(),
            Some("chat_completions")
        );
    }

    #[test]
    fn set_model_configuration_at_creates_model_table() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/config.toml");

        set_model_configuration_at(&path, "https://api.example/v1", "key").unwrap();

        let doc = read_config_document_for_edit(&path).expect("reparse");
        let default_model = xai_grok_shell::models::default_model();
        assert_eq!(doc["model"][default_model]["api_key"].as_str(), Some("key"));
    }

    #[test]
    fn set_model_configuration_at_rejects_unparseable_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let bad = "this is [not valid toml\n";
        fs::write(&path, bad).unwrap();

        assert!(set_model_configuration_at(&path, "https://api.example/v1", "key").is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), bad);
    }

    #[test]
    fn vim_mode_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "[ui]\ncompact_mode = false\n").unwrap();

        let mut doc = read_config_document_for_edit(&path).expect("parse");
        doc["ui"]["vim_mode"] = toml_edit::value(true);
        fs::write(&path, doc.to_string()).unwrap();

        let doc2 = read_config_document_for_edit(&path).expect("reparse");
        let enabled = doc2
            .get("ui")
            .and_then(|h| h.get("vim_mode"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        assert!(enabled, "expected vim_mode = true after round-trip");

        let body = fs::read_to_string(&path).unwrap();
        assert!(
            body.contains("compact_mode"),
            "sibling [ui] keys should be preserved"
        );
    }
}
