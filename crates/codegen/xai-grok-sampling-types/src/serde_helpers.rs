use serde::{Deserialize, Deserializer, de::Error};
use serde_json::Value;

/// Deserialize an optional value while accepting an empty string as absent.
///
/// Some OpenAI-compatible providers emit `""` for fields that are normally
/// nullable, such as `finish_reason`. Parse non-empty values through their
/// regular serde implementation so unknown values still fail loudly.
pub fn empty_string_as_none<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: serde::de::DeserializeOwned,
    D: Deserializer<'de>,
{
    match Option::<Value>::deserialize(deserializer)? {
        None => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(value) => serde_json::from_value(value)
            .map(Some)
            .map_err(D::Error::custom),
    }
}

/// Deserialize `Option<Option<T>>`: absent (`None`) leaves, `null` (`Some(None)`)
/// clears, a value sets. Requires `#[serde(default, deserialize_with = "…")]`.
pub fn double_option<'de, T, D>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    Ok(Some(Option::deserialize(deserializer)?))
}
