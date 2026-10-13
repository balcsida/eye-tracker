//! Versions and the shape of a GitHub release, without any networking.
//!
//! The update check asks GitHub for the latest release (`releases/latest`, which
//! never answers with a draft or a pre-release) and keeps only what it can trust:
//!
//! * the tag must be a plain `X.Y.Z` version;
//! * the page of the release is built here, not taken from the answer;
//! * an asset counts only when its download address is exactly the one GitHub
//!   derives from the repository, the tag and the file name, so a crafted answer
//!   cannot point the installer download somewhere else.
//!
//! Only the Windows installer is installed by the app (see [`super::installer`]); the
//! other packages are installed by hand from the release page.

use serde_json::Value;

use crate::REPO_URL;

/// `owner/name` of the repository.
pub fn repo_slug() -> &'static str {
    REPO_URL
        .strip_prefix("https://github.com/")
        .unwrap_or(REPO_URL)
}

/// The one address the update check asks.
pub fn latest_release_url() -> String {
    format!(
        "https://api.github.com/repos/{}/releases/latest",
        repo_slug()
    )
}

pub fn releases_page() -> String {
    format!("{REPO_URL}/releases")
}

/// The file with the SHA-256 of every package of a release (`release.yml`).
pub const CHECKSUMS_NAME: &str = "SHA256SUMS.txt";
/// Largest installer the app will download (the real one is about 150 MB).
pub const MAX_INSTALLER_BYTES: u64 = 400 * 1024 * 1024;
pub const MAX_CHECKSUMS_BYTES: u64 = 64 * 1024;
/// Largest answer of the release API read.
pub const MAX_API_BYTES: u64 = 1024 * 1024;

pub type Version = (u32, u32, u32);

/// `X.Y.Z` at the start of `text` (each part 1 to `max_digits` ASCII digits) and what follows it.
fn leading_version(text: &str, max_digits: usize) -> Option<(Version, &str)> {
    let mut rest = text.strip_prefix('v').unwrap_or(text);
    let mut parts = [0u32; 3];
    for (index, part) in parts.iter_mut().enumerate() {
        if index > 0 {
            rest = rest.strip_prefix('.')?;
        }
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 || digits > max_digits {
            return None;
        }
        *part = rest[..digits].parse().ok()?;
        rest = &rest[digits..];
    }
    Some(((parts[0], parts[1], parts[2]), rest))
}

/// `"0.2.1"` or `"v0.2.1"` as a tuple; anything else (rc, dev, junk) is `None`.
pub fn parse_stable(text: &str) -> Option<Version> {
    match leading_version(text.trim(), 4) {
        Some((version, "")) => Some(version),
        _ => None,
    }
}

/// The version of this build: its leading numbers (`0.3.0.dev1` is 0.3.0).
pub fn parse_running(text: &str) -> Option<Version> {
    // Digits are unbounded here, like `\d+`; a part too big for u32 is no version.
    leading_version(text.trim(), usize::MAX).map(|(version, _)| version)
}

pub fn format_version(version: Version) -> String {
    format!("{}.{}.{}", version.0, version.1, version.2)
}

/// File name of the Windows installer of `version` (see `installer.iss`).
pub fn installer_name(version: &str) -> String {
    format!("EyeTracker-{version}-windows-x64-setup.exe")
}

/// Where GitHub serves the asset `name` of the release `tag`.
pub fn download_url(tag: &str, name: &str) -> String {
    format!("{REPO_URL}/releases/download/{tag}/{name}")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    pub name: String,
    pub url: String,
    pub size: u64,
}

/// A published release: its version, its page and the assets the app uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseInfo {
    pub version: String,
    pub tag: String,
    pub page_url: String,
    pub installer: Option<Asset>,
    pub checksums: Option<Asset>,
}

impl ReleaseInfo {
    pub fn version_tuple(&self) -> Version {
        // parse_release only builds stable versions
        parse_stable(&self.version).unwrap_or_default()
    }

    /// Whether the app can install this release by itself (installer and checksums).
    pub fn installable(&self) -> bool {
        self.installer.is_some() && self.checksums.is_some()
    }
}

fn asset(payload: &Value, tag: &str, name: &str, limit: u64) -> Option<Asset> {
    let items = payload.as_array()?;
    for item in items {
        if item.get("name").and_then(Value::as_str) != Some(name) {
            continue;
        }
        // as_i64 is None for floats and booleans, like `isinstance(size, int)` without bool.
        let size = item.get("size").and_then(Value::as_i64);
        let url = item.get("browser_download_url").and_then(Value::as_str);
        if let (Some(size), Some(url)) = (size, url)
            && size > 0
            && size as u64 <= limit
            && url == download_url(tag, name)
        {
            return Some(Asset {
                name: name.to_string(),
                url: url.to_string(),
                size: size as u64,
            });
        }
    }
    None
}

/// Python truthiness of a JSON value (`payload.get("draft")`).
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64() != Some(0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(map)) => !map.is_empty(),
    }
}

/// The release described by a `releases/latest` answer; `None` if it is not
/// a usable stable release.
pub fn parse_release(payload: &Value) -> Option<ReleaseInfo> {
    if !payload.is_object() || truthy(payload.get("draft")) || truthy(payload.get("prerelease")) {
        return None;
    }
    let tag = payload.get("tag_name")?.as_str()?;
    let version = parse_stable(tag)?;
    let tag = tag.trim();
    let text = format_version(version);
    let assets = payload.get("assets").unwrap_or(&Value::Null);
    Some(ReleaseInfo {
        installer: asset(assets, tag, &installer_name(&text), MAX_INSTALLER_BYTES),
        checksums: asset(assets, tag, CHECKSUMS_NAME, MAX_CHECKSUMS_BYTES),
        page_url: format!("{REPO_URL}/releases/tag/{tag}"),
        version: text,
        tag: tag.to_string(),
    })
}

/// Whether `release` is newer than the running build `running`.
pub fn is_newer(release: &ReleaseInfo, running: &str) -> bool {
    parse_running(running).is_some_and(|current| release.version_tuple() > current)
}

#[cfg(test)]
#[path = "release_tests.rs"]
mod tests;
