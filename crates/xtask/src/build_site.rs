//! Build the project website (GitHub Pages) from `site/` into an output folder.
//!
//! ```text
//! cargo xtask build-site --release release.json --out _site
//! cargo xtask build-site --out _site              # no release data
//! ```
//!
//! The HTML pages under `site/` contain `{{PLACEHOLDER}}` markers for the latest
//! release (version, date, download links and sizes). `--release` takes the JSON
//! of GitHub's "get the latest release" API (`gh api repos/OWNER/REPO/releases/latest`).
//! Without it, or when an asset is missing, links fall back to the releases page so
//! the site is always deployable. Images are copied from the repository's assets/
//! folder, which stays the single source.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use image::imageops::FilterType;
use regex::Regex;
use serde_json::{Map, Value};

use crate::repo_root;

pub const SITE_URL: &str = "https://bugraskl.github.io/eye-tracker/";
pub const RELEASES_URL: &str = "https://github.com/bugraskl/eye-tracker/releases/latest";

/// Placeholder prefix → regular expression for the release asset's file name.
const ASSETS: [(&str, &str); 6] = [
    ("WIN_SETUP", r"-windows-x64-setup\.exe$"),
    ("WIN_ZIP", r"-windows-x64-portable\.zip$"),
    ("MAC_DMG", r"-macos-arm64\.dmg$"),
    ("LINUX_APPIMAGE", r"-linux-x86_64\.AppImage$"),
    ("LINUX_TAR", r"-linux-x86_64\.tar\.gz$"),
    ("SHA256SUMS", r"^SHA256SUMS\.txt$"),
];

/// Files copied from the repository's assets/ folder: source → published name.
const REPO_ASSETS: [(&str, &str); 1] = [("hero.png", "og.png")];

const MONTHS_EN: &str =
    "January February March April May June July August September October November December";
const MONTHS_TR: &str =
    "Ocak Şubat Mart Nisan Mayıs Haziran Temmuz Ağustos Eylül Ekim Kasım Aralık";

/// `63204271` → `"63 MB"` (decimal units, like most download pages).
pub fn human_size(size: i64) -> String {
    if size <= 0 {
        return String::new();
    }
    if size >= 1_000_000 {
        return format!("{:.0} MB", size as f64 / 1_000_000.0);
    }
    format!("{:.0} kB", (size as f64 / 1000.0).max(1.0))
}

/// `(year, month, day)` of an ISO 8601 date or date-time (`2026-10-01T10:05:12Z`).
fn parse_iso_date(value: &str) -> Option<(i32, usize, u32)> {
    let date = Regex::new(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ].*)?$")
        .ok()?
        .captures(value)?;
    let (year, month, day): (i32, usize, u32) = (
        date[1].parse().ok()?,
        date[2].parse().ok()?,
        date[3].parse().ok()?,
    );
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    ((1..=12).contains(&month) && (1..=days[month - 1]).contains(&day))
        .then_some((year, month, day))
}

pub fn human_date(value: &str, lang: &str) -> String {
    let Some((year, month, day)) = parse_iso_date(value) else {
        return String::new();
    };
    let months = if lang == "tr" { MONTHS_TR } else { MONTHS_EN };
    let month = months.split(' ').nth(month - 1).unwrap_or_default();
    if lang == "tr" {
        format!("{day} {month} {year}")
    } else {
        format!("{month} {day}, {year}")
    }
}

/// The current year (UTC), from the days since the epoch.
fn current_year() -> i64 {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() / 86_400) as i64;
    // Howard Hinnant's civil_from_days, year only.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let day_of_year = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    yoe + era * 400 + i64::from(month_index >= 10)
}

fn text_of(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    }
}

/// Placeholder values for one language from the release JSON (may be empty).
pub fn release_values(release: &Map<String, Value>, lang: &str) -> BTreeMap<String, String> {
    let tag = text_of(release.get("tag_name"));
    let version = tag.strip_prefix('v').unwrap_or(&tag);
    let html_url = text_of(release.get("html_url"));
    let mut values = BTreeMap::from([
        ("SITE_URL".to_string(), SITE_URL.to_string()),
        (
            "VERSION".to_string(),
            if version.is_empty() {
                "latest"
            } else {
                version
            }
            .to_string(),
        ),
        (
            "RELEASE_URL".to_string(),
            if html_url.is_empty() {
                RELEASES_URL.to_string()
            } else {
                html_url
            },
        ),
        (
            "RELEASE_DATE".to_string(),
            human_date(&text_of(release.get("published_at")), lang),
        ),
        ("YEAR".to_string(), current_year().to_string()),
    ]);
    let assets: Vec<&Map<String, Value>> = release
        .get("assets")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_object).collect())
        .unwrap_or_default();
    for (key, pattern) in ASSETS {
        let pattern = Regex::new(pattern).expect("the asset patterns are constants");
        let found = assets
            .iter()
            .find(|a| pattern.is_match(&text_of(a.get("name"))));
        values.insert(
            format!("{key}_URL"),
            found.map_or_else(
                || RELEASES_URL.to_string(),
                |a| text_of(a.get("browser_download_url")),
            ),
        );
        values.insert(
            format!("{key}_SIZE"),
            found.map_or_else(String::new, |a| {
                human_size(a.get("size").and_then(Value::as_i64).unwrap_or(0))
            }),
        );
    }
    values
}

/// `text` with every `{{PLACEHOLDER}}` replaced; an unknown placeholder is an error naming `source`.
pub fn render(
    text: &str,
    values: &BTreeMap<String, String>,
    source: &Path,
) -> Result<String, String> {
    let placeholder = Regex::new(r"\{\{([A-Z0-9_]+)\}\}").expect("constant pattern");
    let mut result = String::with_capacity(text.len());
    let mut last = 0;
    for found in placeholder.captures_iter(text) {
        let whole = found.get(0).expect("group 0 always exists");
        let key = &found[1];
        let value = values
            .get(key)
            .ok_or_else(|| format!("{}: unknown placeholder {{{{{key}}}}}", source.display()))?;
        result.push_str(&text[last..whole.start()]);
        result.push_str(value);
        last = whole.end();
    }
    result.push_str(&text[last..]);
    Ok(result)
}

fn page_language(path: &Path, site: &Path) -> &'static str {
    let first = path
        .strip_prefix(site)
        .ok()
        .and_then(|p| p.components().next());
    if first.is_some_and(|c| c.as_os_str() == "tr") {
        "tr"
    } else {
        "en"
    }
}

fn files_below(dir: &Path, found: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            files_below(&path, found)?;
        } else {
            found.push(path);
        }
    }
    Ok(())
}

/// Write the site into `out` (emptied first) and return the pages written.
pub fn build(release: &Map<String, Value>, out: &Path) -> Result<Vec<PathBuf>, String> {
    build_from(&repo_root(), release, out)
}

fn build_from(
    root: &Path,
    release: &Map<String, Value>,
    out: &Path,
) -> Result<Vec<PathBuf>, String> {
    let site = root.join("site");
    let failed = |e: io::Error| e.to_string();
    if out.exists() {
        fs::remove_dir_all(out).map_err(failed)?;
    }
    fs::create_dir_all(out).map_err(failed)?;
    let mut sources = Vec::new();
    files_below(&site, &mut sources).map_err(failed)?;
    sources.sort();
    let mut pages = Vec::new();
    for source in sources {
        let target = out.join(source.strip_prefix(&site).map_err(|e| e.to_string())?);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(failed)?;
        }
        if source.extension().is_some_and(|e| e == "html") {
            let html = render(
                &fs::read_to_string(&source).map_err(failed)?,
                &release_values(release, page_language(&source, &site)),
                &source,
            )?;
            fs::write(&target, html).map_err(failed)?;
            pages.push(target);
        } else {
            fs::copy(&source, &target).map_err(failed)?;
        }
    }
    fs::create_dir_all(out.join("assets")).map_err(failed)?;
    for (name, published) in REPO_ASSETS {
        fs::copy(
            root.join("assets").join(name),
            out.join("assets").join(published),
        )
        .map_err(failed)?;
    }
    write_icons(root, &out.join("assets"))?;
    // GitHub Pages: serve files as they are (no Jekyll processing).
    fs::write(out.join(".nojekyll"), "").map_err(failed)?;
    Ok(pages)
}

/// Favicons resized from the app icon (`assets/icon.png`, 1024 px).
fn write_icons(root: &Path, folder: &Path) -> Result<(), String> {
    let icon = image::open(root.join("assets").join("icon.png"))
        .map_err(|e| e.to_string())?
        .to_rgba8();
    for (name, size) in [("favicon.png", 64), ("apple-touch-icon.png", 180)] {
        image::imageops::resize(&icon, size, size, FilterType::Lanczos3)
            .save(folder.join(name))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    /// JSON of the latest GitHub release.
    #[arg(long)]
    pub release: Option<PathBuf>,
    /// Output folder (default: _site in the repository).
    #[arg(long)]
    pub out: Option<PathBuf>,
}

/// Entry point of `cargo xtask build-site`; returns the exit code.
pub fn main(args: &Args) -> i32 {
    let mut release = Map::new();
    if let Some(path) = args.release.as_ref().filter(|p| p.is_file()) {
        let text = fs::read_to_string(path).unwrap_or_default();
        let text = if text.is_empty() { "{}" } else { &text };
        match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(loaded)) => release = loaded,
            Ok(_) => {}
            Err(e) => eprintln!("build_site: ignoring unreadable release data: {e}"),
        }
    }
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| repo_root().join("_site"));
    match build(&release, &out) {
        Ok(pages) => {
            let version = match release.get("tag_name") {
                Some(Value::String(tag)) if !tag.is_empty() => tag.as_str(),
                _ => "no release data",
            };
            println!(
                "build_site: {} pages ({version}) -> {}",
                pages.len(),
                out.display()
            );
            0
        }
        Err(e) => {
            eprintln!("build_site: {e}");
            1
        }
    }
}

#[cfg(test)]
#[path = "build_site_tests.rs"]
mod tests;
