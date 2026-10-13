//! Tests for `build_site` (port of `tests/test_site.py`): the project website (`site/`,
//! built for GitHub Pages).

use regex::Regex;
use serde_json::json;

use super::*;

fn release() -> Map<String, Value> {
    let mut assets: Vec<Value> = [
        ("windows-x64-setup.exe", 63_204_271),
        ("windows-x64-portable.zip", 89_695_115),
        ("macos-arm64.dmg", 88_420_854),
        ("linux-x86_64.AppImage", 132_553_208),
        ("linux-x86_64.tar.gz", 143_728_124),
    ]
    .iter()
    .map(|(suffix, size)| {
        json!({
            "name": format!("EyeTracker-1.2.3-{suffix}"),
            "size": size,
            "browser_download_url": format!("https://dl/{suffix}"),
        })
    })
    .collect();
    assets.push(json!({
        "name": "SHA256SUMS.txt", "size": 520, "browser_download_url": "https://dl/SHA256SUMS.txt"
    }));
    let Value::Object(release) = json!({
        "tag_name": "v1.2.3",
        "html_url": "https://github.com/bugraskl/eye-tracker/releases/tag/v1.2.3",
        "published_at": "2026-10-01T10:05:12Z",
        "assets": assets,
    }) else {
        unreachable!()
    };
    release
}

fn built() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("_site");
    build(&release(), &out).unwrap();
    (tmp, out)
}

fn read(out: &Path, page: &str) -> String {
    fs::read_to_string(out.join(page)).unwrap()
}

fn ids(html: &str) -> Vec<String> {
    let mut found: Vec<String> = Regex::new(r#"\bid="([^"]+)""#)
        .unwrap()
        .captures_iter(html)
        .map(|c| c[1].to_string())
        .collect();
    found.sort();
    found
}

const PAGES: [&str; 2] = ["index.html", "tr/index.html"];

#[test]
fn test_both_languages_are_built() {
    let (_tmp, out) = built();
    assert!(read(&out, "index.html").contains("<html lang=\"en\">"));
    assert!(read(&out, "tr/index.html").contains("<html lang=\"tr\">"));
    assert!(out.join(".nojekyll").exists());
}

#[test]
fn test_no_placeholder_survives() {
    let (_tmp, out) = built();
    for page in PAGES {
        assert!(!read(&out, page).contains("{{"), "{page}");
    }
}

/// Both pages carry the same sections, anchors and download cards.
#[test]
fn test_languages_have_the_same_structure() {
    let (_tmp, out) = built();
    let (english, turkish) = (read(&out, "index.html"), read(&out, "tr/index.html"));
    assert_eq!(ids(&english), ids(&turkish));
    for tag in ["<section", "<article", "<details", "data-download"] {
        assert_eq!(
            english.matches(tag).count(),
            turkish.matches(tag).count(),
            "{tag}"
        );
    }
}

#[test]
fn test_local_links_resolve() {
    let (_tmp, out) = built();
    let link = Regex::new(r##"(?:src|href)="([^"#]+)""##).unwrap();
    for page in PAGES {
        let html = read(&out, page);
        let folder = out.join(page).parent().unwrap().to_path_buf();
        let mut missing = Vec::new();
        for found in link.captures_iter(&html) {
            let target = &found[1];
            if target.contains("://")
                || target.starts_with("mailto:")
                || target.starts_with("data:")
            {
                continue;
            }
            let mut resolved = folder.join(target);
            if resolved.is_dir() {
                resolved = resolved.join("index.html");
            }
            if !resolved.exists() {
                missing.push(target.to_string());
            }
        }
        assert!(missing.is_empty(), "{page}: {missing:?}");
    }
}

#[test]
fn test_release_data_is_filled_in() {
    let (_tmp, out) = built();
    for page in PAGES {
        let html = read(&out, page);
        assert!(html.contains("1.2.3"));
        for suffix in [
            "windows-x64-setup.exe",
            "macos-arm64.dmg",
            "linux-x86_64.AppImage",
        ] {
            assert!(
                html.contains(&format!("href=\"https://dl/{suffix}\"")),
                "{suffix}"
            );
            assert!(!html.contains(&format!("data-=\"https://dl/{suffix}\"")));
        }
        assert!(html.contains("href=\"https://dl/SHA256SUMS.txt\""));
        assert!(html.contains("63 MB"));
    }
}

#[test]
fn test_dates_are_localised() {
    let (_tmp, out) = built();
    assert!(read(&out, "index.html").contains("October 1, 2026"));
    assert!(read(&out, "tr/index.html").contains("1 Ekim 2026"));
}

/// The display-arrangement demo's status messages are translated.
#[test]
fn test_hero_demo_speaks_the_page_language() {
    let (_tmp, out) = built();
    let (english, turkish) = (read(&out, "index.html"), read(&out, "tr/index.html"));
    for html in [&english, &turkish] {
        for name in ["look", "away", "pick", "pane"] {
            assert!(html.contains(&format!("data-msg-{name}=\"")), "{name}");
        }
    }
    assert!(english.contains("Looking at display"));
    assert!(!turkish.contains("Looking at display"));
    assert!(turkish.contains("ekrana bakıyorsunuz"));
}

#[test]
fn test_without_release_data_links_fall_back_to_the_releases_page() {
    let tmp = tempfile::tempdir().unwrap();
    build(&Map::new(), &tmp.path().join("_site")).unwrap();
    let html = fs::read_to_string(tmp.path().join("_site/index.html")).unwrap();
    assert!(!html.contains("{{"));
    assert!(html.contains(&format!("href=\"{RELEASES_URL}\"")));
}

/// Like the app: no web fonts, trackers or scripts loaded from other sites.
#[test]
fn test_site_makes_no_third_party_requests() {
    let (_tmp, out) = built();
    let loaded = Regex::new(
        r#"<(?:script[^>]*\ssrc|img[^>]*\ssrc|link[^>]*rel="(?:stylesheet|icon|apple-touch-icon|preload)"[^>]*\shref)="([^"]+)""#,
    )
    .unwrap();
    for page in PAGES {
        let resources: Vec<String> = loaded
            .captures_iter(&read(&out, page))
            .map(|c| c[1].to_string())
            .collect();
        assert!(!resources.is_empty());
        assert!(
            resources.iter().all(|url| !url.contains("://")),
            "{resources:?}"
        );
    }
    let css = read(&out, "assets/style.css");
    assert!(!css.contains("@import"));
    assert!(!css.contains("url(http"));
}

#[test]
fn test_human_size() {
    for (size, text) in [
        (0, ""),
        (520, "1 kB"),
        (63_204_271, "63 MB"),
        (143_728_124, "144 MB"),
    ] {
        assert_eq!(human_size(size), text, "{size}");
    }
}

#[test]
fn test_unknown_placeholder_is_an_error() {
    assert!(render("{{NOPE}}", &BTreeMap::new(), Path::new("x.html")).is_err());
}
