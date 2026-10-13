//! Tests for `gen_config_docs` (port of `tests/test_docs.py`).
//!
//! `test_relative_links_in_markdown_resolve` and `test_images_are_checked_too` check the
//! documentation pages, not the generator; they are ported here too so no docs test is lost.

use std::fs;
use std::path::{Path, PathBuf};

use super::*;

fn modifiers(windows: &str, macos: &str, linux: &str) -> HotkeyModifiers {
    HotkeyModifiers {
        windows: windows.into(),
        macos: macos.into(),
        linux: linux.into(),
    }
}

#[test]
fn test_configuration_reference_is_up_to_date() {
    assert_eq!(
        main(&Args { check: true }),
        0,
        "run: cargo xtask gen-config-docs"
    );
}

#[test]
fn test_hotkey_defaults_are_listed_per_platform_only_when_they_differ() {
    let same = modifiers("ctrl+alt+meta", "ctrl+alt+meta", "ctrl+alt+meta");
    assert_eq!(
        default_cell("hotkeys.toggle_tracking", &"ctrl+alt+meta+t".into(), &same),
        "`ctrl+alt+meta+t`"
    );
    let differ = modifiers("ctrl+alt+meta", "ctrl+alt", "ctrl+alt+meta");
    assert_eq!(
        default_cell("hotkeys.recalibrate", &"ctrl+alt+meta+c".into(), &differ),
        "`ctrl+alt+meta+c` Windows<br>`ctrl+alt+c` macOS<br>`ctrl+alt+meta+c` Linux"
    );
    // Not a hotkey: as it is.
    assert_eq!(default_cell("camera.device", &"0".into(), &same), "`0`");
}

/// The countdown is part of the away time (r2-docs-06): the page says when it starts.
#[test]
fn test_section_notes_take_their_numbers_from_the_defaults() {
    let defaults = NoteDefaults {
        away_timeout_s: 60,
        warning_s: 15,
        ..NoteDefaults::current()
    };
    let notes = section_notes(&defaults);
    let note = |name: &str| notes.iter().find(|(n, _)| *n == name).unwrap().1.clone();
    assert!(
        note("presence").contains("appears after 45 s without you, and the action follows at 60 s")
    );
    assert!(note("hotkeys").contains(&format!("`{}`", HotkeySettings::default().toggle_tracking)));
}

#[test]
fn test_every_setting_is_documented() {
    let undocumented: Vec<String> = config::describe_settings()
        .into_iter()
        .filter(|row| row.doc.trim().is_empty())
        .map(|row| row.key)
        .collect();
    assert!(undocumented.is_empty(), "{undocumented:?}");
}

fn pages() -> Vec<PathBuf> {
    let root = repo_root();
    let mut docs: Vec<PathBuf> = fs::read_dir(root.join("docs"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "md"))
        .collect();
    docs.sort();
    [
        "README.md",
        "README.tr.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CHANGELOG.md",
    ]
    .iter()
    .map(|name| root.join(name))
    .chain(docs)
    .collect()
}

/// Relative link and image targets of a Markdown page: `[text](target)` and the
/// `<img src="target">` of the HTML blocks (the README hero and demo).
fn local_targets(page: &Path) -> Vec<String> {
    let text = fs::read_to_string(page).unwrap();
    let links = regex::Regex::new(r"\]\(([^)\s]+)\)").unwrap();
    let sources = regex::Regex::new(r#"\ssrc="([^"]+)""#).unwrap();
    let mut targets: Vec<String> = Vec::new();
    for pattern in [&links, &sources] {
        targets.extend(pattern.captures_iter(&text).map(|c| c[1].to_string()));
    }
    targets
        .into_iter()
        .filter(|t| !t.contains("://") && !t.starts_with('#') && !t.starts_with("mailto:"))
        .collect()
}

#[test]
fn test_relative_links_in_markdown_resolve() {
    let root = repo_root();
    let mut broken = Vec::new();
    for page in pages() {
        for target in local_targets(&page) {
            let path = target.split('#').next().unwrap_or_default();
            if !path.is_empty() && !page.parent().unwrap().join(path).exists() {
                broken.push(format!(
                    "{} -> {target}",
                    page.strip_prefix(&root).unwrap_or(&page).display()
                ));
            }
        }
    }
    // assets/hero.png is rendered from assets/hero.svg and committed with it.
    assert!(
        broken.is_empty(),
        "missing link or image targets: {broken:?}"
    );
}

/// Regression (r2-docs-11): a missing README hero image went unnoticed, because
/// only Markdown links were checked, not `<img src>`.
#[test]
fn test_images_are_checked_too() {
    let tmp = tempfile::tempdir().unwrap();
    let page = tmp.path().join("README.md");
    fs::write(
        &page,
        "<img src=\"assets/hero.png\" alt=\"x\">\n[docs](docs/a.md) ![shot](shot.png)\n\
         <img src=\"https://example.org/badge.svg\">\n",
    )
    .unwrap();
    assert_eq!(
        local_targets(&page),
        ["docs/a.md", "shot.png", "assets/hero.png"]
    );
}
