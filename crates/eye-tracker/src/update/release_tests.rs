//! Tests for update/release.rs: versions and what is trusted from a release answer.
//!
//! Not ported: the `None` and `7` cases of `test_parse_stable_accepts_only_plain_versions`
//! (`parse_stable` takes a `&str`, so a non-string cannot be passed; the JSON side of that is
//! covered by `test_only_stable_published_releases_count` with `tag_name: null`).

use serde_json::{Value, json};

use super::*;

const TAG: &str = "v0.2.1";
const SETUP: &str = "EyeTracker-0.2.1-windows-x64-setup.exe";

fn payload(changes: Value) -> Value {
    let mut data = json!({
        "tag_name": TAG,
        "draft": false,
        "prerelease": false,
        "html_url": "https://evil.example/release",
        "assets": [
            {
                "name": SETUP,
                "size": 63_308_253,
                "browser_download_url": format!("{REPO_URL}/releases/download/{TAG}/{SETUP}"),
            },
            {
                "name": "SHA256SUMS.txt",
                "size": 520,
                "browser_download_url": format!("{REPO_URL}/releases/download/{TAG}/SHA256SUMS.txt"),
            },
        ],
    });
    for (key, value) in changes.as_object().into_iter().flatten() {
        data[key] = value.clone();
    }
    data
}

// --------------------------------------------------------------------------- versions
#[test]
fn test_parse_stable_accepts_only_plain_versions() {
    let cases: [(&str, Option<Version>); 9] = [
        ("0.2.1", Some((0, 2, 1))),
        ("v1.10.0", Some((1, 10, 0))),
        (" 2.0.0 ", Some((2, 0, 0))),
        ("0.2", None),
        ("0.2.1rc1", None),
        ("0.2.1.dev3", None),
        ("v0.2.1-beta", None),
        ("latest", None),
        ("", None),
    ];
    for (text, expected) in cases {
        assert_eq!(parse_stable(text), expected, "{text:?}");
    }
}

#[test]
fn test_the_running_version_keeps_its_leading_numbers() {
    let cases: [(&str, Option<Version>); 4] = [
        ("0.2.1", Some((0, 2, 1))),
        ("0.3.0.dev1", Some((0, 3, 0))),
        ("v1.0.0rc2", Some((1, 0, 0))),
        ("dev", None),
    ];
    for (text, expected) in cases {
        assert_eq!(parse_running(text), expected, "{text:?}");
    }
}

#[test]
fn test_versions_compare_numerically() {
    let newer = parse_release(&payload(json!({"tag_name": "v0.10.0"}))).unwrap();
    assert!(is_newer(&newer, "0.9.9")); // not as text
    assert!(!is_newer(&newer, "0.10.0"));
    assert!(!is_newer(&newer, "0.10.1"));
    assert!(!is_newer(&newer, "0.10.0.dev1")); // a dev build of it is not older
}

#[test]
fn test_an_unknown_running_version_is_never_updated() {
    let release = parse_release(&payload(json!({}))).unwrap();
    assert!(!is_newer(&release, "unknown"));
}

// -------------------------------------------------------------------------- the release
#[test]
fn test_a_release_keeps_its_installer_and_checksums() {
    let release = parse_release(&payload(json!({}))).unwrap();
    assert_eq!(release.version, "0.2.1");
    assert_eq!(release.tag, TAG);
    assert_eq!(
        release.installer,
        Some(Asset {
            name: SETUP.to_string(),
            url: format!("{REPO_URL}/releases/download/{TAG}/{SETUP}"),
            size: 63_308_253,
        })
    );
    assert_eq!(release.checksums.unwrap().name, CHECKSUMS_NAME);
    assert!(parse_release(&payload(json!({}))).unwrap().installable());
}

#[test]
fn test_the_page_is_built_here_not_taken_from_the_answer() {
    let release = parse_release(&payload(json!({}))).unwrap();
    assert_eq!(release.page_url, format!("{REPO_URL}/releases/tag/{TAG}"));
}

#[test]
fn test_only_stable_published_releases_count() {
    let bad = [
        json!({"draft": true}),
        json!({"prerelease": true}),
        json!({"tag_name": "v0.3.0-rc1"}),
        json!({"tag_name": "nightly"}),
        json!({"tag_name": null}),
        json!({"tag_name": "v1.2.3\nHost: evil"}),
    ];
    for changes in bad {
        assert_eq!(parse_release(&payload(changes.clone())), None, "{changes}");
    }
}

#[test]
fn test_a_junk_answer_is_not_a_release() {
    for junk in [
        json!(null),
        json!([]),
        json!("release"),
        json!(3),
        json!({}),
    ] {
        assert_eq!(parse_release(&junk), None, "{junk}");
    }
}

#[test]
fn test_an_asset_must_be_served_from_the_repository() {
    let mut data = payload(json!({}));
    data["assets"][0]["browser_download_url"] = json!(format!("https://evil.example/{SETUP}"));
    let release = parse_release(&data).unwrap();
    assert_eq!(release.installer, None);
    assert!(!release.installable());
}

#[test]
fn test_an_asset_of_another_tag_is_refused() {
    let mut data = payload(json!({}));
    data["assets"][0]["browser_download_url"] =
        json!(format!("{REPO_URL}/releases/download/v9.9.9/{SETUP}"));
    let release = parse_release(&data).unwrap();
    assert_eq!(release.installer, None);
}

#[test]
fn test_an_asset_needs_a_sane_size() {
    let sizes = [
        json!(0),
        json!(-1),
        json!(true),
        json!("63"),
        json!(null),
        json!(MAX_INSTALLER_BYTES + 1),
    ];
    for size in sizes {
        let mut data = payload(json!({}));
        data["assets"][0]["size"] = size.clone();
        let release = parse_release(&data).unwrap();
        assert_eq!(release.installer, None, "{size}");
    }
}

#[test]
fn test_without_a_checksum_list_nothing_is_installable() {
    let mut data = payload(json!({}));
    let first = data["assets"][0].clone();
    data["assets"] = json!([first]);
    let release = parse_release(&data).unwrap();
    assert!(release.installer.is_some());
    assert_eq!(release.checksums, None);
    assert!(!release.installable());
}

#[test]
fn test_a_release_without_assets_still_names_its_version() {
    let release = parse_release(&payload(json!({"assets": null}))).unwrap();
    assert_eq!(release.installer, None);
    assert_eq!(release.version, "0.2.1");
}

#[test]
fn test_names_and_addresses() {
    assert_eq!(installer_name("0.2.1"), SETUP);
    assert_eq!(
        download_url(TAG, SETUP),
        format!("{REPO_URL}/releases/download/{TAG}/{SETUP}")
    );
    assert_eq!(
        latest_release_url(),
        "https://api.github.com/repos/bugraskl/eye-tracker/releases/latest"
    );
}

#[test]
fn test_the_names_the_app_looks_for_are_the_ones_the_release_workflow_publishes() {
    let workflow = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.github/workflows/release.yml");
    let Ok(text) = std::fs::read_to_string(workflow) else {
        return; // the workflows are not part of this source tree
    };
    // Renaming either file would make every installed app stop finding its update.
    assert!(text.contains(&installer_name("$VERSION")));
    assert!(text.contains(CHECKSUMS_NAME));
}
