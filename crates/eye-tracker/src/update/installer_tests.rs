//! Tests for update/installer.rs: verifying and starting the downloaded setup program.
//!
//! Not ported as such: `test_a_source_checkout_is_not_updated_by_the_app` ran with
//! `is_frozen()` false; the Rust app has no such case, so `install_kind_for(…, frozen = false, …)`
//! stands in for it. `test_the_setup_is_started_silently_and_detached` cannot look at
//! `close_fds`/`stdin` of the child (std's `Command` does not expose them); it checks the program
//! and the arguments.

use sha2::{Digest, Sha256};

use super::super::fetch::tests::{FakeResponse, FakeTransport};
use super::super::release::{Asset, download_url};
use super::*;
use crate::REPO_URL;

const TAG: &str = "v0.2.1";
const NAME: &str = "EyeTracker-0.2.1-windows-x64-setup.exe";

fn body() -> Vec<u8> {
    let mut body = b"MZ".to_vec();
    body.extend([0u8; 5000]);
    body
}

fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

fn release(size: u64, with_checksums: bool) -> ReleaseInfo {
    ReleaseInfo {
        version: "0.2.1".into(),
        tag: TAG.into(),
        page_url: format!("{REPO_URL}/releases/tag/{TAG}"),
        installer: Some(Asset {
            name: NAME.into(),
            url: download_url(TAG, NAME),
            size,
        }),
        checksums: with_checksums.then(|| Asset {
            name: "SHA256SUMS.txt".into(),
            url: download_url(TAG, "SHA256SUMS.txt"),
            size: 200,
        }),
    }
}

fn default_release() -> ReleaseInfo {
    release(body().len() as u64, true)
}

fn sums(digest: &str, name: &str) -> Vec<u8> {
    let other = "a".repeat(64);
    format!("{other}  EyeTracker-0.2.1-linux-x86_64.tar.gz\n{digest}  {name}\n").into_bytes()
}

fn transport(served: &[u8], listing: Option<Vec<u8>>) -> FakeTransport {
    let listing = listing.unwrap_or_else(|| sums(&sha256_hex(&body()), NAME));
    FakeTransport::new(vec![
        (
            (
                "github.com",
                &format!("/bugraskl/eye-tracker/releases/download/{TAG}/SHA256SUMS.txt"),
            ),
            FakeResponse::ok(&listing),
        ),
        (
            (
                "github.com",
                &format!("/bugraskl/eye-tracker/releases/download/{TAG}/{NAME}"),
            ),
            FakeResponse::ok(served),
        ),
    ])
}

fn fetcher(transport: &FakeTransport) -> Fetcher {
    Fetcher::with_transport(Box::new(transport.clone()))
}

fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

// ------------------------------------------------------------------------- checksums
#[test]
fn test_checksum_lists_are_read_like_sha256sum_writes_them() {
    let digest = "AB".repeat(32);
    let parsed = parse_checksums(&format!(
        "{digest}  a.zip\n{} *b.exe\nnot a checksum line\n{}  short.zip\n",
        "cd".repeat(32),
        "e".repeat(63)
    ));
    let expected: HashMap<String, String> = [
        ("a.zip".to_string(), "ab".repeat(32)),
        ("b.exe".to_string(), "cd".repeat(32)),
    ]
    .into();
    assert_eq!(parsed, expected);
}

#[test]
fn test_file_sha256() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f");
    fs::write(&path, body()).unwrap();
    assert_eq!(file_sha256(&path).unwrap(), sha256_hex(&body()));
}

// -------------------------------------------------------------------------- download
#[test]
fn test_a_verified_download_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let mut seen: Vec<(u64, u64)> = Vec::new();
    let path = download_installer(
        &default_release(),
        &mut fetcher(&transport(&body(), None)),
        Some(&mut |done, total| seen.push((done, total))),
        None,
        Some(dir.path()),
    )
    .unwrap();
    assert_eq!(path, dir.path().join(NAME));
    assert_eq!(fs::read(&path).unwrap(), body());
    assert_eq!(names(dir.path()), [NAME]); // no .part left
    let size = body().len() as u64;
    assert_eq!(seen.last(), Some(&(size, size)));
}

#[test]
fn test_a_download_with_the_wrong_checksum_is_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let mut tampered = body();
    *tampered.last_mut().unwrap() = b'X';
    let error = download_installer(
        &default_release(),
        &mut fetcher(&transport(&tampered, None)),
        None,
        None,
        Some(dir.path()),
    )
    .unwrap_err();
    assert!(matches!(error, InstallError::Update(_)));
    assert!(error.to_string().contains("checksum"), "{error}");
    assert!(names(dir.path()).is_empty());
}

#[test]
fn test_a_short_download_is_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let error = download_installer(
        &release(body().len() as u64 + 10, true),
        &mut fetcher(&transport(&body(), None)),
        None,
        None,
        Some(dir.path()),
    )
    .unwrap_err();
    assert!(matches!(error, InstallError::Update(_)));
    assert!(error.to_string().contains("incomplete"), "{error}");
    assert!(names(dir.path()).is_empty());
}

#[test]
fn test_a_download_larger_than_announced_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let error = download_installer(
        &release(body().len() as u64 - 1, true),
        &mut fetcher(&transport(&body(), None)),
        None,
        None,
        Some(dir.path()),
    )
    .unwrap_err();
    assert!(matches!(error, InstallError::Network(_)));
    assert!(error.to_string().contains("larger"), "{error}");
    assert!(names(dir.path()).is_empty());
}

#[test]
fn test_an_installer_missing_from_the_checksum_list_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let listing = sums(&sha256_hex(&body()), "something-else.exe");
    let t = transport(&body(), Some(listing));
    let error = download_installer(
        &default_release(),
        &mut fetcher(&t),
        None,
        None,
        Some(dir.path()),
    )
    .unwrap_err();
    assert!(error.to_string().contains("does not name"), "{error}");
    // The installer itself was never asked for.
    assert!(
        t.asked()
            .iter()
            .all(|(_, path, _)| path.ends_with("SHA256SUMS.txt"))
    );
}

#[test]
fn test_a_release_without_checksums_is_not_installed() {
    let dir = tempfile::tempdir().unwrap();
    let error = download_installer(
        &release(body().len() as u64, false),
        &mut fetcher(&transport(&body(), None)),
        None,
        None,
        Some(dir.path()),
    )
    .unwrap_err();
    assert!(error.to_string().contains("no installer"), "{error}");
}

#[test]
fn test_a_cancelled_download_leaves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let error = download_installer(
        &default_release(),
        &mut fetcher(&transport(&body(), None)),
        None,
        Some(&|| true),
        Some(dir.path()),
    )
    .unwrap_err();
    assert!(matches!(error, InstallError::Cancelled));
    assert!(names(dir.path()).is_empty());
}

#[test]
fn test_an_older_download_is_replaced() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join(NAME), b"old").unwrap();
    fs::write(dir.path().join(format!("{NAME}.part")), b"older").unwrap();
    let path = download_installer(
        &default_release(),
        &mut fetcher(&transport(&body(), None)),
        None,
        None,
        Some(dir.path()),
    )
    .unwrap();
    assert_eq!(fs::read(path).unwrap(), body());
    assert_eq!(names(dir.path()), [NAME]);
}

#[test]
fn test_leftover_downloads_are_removed() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.exe"), b"1").unwrap();
    fs::write(dir.path().join("b.part"), b"2").unwrap();
    remove_downloads_in(dir.path());
    assert!(names(dir.path()).is_empty());
    remove_downloads_in(&dir.path().join("missing")); // nothing to do, no error
}

// ---------------------------------------------------------------------------- kinds
#[test]
fn test_a_source_checkout_is_not_updated_by_the_app() {
    let exe = Path::new("/nowhere/EyeTracker.exe");
    assert_eq!(install_kind_for(true, false, exe), KIND_UNSUPPORTED);
}

#[test]
fn test_other_platforms_are_not_updated_by_the_app() {
    let exe = Path::new("/nowhere/EyeTracker.exe");
    assert_eq!(install_kind_for(false, true, exe), KIND_UNSUPPORTED);
}

#[test]
fn test_the_installed_app_is_told_by_its_uninstaller() {
    let dir = tempfile::tempdir().unwrap();
    let exe = dir.path().join("EyeTracker.exe");
    assert_eq!(install_kind_for(true, true, &exe), KIND_PORTABLE);
    fs::write(dir.path().join("unins000.exe"), b"").unwrap();
    assert_eq!(install_kind_for(true, true, &exe), KIND_SETUP);
}

// --------------------------------------------------------------------------- launch
#[test]
fn test_the_setup_is_started_silently_and_detached() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(NAME);
    let mut calls: Vec<(std::ffi::OsString, Vec<std::ffi::OsString>)> = Vec::new();
    launch(&path, KIND_SETUP, |command| {
        calls.push((
            command.get_program().to_owned(),
            command.get_args().map(ToOwned::to_owned).collect(),
        ));
        Ok(())
    })
    .unwrap();
    let (program, args) = &calls[0];
    assert_eq!(program, path.as_os_str());
    assert_eq!(
        args,
        &["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"].map(std::ffi::OsString::from)
    );
}

#[test]
fn test_the_setup_is_not_started_for_copies_it_does_not_own() {
    for kind in [KIND_PORTABLE, KIND_UNSUPPORTED] {
        let dir = tempfile::tempdir().unwrap();
        let mut started = 0;
        let error = launch(&dir.path().join(NAME), kind, |_| {
            started += 1;
            Ok(())
        })
        .unwrap_err();
        assert!(matches!(error, InstallError::Update(_)));
        assert_eq!(started, 0);
    }
}
