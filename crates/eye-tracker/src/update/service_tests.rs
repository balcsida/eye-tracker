//! Tests for update/service.rs: when to look, what to announce, how an install goes.
//!
//! The Qt event loop (`qapp.processEvents`) is replaced by `Harness::wait_for`, which drains the
//! service's events until a condition holds; `service._tick()` is `scheduled_check`.
//! Not ported: `test_the_release_page_is_the_repository_s` only asserted two constants of
//! release.rs and is covered there by `test_names_and_addresses`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::json;
use sha2::{Digest, Sha256};

use super::super::fetch::tests::{FakeResponse, FakeTransport};
use super::super::installer::{KIND_PORTABLE, KIND_SETUP, KIND_UNSUPPORTED};
use super::super::release::{download_url, latest_release_url};
use super::*;
use crate::testing::{AppDirs, app_dirs};

const TAG: &str = "v0.2.2";
const NAME: &str = "EyeTracker-0.2.2-windows-x64-setup.exe";
const API_HOST: &str = "api.github.com";

fn body() -> Vec<u8> {
    let mut body = b"MZ".to_vec();
    body.extend([1u8; 4000]);
    body
}

fn api_path() -> String {
    latest_release_url()
        .strip_prefix("https://api.github.com")
        .unwrap()
        .to_string()
}

fn base() -> String {
    format!("/bugraskl/eye-tracker/releases/download/{TAG}")
}

fn release_json(tag: &str) -> Vec<u8> {
    let name = format!("EyeTracker-{}-windows-x64-setup.exe", &tag[1..]);
    json!({
        "tag_name": tag,
        "draft": false,
        "prerelease": false,
        "assets": [
            {
                "name": name,
                "size": body().len(),
                "browser_download_url": download_url(tag, &name),
            },
            {
                "name": "SHA256SUMS.txt",
                "size": 150,
                "browser_download_url": download_url(tag, "SHA256SUMS.txt"),
            },
        ],
    })
    .to_string()
    .into_bytes()
}

/// Every request gets an unread copy of the answer (a check can run twice).
fn transport_with(served: &[u8]) -> FakeTransport {
    let name = format!("EyeTracker-{}-windows-x64-setup.exe", &TAG[1..]);
    let listing = format!("{}  {name}\n", hex::encode(Sha256::digest(body())));
    FakeTransport::new(vec![
        (
            (API_HOST, &api_path()),
            FakeResponse::ok(&release_json(TAG)),
        ),
        (
            ("github.com", &format!("{}/SHA256SUMS.txt", base())),
            FakeResponse::ok(listing.as_bytes()),
        ),
        (
            ("github.com", &format!("{}/{name}", base())),
            FakeResponse::ok(served),
        ),
    ])
    .fresh()
}

fn transport() -> FakeTransport {
    transport_with(&body())
}

/// A service over a fake transport, with the events it handed out so far.
struct Harness {
    service: UpdateService,
    dirs: AppDirs,
    phases: Vec<Phase>,
    announced: Vec<String>,
    started: Arc<Mutex<Vec<PathBuf>>>,
    now: Arc<Mutex<f64>>,
}

struct Setup {
    transport: Option<FakeTransport>,
    running: &'static str,
    kind: &'static str,
    chunk: Option<usize>,
    supported: bool,
    launcher_error: Option<&'static str>,
    start: f64,
    /// Text of `updates.json` before the service starts.
    memory: Option<String>,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            transport: None,
            running: "0.2.1",
            kind: KIND_SETUP,
            chunk: None,
            supported: true,
            launcher_error: None,
            start: 1_000_000.0,
            memory: None,
        }
    }
}

fn make(setup: Setup) -> Harness {
    let dirs = app_dirs();
    if let Some(text) = &setup.memory {
        std::fs::write(dirs.path().join("updates.json"), text).unwrap();
    }
    let transport = setup.transport.unwrap_or_else(transport);
    let chunk = setup.chunk;
    let started = Arc::new(Mutex::new(Vec::new()));
    let now = Arc::new(Mutex::new(setup.start));
    let clock_now = Arc::clone(&now);
    let launched = Arc::clone(&started);
    let launcher_error = setup.launcher_error;
    let supported = setup.supported;
    let service = UpdateService::new(UpdateOptions {
        fetcher_factory: Arc::new(move || {
            Fetcher::with_transport(Box::new(transport.clone()))
                .with_chunk_bytes(chunk.unwrap_or(super::super::fetch::CHUNK_BYTES))
        }),
        clock: Box::new(move || *clock_now.lock()),
        running_version: setup.running.to_string(),
        memory_path: Some(dirs.path().join("updates.json")),
        kind: Some(setup.kind.to_string()),
        launcher: Box::new(move |path| {
            if let Some(message) = launcher_error {
                return Err(installer::UpdateError(message.to_string()).into());
            }
            launched.lock().push(path.to_path_buf());
            Ok(())
        }),
        supported: Box::new(move || supported),
    });
    Harness {
        service,
        dirs,
        phases: Vec::new(),
        announced: Vec::new(),
        started,
        now,
    }
}

impl Harness {
    /// Hand out the service's events and remember what they were.
    fn pump(&mut self) {
        for event in self.service.drain_events() {
            match event {
                Event::StateChanged(state) => self.phases.push(state.phase),
                Event::Announce(release) => self.announced.push(release.version),
            }
        }
    }

    fn wait_for(&mut self, mut predicate: impl FnMut(&UpdateService) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            self.pump();
            if predicate(&self.service) {
                return;
            }
            assert!(Instant::now() < deadline, "timed out");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn wait_for_phase(&mut self, phase: Phase) {
        self.wait_for(|service| service.state().phase == phase);
    }

    /// The phase is reached and the worker thread is done.
    fn wait_for_idle_phase(&mut self, phase: Phase) {
        self.wait_for(|service| service.state().phase == phase && !service.busy());
    }

    fn memory_file(&self) -> serde_json::Value {
        serde_json::from_str(
            &std::fs::read_to_string(self.dirs.path().join("updates.json")).unwrap(),
        )
        .unwrap()
    }

    fn started(&self) -> Vec<PathBuf> {
        self.started.lock().clone()
    }
}

fn updates_folder() -> PathBuf {
    installer::updates_dir()
}

fn file_names(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

// ------------------------------------------------------------------------- checking
#[test]
fn test_a_manual_check_finds_a_newer_release() {
    let mut h = make(Setup::default());
    assert!(h.service.check_now());
    h.wait_for_phase(Phase::Available);
    assert_eq!(h.phases, [Phase::Checking, Phase::Available]);
    assert_eq!(h.service.state().release.as_ref().unwrap().version, "0.2.2");
    assert!(h.service.state().manual);
    assert!(h.service.can_install());
}

#[test]
fn test_a_manual_check_of_the_latest_version_says_so() {
    let mut h = make(Setup {
        running: "0.2.2",
        ..Setup::default()
    });
    h.service.check_now();
    h.wait_for_phase(Phase::UpToDate);
    assert!(!h.service.can_install());
}

#[test]
fn test_a_failed_check_reports_why() {
    let fake = FakeTransport::new(vec![((API_HOST, &api_path()), FakeResponse::status(403))]);
    let mut h = make(Setup {
        transport: Some(fake),
        ..Setup::default()
    });
    h.service.check_now();
    h.wait_for_phase(Phase::Failed);
    assert!(h.service.state().message.contains("403"));
    assert_eq!(h.service.state().release, None);
}

#[test]
fn test_a_junk_answer_is_a_failed_check() {
    let fake = FakeTransport::new(vec![(
        (API_HOST, &api_path()),
        FakeResponse::ok(b"<html>captive portal</html>"),
    )]);
    let mut h = make(Setup {
        transport: Some(fake),
        ..Setup::default()
    });
    h.service.check_now();
    h.wait_for_phase(Phase::Failed);
    assert!(h.service.state().message.contains("could not be read"));
}

#[test]
fn test_checking_is_unavailable_where_there_is_no_transport() {
    let mut h = make(Setup {
        supported: false,
        ..Setup::default()
    });
    assert!(!h.service.available());
    assert!(!h.service.check_now());
    h.service.set_auto(true, 0.0);
    assert_eq!(h.service.next_tick(), None);
}

#[test]
fn test_a_second_check_waits_for_the_first() {
    let mut h = make(Setup::default());
    assert!(h.service.check_now());
    assert!(!h.service.check_now());
    h.wait_for_phase(Phase::Available);
}

// ------------------------------------------------------------------- daily schedule
#[test]
fn test_the_daily_check_is_off_unless_asked_for() {
    let mut h = make(Setup::default());
    h.service.start(false, 0.0);
    assert_eq!(h.service.next_tick(), None);
    h.service.set_auto(true, 0.0);
    assert!(h.service.next_tick().is_some());
    h.service.set_auto(false, 0.0);
    assert_eq!(h.service.next_tick(), None);
}

#[test]
fn test_the_first_check_is_not_at_startup() {
    let mut h = make(Setup::default());
    h.service.start(true, 10.0);
    assert!(h.service.next_tick().unwrap() - 10.0 > 60.0);
}

#[test]
fn test_an_automatic_check_announces_a_version_once() {
    let mut h = make(Setup::default());
    h.service.set_auto(true, 0.0);

    h.service.scheduled_check(0.0);
    h.wait_for_phase(Phase::Available);
    h.wait_for(|service| !service.busy());
    assert_eq!(h.announced, ["0.2.2"]);

    *h.now.lock() += CHECK_INTERVAL_S + 1.0; // the next day: found again, not announced again
    h.service.scheduled_check(TICK_INTERVAL_S);
    h.wait_for_idle_phase(Phase::Available);
    h.pump();
    assert_eq!(h.announced, ["0.2.2"]);
    assert_eq!(h.memory_file()["notified"], "0.2.2");
}

#[test]
fn test_the_check_is_not_repeated_within_a_day() {
    let fake = transport();
    let mut h = make(Setup {
        transport: Some(fake.clone()),
        start: 5_000_000.0,
        ..Setup::default()
    });
    h.service.set_auto(true, 0.0);
    h.service.scheduled_check(0.0);
    h.wait_for_idle_phase(Phase::Available);
    let asked = fake.asked().len();
    *h.now.lock() += 3600.0;
    h.service.scheduled_check(3600.0);
    h.pump();
    assert_eq!(fake.asked().len(), asked);
}

#[test]
fn test_a_clock_set_back_does_not_block_checks() {
    let mut h = make(Setup {
        start: 1000.0,
        memory: Some(json!({"last_check": 9e12, "notified": ""}).to_string()),
        ..Setup::default()
    });
    h.service.set_auto(true, 0.0);
    h.service.scheduled_check(0.0);
    h.wait_for_phase(Phase::Available);
}

#[test]
fn test_a_manual_check_does_not_announce() {
    let mut h = make(Setup::default());
    h.service.check_now();
    h.wait_for_phase(Phase::Available);
    h.pump();
    assert!(h.announced.is_empty());
}

#[test]
fn test_the_memory_survives_a_bad_file() {
    let h = make(Setup {
        memory: Some("{not json".into()),
        ..Setup::default()
    });
    assert_eq!(h.service.memory.last_check, 0.0);
    let path = h.dirs.path().join("updates.json");
    std::fs::write(
        &path,
        json!({"last_check": true, "notified": 3}).to_string(),
    )
    .unwrap();
    assert_eq!(Memory::load(path).last_check, 0.0);
}

// --------------------------------------------------------------------------- install
#[test]
fn test_the_installer_is_downloaded_verified_and_started() {
    let mut h = make(Setup::default());
    h.service.check_now();
    h.wait_for_idle_phase(Phase::Available);
    assert!(h.service.install());
    h.wait_for_phase(Phase::Installing);
    assert_eq!(h.started(), [updates_folder().join(NAME)]);
    assert_eq!(std::fs::read(&h.started()[0]).unwrap(), body());
    assert!(h.phases.contains(&Phase::Downloading));
    assert_eq!(h.phases.last(), Some(&Phase::Installing));
}

#[test]
fn test_a_tampered_installer_is_not_started() {
    let mut tampered = body();
    *tampered.last_mut().unwrap() = b'!';
    let mut h = make(Setup {
        transport: Some(transport_with(&tampered)),
        ..Setup::default()
    });
    h.service.check_now();
    h.wait_for_idle_phase(Phase::Available);
    h.service.install();
    h.wait_for_phase(Phase::Failed);
    assert!(h.service.state().message.contains("checksum"));
    assert!(h.started().is_empty());
    assert!(file_names(&updates_folder()).is_empty());
    assert!(h.service.can_install()); // try again
}

#[test]
fn test_copies_the_app_does_not_own_are_never_installed() {
    for kind in [KIND_PORTABLE, KIND_UNSUPPORTED] {
        let mut h = make(Setup {
            kind,
            ..Setup::default()
        });
        h.service.check_now();
        h.wait_for_idle_phase(Phase::Available);
        assert!(!h.service.can_install());
        assert!(!h.service.install());
        assert!(h.started().is_empty());
    }
}

#[test]
fn test_nothing_is_installed_before_a_check_found_something() {
    let mut h = make(Setup::default());
    assert!(!h.service.install());
    assert_eq!(*h.service.state(), UpdateState::default());
}

#[test]
fn test_a_failing_launcher_is_reported() {
    let mut h = make(Setup {
        launcher_error: Some("blocked"),
        ..Setup::default()
    });
    h.service.check_now();
    h.wait_for_idle_phase(Phase::Available);
    h.service.install();
    h.wait_for_phase(Phase::Failed);
    assert_eq!(h.service.state().message, "blocked");
}

#[test]
fn test_a_download_can_be_cancelled() {
    // A body that arrives slowly, so that a download can be cancelled half way.
    let slow = transport();
    slow.insert(
        "github.com",
        &format!("{}/{NAME}", base()),
        FakeResponse::ok(&body()).slow(),
    );
    let mut h = make(Setup {
        transport: Some(slow),
        chunk: Some(100),
        ..Setup::default()
    });
    h.service.check_now();
    h.wait_for_idle_phase(Phase::Available);
    assert!(h.service.install());
    h.service.cancel();
    h.wait_for_phase(Phase::Available);
    assert!(h.started().is_empty());
    h.wait_for(|service| !service.busy());
    assert!(file_names(&updates_folder()).is_empty());
    assert!(h.service.can_install()); // the user may start it again
}

#[test]
fn test_old_downloads_are_cleared_at_start() {
    let mut h = make(Setup::default());
    let folder = updates_folder();
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(folder.join(NAME), b"left over").unwrap();
    h.service.start(false, 0.0);
    assert!(file_names(&folder).is_empty());
}
