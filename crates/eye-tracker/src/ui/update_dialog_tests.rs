//! Tests for the update window and the settings switch (`tests/test_update_ui.py`).
//!
//! The window's state is what [`update_view`] makes of an [`UpdateState`]; `can_install` is
//! what `UpdateService::can_install` reports (its rule is tested in `update/service_tests.rs`).
//!
//! Not ported here (they belong to the tray and the app):
//! * `test_the_tray_names_the_new_version`, `test_the_tray_entry_opens_the_update_window`,
//!   `test_the_tray_entry_is_hidden_where_updates_cannot_be_checked`: `ui/tray_tests.rs`.
//! * `test_the_app_starts_with_the_check_off`, `test_turning_the_setting_on_schedules_the_check`,
//!   `test_a_new_version_is_announced_and_the_notification_opens_the_window`,
//!   `test_being_up_to_date_resets_the_tray_entry`,
//!   `test_the_window_opens_once_and_is_closed_with_the_app`: `app_tests.rs`.

use std::collections::HashMap;

use super::*;
use crate::REPO_URL;
use crate::config::Settings;
use crate::platform::autostart::{AutostartError, Status};
use crate::ui::settings_dialog::{AutostartBackend, SettingsDialog};
use crate::update::release::{Asset, ReleaseInfo, download_url};

const TAG: &str = "v0.2.2";
const NAME: &str = "EyeTracker-0.2.2-windows-x64-setup.exe";

fn release(installable: bool) -> ReleaseInfo {
    ReleaseInfo {
        version: "0.2.2".into(),
        tag: TAG.into(),
        page_url: format!("{REPO_URL}/releases/tag/{TAG}"),
        installer: installable.then(|| Asset {
            name: NAME.into(),
            url: download_url(TAG, NAME),
            size: 100,
        }),
        checksums: installable.then(|| Asset {
            name: "SHA256SUMS.txt".into(),
            url: download_url(TAG, "SHA256SUMS.txt"),
            size: 50,
        }),
    }
}

fn state(phase: Phase, release: Option<ReleaseInfo>) -> UpdateState {
    UpdateState {
        phase,
        release,
        ..Default::default()
    }
}

// ------------------------------------------------------------------------ the window
#[test]
fn test_an_installable_update_offers_install_and_later() {
    let view = update_view(&state(Phase::Available, Some(release(true))), true);
    assert!(view.title.contains("0.2.2"));
    assert_eq!(view.primary, Some(("Install and restart", Choice::Install)));
    assert_eq!(view.secondary, Some(("Later", Choice::Close)));
    // The link to what is new.
    assert!(view.link.unwrap().1.contains(REPO_URL));
}

#[test]
fn test_a_copy_that_cannot_update_itself_points_to_the_page() {
    // A portable copy: the service cannot install.
    let view = update_view(&state(Phase::Available, Some(release(true))), false);
    assert_eq!(
        view.primary,
        Some(("Open the download page", Choice::OpenDownloadPage))
    );
    assert!(view.detail.contains("cannot update itself"));
}

#[test]
fn test_a_release_without_a_verifiable_installer_is_not_installed() {
    let unverifiable = release(false);
    // What UpdateService::can_install says for it.
    assert!(!unverifiable.installable());
    let view = update_view(&state(Phase::Available, Some(unverifiable)), false);
    assert_eq!(
        view.primary,
        Some(("Open the download page", Choice::OpenDownloadPage))
    );
}

#[test]
fn test_the_download_shows_progress_and_can_be_cancelled() {
    let mut downloading = state(Phase::Downloading, Some(release(true)));
    downloading.done = 5_000_000;
    downloading.total = 20_000_000;
    let view = update_view(&downloading, false);
    let Bar::Progress { done, total, text } = &view.bar else {
        panic!("no progress bar: {:?}", view.bar);
    };
    assert_eq!(*total, 20_000_000);
    assert_eq!(*done, 5_000_000);
    assert!(text.contains("5.0 / 20.0 MB"));
    assert_eq!(view.primary, Some(("Cancel", Choice::Cancel)));
    let mut dialog = UpdateDialog::new();
    assert_eq!(dialog.choose(Choice::Cancel), Some(UpdateAction::Cancel));
}

#[test]
fn test_installing_has_no_buttons_and_failing_offers_another_try() {
    let view = update_view(&state(Phase::Installing, Some(release(true))), false);
    assert_eq!(view.primary, None);
    let mut failed = state(Phase::Failed, Some(release(true)));
    failed.message = "The download does not match its checksum.".into();
    let view = update_view(&failed, true);
    assert!(view.detail.contains("checksum"));
    assert_eq!(view.primary, Some(("Try again", Choice::Install)));
    assert!(view.secondary.is_some());
}

#[test]
fn test_checking_and_up_to_date() {
    let view = update_view(&state(Phase::Checking, None), false);
    assert_eq!(view.bar, Bar::Busy); // busy indicator
    let view = update_view(&state(Phase::UpToDate, Some(release(true))), false);
    assert_eq!(view.bar, Bar::Hidden);
    assert!(view.detail.contains("up to date"));
    assert_eq!(view.primary, Some(("Close", Choice::Close)));
}

#[test]
fn test_presenting_looks_for_an_update_only_when_none_is_known() {
    let mut dialog = UpdateDialog::new();
    // Nothing known yet.
    assert_eq!(
        dialog.present(&state(Phase::Idle, None)),
        [UpdateAction::CheckNow]
    );
    assert!(dialog.is_visible());
    // One is known: shown, not asked again.
    assert!(
        dialog
            .present(&state(Phase::Available, Some(release(true))))
            .is_empty()
    );
    // A failed install keeps its release for another try.
    let mut failed = state(Phase::Failed, Some(release(true)));
    failed.message = "boom".into();
    assert!(dialog.present(&failed).is_empty());
    let mut offline = state(Phase::Failed, None);
    offline.message = "offline".into();
    assert_eq!(dialog.present(&offline), [UpdateAction::CheckNow]);
}

#[test]
fn test_a_failed_check_offers_to_check_again() {
    let view = update_view(&state(Phase::Failed, None), false);
    assert_eq!(view.primary, Some(("Try again", Choice::CheckNow)));
}

// ----------------------------------------------------------------------- the settings
struct NoAutostart;

impl AutostartBackend for NoAutostart {
    fn is_supported(&self) -> bool {
        false
    }
    fn is_enabled(&self) -> bool {
        false
    }
    fn enable(&self) -> Result<(), AutostartError> {
        Err(AutostartError("unsupported".into()))
    }
    fn disable(&self) -> Result<(), AutostartError> {
        Err(AutostartError("unsupported".into()))
    }
    fn status(&self) -> Status {
        Status::Disabled
    }
}

fn settings_dialog() -> SettingsDialog {
    SettingsDialog::new(
        &Settings::default(),
        HashMap::new(),
        None,
        Box::new(NoAutostart),
    )
}

#[test]
fn test_the_settings_offer_the_daily_check() {
    let mut dialog = settings_dialog();
    assert!(!dialog.checked("updates.check")); // off by default
    assert!(!dialog.settings().updates.check);
    dialog.set_checked("updates.check", true);
    assert!(dialog.settings().updates.check);
}

#[test]
fn test_where_updates_cannot_be_checked_the_switch_is_shown_but_disabled() {
    let mut dialog = settings_dialog();
    dialog.update_supported = false;
    assert!(!dialog.is_enabled("updates.check"));
    assert!(!dialog.checked("updates.check"));
}

// ============================================================ egui (not in Python)
#[test]
fn test_every_phase_draws_headless() {
    let ctx = egui::Context::default();
    let mut dialog = UpdateDialog::new();
    dialog.present(&UpdateState::default());
    for phase in [
        Phase::Idle,
        Phase::Checking,
        Phase::UpToDate,
        Phase::Available,
        Phase::Downloading,
        Phase::Installing,
        Phase::Failed,
    ] {
        let state = state(phase, Some(release(true)));
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let inputs = UpdateInputs {
                state: &state,
                can_install: true,
            };
            assert!(dialog.show(ui.ctx(), &inputs).is_empty());
        });
        output.textures_delta.clear();
    }
}
