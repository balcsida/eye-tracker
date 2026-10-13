//! Start at login ("autostart") on Windows, macOS and Linux.
//!
//! Each OS uses its native per-user mechanism; none needs administrator rights:
//!
//! * **Windows**: a `HKCU\...\CurrentVersion\Run` value named `EyeTracker` (the same value the
//!   installer's "Start at login" task writes).
//! * **macOS**: a LaunchAgent plist in `~/Library/LaunchAgents`. It is only written, never
//!   loaded with `launchctl`: launchd picks it up at next login, which is exactly when it is
//!   needed. A switch-off that launchd records in its own database (`launchctl disable`,
//!   possibly the Login Items switch in System Settings) leaves the plist untouched; it is read
//!   back with `launchctl print-disabled` and cleared by [`enable`] with `launchctl enable`,
//!   which does not load the agent either.
//! * **Linux**: an XDG autostart entry, `$XDG_CONFIG_HOME/autostart/eye-tracker.desktop`.
//!
//! # Profiles
//!
//! Every function takes `config_dir`: the `--config-dir` profile the entry is for. `None` means
//! the profile of the running process (its `--config-dir`, if it was started with one). A
//! non-default profile's entry carries `--config-dir <dir>` in its command, so the login launch
//! opens the same settings and calibration and is recognised as the same instance.
//!
//! There is still only one entry per user: the camera and the cursor can serve one tracker, so
//! turning start at login on in one profile replaces another profile's entry, and [`status`]
//! reports such an entry as [`Status::OtherProfile`] rather than as enabled.
//!
//! # Stale entries
//!
//! An entry records the path of the program that wrote it. That path can go away: a replaced
//! AppImage, a moved portable folder, or a macOS app that ran from its disk image or from
//! Gatekeeper's App Translocation mount. [`status`] reports those as [`Status::Stale`],
//! [`enable`] refuses to register a macOS app from such a location, and [`refresh`] (called by
//! the app at startup) re-points a stale entry at the running copy.
//!
//! Everything the functions touch outside their arguments (the OS, the registry, the
//! filesystem locations, `launchctl`, the command line of the running copy) is held by a
//! private `Host`, so tests can redirect it.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::config::atomic_write_text;
use crate::platform::base::which;
use crate::{APP_ID, APP_NAME, APP_SLUG, cli, paths};

pub const BACKGROUND_FLAG: &str = "--background";
pub const CONFIG_DIR_FLAG: &str = "--config-dir";
pub const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
/// Where Task Manager's "Startup apps" page records entries the user disabled.
pub const STARTUP_APPROVED_KEY: &str =
    r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run";
pub const RUN_VALUE_NAME: &str = "EyeTracker";
pub const LAUNCH_AGENT_LABEL: &str = APP_ID;
pub const DESKTOP_FILE_NAME: &str = "eye-tracker.desktop";

/// Characters that force quoting inside a Desktop Entry `Exec` value.
const DESKTOP_RESERVED: &str = " \t\n\"'\\><~|&;$*?#()`";
/// Characters escaped with a backslash inside a quoted `Exec` argument.
const DESKTOP_QUOTED_ESCAPES: &str = "\"`$\\";

const TRANSIENT_LOCATION_MESSAGE: &str = "Eye Tracker is running from its disk image or from a \
temporary location, which will not exist at the next login. Move it to the Applications folder, \
open it from there, then turn on Start at login.";
const LOGIN_ITEMS_MESSAGE: &str = "macOS keeps Eye Tracker from starting at login. Allow it in \
System Settings › General › Login Items (under “Allow in the Background”).";

/// launchd's command-line tool, by absolute path (an app may start with a bare PATH).
const LAUNCHCTL: &str = "/bin/launchctl";
const LAUNCHCTL_TIMEOUT: Duration = Duration::from_secs(2);

/// Enabling or disabling start at login failed (the message is user-presentable).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct AutostartError(pub String);

/// State of the login entry, as seen from one `--config-dir` profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The entry starts this profile, and its program exists.
    Enabled,
    /// No entry, or one that was switched off (Task Manager, GNOME Tweaks, launchd).
    Disabled,
    /// The entry starts this profile, but its program is gone or sits in a temporary location
    /// (macOS disk image, App Translocation).
    Stale,
    /// The entry starts another `--config-dir` profile.
    OtherProfile,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Enabled => "enabled",
            Status::Disabled => "disabled",
            Status::Stale => "stale",
            Status::OtherProfile => "other-profile",
        }
    }
}

/// A login entry as found on disk or in the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
    /// Its command line split into arguments (empty when there is none).
    command: Vec<String>,
    /// `false` when the OS or desktop switched it off without deleting it.
    active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum System {
    Windows,
    MacOs,
    Linux,
    Unsupported,
}

/// A registry value, as far as this module distinguishes them.
// Only the Windows registry (and the tests) produce values.
#[cfg_attr(not(windows), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
enum RegValue {
    Str(String),
    Bytes(Vec<u8>),
    /// Any other type (`REG_DWORD`, `REG_MULTI_SZ` ...).
    Other,
}

/// `HKCU` access; tests replace it with a dictionary.
trait Registry {
    /// Value `name` under `HKCU\key_path`, `None` when missing.
    fn read(&self, key_path: &str, name: &str) -> io::Result<Option<RegValue>>;
    /// Write a `REG_SZ` value under `HKCU\key_path` (creating the key).
    fn write_str(&self, key_path: &str, name: &str, value: &str) -> io::Result<()>;
    /// Delete value `name` under `HKCU\key_path`; missing is fine.
    fn delete(&self, key_path: &str, name: &str) -> io::Result<()>;
}

type LaunchctlFn = dyn Fn(&[&str]) -> Option<String>;
type WriteTextFn = dyn Fn(&Path, &str) -> io::Result<()>;

/// What the module acts on; [`Host::real`] is the running process on the running OS.
struct Host {
    system: System,
    registry: Box<dyn Registry>,
    mac_plist_path: PathBuf,
    linux_desktop_path: PathBuf,
    /// Runs `launchctl` and returns its output; `None` when it failed.
    launchctl: Box<LaunchctlFn>,
    /// launchd's domain for this user's login session (`gui/<uid>`); `None` off POSIX.
    launchd_domain: Option<String>,
    /// The command line that starts the running copy ([`cli::app_command`]).
    app_command: Box<dyn Fn() -> Vec<String>>,
    /// The `--config-dir` of this process; `None` for the default profile.
    active_profile: Option<PathBuf>,
    /// Whether a bare program name is found on the PATH.
    on_path: Box<dyn Fn(&str) -> bool>,
    /// Whether a path is on a read-only filesystem (a mounted disk image).
    is_read_only_volume: Box<dyn Fn(&str) -> bool>,
    write_text: Box<WriteTextFn>,
}

impl Host {
    fn real() -> Host {
        let home = home_dir();
        Host {
            system: system_for(std::env::consts::OS),
            registry: Box::new(RealRegistry),
            mac_plist_path: mac_plist_path_in(&home),
            linux_desktop_path: linux_desktop_path_in(
                &std::env::var("XDG_CONFIG_HOME").unwrap_or_default(),
                &home,
            ),
            launchctl: Box::new(|args| run_program(LAUNCHCTL, args, LAUNCHCTL_TIMEOUT)),
            launchd_domain: launchd_domain(),
            app_command: Box::new(cli::app_command),
            active_profile: paths::base_override(),
            on_path: Box::new(|name| which(name).is_some()),
            is_read_only_volume: Box::new(is_read_only_volume),
            write_text: Box::new(write_text),
        }
    }
}

// ------------------------------------------------------------------ public API

/// Whether this OS has an autostart implementation.
pub fn is_supported() -> bool {
    Host::real().system != System::Unsupported
}

/// Command line that starts this installation of the app ([`cli::app_command`]: the
/// `.AppImage` file, or the windowed executable, which is inside the `.app` bundle on macOS).
///
/// A non-default profile (`config_dir`, see the module docs) adds `--config-dir <dir>`.
/// `background` appends `--background` (no first-run wizard or startup notifications), which is
/// what a login launch wants.
pub fn launch_command(background: bool, config_dir: Option<&Path>) -> Vec<String> {
    Host::real().launch_command(background, config_dir)
}

/// The command of the login entry, split into arguments; `None` without one.
///
/// Includes entries that were switched off in Task Manager or the desktop's settings. Never
/// fails.
pub fn registered_command() -> Option<Vec<String>> {
    Host::real().registered_command()
}

/// State of the login entry for the `config_dir` profile. Never fails.
pub fn status(config_dir: Option<&Path>) -> Status {
    Host::real().status(config_dir)
}

/// Whether the app is registered to start the `config_dir` profile at login.
///
/// `false` for an entry whose program no longer exists or that starts another profile (see
/// [`status`]). Never fails.
pub fn is_enabled(config_dir: Option<&Path>) -> bool {
    status(config_dir) == Status::Enabled
}

/// Register the app to start the `config_dir` profile at login.
///
/// Replaces any previous entry, including another profile's. Fails with [`AutostartError`] on
/// failure, on an unsupported OS, or on macOS when the app runs from its disk image or a
/// translocated copy.
pub fn enable(background: bool, config_dir: Option<&Path>) -> Result<(), AutostartError> {
    Host::real().enable(background, config_dir)
}

/// Remove the login entry of the `config_dir` profile. A missing entry is not an error.
///
/// An entry that starts another profile is left alone (and logged): it is not this profile's to
/// remove. Fails with [`AutostartError`] when an existing entry cannot be removed.
pub fn disable(config_dir: Option<&Path>) -> Result<(), AutostartError> {
    Host::real().disable(config_dir)
}

/// Point this profile's login entry at the running copy of the app, if it moved.
///
/// Meant for app startup. The entry is rewritten when it is active, starts the `config_dir`
/// profile and records a different command than [`launch_command`] gives now, and
///
/// * its program is gone or in a temporary location (a replaced AppImage, a moved folder, a
///   macOS app first run from its disk image), or
/// * the program exists: this is a packaged build, which then replaces what the entry started
///   (e.g. an older AppImage).
///
/// It is never pointed at a temporary location. With several packaged copies of one profile,
/// the one started last wins. The `--background` choice of the entry is kept. Returns `true`
/// when the entry was rewritten. Never fails.
pub fn refresh(config_dir: Option<&Path>) -> bool {
    Host::real().refresh(config_dir)
}

/// Human-readable location of the login entry (for diagnostics).
pub fn location() -> Option<String> {
    Host::real().location()
}

// ------------------------------------------------------------------- UI seam

/// What the app, the tray, the settings and the setup assistant need from this module (tests
/// fake it).
pub trait AutostartBackend {
    fn is_supported(&self) -> bool;
    fn is_enabled(&self) -> bool;
    /// `None` when the backend cannot say more than "enabled or not".
    fn status(&self) -> Option<Status> {
        None
    }
    /// An [`AutostartError`] carries a message meant for the user; any other error is reported
    /// generically.
    fn enable(&self, background: bool) -> Result<(), Box<dyn std::error::Error>>;
    fn disable(&self) -> Result<(), Box<dyn std::error::Error>>;
    /// [`refresh`]: point the login item at this copy if it moved. Returns whether it changed.
    fn refresh(&self) -> bool {
        false
    }
}

/// The real login item of this process's settings profile.
pub struct SystemAutostart;

impl AutostartBackend for SystemAutostart {
    fn is_supported(&self) -> bool {
        is_supported()
    }

    fn is_enabled(&self) -> bool {
        is_enabled(None)
    }

    fn status(&self) -> Option<Status> {
        Some(status(None))
    }

    fn enable(&self, background: bool) -> Result<(), Box<dyn std::error::Error>> {
        enable(background, None).map_err(Into::into)
    }

    fn disable(&self) -> Result<(), Box<dyn std::error::Error>> {
        disable(None).map_err(Into::into)
    }

    fn refresh(&self) -> bool {
        refresh(None)
    }
}

impl Host {
    fn launch_command(&self, background: bool, config_dir: Option<&Path>) -> Vec<String> {
        // The profile of the running process is replaced by the one asked for.
        let base = (self.app_command)();
        let mut command: Vec<String> = base.first().cloned().into_iter().collect();
        command.extend(without_profile(&base));
        if let Some(profile) = self.profile_dir(config_dir) {
            command.push(CONFIG_DIR_FLAG.to_string());
            command.push(profile.to_string_lossy().into_owned());
        }
        if background {
            command.push(BACKGROUND_FLAG.to_string());
        }
        command
    }

    fn registered_command(&self) -> Option<Vec<String>> {
        match self.read_entry() {
            Ok(Some(entry)) if !entry.command.is_empty() => Some(entry.command),
            Ok(_) => None,
            Err(err) => {
                log::warn!("Could not read the start-at-login entry: {err}");
                None
            }
        }
    }

    fn status(&self, config_dir: Option<&Path>) -> Status {
        let entry = match self.read_entry() {
            Ok(entry) => entry,
            Err(err) => {
                log::warn!("Could not read the start-at-login state: {err}");
                return Status::Disabled;
            }
        };
        let Some(entry) = entry.filter(|e| e.active && !e.command.is_empty()) else {
            return Status::Disabled;
        };
        let profile = self.profile_dir(config_dir);
        if !self.same_path(
            command_profile(&entry.command).as_deref(),
            profile.as_deref().map(path_text).as_deref(),
        ) {
            return Status::OtherProfile;
        }
        if !self.program_available(&entry.command[0]) {
            return Status::Stale;
        }
        Status::Enabled
    }

    fn enable(&self, background: bool, config_dir: Option<&Path>) -> Result<(), AutostartError> {
        let command = self.launch_command(background, config_dir);
        if command
            .first()
            .is_some_and(|p| self.is_transient_location(p))
        {
            return Err(AutostartError(TRANSIENT_LOCATION_MESSAGE.to_string()));
        }
        let failed =
            |err: io::Error| AutostartError(format!("Could not enable start at login: {err}"));
        match self.system {
            System::Windows => self.win_enable(&command).map_err(failed)?,
            System::MacOs => self.mac_enable(&command)?,
            System::Linux => self.linux_enable(&command).map_err(failed)?,
            System::Unsupported => {
                return Err(AutostartError(format!(
                    "Start at login is not supported on {}",
                    std::env::consts::OS
                )));
            }
        }
        log::info!(
            "Start at login enabled ({})",
            self.location().unwrap_or_default()
        );
        Ok(())
    }

    fn disable(&self, config_dir: Option<&Path>) -> Result<(), AutostartError> {
        if self.system == System::Unsupported {
            return Ok(());
        }
        // An unreadable entry is removed all the same: it is the only sensible thing to do.
        if let Ok(Some(entry)) = self.read_entry()
            && !entry.command.is_empty()
        {
            let owner = command_profile(&entry.command);
            let profile = self.profile_dir(config_dir);
            if !self.same_path(
                owner.as_deref(),
                profile.as_deref().map(path_text).as_deref(),
            ) {
                log::info!(
                    "Start at login starts another profile ({}); left alone",
                    owner.as_deref().unwrap_or("default")
                );
                return Ok(());
            }
        }
        let removed = match self.system {
            System::Windows => self.win_disable(),
            System::MacOs => unlink(&self.mac_plist_path),
            _ => unlink(&self.linux_desktop_path),
        };
        removed
            .map_err(|err| AutostartError(format!("Could not disable start at login: {err}")))?;
        log::info!("Start at login disabled");
        Ok(())
    }

    fn refresh(&self, config_dir: Option<&Path>) -> bool {
        match self.refresh_entry(config_dir) {
            Ok(rewritten) => rewritten,
            Err(err) => {
                log::warn!("Could not update the start-at-login entry: {err}");
                false
            }
        }
    }

    fn refresh_entry(&self, config_dir: Option<&Path>) -> io::Result<bool> {
        let Some(entry) = self
            .read_entry()?
            .filter(|e| e.active && !e.command.is_empty())
        else {
            return Ok(false);
        };
        let profile = self.profile_dir(config_dir);
        let profile_text = profile.as_deref().map(path_text);
        if !self.same_path(
            command_profile(&entry.command).as_deref(),
            profile_text.as_deref(),
        ) {
            return Ok(false);
        }
        let background = entry.command[1..].iter().any(|a| a == BACKGROUND_FLAG);
        let wanted = self.launch_command(background, profile.as_deref());
        if self.same_command(&entry.command, &wanted) || self.is_transient_location(&wanted[0]) {
            return Ok(false);
        }
        // Every build is a packaged one, so a working entry is replaced by the running copy.
        self.enable(background, profile.as_deref())
            .map_err(|err| io::Error::other(err.0))?;
        log::info!(
            "Start at login updated: {} (was {})",
            wanted[0],
            entry.command[0]
        );
        Ok(true)
    }

    fn location(&self) -> Option<String> {
        match self.system {
            System::Windows => Some(format!("HKCU\\{RUN_KEY}\\{RUN_VALUE_NAME}")),
            System::MacOs => Some(self.mac_plist_path.display().to_string()),
            System::Linux => Some(self.linux_desktop_path.display().to_string()),
            System::Unsupported => None,
        }
    }

    // ------------------------------------------------------------------ command

    /// The profile an entry is for: `config_dir` resolved like `paths` does, or the running
    /// process's profile when `None`.
    fn profile_dir(&self, config_dir: Option<&Path>) -> Option<PathBuf> {
        match config_dir {
            None => self.active_profile.clone(),
            Some(dir) => Some(resolve(&expand_user(dir))),
        }
    }

    /// Comparable form of a path; case-insensitive for Windows entries.
    fn path_key(&self, path: &str) -> String {
        let text = normpath(&expand_user_text(path));
        if self.system == System::Windows {
            text.to_lowercase()
        } else {
            text
        }
    }

    fn same_path(&self, a: Option<&str>, b: Option<&str>) -> bool {
        match (a, b) {
            (Some(a), Some(b)) => self.path_key(a) == self.path_key(b),
            (a, b) => a.is_none() && b.is_none(),
        }
    }

    /// Whether two commands start the same program with the same profile and options.
    fn same_command(&self, recorded: &[String], wanted: &[String]) -> bool {
        !recorded.is_empty()
            && !wanted.is_empty()
            && self.same_path(Some(&recorded[0]), Some(&wanted[0]))
            && self.same_path(
                command_profile(recorded).as_deref(),
                command_profile(wanted).as_deref(),
            )
            && without_profile(recorded) == without_profile(wanted)
    }

    /// Whether an entry's program will still be there at the next login.
    fn program_available(&self, program: &str) -> bool {
        if program.is_empty() || self.is_transient_location(program) {
            return false;
        }
        let program = if self.system == System::Windows {
            expand_vars(program) // a hand-made REG_EXPAND_SZ entry
        } else {
            program.to_string()
        };
        if Path::new(&program).is_absolute() {
            return Path::new(&program).exists();
        }
        (self.on_path)(&program)
    }

    /// Whether `program` runs from a place that will be gone at the next login (macOS).
    ///
    /// Gatekeeper's App Translocation runs a quarantined app from a random mount under
    /// `/private/var/folders/.../AppTranslocation/`, and an app opened straight from its disk
    /// image lives on a read-only `/Volumes/` mount that disappears when the image is ejected.
    /// (Apps on an external drive also live under `/Volumes/`, but on a writable volume.)
    fn is_transient_location(&self, program: &str) -> bool {
        if self.system != System::MacOs {
            return false;
        }
        if program.contains("/AppTranslocation/") {
            return true;
        }
        program.starts_with("/Volumes/") && (self.is_read_only_volume)(program)
    }

    /// The login entry of the current OS, `None` when there is none.
    fn read_entry(&self) -> io::Result<Option<Entry>> {
        match self.system {
            System::Windows => self.win_entry(),
            System::MacOs => self.mac_entry(),
            System::Linux => self.linux_entry(),
            System::Unsupported => Ok(None),
        }
    }

    // ------------------------------------------------------------------ Windows

    fn win_entry(&self) -> io::Result<Option<Entry>> {
        let Some(RegValue::Str(value)) = self.registry.read(RUN_KEY, RUN_VALUE_NAME)? else {
            return Ok(None);
        };
        if value.trim().is_empty() {
            return Ok(None);
        }
        Ok(Some(Entry {
            command: win_split(&value),
            active: !self.win_disabled_in_task_manager()?,
        }))
    }

    /// Task Manager keeps the Run value but flags it; the first byte is odd when disabled.
    fn win_disabled_in_task_manager(&self) -> io::Result<bool> {
        let data = self.registry.read(STARTUP_APPROVED_KEY, RUN_VALUE_NAME)?;
        Ok(
            matches!(data, Some(RegValue::Bytes(bytes)) if bytes.first().is_some_and(|b| b & 1 == 1)),
        )
    }

    fn win_enable(&self, command: &[String]) -> io::Result<()> {
        self.registry
            .write_str(RUN_KEY, RUN_VALUE_NAME, &list2cmdline(command))?;
        // Clearing Task Manager's "disabled" flag makes the new entry effective; otherwise
        // the toggle in our UI would silently do nothing.
        self.registry.delete(STARTUP_APPROVED_KEY, RUN_VALUE_NAME)
    }

    fn win_disable(&self) -> io::Result<()> {
        self.registry.delete(RUN_KEY, RUN_VALUE_NAME)?;
        self.registry.delete(STARTUP_APPROVED_KEY, RUN_VALUE_NAME)
    }

    // ------------------------------------------------------------------ macOS

    fn mac_enable(&self, command: &[String]) -> Result<(), AutostartError> {
        (self.write_text)(&self.mac_plist_path, &mac_plist(command))
            .map_err(|err| AutostartError(format!("Could not enable start at login: {err}")))?;
        if !self.mac_disabled_in_launchd() {
            return Ok(());
        }
        // Switched off in launchd's records, which outlive the plist: without this the
        // checkbox would silently do nothing. `enable` only clears that record; it does not
        // load the agent, which still starts at the next login.
        if let Some(domain) = &self.launchd_domain {
            (self.launchctl)(&["enable", &format!("{domain}/{LAUNCH_AGENT_LABEL}")]);
        }
        if self.mac_disabled_in_launchd() {
            return Err(AutostartError(LOGIN_ITEMS_MESSAGE.to_string()));
        }
        Ok(())
    }

    /// Whether launchd itself keeps the agent from starting, whatever the plist says.
    ///
    /// `launchctl disable` (and `unload -w`) record a switch-off in launchd's per-user
    /// database and leave the plist untouched; the Login Items switch in System Settings may do
    /// so as well. `false` when that cannot be read.
    fn mac_disabled_in_launchd(&self) -> bool {
        let Some(domain) = &self.launchd_domain else {
            return false;
        };
        (self.launchctl)(&["print-disabled", domain])
            .is_some_and(|listing| launchd_lists_disabled(&listing, LAUNCH_AGENT_LABEL))
    }

    fn mac_entry(&self) -> io::Result<Option<Entry>> {
        let path = &self.mac_plist_path;
        let text = match fs::read(path) {
            Ok(bytes) => String::from_utf8(bytes).map_err(|e| e.to_string()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => Err(err.to_string()),
        };
        // launchd would ignore an unreadable agent as well.
        let unreadable = Entry {
            command: Vec::new(),
            active: false,
        };
        let agent = match text.and_then(|text| parse_plist(&text)) {
            Ok(Plist::Dict(agent)) => agent,
            Ok(_) => return Ok(Some(unreadable)),
            Err(err) => {
                log::debug!("Unreadable LaunchAgent {}: {err}", path.display());
                return Ok(Some(unreadable));
            }
        };
        let command = match agent.get("ProgramArguments") {
            Some(Plist::Array(items)) if items.iter().all(|i| matches!(i, Plist::String(_))) => {
                items
                    .iter()
                    .filter_map(|i| match i {
                        Plist::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect()
            }
            _ => Vec::new(),
        };
        let mut active = agent.get("Disabled") != Some(&Plist::Bool(true));
        if active && !command.is_empty() && self.mac_disabled_in_launchd() {
            active = false; // launchd will not start it at login
        }
        Ok(Some(Entry { command, active }))
    }

    // ------------------------------------------------------------------ Linux

    fn linux_enable(&self, command: &[String]) -> io::Result<()> {
        (self.write_text)(&self.linux_desktop_path, &desktop_entry(command))
    }

    fn linux_entry(&self) -> io::Result<Option<Entry>> {
        let text = match fs::read_to_string(&self.linux_desktop_path) {
            Ok(text) => text,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err),
        };
        let entry = parse_desktop_entry(&text);
        // Desktop environments "turn off" an autostart entry by editing these keys (GNOME
        // Tweaks, KDE System Settings) rather than deleting the file.
        let flag = |key: &str, default: &str| {
            entry
                .get(key)
                .map_or_else(|| default.to_string(), |v| v.to_lowercase())
        };
        let active = flag("Hidden", "false") != "true"
            && flag("X-GNOME-Autostart-enabled", "true") != "false";
        let exec = entry.get("Exec").map_or("", String::as_str);
        Ok(Some(Entry {
            command: if exec.is_empty() {
                Vec::new()
            } else {
                desktop_exec_split(exec)
            },
            active,
        }))
    }
}

// -------------------------------------------------------------------- helpers

/// `Windows`, `MacOs`, `Linux` or `Unsupported` for a `std::env::consts::OS` value.
fn system_for(os: &str) -> System {
    match os {
        "windows" => System::Windows,
        "macos" => System::MacOs,
        _ if os.starts_with("linux") => System::Linux,
        _ => System::Unsupported,
    }
}

fn home_dir() -> PathBuf {
    std::env::home_dir().unwrap_or_default()
}

fn path_text(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn expand_user_text(text: &str) -> String {
    match text.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with(['/', '\\']) => {
            format!("{}{rest}", home_dir().display())
        }
        _ => text.to_string(),
    }
}

fn expand_user(path: &Path) -> PathBuf {
    PathBuf::from(expand_user_text(&path_text(path)))
}

/// Absolute path, with symlinks resolved as far as the path exists (`Path.resolve`).
fn resolve(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    fs::canonicalize(&absolute).unwrap_or_else(|_| PathBuf::from(normpath(&path_text(&absolute))))
}

/// `os.path.normpath`: collapses separators and `.`/`..` without touching the filesystem.
fn normpath(text: &str) -> String {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in Path::new(text).components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let last_is_name =
                    matches!(out.components().next_back(), Some(Component::Normal(_)));
                if last_is_name {
                    out.pop();
                } else if !out.has_root() {
                    out.push("..");
                }
            }
            other => out.push(other),
        }
    }
    if out.as_os_str().is_empty() {
        ".".to_string()
    } else {
        path_text(&out)
    }
}

/// `%NAME%` references replaced by the environment (`ntpath.expandvars`, without `$`).
fn expand_vars(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find('%') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        match after.find('%') {
            Some(end) => match std::env::var(&after[..end]) {
                Ok(value) if end > 0 => {
                    out.push_str(&value);
                    rest = &after[end + 1..];
                }
                _ => {
                    out.push('%');
                    rest = after;
                }
            },
            None => {
                out.push('%');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// The `--config-dir` value in a command line, `None` for the default profile.
fn command_profile(command: &[String]) -> Option<String> {
    let args = command.get(1..).unwrap_or_default();
    for (index, arg) in args.iter().enumerate() {
        if arg == CONFIG_DIR_FLAG && index + 1 < args.len() {
            return Some(args[index + 1].clone());
        }
        if let Some(value) = arg.strip_prefix(&format!("{CONFIG_DIR_FLAG}=")) {
            return Some(value.to_string());
        }
    }
    None
}

/// The arguments of `command` (after the program) other than `--config-dir`.
fn without_profile(command: &[String]) -> Vec<String> {
    let mut rest = Vec::new();
    let mut args = command.iter().skip(1);
    while let Some(arg) = args.next() {
        if arg == CONFIG_DIR_FLAG {
            args.next();
        } else if !arg.starts_with(&format!("{CONFIG_DIR_FLAG}=")) {
            rest.push(arg.clone());
        }
    }
    rest
}

fn write_text(path: &Path, text: &str) -> io::Result<()> {
    atomic_write_text(path, text)?;
    // The temporary file is created 0600; login entries are conventionally 0644.
    #[cfg(unix)]
    if let Err(err) = fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o644))
    {
        log::debug!("Could not chmod {}: {err}", path.display());
    }
    Ok(())
}

fn unlink(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(err) if err.kind() != io::ErrorKind::NotFound => Err(err),
        _ => Ok(()),
    }
}

/// Whether `path` is on a read-only filesystem (a mounted disk image).
fn is_read_only_volume(path: &str) -> bool {
    #[cfg(unix)]
    {
        const ST_RDONLY: u64 = 1;
        let Ok(c_path) = std::ffi::CString::new(path) else {
            return false;
        };
        // SAFETY: `c_path` is a valid NUL-terminated string and `stat` is a plain C struct the
        // call fills in.
        let flags = unsafe {
            let mut stat: libc::statvfs = std::mem::zeroed();
            if libc::statvfs(c_path.as_ptr(), &mut stat) != 0 {
                return false;
            }
            stat.f_flag
        };
        flags & ST_RDONLY != 0
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        false
    }
}

/// launchd's domain for this user's login session (`gui/<uid>`); `None` off POSIX.
fn launchd_domain() -> Option<String> {
    #[cfg(unix)]
    {
        // SAFETY: `getuid` has no preconditions.
        Some(format!("gui/{}", unsafe { libc::getuid() }))
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Run a program and return its output; `None` when it failed or took longer than
/// `timeout`.
fn run_program(program: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let line = format!("{program} {}", args.join(" "));
    let mut child = match Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            log::debug!("{line} failed: {err}");
            return None;
        }
    };
    let read_all = |pipe: Option<Box<dyn Read + Send>>| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut bytes);
            }
            String::from_utf8_lossy(&bytes).into_owned()
        })
    };
    let stdout = read_all(child.stdout.take().map(|p| Box::new(p) as _));
    let stderr = read_all(child.stderr.take().map(|p| Box::new(p) as _));
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(5)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                log::debug!("{line} failed: timed out");
                return None;
            }
            Err(err) => {
                log::debug!("{line} failed: {err}");
                return None;
            }
        }
    };
    let out = stdout.join().unwrap_or_default();
    if !status.success() {
        let err = stderr.join().unwrap_or_default();
        let err: String = err.trim().chars().take(200).collect();
        match status.code() {
            Some(code) => log::debug!("{line} exited with {code}: {err}"),
            None => log::debug!("{line} was terminated: {err}"),
        }
        return None;
    }
    Some(out)
}

/// Whether `launchctl print-disabled` output switches `label` off.
///
/// Its lines read `"<label>" => disabled` (`=> true` on older macOS); other sections of the
/// output map labels to app identifiers instead.
fn launchd_lists_disabled(listing: &str, label: &str) -> bool {
    listing.lines().any(|line| {
        let Some(rest) = line.trim().strip_prefix('"') else {
            return false;
        };
        let Some((name, rest)) = rest.split_once('"') else {
            return false;
        };
        let Some(state) = rest.trim().strip_prefix("=>") else {
            return false;
        };
        let state = state.trim();
        name == label
            && !state.is_empty()
            && !state.contains(char::is_whitespace)
            && matches!(state.to_lowercase().as_str(), "disabled" | "true")
    })
}

// -------------------------------------------------------------------- Windows

/// `subprocess.list2cmdline`: the quoting rules of the MS C runtime.
fn list2cmdline(parts: &[String]) -> String {
    let mut result = String::new();
    for (i, arg) in parts.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }
        let need_quote = arg.contains([' ', '\t']) || arg.is_empty();
        if need_quote {
            result.push('"');
        }
        let mut backslashes = 0;
        for c in arg.chars() {
            match c {
                '\\' => backslashes += 1,
                '"' => {
                    result.push_str(&"\\".repeat(backslashes * 2));
                    backslashes = 0;
                    result.push_str("\\\"");
                }
                _ => {
                    result.push_str(&"\\".repeat(backslashes));
                    backslashes = 0;
                    result.push(c);
                }
            }
        }
        result.push_str(&"\\".repeat(backslashes));
        if need_quote {
            result.push_str(&"\\".repeat(backslashes));
            result.push('"');
        }
    }
    result
}

/// Split a command line the way `CommandLineToArgvW` and the C runtime do.
///
/// The program name ends at the next space, or at the closing quote when it is quoted;
/// backslashes mean nothing there. For the other arguments, `2n` backslashes before a quote
/// become `n` and the quote delimits, `2n+1` become `n` and a literal quote, and `""` inside
/// quotes is a literal quote. This is the inverse of [`list2cmdline`].
fn win_split(command_line: &str) -> Vec<String> {
    let text: Vec<char> = command_line
        .trim_start_matches([' ', '\t'])
        .chars()
        .collect();
    if text.is_empty() {
        return Vec::new();
    }
    let mut args: Vec<String>;
    let mut index;
    if text[0] == '"' {
        let end = text[1..]
            .iter()
            .position(|&c| c == '"')
            .map_or(text.len(), |p| p + 1);
        args = vec![text[1..end].iter().collect()];
        index = end + 1;
    } else {
        let end = text
            .iter()
            .position(|&c| c == ' ' || c == '\t')
            .unwrap_or(text.len());
        args = vec![text[..end].iter().collect()];
        index = end;
    }
    let length = text.len();
    let mut current = String::new();
    let (mut in_quotes, mut started) = (false, false);
    while index < length {
        let c = text[index];
        if c == '\\' {
            let mut run = index;
            while run < length && text[run] == '\\' {
                run += 1;
            }
            let count = run - index;
            if run < length && text[run] == '"' {
                current.push_str(&"\\".repeat(count / 2));
                if count % 2 == 1 {
                    current.push('"');
                    index = run + 1;
                } else {
                    index = run; // the quote is a delimiter, handled next
                }
            } else {
                current.push_str(&"\\".repeat(count));
                index = run;
            }
            started = true;
            continue;
        }
        if c == '"' {
            if in_quotes && index + 1 < length && text[index + 1] == '"' {
                current.push('"');
                index += 2;
                continue;
            }
            in_quotes = !in_quotes;
            started = true;
        } else if (c == ' ' || c == '\t') && !in_quotes {
            if started {
                args.push(std::mem::take(&mut current));
                started = false;
            }
        } else {
            current.push(c);
            started = true;
        }
        index += 1;
    }
    if started {
        args.push(current);
    }
    args
}

/// The registry of the running OS.
struct RealRegistry;

#[cfg(not(windows))]
impl Registry for RealRegistry {
    fn read(&self, _key_path: &str, _name: &str) -> io::Result<Option<RegValue>> {
        Err(registry_unavailable())
    }

    fn write_str(&self, _key_path: &str, _name: &str, _value: &str) -> io::Result<()> {
        Err(registry_unavailable())
    }

    fn delete(&self, _key_path: &str, _name: &str) -> io::Result<()> {
        Err(registry_unavailable())
    }
}

#[cfg(not(windows))]
fn registry_unavailable() -> io::Error {
    io::Error::other("the Windows registry is not available on this OS")
}

#[cfg(windows)]
mod win_registry {
    use std::io;

    use windows::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS, WIN32_ERROR};
    use windows::Win32::System::Registry::{
        HKEY, HKEY_CURRENT_USER, KEY_READ, KEY_SET_VALUE, REG_BINARY, REG_EXPAND_SZ,
        REG_OPTION_NON_VOLATILE, REG_SZ, REG_VALUE_TYPE, RegCloseKey, RegCreateKeyExW,
        RegDeleteValueW, RegOpenKeyExW, RegQueryValueExW, RegSetValueExW,
    };
    use windows::core::{HSTRING, PCWSTR};

    use super::{RealRegistry, RegValue, Registry};

    fn check(code: WIN32_ERROR) -> io::Result<()> {
        if code == ERROR_SUCCESS {
            Ok(())
        } else {
            Err(io::Error::from_raw_os_error(code.0 as i32))
        }
    }

    /// An open key, closed on drop.
    struct Key(HKEY);

    impl Key {
        fn open(
            key_path: &str,
            access: windows::Win32::System::Registry::REG_SAM_FLAGS,
        ) -> io::Result<Key> {
            let mut key = HKEY::default();
            // SAFETY: the path is a NUL-terminated UTF-16 string that outlives the call and
            // `key` is a valid out pointer.
            check(unsafe {
                RegOpenKeyExW(
                    HKEY_CURRENT_USER,
                    PCWSTR(HSTRING::from(key_path).as_ptr()),
                    None,
                    access,
                    &mut key,
                )
            })?;
            Ok(Key(key))
        }
    }

    impl Drop for Key {
        fn drop(&mut self) {
            // SAFETY: the handle was opened by `Key::open` or `RegCreateKeyExW` and is closed once.
            let _ = unsafe { RegCloseKey(self.0) };
        }
    }

    fn not_found(err: &io::Error) -> bool {
        err.raw_os_error() == Some(ERROR_FILE_NOT_FOUND.0 as i32)
    }

    impl Registry for RealRegistry {
        fn read(&self, key_path: &str, name: &str) -> io::Result<Option<RegValue>> {
            let key = match Key::open(key_path, KEY_READ) {
                Ok(key) => key,
                Err(err) if not_found(&err) => return Ok(None),
                Err(err) => return Err(err),
            };
            let name = HSTRING::from(name);
            let name = PCWSTR(name.as_ptr());
            let mut kind = REG_VALUE_TYPE::default();
            let mut size = 0u32;
            // SAFETY: `name` is NUL-terminated; the out pointers are valid for the call.
            let probe = unsafe {
                RegQueryValueExW(key.0, name, None, Some(&mut kind), None, Some(&mut size))
            };
            match check(probe) {
                Ok(()) => {}
                Err(err) if not_found(&err) => return Ok(None),
                Err(err) => return Err(err),
            }
            let mut data = vec![0u8; size as usize];
            // SAFETY: `data` holds `size` bytes, which the call is told.
            check(unsafe {
                RegQueryValueExW(
                    key.0,
                    name,
                    None,
                    Some(&mut kind),
                    Some(data.as_mut_ptr()),
                    Some(&mut size),
                )
            })?;
            data.truncate(size as usize);
            Ok(Some(match kind {
                REG_SZ | REG_EXPAND_SZ => {
                    let units: Vec<u16> = data
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| u16::from_le_bytes(*b))
                        .collect();
                    let end = units.iter().position(|&u| u == 0).unwrap_or(units.len());
                    RegValue::Str(String::from_utf16_lossy(&units[..end]))
                }
                REG_BINARY => RegValue::Bytes(data),
                _ => RegValue::Other,
            }))
        }

        fn write_str(&self, key_path: &str, name: &str, value: &str) -> io::Result<()> {
            let mut handle = HKEY::default();
            // SAFETY: the path is NUL-terminated and outlives the call; `handle` is a valid
            // out pointer.
            check(unsafe {
                RegCreateKeyExW(
                    HKEY_CURRENT_USER,
                    PCWSTR(HSTRING::from(key_path).as_ptr()),
                    None,
                    PCWSTR::null(),
                    REG_OPTION_NON_VOLATILE,
                    KEY_SET_VALUE,
                    None,
                    &mut handle,
                    None,
                )
            })?;
            let key = Key(handle);
            let bytes: Vec<u8> = value
                .encode_utf16()
                .chain(std::iter::once(0))
                .flat_map(u16::to_le_bytes)
                .collect();
            // SAFETY: `name` is NUL-terminated and `bytes` is the UTF-16 value with its NUL.
            check(unsafe {
                RegSetValueExW(
                    key.0,
                    PCWSTR(HSTRING::from(name).as_ptr()),
                    None,
                    REG_SZ,
                    Some(&bytes),
                )
            })
        }

        fn delete(&self, key_path: &str, name: &str) -> io::Result<()> {
            let key = match Key::open(key_path, KEY_SET_VALUE) {
                Ok(key) => key,
                Err(err) if not_found(&err) => return Ok(()),
                Err(err) => return Err(err),
            };
            // SAFETY: `name` is NUL-terminated and outlives the call.
            match check(unsafe { RegDeleteValueW(key.0, PCWSTR(HSTRING::from(name).as_ptr())) }) {
                Err(err) if !not_found(&err) => Err(err),
                _ => Ok(()),
            }
        }
    }
}

// ---------------------------------------------------------------------- macOS

fn mac_plist_path_in(home: &Path) -> PathBuf {
    home.join("Library")
        .join("LaunchAgents")
        .join(format!("{LAUNCH_AGENT_LABEL}.plist"))
}

fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The LaunchAgent as XML plist text, byte for byte what `plistlib.dumps(..., sort_keys=True)`
/// wrote.
fn mac_plist(command: &[String]) -> String {
    let mut text = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \
\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
<plist version=\"1.0\">\n<dict>\n",
    );
    text.push_str(&format!(
        "\t<key>Label</key>\n\t<string>{}</string>\n",
        xml_escape(LAUNCH_AGENT_LABEL)
    ));
    // Only in the graphical login session (not SSH or the login window).
    text.push_str("\t<key>LimitLoadToSessionType</key>\n\t<string>Aqua</string>\n");
    // A user-facing app: no background QoS throttling of the camera loop.
    text.push_str("\t<key>ProcessType</key>\n\t<string>Interactive</string>\n");
    text.push_str("\t<key>ProgramArguments</key>\n");
    if command.is_empty() {
        text.push_str("\t<array/>\n");
    } else {
        text.push_str("\t<array>\n");
        for arg in command {
            text.push_str(&format!("\t\t<string>{}</string>\n", xml_escape(arg)));
        }
        text.push_str("\t</array>\n");
    }
    text.push_str("\t<key>RunAtLoad</key>\n\t<true/>\n</dict>\n</plist>\n");
    text
}

/// The property list values this module reads.
#[derive(Debug, Clone, PartialEq)]
enum Plist {
    Dict(BTreeMap<String, Plist>),
    Array(Vec<Plist>),
    String(String),
    Bool(bool),
    /// Numbers, dates and data: never needed.
    Other,
}

enum Tag {
    Open(String),
    Close(String),
    Empty(String),
}

/// A reader of XML property lists (the format [`mac_plist`] writes and `plutil -convert xml1`
/// produces; binary property lists are not understood).
struct PlistReader<'a> {
    text: &'a str,
    pos: usize,
}

impl PlistReader<'_> {
    /// The next tag, skipping text, comments, processing instructions and the DOCTYPE.
    fn next_tag(&mut self) -> Result<Tag, String> {
        loop {
            let rest = &self.text[self.pos..];
            let start = rest.find('<').ok_or("unexpected end of file")?;
            let rest = &rest[start..];
            let (skip_to, closer) = if rest.starts_with("<?") {
                (2, "?>")
            } else if rest.starts_with("<!--") {
                (4, "-->")
            } else if rest.starts_with("<!") {
                (2, ">")
            } else {
                let end = rest.find('>').ok_or("unterminated tag")?;
                let inner = &rest[1..end];
                self.pos += start + end + 1;
                let name = |s: &str| s.split_whitespace().next().unwrap_or("").to_string();
                return Ok(if let Some(closing) = inner.strip_prefix('/') {
                    Tag::Close(name(closing))
                } else if let Some(empty) = inner.strip_suffix('/') {
                    Tag::Empty(name(empty))
                } else {
                    Tag::Open(name(inner))
                });
            };
            let end = rest[skip_to..].find(closer).ok_or("unterminated markup")?;
            self.pos += start + skip_to + end + closer.len();
        }
    }

    /// The text of an element whose opening tag was just read, up to its closing tag.
    fn text(&mut self, name: &str) -> Result<String, String> {
        let rest = &self.text[self.pos..];
        let end = rest.find('<').ok_or("unexpected end of file")?;
        let raw = &rest[..end];
        self.pos += end;
        match self.next_tag()? {
            Tag::Close(closed) if closed == name => Ok(xml_unescape(raw)),
            _ => Err(format!("<{name}> is not closed")),
        }
    }

    fn value(&mut self, tag: Tag) -> Result<Plist, String> {
        match tag {
            Tag::Open(name) => match name.as_str() {
                "dict" => {
                    let mut dict = BTreeMap::new();
                    loop {
                        match self.next_tag()? {
                            Tag::Close(closed) if closed == "dict" => break,
                            Tag::Open(key) if key == "key" => {
                                let key = self.text("key")?;
                                let tag = self.next_tag()?;
                                dict.insert(key, self.value(tag)?);
                            }
                            _ => return Err("unexpected content in <dict>".to_string()),
                        }
                    }
                    Ok(Plist::Dict(dict))
                }
                "array" => {
                    let mut items = Vec::new();
                    loop {
                        match self.next_tag()? {
                            Tag::Close(closed) if closed == "array" => break,
                            tag => items.push(self.value(tag)?),
                        }
                    }
                    Ok(Plist::Array(items))
                }
                "string" => Ok(Plist::String(self.text("string")?)),
                _ => {
                    self.text(&name)?;
                    Ok(Plist::Other)
                }
            },
            Tag::Empty(name) => Ok(match name.as_str() {
                "dict" => Plist::Dict(BTreeMap::new()),
                "array" => Plist::Array(Vec::new()),
                "string" => Plist::String(String::new()),
                "true" => Plist::Bool(true),
                "false" => Plist::Bool(false),
                _ => Plist::Other,
            }),
            Tag::Close(name) => Err(format!("unexpected </{name}>")),
        }
    }
}

fn parse_plist(text: &str) -> Result<Plist, String> {
    let mut reader = PlistReader { text, pos: 0 };
    match reader.next_tag()? {
        Tag::Open(name) if name == "plist" => {}
        _ => return Err("not a property list".to_string()),
    }
    let tag = reader.next_tag()?;
    let value = reader.value(tag)?;
    match reader.next_tag()? {
        Tag::Close(name) if name == "plist" => Ok(value),
        _ => Err("<plist> is not closed".to_string()),
    }
}

fn xml_unescape(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        let entity_end = rest[start..].find(';');
        let decoded = entity_end.and_then(|end| {
            let name = &rest[start + 1..start + end];
            let c = match name {
                "amp" => '&',
                "lt" => '<',
                "gt" => '>',
                "quot" => '"',
                "apos" => '\'',
                _ => {
                    let digits = name.strip_prefix('#')?;
                    let code = match digits.strip_prefix(['x', 'X']) {
                        Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                        None => digits.parse().ok()?,
                    };
                    char::from_u32(code)?
                }
            };
            Some((c, start + end + 1))
        });
        match decoded {
            Some((c, next)) => {
                out.push(c);
                rest = &rest[next..];
            }
            None => {
                out.push('&');
                rest = &rest[start + 1..];
            }
        }
    }
    out.push_str(rest);
    out
}

// ---------------------------------------------------------------------- Linux

/// `$XDG_CONFIG_HOME/autostart/eye-tracker.desktop`, or under `~/.config`.
fn linux_desktop_path_in(xdg_config_home: &str, home: &Path) -> PathBuf {
    // The XDG spec says relative values are invalid and must be ignored.
    let root = if !xdg_config_home.is_empty() && Path::new(xdg_config_home).is_absolute() {
        PathBuf::from(xdg_config_home)
    } else {
        home.join(".config")
    };
    root.join("autostart").join(DESKTOP_FILE_NAME)
}

/// Quote one argument following the Desktop Entry `Exec` rules.
///
/// Only double quotes are recognised (`shlex.quote`'s single quotes are not), and `%` must be
/// doubled because it introduces field codes.
fn desktop_exec_arg(arg: &str) -> String {
    let quoted = if !arg.is_empty() && !arg.chars().any(|c| DESKTOP_RESERVED.contains(c)) {
        arg.to_string()
    } else {
        let mut escaped = String::from('"');
        for c in arg.chars() {
            if DESKTOP_QUOTED_ESCAPES.contains(c) {
                escaped.push('\\');
            }
            escaped.push(c);
        }
        escaped.push('"');
        escaped
    };
    quoted.replace('%', "%%")
}

/// Escaping of the key file's `string` type, applied on top of `Exec` quoting.
fn desktop_string_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
}

fn desktop_exec(command: &[String]) -> String {
    let args: Vec<String> = command.iter().map(|a| desktop_exec_arg(a)).collect();
    desktop_string_escape(&args.join(" "))
}

/// Split an `Exec` value into arguments (the inverse of [`desktop_exec`]).
///
/// Undoes the key-file string escapes, then the `Exec` quoting (only double quotes; `\"`,
/// `` \` ``, `\$` and `\\` inside them), then `%%`.
fn desktop_exec_split(value: &str) -> Vec<String> {
    let mut unescaped = String::new();
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('s') => unescaped.push(' '),
                Some('n') => unescaped.push('\n'),
                Some('t') => unescaped.push('\t'),
                Some('r') => unescaped.push('\r'),
                Some('\\') => unescaped.push('\\'),
                Some(other) => {
                    unescaped.push('\\');
                    unescaped.push(other);
                }
                None => unescaped.push('\\'),
            }
        } else {
            unescaped.push(c);
        }
    }
    let text: Vec<char> = unescaped.chars().collect();
    let mut args = Vec::new();
    let mut current = String::new();
    let (mut in_quotes, mut started) = (false, false);
    let mut index = 0;
    while index < text.len() {
        let c = text[index];
        let following = text.get(index + 1).copied();
        if in_quotes {
            match following {
                Some(next) if c == '\\' && DESKTOP_QUOTED_ESCAPES.contains(next) => {
                    current.push(next);
                    index += 1;
                }
                _ if c == '"' => in_quotes = false,
                _ => current.push(c),
            }
        } else if c == '"' {
            in_quotes = true;
            started = true;
        } else if c == ' ' || c == '\t' {
            if started {
                args.push(std::mem::take(&mut current));
                started = false;
            }
        } else {
            current.push(c);
            started = true;
        }
        index += 1;
    }
    if started {
        args.push(current);
    }
    args.iter().map(|a| a.replace("%%", "%")).collect()
}

fn desktop_entry(command: &[String]) -> String {
    let lines = [
        "[Desktop Entry]".to_string(),
        "Type=Application".to_string(),
        "Version=1.5".to_string(),
        format!("Name={APP_NAME}"),
        "Comment=Moves the mouse cursor to the monitor you are looking at".to_string(),
        format!("Exec={}", desktop_exec(command)),
        format!("Icon={APP_SLUG}"),
        "Terminal=false".to_string(),
        "Hidden=false".to_string(),
        "NoDisplay=false".to_string(),
        "X-GNOME-Autostart-enabled=true".to_string(),
        // Give the panel a moment so the tray icon has somewhere to go.
        "X-GNOME-Autostart-Delay=3".to_string(),
    ];
    lines.join("\n") + "\n"
}

/// Keys of the `[Desktop Entry]` group (values unescaped only as far as needed here).
fn parse_desktop_entry(text: &str) -> HashMap<String, String> {
    let mut entry = HashMap::new();
    let mut group = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            group = Some(&line[1..line.len() - 1]);
            continue;
        }
        if group == Some("Desktop Entry")
            && let Some((key, value)) = line.split_once('=')
        {
            entry
                .entry(key.trim().to_string())
                .or_insert_with(|| value.trim().to_string());
        }
    }
    entry
}

#[cfg(test)]
#[path = "autostart_tests.rs"]
mod tests;
