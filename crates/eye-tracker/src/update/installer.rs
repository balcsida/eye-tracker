//! Downloading, verifying and starting the Windows installer of a newer release.
//!
//! The app installs an update only on Windows, and only when it was itself
//! installed by the setup program (it sits next to `unins000.exe`): the new setup
//! replaces the files of that very installation, closes the running app and starts
//! it again, the way `winget upgrade` does (see `installer.iss`). The portable
//! zip, macOS and Linux get a notice and the address of the release page instead.
//!
//! What is checked before the setup program is started:
//!
//! * the file comes from the release's own download address (see
//!   [`super::release`]), over HTTPS, from GitHub's hosts only (see [`super::fetch`]);
//! * its size is the size GitHub announced;
//! * its SHA-256 equals the one in the release's `SHA256SUMS.txt`.
//!
//! The checksum file lives next to the installer, so it protects against a damaged
//! or truncated download, not against someone who can replace both files on GitHub.
//! Releases also carry Sigstore build attestations, which `gh attestation verify`
//! checks (see `docs/privacy.md`); the app does not run that check itself.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use sha2::{Digest, Sha256};

use super::fetch::{DownloadError, Fetcher, NetworkError};
use super::release::{MAX_CHECKSUMS_BYTES, ReleaseInfo};
use crate::paths;

/// `install_kind()` answers.
pub const KIND_SETUP: &str = "setup";
pub const KIND_PORTABLE: &str = "portable";
pub const KIND_UNSUPPORTED: &str = "unsupported";

const UNINSTALLER: &str = "unins000.exe";
/// Switches of the Inno Setup program: no windows, no prompts, no restart.
pub const SETUP_ARGUMENTS: [&str; 3] = ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"];
#[cfg(windows)]
const DETACHED_PROCESS: u32 = 0x0000_0008;
#[cfg(windows)]
const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;

/// An update that cannot be installed (the message is meant for the user).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct UpdateError(pub String);

/// Everything [`download_installer`] and [`launch_installer`] can fail with.
#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    #[error(transparent)]
    Update(#[from] UpdateError),
    #[error(transparent)]
    Network(#[from] NetworkError),
    /// The user cancelled the download.
    #[error("The download was cancelled.")]
    Cancelled,
    #[error(transparent)]
    Io(#[from] io::Error),
}

impl From<DownloadError> for InstallError {
    fn from(error: DownloadError) -> Self {
        match error {
            DownloadError::Network(error) => Self::Network(error),
            DownloadError::Cancelled => Self::Cancelled,
            DownloadError::Io(error) => Self::Io(error),
        }
    }
}

fn update_error(message: &str) -> InstallError {
    UpdateError(message.to_string()).into()
}

/// How this copy of the app can be updated: by its setup program, by hand
/// (portable zip), or not by the app (other platforms).
///
/// Unlike the Python app there is no "source checkout" case: every build of the app is a binary.
pub fn install_kind() -> &'static str {
    let executable = std::env::current_exe()
        .and_then(fs::canonicalize)
        .unwrap_or_default();
    install_kind_for(cfg!(windows), true, &executable)
}

fn install_kind_for(windows: bool, frozen: bool, executable: &Path) -> &'static str {
    if !windows || !frozen {
        return KIND_UNSUPPORTED;
    }
    if executable
        .parent()
        .is_some_and(|folder| folder.join(UNINSTALLER).is_file())
    {
        return KIND_SETUP;
    }
    KIND_PORTABLE
}

pub fn updates_dir() -> PathBuf {
    paths::data_dir().join("updates")
}

/// `{file name: lower-case SHA-256}` from a `sha256sum` listing.
pub fn parse_checksums(text: &str) -> HashMap<String, String> {
    let mut sums = HashMap::new();
    for line in text.lines() {
        // 64 hex digits, a space, a space or `*` (binary mode), then the name.
        let Some((digest, rest)) = line
            .is_char_boundary(64)
            .then(|| line.split_at(64))
            .filter(|(digest, _)| digest.bytes().all(|b| b.is_ascii_hexdigit()))
        else {
            continue;
        };
        let Some(name) = rest
            .strip_prefix(' ')
            .and_then(|rest| rest.strip_prefix([' ', '*']))
            .filter(|name| name.chars().next().is_some_and(|c| !c.is_whitespace()))
        else {
            continue;
        };
        sums.insert(name.trim_end().to_string(), digest.to_ascii_lowercase());
    }
    sums
}

pub fn file_sha256(path: &Path) -> io::Result<String> {
    let mut digest = Sha256::new();
    let mut handle = File::open(path)?;
    let mut block = vec![0u8; 1024 * 1024];
    loop {
        let read = handle.read(&mut block)?;
        if read == 0 {
            return Ok(hex::encode(digest.finalize()));
        }
        digest.update(&block[..read]);
    }
}

/// Compares without stopping at the first difference.
fn digests_equal(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0, |acc, (x, y)| acc | (x ^ y))
            == 0
}

/// Download the installer of `release` and verify it; returns its path.
///
/// Fails with [`UpdateError`] when it cannot be trusted, [`InstallError::Cancelled`] when
/// `cancelled()` turns true, and [`InstallError::Network`] when the download fails. A
/// file that fails a check is deleted.
pub fn download_installer(
    release: &ReleaseInfo,
    fetcher: &mut Fetcher,
    progress: Option<&mut dyn FnMut(u64, u64)>,
    cancelled: Option<&dyn Fn() -> bool>,
    directory: Option<&Path>,
) -> Result<PathBuf, InstallError> {
    let (Some(installer), Some(checksums)) = (&release.installer, &release.checksums) else {
        return Err(update_error(
            "This release has no installer the app can verify.",
        ));
    };
    let body = fetcher.get(&checksums.url, MAX_CHECKSUMS_BYTES, None)?;
    let text = String::from_utf8_lossy(&body);
    let Some(expected) = parse_checksums(&text).remove(&installer.name) else {
        return Err(update_error(
            "The release's checksum list does not name the installer.",
        ));
    };

    let folder = directory.map_or_else(updates_dir, Path::to_path_buf);
    fs::create_dir_all(&folder)?;
    let finalpath = folder.join(&installer.name);
    let part = folder.join(format!("{}.part", installer.name));
    for stale in [&part, &finalpath] {
        let _ = fs::remove_file(stale);
    }
    let fetched = (|| {
        let size = fetcher.download(&installer.url, &part, installer.size, progress, cancelled)?;
        if size != installer.size {
            return Err(update_error("The download is incomplete."));
        }
        if !digests_equal(&file_sha256(&part)?, &expected) {
            return Err(update_error(
                "The download does not match its checksum and was discarded.",
            ));
        }
        fs::rename(&part, &finalpath)?;
        Ok(())
    })();
    if let Err(error) = fetched {
        let _ = fs::remove_file(&part);
        return Err(error);
    }
    Ok(finalpath)
}

/// Start the setup program in the background. It asks the running app to quit,
/// replaces the files and starts the app again; this process does not wait.
pub fn launch_installer(installer: &Path) -> Result<(), InstallError> {
    launch(installer, install_kind(), |mut command| {
        command.spawn().map(drop)
    })
}

fn launch(
    installer: &Path,
    kind: &str,
    spawn: impl FnOnce(Command) -> io::Result<()>,
) -> Result<(), InstallError> {
    if kind != KIND_SETUP {
        return Err(update_error(
            "This copy of the app is not installed by its setup program.",
        ));
    }
    let mut command = Command::new(installer);
    command
        .args(SETUP_ARGUMENTS)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }
    log::info!(
        "Starting the installer {}",
        installer
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
    );
    spawn(command)?;
    Ok(())
}

/// Delete what earlier updates left in the downloads folder (best effort).
pub fn remove_downloads() {
    remove_downloads_in(&updates_dir());
}

fn remove_downloads_in(folder: &Path) {
    let Ok(items) = fs::read_dir(folder) else {
        return;
    };
    for item in items.flatten() {
        if item.path().is_file() {
            let _ = fs::remove_file(item.path());
        }
    }
}

#[cfg(test)]
#[path = "installer_tests.rs"]
mod tests;
