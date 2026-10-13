//! Download and verify the face models that ship with Eye Tracker.
//!
//! The models are committed to the repository, so most people never need to run
//! this. It exists to recreate them in a fresh or damaged checkout, to verify them
//! in CI, and to record exactly where each file comes from and under which licence
//! (see also `crates/eye-tracker/models/NOTICE.md`).
//!
//! ```text
//! cargo xtask fetch-models            # download missing or corrupt models
//! cargo xtask fetch-models --check    # verify only; never touches the network
//! cargo xtask fetch-models --force    # download again even if valid
//! ```
//!
//! Every download is checked against a pinned SHA-256 before anything is moved
//! into place, so a failed or tampered download can never replace a good model.
//! Some models come inside an archive (MediaPipe's `face_landmarker.task` is a
//! zip file): the archive is verified, only the members the app needs are
//! extracted, and each member is verified against its own pinned SHA-256. The
//! archive itself is not kept.
//!
//! The models' licence texts (`licenses/` next to the models) must ship with
//! them. They are committed like the models but never downloaded: `--check`
//! fails when one is missing or altered, the other modes only warn.
//!
//! This is developer tooling: the application itself never touches the network.

use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::repo_root;

const USER_AGENT: &str = "eye-tracker-fetch-models/1.0 (+https://github.com/bugraskl/eye-tracker)";

/// Refuse absurdly large responses (the real files are a few MiB).
pub const MAX_BYTES: u64 = 64 * 1024 * 1024;
const CHUNK: usize = 1024 * 256;

/// The model directory of the app crate.
pub fn default_dest() -> PathBuf {
    repo_root()
        .join("crates")
        .join("eye-tracker")
        .join("models")
}

/// A file installed from a downloaded archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    /// Path inside the archive; also the installed file name.
    pub name: String,
    pub sha256: String,
}

impl Member {
    pub fn new(name: &str, sha256: &str) -> Self {
        Self {
            name: name.into(),
            sha256: sha256.into(),
        }
    }
}

/// A model download with its canonical source and pinned checksum.
///
/// With `members` the download is a zip archive and only those members are
/// installed; otherwise the downloaded file is installed as `filename`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    pub filename: String,
    pub url: String,
    pub sha256: String,
    pub license: String,
    pub homepage: String,
    pub description: String,
    pub members: Vec<Member>,
}

impl Model {
    /// `(file name, sha256)` of every file this model puts into the model directory.
    pub fn installed(&self) -> Vec<(&str, &str)> {
        if self.members.is_empty() {
            vec![(&self.filename, &self.sha256)]
        } else {
            self.members
                .iter()
                .map(|m| (m.name.as_str(), m.sha256.as_str()))
                .collect()
        }
    }
}

pub fn models() -> Vec<Model> {
    vec![
        Model {
            filename: "face_landmarker.task".into(),
            url: "https://storage.googleapis.com/mediapipe-models/face_landmarker/\
                  face_landmarker/float16/1/face_landmarker.task"
                .into(),
            sha256: "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff".into(),
            license: "Apache-2.0".into(),
            homepage: "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"
                .into(),
            description: "MediaPipe Face Landmarker bundle: the 478-point face landmark network \
                          (with irises) and the canonical face geometry, run with OpenCV DNN"
                .into(),
            members: vec![
                Member::new(
                    "face_landmarks_detector.tflite",
                    "c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5",
                ),
                Member::new(
                    "geometry_pipeline_metadata_landmarks.binarypb",
                    "bdbcda96dfcb7da883da124aaa2c55dee49770d934f0fcc71747f8c21bdc75b4",
                ),
            ],
        },
        Model {
            filename: "face_detection_yunet_2023mar.onnx".into(),
            url: "https://github.com/opencv/opencv_zoo/raw/main/models/\
                  face_detection_yunet/face_detection_yunet_2023mar.onnx"
                .into(),
            sha256: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4".into(),
            license: "MIT".into(),
            homepage: "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
                .into(),
            description: "OpenCV Zoo YuNet face detector (face finding and the lite backend)"
                .into(),
            members: vec![],
        },
    ]
}

/// A licence text that ships next to the models it covers (see NOTICE.md).
///
/// Apache-2.0 requires giving recipients a copy of the licence, and MIT
/// requires its copyright and permission notice in all copies, so a link is not
/// enough. The texts are committed to the repository and only verified.
#[derive(Debug, Clone, Copy)]
pub struct LicenceText {
    /// Path relative to the model directory.
    pub path: &'static str,
    pub sha256: &'static str,
    /// Installed model files the licence covers (checked by the tests).
    #[cfg_attr(not(test), allow(dead_code))]
    pub covers: &'static [&'static str],
}

/// Sources: <https://www.apache.org/licenses/LICENSE-2.0.txt> and
/// <https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE>.
pub const LICENCE_TEXTS: [LicenceText; 2] = [
    LicenceText {
        path: "licenses/LICENSE-APACHE-2.0.txt",
        sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        covers: &[
            "face_landmarks_detector.tflite",
            "geometry_pipeline_metadata_landmarks.binarypb",
        ],
    },
    LicenceText {
        path: "licenses/LICENSE-YUNET.txt",
        sha256: "2ad92c7a6eb7aebede4e19f5ec4930c8bd9d614dbb8e75be1f740edae346734c",
        covers: &["face_detection_yunet_2023mar.onnx"],
    },
];

/// A model could not be downloaded or failed verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadError(pub String);

impl fmt::Display for DownloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DownloadError {}

/// Why opening a URL failed; an HTTP status is told apart because 4xx is not retried.
#[derive(Debug)]
pub enum FetchError {
    Http(u16),
    Other(String),
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FetchError::Http(code) => write!(f, "HTTP Error {code}"),
            FetchError::Other(reason) => f.write_str(reason),
        }
    }
}

/// The network, replaceable in tests (the Python tests patched `urlopen` and `time.sleep`).
pub trait Transport {
    fn open(&mut self, url: &str, timeout: Duration) -> Result<Box<dyn Read>, FetchError>;

    fn sleep(&mut self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

/// `ureq`, with the project's user agent.
pub struct UreqTransport;

impl Transport for UreqTransport {
    fn open(&mut self, url: &str, timeout: Duration) -> Result<Box<dyn Read>, FetchError> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .build()
            .into();
        match agent.get(url).header("User-Agent", USER_AGENT).call() {
            Ok(response) => Ok(Box::new(response.into_body().into_reader())),
            Err(ureq::Error::StatusCode(code)) => Err(FetchError::Http(code)),
            Err(other) => Err(FetchError::Other(other.to_string())),
        }
    }
}

// ------------------------------------------------------------------------------ helpers
fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; CHUNK];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            return Ok(hex::encode(digest.finalize()));
        }
        digest.update(&buffer[..n]);
    }
}

/// `"ok"`, `"missing"` or `"mismatch"` for the model's files in `dest`.
pub fn verify(model: &Model, dest: &Path) -> &'static str {
    let mut status = "ok";
    for (name, sha256) in model.installed() {
        let path = dest.join(name);
        if !path.is_file() {
            return "missing";
        }
        if sha256_file(&path).ok().as_deref() != Some(sha256) {
            status = "mismatch";
        }
    }
    status
}

/// `"ok"`, `"missing"` or `"altered"` for a licence text in `dest`.
pub fn verify_licence(text: &LicenceText, dest: &Path) -> &'static str {
    let path = dest.join(text.path);
    if !path.is_file() {
        return "missing";
    }
    if sha256_file(&path).ok().as_deref() == Some(text.sha256) {
        "ok"
    } else {
        "altered"
    }
}

/// Report every licence text's status; returns how many are missing or altered.
pub fn check_licences(dest: &Path, out: &mut dyn Write) -> usize {
    let mut problems = 0;
    for text in &LICENCE_TEXTS {
        let status = verify_licence(text, dest);
        if status == "ok" {
            let _ = writeln!(out, "  licence  {} (sha256 verified)", text.path);
            continue;
        }
        problems += 1;
        let _ = writeln!(
            out,
            "  {status:<8} {}  (restore it from git: it must ship with the models)",
            text.path
        );
    }
    problems
}

fn human_size(n: u64) -> String {
    let mut value = n as f64;
    for unit in ["B", "KiB", "MiB"] {
        if value < 1024.0 || unit == "MiB" {
            return if unit == "B" {
                format!("{value:.0} {unit}")
            } else {
                format!("{value:.1} {unit}")
            };
        }
        value /= 1024.0;
    }
    format!("{n} B")
}

fn installed_size(model: &Model, dest: &Path) -> u64 {
    model
        .installed()
        .iter()
        .filter_map(|(name, _)| fs::metadata(dest.join(name)).ok())
        .map(|m| m.len())
        .sum()
}

/// A temporary file next to its destination, removed on drop unless it was renamed away.
struct Partial(PathBuf);

impl Partial {
    fn new(dir: &Path, final_name: &str) -> Self {
        Self(dir.join(format!(".{final_name}.{}.part", process::id())))
    }
}

impl Drop for Partial {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn io_error(model: &Model, error: io::Error) -> DownloadError {
    DownloadError(format!("{}: {error}", model.filename))
}

/// Why one download attempt failed.
enum Attempt {
    /// A checksum mismatch or an oversized response will not fix itself on retry.
    Fatal(DownloadError),
    /// A network or disk error might.
    Retry(String),
}

/// Stream the response into `target`, checking size and checksum.
fn download_once(
    model: &Model,
    target: &Path,
    mut body: Box<dyn Read>,
    max_bytes: u64,
) -> Result<(), Attempt> {
    let retry = |e: io::Error| Attempt::Retry(e.to_string());
    let mut out = fs::File::create(target).map_err(retry)?;
    let mut digest = Sha256::new();
    let mut received = 0u64;
    let mut buffer = vec![0u8; CHUNK];
    loop {
        let n = body.read(&mut buffer).map_err(retry)?;
        if n == 0 {
            break;
        }
        received += n as u64;
        if received > max_bytes {
            return Err(Attempt::Fatal(DownloadError(format!(
                "{}: response larger than {max_bytes} bytes",
                model.filename
            ))));
        }
        digest.update(&buffer[..n]);
        out.write_all(&buffer[..n]).map_err(retry)?;
    }
    let actual = hex::encode(digest.finalize());
    if actual != model.sha256 {
        return Err(Attempt::Fatal(DownloadError(format!(
            "{}: checksum mismatch (got {actual}, expected {}); the upstream file may have changed",
            model.filename, model.sha256
        ))));
    }
    Ok(())
}

fn write_atomic(data: &[u8], final_path: &Path) -> io::Result<()> {
    let name = final_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = Partial::new(final_path.parent().unwrap_or(Path::new(".")), &name);
    fs::write(&tmp.0, data)?;
    fs::rename(&tmp.0, final_path)
}

/// Install the verified members of a downloaded (and verified) zip archive.
///
/// Every member is read and checked before any file is written, so a bad
/// archive leaves the model directory untouched.
///
/// # Errors
///
/// The archive is not a zip file, lacks a member or a member's checksum does
/// not match.
pub fn extract_members(
    model: &Model,
    archive: &Path,
    dest: &Path,
) -> Result<Vec<PathBuf>, DownloadError> {
    let unreadable = |e: &dyn fmt::Display| {
        DownloadError(format!(
            "{}: not a readable zip archive: {e}",
            model.filename
        ))
    };
    let file = fs::File::open(archive).map_err(|e| unreadable(&e))?;
    let mut bundle = zip::ZipArchive::new(file).map_err(|e| unreadable(&e))?;
    let mut contents = Vec::new();
    for member in &model.members {
        let mut entry = match bundle.by_name(&member.name) {
            Ok(entry) => entry,
            Err(zip::result::ZipError::FileNotFound) => {
                return Err(DownloadError(format!(
                    "{}: archive has no member '{}'",
                    model.filename, member.name
                )));
            }
            Err(e) => return Err(unreadable(&e)),
        };
        let mut data = Vec::new();
        entry.read_to_end(&mut data).map_err(|e| unreadable(&e))?;
        let actual = sha256_hex(&data);
        if actual != member.sha256 {
            return Err(DownloadError(format!(
                "{}: checksum mismatch for {} (got {actual}, expected {})",
                model.filename, member.name, member.sha256
            )));
        }
        contents.push((member, data));
    }
    let mut installed = Vec::new();
    for (member, data) in contents {
        // Only the pinned base name is used: archive paths never reach the disk.
        let base = Path::new(&member.name).file_name().unwrap_or_default();
        let final_path = dest.join(base);
        write_atomic(&data, &final_path).map_err(|e| io_error(model, e))?;
        installed.push(final_path);
    }
    Ok(installed)
}

/// Tunables of [`download`].
#[derive(Debug, Clone, Copy)]
pub struct DownloadOptions {
    pub timeout: Duration,
    pub attempts: u32,
    pub max_bytes: u64,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            attempts: 3,
            max_bytes: MAX_BYTES,
        }
    }
}

fn pause_before_retry(
    transport: &mut dyn Transport,
    model: &Model,
    error: &dyn fmt::Display,
    attempt: u32,
    out: &mut dyn Write,
) {
    let delay = 2.0 * f64::from(attempt);
    let _ = writeln!(
        out,
        "  retry    {}: {error} (again in {delay:.0} s)",
        model.filename
    );
    transport.sleep(Duration::from_secs_f64(delay));
}

/// Download `model` into `dest` atomically; returns the (first) installed path.
///
/// The data goes to a temporary file in the same directory and is installed
/// only after the checksum matches (for archives: after every member matched).
pub fn download(
    model: &Model,
    dest: &Path,
    transport: &mut dyn Transport,
    options: DownloadOptions,
    out: &mut dyn Write,
) -> Result<PathBuf, DownloadError> {
    fs::create_dir_all(dest).map_err(|e| io_error(model, e))?;
    let final_path = dest.join(&model.filename);
    let mut last_error = String::new();
    for attempt in 1..=options.attempts {
        let tmp = Partial::new(dest, &model.filename);
        let opened = match transport.open(&model.url, options.timeout) {
            // a client error will not fix itself on retry
            Err(FetchError::Http(code)) if (400..500).contains(&code) => {
                return Err(DownloadError(format!(
                    "{}: {} returned HTTP Error {code}",
                    model.filename, model.url
                )));
            }
            other => other.map_err(|e| e.to_string()),
        };
        let attempted = opened
            .map_err(Attempt::Retry)
            .and_then(|body| download_once(model, &tmp.0, body, options.max_bytes));
        match attempted {
            Ok(()) => {}
            Err(Attempt::Fatal(e)) => return Err(e),
            Err(Attempt::Retry(reason)) => {
                last_error = reason;
                if attempt < options.attempts {
                    pause_before_retry(transport, model, &last_error, attempt, out);
                }
                continue;
            }
        }
        if !model.members.is_empty() {
            let installed = extract_members(model, &tmp.0, dest)?;
            return Ok(installed.into_iter().next().unwrap_or(final_path));
        }
        fs::rename(&tmp.0, &final_path).map_err(|e| io_error(model, e))?;
        return Ok(final_path);
    }
    Err(DownloadError(format!(
        "{}: download failed: {last_error}",
        model.filename
    )))
}

fn describe(model: &Model) -> String {
    if model.members.is_empty() {
        return model.filename.clone();
    }
    let names: Vec<&str> = model.members.iter().map(|m| m.name.as_str()).collect();
    format!("{} -> {}", model.filename, names.join(", "))
}

// --------------------------------------------------------------------------------- main
#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    /// Verify only; exit 1 if anything is wrong.
    #[arg(long, conflicts_with = "force")]
    pub check: bool,
    /// Download even if the files are valid.
    #[arg(long)]
    pub force: bool,
    /// Model directory (default: crates/eye-tracker/models).
    #[arg(long)]
    pub dest: Option<PathBuf>,
    /// Network timeout in seconds.
    #[arg(long, default_value_t = 60.0)]
    pub timeout: f64,
}

/// Entry point of `cargo xtask fetch-models`; returns the exit code.
pub fn main(args: &Args) -> i32 {
    run(
        args,
        &models(),
        &mut UreqTransport,
        &mut io::stdout(),
        &mut io::stderr(),
    )
}

/// [`main`] with the models, network and output streams injected.
pub fn run(
    args: &Args,
    models: &[Model],
    transport: &mut dyn Transport,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    let dest = args.dest.clone().unwrap_or_else(default_dest);
    let options = DownloadOptions {
        timeout: Duration::from_secs_f64(args.timeout),
        ..DownloadOptions::default()
    };

    let mut failures = 0;
    for model in models {
        let status = if args.force {
            "stale"
        } else {
            verify(model, &dest)
        };
        if status == "ok" {
            let size = human_size(installed_size(model, &dest));
            let _ = writeln!(
                out,
                "  ok       {} ({size}, sha256 verified)",
                describe(model)
            );
            continue;
        }
        if args.check {
            let _ = writeln!(
                out,
                "  {status:<8} {}  (run cargo xtask fetch-models to fix)",
                describe(model)
            );
            failures += 1;
            continue;
        }
        let _ = writeln!(
            out,
            "  fetch    {} [{}] from {}",
            model.filename, model.license, model.url
        );
        if let Err(e) = download(model, &dest, transport, options, out) {
            let _ = writeln!(err, "  FAILED   {e}");
            failures += 1;
            continue;
        }
        let size = human_size(installed_size(model, &dest));
        let _ = writeln!(out, "  ok       {} ({size}, verified)", describe(model));
    }

    let licence_problems = check_licences(&dest, out);
    if failures > 0 {
        let _ = writeln!(
            err,
            "{failures} model(s) missing or invalid in {}",
            dest.display()
        );
    }
    if licence_problems > 0 {
        // Committed files, never downloaded: only --check (CI, release builds)
        // fails on them; fetching models into another folder merely warns.
        let prefix = if args.check { "" } else { "warning: " };
        let _ = writeln!(
            err,
            "{prefix}{licence_problems} licence text(s) missing or altered in {}",
            dest.display()
        );
    }
    i32::from(failures > 0 || (args.check && licence_problems > 0))
}

#[cfg(test)]
#[path = "fetch_models_tests.rs"]
mod tests;
