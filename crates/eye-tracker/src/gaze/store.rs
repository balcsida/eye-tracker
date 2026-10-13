//! Persistence of calibrations as a small, human-readable JSON file.
//!
//! Only numbers are stored: the feature vectors (head pose angles, head position,
//! iris offsets), the screen points they were labelled with and the fitted model.
//! Camera frames are never part of a calibration.
//!
//! A calibration is only valid for the setup it was recorded on: the monitor
//! layout, the vision backend (and the version of its measurements) and the
//! camera. A laptop that moves between desks therefore needs one calibration per
//! desk, so the file holds a small [`CalibrationLibrary`] of *profiles*, one
//! per setup, most recently used first. Version 1 files, which held a single
//! calibration, are read as a library with one profile.
//!
//! The file is pretty-printed with one sample per line so it stays readable and
//! reasonably small (a two-monitor calibration is roughly 50 KB). Loading never
//! fails: a missing, unreadable, corrupt or too new file yields an empty library
//! (`None` from [`load_calibration`]) and the app asks for a new calibration;
//! a single corrupt profile is skipped and the others are kept.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

use super::calibration::{
    CalibrationSample, axis_errors_from_dict, json_float, json_i32, json_int, per_axis_errors,
    samples_to_arrays,
};
use super::model::{GazeModel, SUPPORTED_DEGREES, lopo_predictions};
use crate::config::atomic_write_text;
use crate::types::{Monitor, Rect, layout_signature, virtual_bounds};

/// Version 2: a list of profiles. Version 1 (a single calibration) is still read.
pub const CALIBRATION_VERSION: i64 = 2;

/// Marker that identifies the file type.
pub const FILE_FORMAT: &str = "eye-tracker-calibration";

/// Profiles kept by a [`CalibrationLibrary`]; the least recently used go first.
pub const MAX_PROFILES: usize = 8;

/// Camera frames whose aspect ratios differ by more than this (relative) show a
/// different field of view. The resolution alone does not matter: every backend
/// normalises its measurements by the frame size.
pub const ASPECT_TOLERANCE: f64 = 0.01;

// Samples are measurements, so six decimals are far below their noise. Model
// parameters keep ten significant digits so a reloaded model predicts the same
// pixels as the one that was saved.
const SAMPLE_DECIMALS: usize = 6;
const MODEL_DIGITS: usize = 10;

/// `CalibrationData::key`: (layout signature, backend, feature version, camera).
pub type ProfileKey = (String, String, String, String);

/// Backend names of earlier versions whose measurements did not change with the
/// new name: the OpenCV/YuNet backend is now called `lite` (same features, same
/// `feature_version`), so its calibrations stay valid. The MediaPipe backend's
/// successor, `facemesh`, measures differently (another `feature_version`),
/// so `mediapipe` calibrations are deliberately not carried over.
pub const BACKEND_ALIASES: [(&str, &str); 1] = [("opencv", "lite")];

/// The current name of a vision backend (see [`BACKEND_ALIASES`]).
pub fn canonical_backend(name: &str) -> &str {
    BACKEND_ALIASES
        .iter()
        .find(|(old, _)| *old == name)
        .map_or(name, |(_, new)| new)
}

/// Current time as ISO 8601 UTC with second precision, e.g. `2026-09-30T12:00:00+00:00`.
pub fn utc_now_iso() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    iso_from_unix(seconds)
}

fn iso_from_unix(seconds: i64) -> String {
    let (days, rest) = (seconds.div_euclid(86_400), seconds.rem_euclid(86_400));
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}+00:00",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    )
}

/// A calibration and everything needed to decide whether it is still valid.
///
/// `camera` is the camera device setting (`CameraSettings::device`, e.g.
/// `"0"`) and `frame_size` the `(w, h)` of the camera frames the samples
/// were measured on (`CalibrationCollector::frame_size`). Head pose and face
/// position are measured relative to the camera, so another camera, or the same
/// one with another field of view, needs another calibration. `""` and
/// `(0, 0)` mean unknown (files written before they were recorded) and are
/// never a reason to reject the calibration.
#[derive(Debug, Clone)]
pub struct CalibrationData {
    pub backend: String,
    pub feature_version: String,
    pub layout_signature: String,
    pub monitors: Vec<Monitor>,
    pub samples: Vec<CalibrationSample>,
    pub implicit_samples: Vec<CalibrationSample>,
    pub model: GazeModel,
    pub report: Map<String, Value>,
    pub created_at: String,
    pub camera: String,
    pub frame_size: (u32, u32),
    /// Per-axis errors already worked out for this calibration (loaded, recomputed or found
    /// impossible to recompute: empty), so each profile is looked at once ([`axis_error`]).
    axis_errors: Option<BTreeMap<i32, (f64, f64)>>,
}

impl CalibrationData {
    /// A calibration with an empty report, the current time as `created_at` and an unknown
    /// camera and frame size.
    pub fn new(
        backend: impl Into<String>,
        feature_version: impl Into<String>,
        layout_signature: impl Into<String>,
        monitors: Vec<Monitor>,
        samples: Vec<CalibrationSample>,
        implicit_samples: Vec<CalibrationSample>,
        model: GazeModel,
    ) -> Self {
        CalibrationData {
            backend: backend.into(),
            feature_version: feature_version.into(),
            layout_signature: layout_signature.into(),
            monitors,
            samples,
            implicit_samples,
            model,
            report: Map::new(),
            created_at: utc_now_iso(),
            camera: String::new(),
            frame_size: (0, 0),
            axis_errors: None,
        }
    }

    /// The setup this calibration belongs to; a library keeps one profile per key.
    ///
    /// The backend is given by its current name ([`canonical_backend`]), so a
    /// profile of the renamed `opencv` backend and a new `lite` one are the same.
    pub fn key(&self) -> ProfileKey {
        (
            self.layout_signature.clone(),
            canonical_backend(&self.backend).to_string(),
            self.feature_version.clone(),
            self.camera.trim().to_string(),
        )
    }

    /// `(true, "")` if usable with this backend, monitor layout and camera, else
    /// `(false, reason)` where `reason` is a short lower-case phrase such as
    /// `"the monitor layout changed"`.
    ///
    /// `camera` and `frame_size` are the current camera device setting and
    /// frame size; `None`, `""` or `(0, 0)` (here or stored) skip that check.
    /// Backend names are compared by their current names ([`canonical_backend`]).
    pub fn is_compatible(
        &self,
        backend_name: &str,
        feature_version: &str,
        monitors: &[Monitor],
        camera: Option<&str>,
        frame_size: Option<(u32, u32)>,
    ) -> (bool, String) {
        let (current, stored) = (
            canonical_backend(backend_name),
            canonical_backend(&self.backend),
        );
        if current != stored {
            return (
                false,
                format!("calibrated with the {stored} backend, but {current} is in use"),
            );
        }
        if feature_version != self.feature_version {
            return (
                false,
                format!(
                    "the {backend_name} backend's measurements changed ({} → {feature_version})",
                    self.feature_version
                ),
            );
        }
        if layout_signature(monitors) != self.layout_signature {
            return (false, "the monitor layout changed".into());
        }
        let (ok, reason) = self.camera_matches(camera, frame_size);
        if !ok {
            return (false, reason);
        }
        if !self.model.is_fitted() {
            return (false, "the calibration has no fitted model".into());
        }
        (true, String::new())
    }

    /// The camera part of [`is_compatible`](Self::is_compatible), e.g. for the first frame after
    /// the camera was (re)opened: `(false, "the camera changed")` for another
    /// device, `(false, "the camera's aspect ratio changed (…)")` for another
    /// field of view. Unknown values on either side pass.
    pub fn camera_matches(
        &self,
        camera: Option<&str>,
        frame_size: Option<(u32, u32)>,
    ) -> (bool, String) {
        let (stored, current) = (self.camera.trim(), camera.unwrap_or("").trim());
        if !stored.is_empty() && !current.is_empty() && stored != current {
            return (false, "the camera changed".into());
        }
        if let Some(frame_size) = frame_size
            && !same_aspect(self.frame_size, frame_size)
        {
            let ((w0, h0), (w1, h1)) = (self.frame_size, frame_size);
            return (
                false,
                format!("the camera's aspect ratio changed ({w0}x{h0} → {w1}x{h1})"),
            );
        }
        (true, String::new())
    }

    /// Quality grade from the report (`"excellent"` … `"poor"`), if known.
    pub fn grade(&self) -> Option<&str> {
        self.report.get("grade").and_then(Value::as_str)
    }
}

/// Whether frame sizes `a` and `b` `(w, h)` have the same aspect ratio
/// (within [`ASPECT_TOLERANCE`]); true if either is unknown (not positive).
pub fn same_aspect(a: (u32, u32), b: (u32, u32)) -> bool {
    let ((wa, ha), (wb, hb)) = (a, b);
    if wa.min(ha).min(wb).min(hb) == 0 {
        return true;
    }
    let ratio = (f64::from(wa) / f64::from(ha)) / (f64::from(wb) / f64::from(hb));
    (ratio - 1.0).abs() <= ASPECT_TOLERANCE
}

// ------------------------------------------------------------ gaze accuracy
/// How far off the gaze estimate typically is on `monitor`: `(x_px, y_px)`.
///
/// The answer comes, in this order, from
///
/// 1. the report's `per_monitor_error_px` (calibrations made by this version),
/// 2. the leave-one-point-out predictions recomputed from the calibration's own
///    samples with the report's `alpha` and `degree` (older calibrations;
///    the result is added to [`CalibrationData::report`], so it is saved
///    with the profile's next write), and
/// 3. the report's mean error divided by √2 on both axes.
///
/// `monitor` is matched to the calibration's monitors by its rectangle (the
/// system may number the same screens differently after a restart), then by index.
/// `None` when nothing is known.
///
/// The first call for an old profile recomputes on the calling thread (the UI
/// thread, from the controller), on purpose: the leave-one-point-out
/// predictions cost one design matrix and one small `p x p` solve per
/// calibration dot, about 2 ms for a two-monitor calibration of 360 samples and
/// 4 ms for 960 (degree 3, measured on a desktop; the tests keep a generous
/// bound). It happens once per profile and session; later calls are lookups. A
/// background thread would cost more than it saves.
pub fn axis_error(data: &mut CalibrationData, monitor: &Monitor) -> Option<(f64, f64)> {
    if data.axis_errors.is_none() {
        data.axis_errors = Some(axis_errors_of(data));
    }
    let index = data
        .monitors
        .iter()
        .find(|m| m.rect == monitor.rect)
        .map_or(monitor.index, |m| m.index);
    if let Some(found) = data.axis_errors.as_ref().and_then(|e| e.get(&index)) {
        return Some(*found);
    }
    let mean = data.report.get("mean_error_px").and_then(Value::as_f64);
    if let Some(mean) = mean {
        let value = mean / 2.0_f64.sqrt();
        if value.is_finite() && value > 0.0 {
            return Some((value, value));
        }
    }
    None
}

fn axis_errors_of(data: &mut CalibrationData) -> BTreeMap<i32, (f64, f64)> {
    let stored = axis_errors_from_dict(data.report.get("per_monitor_error_px")).unwrap_or_default();
    if !stored.is_empty() {
        return stored;
    }
    let computed = recompute_axis_errors(data);
    if !computed.is_empty() {
        let doc: Map<String, Value> = computed
            .iter()
            .map(|(k, &(x, y))| (k.to_string(), json!([x, y])))
            .collect();
        data.report
            .insert("per_monitor_error_px".into(), Value::Object(doc));
        log::debug!(
            "Gaze error per axis recomputed for the calibration of {}",
            data.created_at
        );
    }
    computed
}

/// Leave-one-point-out errors per monitor and axis of an older calibration.
fn recompute_axis_errors(data: &CalibrationData) -> BTreeMap<i32, (f64, f64)> {
    let samples: Vec<CalibrationSample> = data
        .samples
        .iter()
        .filter(|s| data.monitors.iter().any(|m| m.index == s.monitor_index))
        .cloned()
        .collect();
    let groups: Vec<i64> = samples.iter().map(|s| s.point_id).collect();
    let mut distinct = groups.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if samples.is_empty() || distinct.len() < 2 {
        return BTreeMap::new();
    }
    let alpha = data
        .report
        .get("alpha")
        .and_then(Value::as_f64)
        .filter(|a| *a >= 0.0)
        .unwrap_or(data.model.alpha);
    let degree = data
        .report
        .get("degree")
        .and_then(Value::as_i64)
        .and_then(|d| usize::try_from(d).ok())
        .filter(|d| SUPPORTED_DEGREES.contains(d))
        .unwrap_or(data.model.degree);
    let attempt = || -> Result<_, String> {
        let (x, y, w) = samples_to_arrays(&samples).map_err(|e| e.0)?;
        let bounds = virtual_bounds(&data.monitors).map_err(|e| e.to_string())?;
        let options = super::model::FitOptions {
            bounds: Some(bounds),
            weights: Some(&w),
            nonlinear: data.model.nonlinear(),
        };
        let preds = lopo_predictions(&x, &y, &groups, alpha, degree, &options)
            .map_err(|e| e.to_string())?;
        let owners: Vec<i32> = samples.iter().map(|s| s.monitor_index).collect();
        per_axis_errors(&preds, &y, &owners).map_err(|e| e.0)
    };
    attempt().unwrap_or_else(|exc| {
        log::debug!("Cannot recompute the per-axis gaze error: {exc}");
        BTreeMap::new()
    })
}

// --------------------------------------------------------------------- library
/// The saved calibrations, one profile per setup ([`CalibrationData::key`]),
/// most recently used first.
///
/// Typical use by the engine:
///
/// ```text
/// library = CalibrationLibrary::load(path)
/// data = library.best(backend, feature_version, monitors, camera)
/// ...after a calibration or a refit:   library.put(data); library.save(path)
/// ...after switching to a profile:     library.mark_used(key); library.save(path)
/// ```
///
/// At most `max_profiles` profiles are kept; adding one more drops the least
/// recently used. The library owns its profiles (Python held them by reference): a
/// profile is addressed by its [`ProfileKey`], and changing its model or implicit
/// samples through [`get_mut`](Self::get_mut) and calling [`save`](Self::save) persists the
/// change.
#[derive(Debug, Clone)]
pub struct CalibrationLibrary {
    pub max_profiles: usize,
    profiles: Vec<CalibrationData>,
}

impl Default for CalibrationLibrary {
    fn default() -> Self {
        CalibrationLibrary {
            max_profiles: MAX_PROFILES,
            profiles: Vec::new(),
        }
    }
}

impl CalibrationLibrary {
    /// A library of `profiles`, given most recent first.
    pub fn new(
        profiles: impl IntoIterator<Item = CalibrationData>,
        max_profiles: usize,
    ) -> Result<Self, super::calibration::CalibrationError> {
        if max_profiles < 1 {
            return Err(super::calibration::CalibrationError(
                "max_profiles must be >= 1".into(),
            ));
        }
        let mut library = CalibrationLibrary {
            max_profiles,
            profiles: Vec::new(),
        };
        // Given most recent first; adding in reverse keeps that order and drops
        // later duplicates of a key.
        let profiles: Vec<CalibrationData> = profiles.into_iter().collect();
        for data in profiles.into_iter().rev() {
            library.put(data);
        }
        Ok(library)
    }

    // ------------------------------------------------------------- persistence
    /// Read the calibration file; never fails (a bad file gives an empty library).
    pub fn load(path: &Path) -> Self {
        Self::load_limited(path, MAX_PROFILES).unwrap_or_default()
    }

    /// [`load`](Self::load) keeping at most `max_profiles` profiles.
    pub fn load_limited(
        path: &Path,
        max_profiles: usize,
    ) -> Result<Self, super::calibration::CalibrationError> {
        Self::new(read_profiles(path), max_profiles)
    }

    /// The file's content.
    pub fn to_text(&self) -> String {
        let mut doc = Map::new();
        doc.insert("format".into(), json!(FILE_FORMAT));
        doc.insert("version".into(), json!(CALIBRATION_VERSION));
        doc.insert(
            "profiles".into(),
            Value::Array(
                self.profiles
                    .iter()
                    .map(|p| Value::Object(profile_to_doc(p)))
                    .collect(),
            ),
        );
        dumps(&doc)
    }

    /// Write every profile to `path` atomically. Fails if writing fails.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        atomic_write_text(path, &self.to_text())?;
        log::debug!(
            "Saved {} calibration profile(s) to {}",
            self.profiles.len(),
            path.display()
        );
        Ok(())
    }

    // ------------------------------------------------------------- access
    /// All profiles, most recently used first.
    pub fn profiles(&self) -> &[CalibrationData] {
        &self.profiles
    }

    /// The most recently used profile (`None` when empty).
    pub fn latest(&self) -> Option<&CalibrationData> {
        self.profiles.first()
    }

    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, CalibrationData> {
        self.profiles.iter()
    }

    /// Whether a profile is stored for `key`.
    pub fn contains(&self, key: &ProfileKey) -> bool {
        self.get(key).is_some()
    }

    /// The profile stored for `key` (see [`CalibrationData::key`]).
    pub fn get(&self, key: &ProfileKey) -> Option<&CalibrationData> {
        self.profiles.iter().find(|p| p.key() == *key)
    }

    pub fn get_mut(&mut self, key: &ProfileKey) -> Option<&mut CalibrationData> {
        self.profiles.iter_mut().find(|p| p.key() == *key)
    }

    /// The best profile for the current setup, or `(None, reason)`.
    ///
    /// Among compatible profiles ([`CalibrationData::is_compatible`]), one
    /// whose camera and frame size are known to match beats one where they are
    /// unknown; otherwise the most recently used wins. Without any, `reason`
    /// explains what is wrong with the most recently used profile
    /// (`"not calibrated yet"` for an empty library).
    pub fn match_profile(
        &self,
        backend_name: &str,
        feature_version: &str,
        monitors: &[Monitor],
        camera: Option<&str>,
        frame_size: Option<(u32, u32)>,
    ) -> (Option<&CalibrationData>, String) {
        let mut best: Option<(usize, usize)> = None;
        for (position, data) in self.profiles.iter().enumerate() {
            let (ok, _) =
                data.is_compatible(backend_name, feature_version, monitors, camera, frame_size);
            if !ok {
                continue;
            }
            let mut specific = usize::from(
                !data.camera.trim().is_empty() && !camera.unwrap_or("").trim().is_empty(),
            );
            specific +=
                usize::from(frame_size.is_some_and(|f| {
                    data.frame_size.0.min(data.frame_size.1).min(f.0).min(f.1) > 0
                }));
            // (-specific, position) minimal: more specific first, then the most recent.
            if best.is_none_or(|(s, p)| {
                (std::cmp::Reverse(specific), position) < (std::cmp::Reverse(s), p)
            }) {
                best = Some((specific, position));
            }
        }
        if let Some((_, position)) = best {
            return (Some(&self.profiles[position]), String::new());
        }
        let Some(latest) = self.latest() else {
            return (None, "not calibrated yet".into());
        };
        let (_, reason) =
            latest.is_compatible(backend_name, feature_version, monitors, camera, frame_size);
        (None, reason)
    }

    /// The profile [`match_profile`](Self::match_profile) chooses, or `None`.
    pub fn best(
        &self,
        backend_name: &str,
        feature_version: &str,
        monitors: &[Monitor],
        camera: Option<&str>,
        frame_size: Option<(u32, u32)>,
    ) -> Option<&CalibrationData> {
        self.match_profile(backend_name, feature_version, monitors, camera, frame_size)
            .0
    }

    // ------------------------------------------------------------- changes
    /// Add `data` as the most recently used profile; returns the profiles it evicted.
    ///
    /// It replaces the profile with the same [`key`](CalibrationData::key), and a
    /// profile of the same setup whose camera is unknown (`""`, from an older
    /// file), which was most likely made with the same camera. Profiles beyond
    /// `max_profiles` are then dropped, least recently used first. The returned
    /// profiles are the replaced ones (also a stored profile of the same key, which Python
    /// recognised by identity and did not return), then the dropped ones.
    pub fn put(&mut self, data: CalibrationData) -> Vec<CalibrationData> {
        let (layout, backend, version, camera) = data.key();
        let is_replaced = |p: &CalibrationData| {
            let key = p.key();
            key == data.key()
                || (!camera.is_empty()
                    && key.0 == layout
                    && key.1 == backend
                    && key.2 == version
                    && key.3.is_empty())
        };
        let (replaced, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.profiles)
            .into_iter()
            .partition(is_replaced);
        self.profiles = std::iter::once(data).chain(kept).collect();
        let evicted = self
            .profiles
            .split_off(self.max_profiles.min(self.profiles.len()));
        for p in &evicted {
            log::info!(
                "Dropping the least recently used calibration ({})",
                p.created_at
            );
        }
        replaced.into_iter().chain(evicted).collect()
    }

    /// Make a stored profile the most recently used; false if it is not stored.
    pub fn mark_used(&mut self, key: &ProfileKey) -> bool {
        let Some(position) = self.profiles.iter().position(|p| p.key() == *key) else {
            return false;
        };
        let data = self.profiles.remove(position);
        self.profiles.insert(0, data);
        true
    }

    /// Remove a stored profile; false if it is not stored.
    pub fn remove(&mut self, key: &ProfileKey) -> bool {
        let Some(position) = self.profiles.iter().position(|p| p.key() == *key) else {
            return false;
        };
        self.profiles.remove(position);
        true
    }

    /// Remove every profile.
    pub fn clear(&mut self) {
        self.profiles.clear();
    }
}

impl<'a> IntoIterator for &'a CalibrationLibrary {
    type Item = &'a CalibrationData;
    type IntoIter = std::slice::Iter<'a, CalibrationData>;

    fn into_iter(self) -> Self::IntoIter {
        self.profiles.iter()
    }
}

// ------------------------------------------------------------ single profile
/// Store `data` in the calibration file at `path` as the most recently used
/// profile, replacing the profile of the same setup and keeping the others.
///
/// Writes atomically; fails if writing fails. A caller that holds a
/// [`CalibrationLibrary`] should use [`CalibrationLibrary::put`] and
/// [`CalibrationLibrary::save`] instead, which avoids re-reading the file.
pub fn save_calibration(path: &Path, data: &CalibrationData) -> io::Result<()> {
    let mut library = CalibrationLibrary::load(path);
    library.put(data.clone());
    library.save(path)
}

/// The most recently used calibration in the file at `path`; `None` if the file
/// is missing, corrupt or unsupported. Never fails.
pub fn load_calibration(path: &Path) -> Option<CalibrationData> {
    CalibrationLibrary::load(path).latest().cloned()
}

// ------------------------------------------------------------------ reading
enum ReadError {
    /// Written by a newer Eye Tracker.
    Newer(String),
    Invalid(String),
}

impl From<String> for ReadError {
    fn from(message: String) -> Self {
        ReadError::Invalid(message)
    }
}

impl From<&str> for ReadError {
    fn from(message: &str) -> Self {
        ReadError::Invalid(message.to_string())
    }
}

fn read_profiles(path: &Path) -> Vec<CalibrationData> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(exc) if exc.kind() == io::ErrorKind::NotFound => return Vec::new(),
        Err(exc) => {
            log::warn!("Could not read calibration {}: {exc}", path.display());
            return Vec::new();
        }
    };
    let text = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(exc) => {
            log::warn!("Could not read calibration {}: {exc}", path.display());
            return Vec::new();
        }
    };
    // A file saved by an editor (or PowerShell 5.1) with a BOM is still the user's
    // calibration, not a corrupt file.
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let parsed = serde_json::from_str::<Value>(text)
        .map_err(|exc| ReadError::Invalid(exc.to_string()))
        .and_then(|doc| profiles_from_doc(&doc, path));
    match parsed {
        Ok(profiles) => profiles,
        Err(ReadError::Newer(exc)) => {
            log::warn!(
                "Calibration {} was written by a newer Eye Tracker ({exc}); ignored",
                path.display()
            );
            Vec::new()
        }
        Err(ReadError::Invalid(exc)) => {
            log::warn!(
                "Ignoring corrupt calibration file {}: {exc}",
                path.display()
            );
            Vec::new()
        }
    }
}

fn profiles_from_doc(doc: &Value, path: &Path) -> Result<Vec<CalibrationData>, ReadError> {
    let Some(map) = doc.as_object() else {
        return Err("root is not an object".into());
    };
    if let Some(format) = map.get("format")
        && format.as_str() != Some(FILE_FORMAT)
    {
        return Err(format!("not a calibration file (format {})", py_repr(format)).into());
    }
    let version = map.get("version");
    let number = version.and_then(Value::as_number);
    let version = match number {
        Some(n) if n.is_i64() => n.as_i64().filter(|v| *v >= 1),
        // Above i64: certainly newer than this version.
        Some(n) if n.is_u64() => Some(i64::MAX),
        _ => None,
    };
    let Some(version) = version else {
        let shown = map.get("version").map_or("None".to_string(), py_repr);
        return Err(format!("invalid version {shown}").into());
    };
    if version > CALIBRATION_VERSION {
        return Err(ReadError::Newer(format!("version {version}")));
    }
    if version == 1 {
        return Ok(vec![profile_from_doc(doc)?]);
    }

    let profiles = list(
        map.get("profiles").ok_or("missing key 'profiles'")?,
        "profiles",
    )?;
    let mut out = Vec::new();
    for (i, item) in profiles.iter().enumerate() {
        match profile_from_doc(item) {
            Ok(profile) => out.push(profile),
            Err(exc) => log::warn!(
                "Ignoring corrupt calibration profile {} in {}: {exc}",
                i + 1,
                path.display()
            ),
        }
    }
    Ok(out)
}

fn profile_from_doc(doc: &Value) -> Result<CalibrationData, String> {
    let Some(map) = doc.as_object() else {
        return Err("profile is not an object".into());
    };
    let required = |key: &str| map.get(key).ok_or_else(|| format!("missing key '{key}'"));
    let model = GazeModel::from_dict(required("model")?).map_err(|e| e.to_string())?;
    let empty = Value::Array(Vec::new());
    let samples = samples_from_list(map.get("samples").unwrap_or(&empty), "samples")?;
    let implicit = samples_from_list(
        map.get("implicit_samples").unwrap_or(&empty),
        "implicit_samples",
    )?;
    let mut dims: Vec<usize> = samples
        .iter()
        .chain(&implicit)
        .map(|s| s.features.len())
        .collect();
    if model.is_fitted() {
        dims.push(model.n_features());
    }
    dims.sort_unstable();
    dims.dedup();
    if dims.len() > 1 {
        return Err(format!("inconsistent feature lengths {dims:?}"));
    }

    let report = match map.get("report") {
        Some(Value::Object(report)) => report.clone(),
        _ => Map::new(),
    };
    let monitors = list(required("monitors")?, "monitors")?
        .iter()
        .map(monitor_from_dict)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CalibrationData {
        // Stored under the current name, so the next save writes that.
        backend: canonical_backend(string(required("backend")?, "backend")?).to_string(),
        feature_version: string(required("feature_version")?, "feature_version")?.to_string(),
        layout_signature: string(required("layout_signature")?, "layout_signature")?.to_string(),
        monitors,
        samples,
        implicit_samples: implicit,
        model,
        report,
        created_at: map.get("created_at").map_or(String::new(), py_str),
        camera: camera_from_doc(map.get("camera")),
        frame_size: frame_size_from_doc(map.get("frame_size")),
        axis_errors: None,
    })
}

/// Optional metadata: a malformed value means "unknown", not a lost calibration.
fn camera_from_doc(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or("").to_string()
}

fn frame_size_from_doc(value: Option<&Value>) -> (u32, u32) {
    let Some([w, h]) = value.and_then(Value::as_array).map(Vec::as_slice) else {
        return (0, 0);
    };
    // Integers only: 640.0 and true are not sizes.
    let size = |v: &Value| {
        v.as_u64()
            .filter(|v| *v > 0)
            .and_then(|v| u32::try_from(v).ok())
    };
    match (size(w), size(h)) {
        (Some(w), Some(h)) => (w, h),
        _ => (0, 0),
    }
}

// ------------------------------------------------------------------ writing
fn profile_to_doc(data: &CalibrationData) -> Map<String, Value> {
    let (width, height) = data.frame_size;
    let mut doc = Map::new();
    doc.insert("created_at".into(), json!(data.created_at));
    doc.insert("backend".into(), json!(data.backend));
    doc.insert("feature_version".into(), json!(data.feature_version));
    doc.insert("layout_signature".into(), json!(data.layout_signature));
    doc.insert("camera".into(), json!(data.camera));
    doc.insert("frame_size".into(), json!([width, height]));
    doc.insert(
        "monitors".into(),
        Value::Array(data.monitors.iter().map(monitor_to_dict).collect()),
    );
    doc.insert("report".into(), Value::Object(data.report.clone()));
    doc.insert("model".into(), round_model(&data.model.to_dict()));
    doc.insert("samples".into(), samples_to_list(&data.samples));
    doc.insert(
        "implicit_samples".into(),
        samples_to_list(&data.implicit_samples),
    );
    doc
}

// ------------------------------------------------------------------- monitors
fn monitor_to_dict(m: &Monitor) -> Value {
    json!({
        "index": m.index,
        "name": m.name,
        "rect": m.rect.to_list(),
        "primary": m.primary,
        "scale": m.scale,
    })
}

fn monitor_from_dict(d: &Value) -> Result<Monitor, String> {
    let Some(map) = d.as_object() else {
        return Err("monitor entry is not an object".into());
    };
    let required = |key: &str| map.get(key).ok_or_else(|| format!("missing key '{key}'"));
    let scale = map.get("scale").map_or(Ok(1.0), json_float)?;
    if !scale.is_finite() {
        return Err("monitor scale is not a finite number".into());
    }
    let rect = list(required("rect")?, "rect")?;
    if rect.len() != 4 {
        return Err("rect must be a list of 4 integers".into());
    }
    let mut values = [0i32; 4];
    for (slot, item) in values.iter_mut().zip(rect) {
        *slot = json_i32(item)?;
    }
    Ok(Monitor {
        index: json_i32(required("index")?)?,
        name: map.get("name").map_or(String::new(), py_str),
        rect: Rect::from_list(values),
        primary: map.get("primary").is_some_and(py_truthy),
        scale,
    })
}

// -------------------------------------------------------------------- samples
/// `round(value, 6)`.
fn round_sample(value: f64) -> f64 {
    format!("{value:.SAMPLE_DECIMALS$}")
        .parse()
        .unwrap_or(value)
}

fn samples_to_list(samples: &[CalibrationSample]) -> Value {
    let mut out = Vec::with_capacity(samples.len());
    let mut dropped = 0;
    for s in samples {
        let finite = s.features.iter().all(|v| v.is_finite())
            && s.x.is_finite()
            && s.y.is_finite()
            && s.weight.is_finite();
        if !finite {
            dropped += 1;
            continue;
        }
        out.push(json!({
            "features": s.features.iter().map(|&v| round_sample(v)).collect::<Vec<_>>(),
            "x": round_sample(s.x),
            "y": round_sample(s.y),
            "monitor": s.monitor_index,
            "point": s.point_id,
            "weight": round_sample(s.weight),
        }));
    }
    if dropped > 0 {
        log::warn!("Not saving {dropped} calibration samples with non-finite values");
    }
    Value::Array(out)
}

fn samples_from_list(items: &Value, name: &str) -> Result<Vec<CalibrationSample>, String> {
    let mut out = Vec::new();
    for item in list(items, name)? {
        let Some(map) = item.as_object() else {
            return Err(format!("{name}: entry is not an object"));
        };
        let required = |key: &str| map.get(key).ok_or_else(|| format!("missing key '{key}'"));
        let raw = required("features")?
            .as_array()
            .ok_or_else(|| format!("{name}: features must be a non-empty list of numbers"))?;
        if raw.is_empty() {
            return Err(format!(
                "{name}: features must be a non-empty list of numbers"
            ));
        }
        // `null` reads as NaN, as numpy converts `None`; the finiteness check below rejects it.
        let features = raw
            .iter()
            .map(|v| match v {
                Value::Null => Ok(f64::NAN),
                other => json_float(other),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sample = CalibrationSample {
            features,
            x: json_float(required("x")?)?,
            y: json_float(required("y")?)?,
            monitor_index: json_i32(required("monitor")?)?,
            point_id: json_int(required("point")?)?,
            weight: map.get("weight").map_or(Ok(1.0), json_float)?,
        };
        if !(sample.features.iter().all(|v| v.is_finite())
            && sample.x.is_finite()
            && sample.y.is_finite()
            && sample.weight.is_finite()
            && sample.weight >= 0.0)
        {
            return Err(format!("{name}: invalid numbers"));
        }
        out.push(sample);
    }
    Ok(out)
}

// ------------------------------------------------------------------- helpers
/// Floats of the model keep ten significant digits (`float(f"{v:.10g}")`).
fn round_model(value: &Value) -> Value {
    match value {
        Value::Number(n) if n.is_f64() => {
            let v = n.as_f64().unwrap_or(0.0);
            let rounded = format!("{v:.prec$e}", prec = MODEL_DIGITS - 1)
                .parse()
                .unwrap_or(v);
            json!(rounded)
        }
        Value::Array(items) => Value::Array(items.iter().map(round_model).collect()),
        // The look-away regions.
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), round_model(v)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn compact(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// Pretty JSON: each profile's bulky parts (samples, model arrays) one item per line.
fn dumps(doc: &Map<String, Value>) -> String {
    let mut lines = Vec::new();
    for (key, value) in doc {
        let body = match value {
            Value::Array(profiles) if key == "profiles" && !profiles.is_empty() => {
                let items: Vec<String> = profiles
                    .iter()
                    .filter_map(Value::as_object)
                    .map(|p| format!("    {}", dump_profile(p, "    ")))
                    .collect();
                format!("[\n{}\n  ]", items.join(",\n"))
            }
            other => compact_spaced(other),
        };
        lines.push(format!("  {}: {body}", compact(&json!(key))));
    }
    format!("{{\n{}\n}}\n", lines.join(",\n"))
}

/// `json.dumps(value)` with its default separators (`", "` and `": "`) for the scalars and the
/// empty list of the file's top level.
fn compact_spaced(value: &Value) -> String {
    match value {
        Value::Array(items) if items.is_empty() => "[]".into(),
        other => compact(other),
    }
}

fn dump_profile(profile: &Map<String, Value>, indent: &str) -> String {
    let inner = format!("{indent}  ");
    let mut lines = Vec::new();
    for (key, value) in profile {
        let body = match value {
            Value::Array(rows)
                if matches!(key.as_str(), "samples" | "implicit_samples" | "monitors")
                    && !rows.is_empty() =>
            {
                let rows: Vec<String> = rows
                    .iter()
                    .map(|v| format!("{inner}  {}", compact(v)))
                    .collect();
                format!("[\n{}\n{inner}]", rows.join(",\n"))
            }
            Value::Object(rows) if key == "model" && !rows.is_empty() => {
                let rows: Vec<String> = rows
                    .iter()
                    .map(|(k, v)| format!("{inner}  {}: {}", compact(&json!(k)), compact(v)))
                    .collect();
                format!("{{\n{}\n{inner}}}", rows.join(",\n"))
            }
            other if key == "frame_size" => compact(other),
            other => serde_json::to_string_pretty(other)
                .unwrap_or_default()
                .replace('\n', &format!("\n{inner}")),
        };
        lines.push(format!("{inner}{}: {body}", compact(&json!(key))));
    }
    format!("{{\n{}\n{indent}}}", lines.join(",\n"))
}

fn list<'a>(value: &'a Value, name: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{name} must be a list"))
}

fn string<'a>(value: &'a Value, name: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{name} must be a string"))
}

/// Python's `repr` of a JSON value, as far as messages need it.
fn py_repr(value: &Value) -> String {
    match value {
        Value::String(s) => format!("'{s}'"),
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        other => other.to_string(),
    }
}

/// Python's `str` of a JSON value.
fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python's `bool` of a JSON value.
fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
