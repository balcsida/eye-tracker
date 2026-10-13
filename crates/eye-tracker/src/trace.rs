//! Opt-in tracking trace for tuning and bug reports (`eye-tracker run --trace FILE`).
//!
//! Each analysed frame becomes one JSON line of *numbers*: the backend's feature vector, the
//! estimated gaze point, the cursor position and the switching decision. No image data is ever
//! written. The file is only created when the user asks for it on the command line, and it
//! stays on the local disk.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use serde::Serialize;

/// Stop writing after this many lines (~100 MB) so a forgotten trace cannot fill a disk.
pub const MAX_LINES: usize = 500_000;

/// Rounding used when a caller has no opinion (`digits=4` in the Python).
pub const DEFAULT_DIGITS: i32 = 4;

/// JSON-safe rounded number (`None` for NaN/inf/missing).
fn num(value: Option<f64>, digits: i32) -> Option<f64> {
    let f = value.filter(|f| f.is_finite())?;
    let scale = 10f64.powi(digits);
    Some((f * scale).round() / scale)
}

#[derive(Default)]
struct State {
    file: Option<BufWriter<File>>,
    lines: usize,
    failed: bool,
}

/// Appends JSON lines to a file; thread-safe and failure-tolerant.
pub struct TraceWriter {
    pub path: PathBuf,
    max_lines: usize,
    state: Mutex<State>,
}

impl TraceWriter {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self::with_max_lines(path, MAX_LINES)
    }

    fn with_max_lines(path: impl AsRef<Path>, max_lines: usize) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            max_lines,
            state: Mutex::default(),
        }
    }

    /// Append one record. Errors are logged once and then ignored.
    pub fn write(&self, record: &impl Serialize) {
        let mut state = self.state.lock();
        if state.failed || state.lines >= self.max_lines {
            return;
        }
        if let Err(err) = self.append(&mut state, record) {
            state.failed = true;
            log::warn!("Tracking trace disabled: {err}");
        }
    }

    fn append(
        &self,
        state: &mut State,
        record: &impl Serialize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if state.file.is_none() {
            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent)?;
            }
            let file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(&self.path)?;
            state.file = Some(BufWriter::new(file));
            log::info!("Writing tracking trace to {}", self.path.display());
        }
        let mut line = serde_json::to_string(record)?;
        line.push('\n');
        let Some(file) = state.file.as_mut() else {
            return Ok(());
        };
        file.write_all(line.as_bytes())?;
        state.lines += 1;
        if state.lines.is_multiple_of(50) {
            file.flush()?;
        }
        if state.lines == self.max_lines {
            log::warn!(
                "Trace reached {} lines; no further lines are written",
                self.max_lines
            );
        }
        Ok(())
    }

    pub fn close(&self) {
        if let Some(mut file) = self.state.lock().file.take() {
            let _ = file.flush();
        }
    }

    pub fn number(value: Option<f64>, digits: i32) -> Option<f64> {
        num(value, digits)
    }

    pub fn vector(values: Option<&[f64]>, digits: i32) -> Option<Vec<Option<f64>>> {
        values.map(|v| v.iter().map(|x| num(Some(*x), digits)).collect())
    }
}

impl Drop for TraceWriter {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
