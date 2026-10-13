//! Running a terminal's own command-line tool, and looking at process trees.
//!
//! Providers never run a shell: the argument vector goes straight to the program,
//! with a timeout, no console window (Windows) and no standard input. Everything
//! here is injectable so that tests never start a process.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

use crate::types::{WindowHandle, WindowRef};

use super::super::types::PaneError;

/// Default time a provider command may take.
pub const COMMAND_TIMEOUT_S: f64 = 1.0;

/// Windows' `CREATE_NO_WINDOW` process creation flag.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    pub returncode: i32,
    pub stdout: String,
    pub stderr: String,
}

impl CommandResult {
    pub fn new(returncode: i32, stdout: impl Into<String>, stderr: impl Into<String>) -> Self {
        Self {
            returncode,
            stdout: stdout.into(),
            stderr: stderr.into(),
        }
    }

    pub fn ok(&self) -> bool {
        self.returncode == 0
    }
}

/// Runs an argument vector with a timeout in seconds (see [`run_command`]); tests pass a fake.
pub trait CommandRunner: Send + Sync {
    fn run(&self, argv: &[String], timeout: f64) -> Result<CommandResult, PaneError>;
}

impl<F> CommandRunner for F
where
    F: Fn(&[String], f64) -> Result<CommandResult, PaneError> + Send + Sync,
{
    fn run(&self, argv: &[String], timeout: f64) -> Result<CommandResult, PaneError> {
        self(argv, timeout)
    }
}

fn read_to_string(stream: Option<impl Read>) -> String {
    let mut bytes = Vec::new();
    if let Some(mut stream) = stream {
        let _ = stream.read_to_end(&mut bytes);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Run `argv` (no shell) and return its exit code and output.
///
/// Fails with [`PaneError`] when the program cannot be started or does not finish within
/// `timeout` seconds (it is killed then).
pub fn run_command(argv: &[String], timeout: f64) -> Result<CommandResult, PaneError> {
    let Some(program) = argv.first() else {
        return Err(PaneError::new("no command to run"));
    };
    let mut command = Command::new(program);
    command
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = command
        .spawn()
        .map_err(|e| PaneError::new(format!("{program} could not be started: {e}")))?;
    // Read both pipes on their own threads so a chatty child cannot block on a full pipe.
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout = thread::spawn(move || read_to_string(stdout));
    let stderr = thread::spawn(move || read_to_string(stderr));

    let deadline = Instant::now() + Duration::from_secs_f64(timeout.clamp(0.0, 3600.0));
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(2)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(PaneError::new(format!(
                    "{program} did not answer within {timeout} s"
                )));
            }
            Err(e) => {
                return Err(PaneError::new(format!(
                    "{program} could not be started: {e}"
                )));
            }
        }
    };
    Ok(CommandResult {
        returncode: status.code().unwrap_or(-1),
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
    })
}

/// A dictionary key for the window of a [`WindowRef`]: its handle and its process id, so a
/// handle reused by another process's window is another key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WindowKey(String, Option<u32>);

pub fn window_key(window: &WindowRef) -> WindowKey {
    // `WindowHandle` has no `Hash`; its Debug form identifies it all the same.
    let handle = match &window.handle {
        WindowHandle::Hwnd(h) => format!("hwnd:{h}"),
        WindowHandle::X11(id) => format!("x11:{id}"),
        WindowHandle::Ax { pid, element } => format!("ax:{pid}:{element}"),
    };
    WindowKey(handle, window.pid)
}

/// Finds a program by name (injectable for tests; [`which`] by default).
pub type Which = Box<dyn Fn(&str) -> Option<String> + Send>;

/// The program called `name`, as `shutil.which` finds it: a path to an executable file, or
/// a name looked up in `PATH` (with the `PATHEXT` extensions on Windows).
pub fn which(name: &str) -> Option<String> {
    let has_dir = name.contains('/') || (cfg!(windows) && name.contains('\\'));
    if has_dir {
        return executable_candidate(Path::new(name));
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .filter(|dir| !dir.as_os_str().is_empty())
        .find_map(|dir| executable_candidate(&dir.join(name)))
}

fn executable_candidate(path: &Path) -> Option<String> {
    if cfg!(windows) {
        pathext_candidate(path)
    } else {
        unix_candidate(path)
    }
}

/// `path` as is, or with one of the `PATHEXT` extensions, if that is a file (Windows).
fn pathext_candidate(path: &Path) -> Option<String> {
    let extensions = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into());
    let extensions: Vec<&str> = extensions.split(';').filter(|e| !e.is_empty()).collect();
    let lower = path.to_string_lossy().to_lowercase();
    let candidates: Vec<std::path::PathBuf> = if extensions
        .iter()
        .any(|e| lower.ends_with(&e.to_lowercase()))
    {
        vec![path.to_path_buf()]
    } else {
        extensions
            .iter()
            .map(|e| {
                let mut full = path.as_os_str().to_owned();
                full.push(e);
                full.into()
            })
            .collect()
    };
    candidates
        .into_iter()
        .find(|c| c.is_file())
        .map(|c| c.to_string_lossy().into_owned())
}

/// `path` if it is an executable file (Unix).
fn unix_candidate(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    #[cfg(unix)]
    let executable = {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    };
    #[cfg(not(unix))]
    let executable = true;
    (meta.is_file() && executable).then(|| path.to_string_lossy().into_owned())
}

/// Full path of the program `pid` runs (`None` when gone or unreadable).
pub fn executable_of(pid: u32) -> Option<String> {
    let mut system = System::new();
    let pid = Pid::from_u32(pid);
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_exe(UpdateKind::Always),
    );
    let exe = system.process(pid)?.exe()?;
    let exe = exe.to_string_lossy().into_owned();
    (!exe.is_empty()).then_some(exe)
}

/// What the process list says about one process.
struct ProcessEntry {
    parent: Option<u32>,
    name: String,
}

/// Parent and name of every running process.
struct ProcessTable(HashMap<u32, ProcessEntry>);

impl ProcessTable {
    fn read() -> Self {
        let mut system = System::new();
        system.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing(),
        );
        Self(
            system
                .processes()
                .iter()
                .map(|(pid, process)| {
                    (
                        pid.as_u32(),
                        ProcessEntry {
                            parent: process.parent().map(Pid::as_u32),
                            name: process.name().to_string_lossy().into_owned(),
                        },
                    )
                })
                .collect(),
        )
    }

    /// The ancestors of `pid`, nearest first (`[]` when it is gone).
    fn parents(&self, pid: u32) -> Vec<u32> {
        let mut found = Vec::new();
        let mut seen = HashSet::from([pid]);
        let mut next = self.0.get(&pid).and_then(|p| p.parent);
        while let Some(parent) = next {
            if !seen.insert(parent) {
                break;
            }
            found.push(parent);
            next = self.0.get(&parent).and_then(|p| p.parent);
        }
        found
    }

    /// Process names (see [`process_basename`]) of everything `pid` started.
    fn descendant_names(&self, pid: u32) -> HashSet<String> {
        if !self.0.contains_key(&pid) {
            return HashSet::new();
        }
        let mut names = HashSet::new();
        let mut seen = HashSet::from([pid]);
        let mut frontier = vec![pid];
        while let Some(parent) = frontier.pop() {
            for (child, entry) in &self.0 {
                if entry.parent == Some(parent) && seen.insert(*child) {
                    names.insert(process_basename(&entry.name));
                    frontier.push(*child);
                }
            }
        }
        names
    }
}

/// The ancestors of `pid`, nearest first (`[]` when it is gone or unreadable).
pub fn parent_pids(pid: u32) -> Vec<u32> {
    ProcessTable::read().parents(pid)
}

/// Process names (see [`process_basename`]) of everything `pid` started.
pub fn descendant_names(pid: u32) -> HashSet<String> {
    ProcessTable::read().descendant_names(pid)
}

// ---- TODO(integrator): unify with platform::base::process_basename (ported by another worker)
/// Executable extensions dropped from process names.
const EXECUTABLE_SUFFIXES: [&str; 3] = [".exe", ".app", ".bin"];

/// `"WindowsTerminal.exe"` → `"windowsterminal"`, `"/usr/bin/kitty"` → `"kitty"`: the lower-cased
/// file name without directory or executable extension.
pub fn process_basename(name: &str) -> String {
    let base = name.replace('/', "\\");
    let base = base.rsplit('\\').next().unwrap_or("").trim().to_lowercase();
    for suffix in EXECUTABLE_SUFFIXES {
        if let Some(stem) = base.strip_suffix(suffix) {
            return stem.to_string();
        }
    }
    base
}
// ---- end of the platform::base helper

#[cfg(test)]
#[path = "command_tests.rs"]
mod tests;
