//! Running a terminal's own command-line tool, and looking at process trees.
//!
//! Providers never run a shell: the argument vector goes straight to the program,
//! with a timeout, no console window (Windows) and no standard input. Everything
//! here is injectable so that tests never start a process.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

use crate::platform::base::process_basename;
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
pub struct WindowKey(WindowHandle, Option<u32>);

pub fn window_key(window: &WindowRef) -> WindowKey {
    WindowKey(window.handle.clone(), window.pid)
}

/// Finds a program by name (injectable for tests; [`which`](crate::platform::base::which) by default).
pub type Which = Box<dyn Fn(&str) -> Option<String> + Send>;

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

#[cfg(test)]
#[path = "command_tests.rs"]
mod tests;
