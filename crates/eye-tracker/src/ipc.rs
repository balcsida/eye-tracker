//! Single-instance guard and local command channel.
//!
//! A running app listens on a per-user local socket (a named pipe on Windows, a Unix domain
//! socket elsewhere). Launching the app a second time, `eye-tracker ctl <command>` and desktop
//! keyboard shortcuts (the only way to bind hotkeys on Wayland) all talk to it through this
//! channel.
//!
//! Which process *is* the running instance is decided by [`InstanceLock`], an OS file lock, not
//! by the socket: two servers can listen on one name (Windows creates a second pipe instance), so
//! "nobody answered, so I listen" is a race between two launches.
//!
//! Protocol: the client sends one UTF-8 line with a command from [`COMMANDS`] and receives one
//! line back: `ok`, `error: <reason>` or, for `status`, a JSON object. The server then closes the
//! connection.
//!
//! This is the only networking in the app, and it is local-only: local sockets never touch the
//! network, and the socket is restricted to the current user. The socket name is predictable, so
//! a client also checks that the server that answered runs under the same user account before it
//! sends anything (see [`server_is_trusted`]).
//!
//! On Windows that restriction is not enough by itself: pipe names are machine-wide and a pipe
//! keeps the security descriptor of whoever created the name *first*, so another account that
//! creates our name before we listen would decide who may connect. The server therefore refuses
//! to listen on a name that already exists while it holds the [`InstanceLock`], and it also
//! checks the account of every client before reading a request (see [`client_is_trusted`]). On
//! Unix the socket lives in a directory only this user can enter, which already keeps other
//! accounts out.
//!
//! The server runs on its own threads. It reports each request to its owner as an
//! [`Event::CommandReceived`] on a channel; the owner answers through the [`Reply`] that comes
//! with it, from any thread. A request nobody answers is answered with an error when the client's
//! idle time runs out.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use interprocess::local_socket::prelude::*;
use interprocess::local_socket::{Listener, ListenerNonblockingMode, ListenerOptions, Stream};
use parking_lot::Mutex;

use crate::config::Platform;
use crate::paths;

/// Commands understood by a running instance.
pub const COMMANDS: &[&str] = &[
    "show",
    "settings",
    "pause",
    "resume",
    "toggle",
    "privacy-on",
    "privacy-off",
    "privacy-toggle",
    "calibrate",
    "status",
    "quit",
];

pub const DEFAULT_TIMEOUT_MS: u64 = 1500;
/// How long a launch that lost the [`InstanceLock`] keeps asking the winner, which takes the lock
/// before it starts listening.
pub const STARTUP_WAIT_MS: u64 = 5000;
/// Pause between two attempts while waiting for an instance to answer.
pub const RETRY_INTERVAL_MS: u64 = 250;
/// How long [`is_running`] and the pre-listen probe wait for a connection.
pub const PROBE_TIMEOUT_MS: u64 = 300;
/// A client that has not sent a complete request by then is disconnected.
pub const CLIENT_IDLE_TIMEOUT_MS: u64 = 3000;
/// Requests are single short words; anything longer is garbage or abuse.
pub const MAX_REQUEST_BYTES: usize = 1024;
/// Replies are one line; `status` JSON stays far below this.
pub const MAX_REPLY_BYTES: usize = 64 * 1024;
/// Unix socket paths are limited to ~104-108 bytes depending on the OS.
const MAX_SOCKET_PATH: usize = 100;
/// How long a pipe name that exists before we listen may take to disappear (Windows): a client
/// still holding a pipe of an instance that just exited keeps the name alive for a moment. A name
/// that stays belongs to someone else.
pub const PIPE_NAME_GRACE_S: f64 = 1.0;
#[cfg(windows)]
const PIPE_NAME_POLL: Duration = Duration::from_millis(100);
#[cfg(windows)]
const PIPE_PREFIX: &str = "\\\\.\\pipe\\";
/// `GetLastError` of a connect to a pipe whose instances are all in use.
const ERROR_PIPE_BUSY: i32 = 231;
/// How often the polling loops look at their socket and deadline.
const POLL_INTERVAL: Duration = Duration::from_millis(2);
const ACCEPT_INTERVAL: Duration = Duration::from_millis(5);

/// Raised for a command that cannot be sent as one line (Python's `ValueError`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid command {0:?}")]
pub struct InvalidCommand(pub String);

/// What the server reports to its owner.
#[derive(Debug)]
pub enum Event {
    /// A client sent a known command; answer it through `reply` (only the first answer counts).
    CommandReceived { command: String, reply: Reply },
}

/// The way to answer one request. Cloneable and sendable to any thread.
#[derive(Debug, Clone)]
pub struct Reply(Sender<String>);

impl Reply {
    /// Send `text` as the one-line reply and close. Later calls are ignored.
    pub fn send(&self, text: impl Into<String>) {
        let _ = self.0.try_send(text.into());
    }
}

/// Name (or, on Linux, full socket path) of this user's instance socket.
///
/// On Linux the socket goes into `$XDG_RUNTIME_DIR` (private to the user and cleaned at logout)
/// or, without one, into a private `/tmp/eye-tracker-<uid>` directory, instead of the shared
/// `/tmp`, where another account could create the socket first.
pub fn server_name() -> String {
    let runtime = std::env::var("XDG_RUNTIME_DIR").unwrap_or_default();
    server_name_for(
        Platform::CURRENT,
        &paths::ipc_name(),
        &runtime,
        private_temp_dir,
        MAX_SOCKET_PATH,
    )
}

fn server_name_for(
    platform: Platform,
    name: &str,
    runtime_dir: &str,
    private_temp: impl FnOnce() -> Option<String>,
    max_socket_path: usize,
) -> String {
    if platform == Platform::Other
        && let Some(directory) = private_runtime_dir(runtime_dir, private_temp)
    {
        let full = Path::new(&directory)
            .join(name)
            .to_string_lossy()
            .into_owned();
        if full.len() <= max_socket_path {
            return full;
        }
    }
    name.to_string()
}

/// A directory only this user can create files in (Linux socket location).
fn private_runtime_dir(
    runtime_dir: &str,
    private_temp: impl FnOnce() -> Option<String>,
) -> Option<String> {
    let runtime = Path::new(runtime_dir);
    if !runtime_dir.is_empty() && runtime.is_absolute() && runtime.is_dir() {
        return Some(runtime_dir.to_string());
    }
    private_temp()
}

/// `<tmp>/eye-tracker-<uid>`, created with mode 0700.
///
/// `None` when it cannot be created or is not safely ours: another account may have created it
/// first, which must not let it choose our socket.
fn private_temp_dir() -> Option<String> {
    private_temp_dir_in(&std::env::temp_dir())
}

#[cfg(unix)]
fn private_temp_dir_in(temp: &Path) -> Option<String> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};

    use crate::APP_SLUG;

    // SAFETY: getuid has no preconditions and cannot fail.
    let uid = unsafe { libc::getuid() };
    let path = temp.join(format!("{APP_SLUG}-{uid}"));
    match fs::DirBuilder::new().mode(0o700).create(&path) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {}
        Err(err) => {
            log::debug!("Cannot create {}: {err}", path.display());
            return None;
        }
    }
    let info = fs::symlink_metadata(&path).ok()?;
    if !info.is_dir() || info.uid() != uid || info.mode() & 0o077 != 0 {
        log::warn!(
            "Not using {} for the instance socket: it is not a private directory",
            path.display()
        );
        return None;
    }
    Some(path.to_string_lossy().into_owned())
}

#[cfg(not(unix))]
fn private_temp_dir_in(_temp: &Path) -> Option<String> {
    None
}

/// Where the Unix socket called `name` lives: an absolute name is the path itself, a bare one goes
/// into the temporary directory (where Qt put it, so Python and Rust copies find each other).
#[cfg(unix)]
fn socket_path(name: &str) -> PathBuf {
    let path = Path::new(name);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::temp_dir().join(name)
    }
}

#[cfg(unix)]
fn local_name(name: &str) -> io::Result<interprocess::local_socket::Name<'static>> {
    socket_path(name).to_fs_name::<interprocess::local_socket::GenericFilePath>()
}

/// A pipe called `name` (`\\.\pipe\<name>`).
#[cfg(windows)]
fn local_name(name: &str) -> io::Result<interprocess::local_socket::Name<'static>> {
    name.to_string()
        .to_ns_name::<interprocess::local_socket::GenericNamespaced>()
}

/// What a client does around its attempts and how it judges the server; replaceable by tests.
struct ClientHooks<'a> {
    before_attempt: &'a dyn Fn(),
    server_trusted: &'a dyn Fn(&Stream, &str) -> bool,
}

impl Default for ClientHooks<'_> {
    fn default() -> Self {
        ClientHooks {
            before_attempt: &allow_foreground_handoff,
            server_trusted: &server_is_trusted,
        }
    }
}

/// Send `command` to the running instance and return its one-line reply.
///
/// Returns `Ok(None)` when no instance is running, or when one accepted the connection but did
/// not answer within `timeout_ms`.
///
/// `wait_ms` keeps trying for that long while nothing answers: a launch that lost the
/// [`InstanceLock`] uses it (with [`STARTUP_WAIT_MS`]) to reach an instance that holds the lock
/// but is still starting up. A server that accepts the connection but never replies is not asked
/// twice.
pub fn send_command(
    command: &str,
    timeout_ms: u64,
    wait_ms: u64,
) -> Result<Option<String>, InvalidCommand> {
    send_command_with(command, timeout_ms, wait_ms, &ClientHooks::default())
}

fn send_command_with(
    command: &str,
    timeout_ms: u64,
    wait_ms: u64,
    hooks: &ClientHooks<'_>,
) -> Result<Option<String>, InvalidCommand> {
    let text = command.trim().to_lowercase();
    if text.is_empty() || text.contains('\n') || text.contains('\r') {
        return Err(InvalidCommand(command.to_string()));
    }
    let request = format!("{text}\n");
    let deadline = Instant::now() + Duration::from_millis(wait_ms);
    let data = loop {
        (hooks.before_attempt)();
        let (connected, data) = exchange(
            &server_name(),
            Some(request.as_bytes()),
            timeout_ms,
            hooks.server_trusted,
        );
        if connected {
            break data;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(None);
        }
        thread::sleep(remaining.min(Duration::from_millis(RETRY_INTERVAL_MS)));
    };
    if data.is_empty() {
        log::warn!("The running instance did not answer {text:?} within {timeout_ms} ms");
        return Ok(None);
    }
    let line = data.split(|b| *b == b'\n').next().unwrap_or_default();
    Ok(Some(
        String::from_utf8_lossy(line)
            .trim_end_matches('\r')
            .to_string(),
    ))
}

/// Let the running instance bring a window to the front (Windows).
///
/// Windows only lets the process the user is interacting with take the foreground. The client (a
/// terminal running `ctl`, or a second launch from the Start menu) is that process, so it passes
/// the right on; otherwise "show"/"settings" would only flash the taskbar button.
fn allow_foreground_handoff() {
    winsec::allow_foreground_handoff();
}

/// Whether an instance for this user (and config directory) is listening.
pub fn is_running(timeout_ms: u64) -> bool {
    exchange(&server_name(), None, timeout_ms, &server_is_trusted).0
}

// ------------------------------------------------------------------------ lock
/// Exclusive lock that makes one process *the* running instance.
///
/// An OS file lock (`flock` on Unix, a byte-range lock on Windows) on a file named after the
/// socket, so every `--config-dir` profile has its own. Taking it is atomic, so of two launches
/// racing each other exactly one wins, and the OS drops it when the holder exits or crashes, so
/// it never goes stale. Hold it for the life of the process ([`InstanceServer`] releases it in
/// [`InstanceServer::close`]).
///
/// The file lives on a local disk, next to the socket where that is a file: a lock in a home
/// directory on a network share would also stop the same user's tracker on *another* computer.
///
/// Clones share the lock: the server that is given one releases the caller's.
#[derive(Clone)]
pub struct InstanceLock {
    name: String,
    state: Arc<Mutex<LockState>>,
    locker: fn(&File) -> io::Result<bool>,
}

struct LockState {
    file: Option<File>,
    enforced: bool,
}

impl InstanceLock {
    /// `name` is the socket name the lock belongs to; defaults to [`server_name`].
    pub fn new(name: Option<&str>) -> Self {
        InstanceLock {
            name: name.map_or_else(server_name, str::to_string),
            state: Arc::new(Mutex::new(LockState {
                file: None,
                enforced: true,
            })),
            locker: try_lock,
        }
    }

    #[cfg(test)]
    fn with_locker(mut self, locker: fn(&File) -> io::Result<bool>) -> Self {
        self.locker = locker;
        self
    }

    /// The lock file.
    pub fn path(&self) -> PathBuf {
        lock_path(
            &self.name,
            Platform::CURRENT,
            &std::env::temp_dir(),
            &paths::config_dir(),
        )
    }

    /// True while this object holds the lock.
    pub fn is_held(&self) -> bool {
        self.state.lock().file.is_some()
    }

    /// False after [`acquire`](Self::acquire) found that this system cannot lock the file.
    pub fn enforced(&self) -> bool {
        self.state.lock().enforced
    }

    /// Take the lock without waiting.
    ///
    /// Returns `false` only when another process (or another `InstanceLock` in this one) holds
    /// it. If the file cannot be locked at all (an odd filesystem), logs a warning, makes
    /// [`enforced`](Self::enforced) `false` and returns `true`: callers then fall back to probing
    /// the socket.
    pub fn acquire(&self) -> bool {
        let mut state = self.state.lock();
        if state.file.is_some() {
            return true;
        }
        let path = self.path();
        let opened = path
            .parent()
            .map_or(Ok(()), fs::create_dir_all)
            .and_then(|()| open_lock_file(&path));
        let file = match opened {
            Ok(file) => file,
            Err(err) => {
                log::warn!("Cannot create the instance lock ({err}); relying on the socket alone");
                state.enforced = false;
                return true;
            }
        };
        match (self.locker)(&file) {
            Err(err) => {
                log::warn!(
                    "Cannot lock {} ({err}); relying on the socket alone",
                    path.display()
                );
                state.enforced = false;
                true
            }
            Ok(false) => false,
            Ok(true) => {
                state.file = Some(file);
                state.enforced = true;
                log::debug!("Holding the instance lock {}", path.display());
                true
            }
        }
    }

    /// Give the lock up. Safe to call when it is not held.
    ///
    /// The file itself stays: deleting it would let a later launch lock a new file while a slower
    /// one still locks the old, unlinked one.
    pub fn release(&self) {
        let Some(file) = self.state.lock().file.take() else {
            return;
        };
        if let Err(err) = unlock(&file) {
            log::debug!("Unlocking the instance lock failed: {err}");
        }
    }
}

/// Where the lock file for the socket `name` lives on `platform`.
fn lock_path(name: &str, platform: Platform, temp: &Path, config_dir: &Path) -> PathBuf {
    let file = format!("{name}.lock");
    if Path::new(name).is_absolute() {
        // a socket path (Linux): keep them together
        PathBuf::from(file)
    } else if platform == Platform::MacOs {
        // The socket goes into $TMPDIR, which is private to the user and always local; the
        // config directory may be in a network home.
        temp.join(file)
    } else {
        // Windows: pipes have no file. The config directory is the local (not roaming) AppData
        // one, or the --config-dir the name is derived from.
        config_dir.join(file)
    }
}

#[cfg(unix)]
fn open_lock_file(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn open_lock_file(path: &Path) -> io::Result<File> {
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

/// Lock `file` exclusively without waiting; `false` when someone else holds it.
///
/// Fails when the file cannot be locked at all.
#[cfg(unix)]
fn try_lock(file: &File) -> io::Result<bool> {
    use std::os::fd::AsRawFd;

    // flock, not fcntl locks: its locks also conflict between two descriptors of one process, and
    // they belong to the open file, not the process.
    // SAFETY: the descriptor is open for the duration of the call.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }
    let err = io::Error::last_os_error();
    if err.kind() == io::ErrorKind::WouldBlock {
        Ok(false)
    } else {
        Err(err)
    }
}

/// Drop the lock right away (closing the file does it too).
#[cfg(unix)]
fn unlock(file: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;

    // SAFETY: the descriptor is open for the duration of the call.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(windows)]
fn try_lock(file: &File) -> io::Result<bool> {
    winsec::lock_first_byte(file)
}

#[cfg(windows)]
fn unlock(file: &File) -> io::Result<()> {
    winsec::unlock_first_byte(file)
}

// ---------------------------------------------------------------------- server
type ClientTrust = dyn Fn(&Stream) -> bool + Send + Sync;

/// Local socket server of the running instance.
///
/// Each request is announced through the event channel as [`Event::CommandReceived`]. Unknown
/// commands are rejected before the owner is involved. When the owner is gone (the receiving end
/// of the channel is dropped) the server answers `error: no handler`.
///
/// Only the holder of the [`InstanceLock`] listens: [`listen`](Self::listen) takes the lock (or
/// uses one the caller already took) and [`close`](Self::close) releases it.
pub struct InstanceServer {
    events: Sender<Event>,
    name: String,
    lock: Option<InstanceLock>,
    idle_timeout: Duration,
    client_trusted: Arc<ClientTrust>,
    pipe_name_grace_s: f64,
    listening: Option<Listening>,
    another_instance: bool,
}

struct Listening {
    shared: Arc<Shared>,
    acceptor: JoinHandle<()>,
}

/// State the accept loop and the connection threads share.
struct Shared {
    events: Sender<Event>,
    idle_timeout: Duration,
    client_trusted: Arc<ClientTrust>,
    stop: AtomicBool,
    /// Connections refused by the account check (only the first is logged loudly).
    refused_clients: AtomicUsize,
    connections: Mutex<Vec<JoinHandle<()>>>,
}

impl InstanceServer {
    pub fn new(events: Sender<Event>) -> Self {
        InstanceServer {
            events,
            name: server_name(),
            lock: None,
            idle_timeout: Duration::from_millis(CLIENT_IDLE_TIMEOUT_MS),
            client_trusted: Arc::new(client_is_trusted),
            pipe_name_grace_s: PIPE_NAME_GRACE_S,
            listening: None,
            another_instance: false,
        }
    }

    /// Server name override (tests); defaults to [`server_name`].
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// The instance lock, typically acquired at startup before anything else; by default
    /// [`listen`](Self::listen) creates and takes one for the server's name.
    pub fn with_lock(mut self, lock: InstanceLock) -> Self {
        self.lock = Some(lock);
        self
    }

    #[cfg(test)]
    fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    #[cfg(test)]
    fn with_client_trust(
        mut self,
        trusted: impl Fn(&Stream) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.client_trusted = Arc::new(trusted);
        self
    }

    #[cfg(all(test, windows))]
    fn with_pipe_name_grace(mut self, seconds: f64) -> Self {
        self.pipe_name_grace_s = seconds;
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The instance lock (created by [`listen`](Self::listen) unless one was passed in).
    pub fn lock(&self) -> Option<&InstanceLock> {
        self.lock.as_ref()
    }

    pub fn is_listening(&self) -> bool {
        self.listening.is_some()
    }

    /// True when [`listen`](Self::listen) failed because another instance holds the lock (or,
    /// where files cannot be locked, answers on the socket).
    pub fn another_instance_running(&self) -> bool {
        self.another_instance
    }

    /// Become the running instance and start listening.
    ///
    /// Returns `false` if another instance holds the [`InstanceLock`] (see
    /// [`another_instance_running`](Self::another_instance_running)) or the socket cannot be
    /// created; in the latter case the lock stays held, since this process still is the instance,
    /// just one that `ctl` cannot reach. That includes a Windows pipe name some other process
    /// created first (see the module docs).
    pub fn listen(&mut self) -> bool {
        if self.is_listening() {
            return true;
        }
        self.another_instance = false;
        let lock = self
            .lock
            .get_or_insert_with(|| InstanceLock::new(Some(&self.name)))
            .clone();
        if !lock.acquire() {
            self.another_instance = true;
            log::info!("Another instance holds {}", lock.path().display());
            return false;
        }
        // Without a file lock, fall back to asking the socket. (Two servers can listen on one
        // name, so listen() itself would not fail.)
        if !lock.is_held() && exchange(&self.name, None, PROBE_TIMEOUT_MS, &server_is_trusted).0 {
            self.another_instance = true;
            log::info!("Another instance is already listening on {}", self.name);
            return false;
        }
        if winsec::pipe_name_taken(&self.name, self.pipe_name_grace_s) {
            // Our own instance would have answered the probe or held the lock, so this is
            // another account's pipe. Joining it would give our pipe instances that account's
            // access rules.
            log::warn!(
                "The command channel {} already exists and does not belong to this instance \
                 (most likely another user account created it); '{}' and desktop shortcuts \
                 cannot reach this instance",
                self.name,
                crate::cli::cli_command_text(&["ctl"])
            );
            return false;
        }
        let Some(listener) = create_listener(&self.name) else {
            return false;
        };
        let shared = Arc::new(Shared {
            events: self.events.clone(),
            idle_timeout: self.idle_timeout,
            client_trusted: Arc::clone(&self.client_trusted),
            stop: AtomicBool::new(false),
            refused_clients: AtomicUsize::new(0),
            connections: Mutex::new(Vec::new()),
        });
        let acceptor = thread::Builder::new().name("ipc-server".into()).spawn({
            let shared = Arc::clone(&shared);
            move || accept_loop(&listener, &shared)
        });
        match acceptor {
            Ok(acceptor) => {
                self.listening = Some(Listening { shared, acceptor });
                true
            }
            Err(err) => {
                log::warn!("Cannot start the command server thread: {err}");
                false
            }
        }
    }

    /// Stop listening, drop open connections and release the instance lock.
    ///
    /// Safe to call more than once.
    pub fn close(&mut self) {
        if let Some(listening) = self.listening.take() {
            listening.shared.stop.store(true, Ordering::SeqCst);
            let _ = listening.acceptor.join();
            let connections = std::mem::take(&mut *listening.shared.connections.lock());
            for connection in connections {
                let _ = connection.join();
            }
        }
        if let Some(lock) = &self.lock {
            lock.release();
        }
    }
}

impl Drop for InstanceServer {
    fn drop(&mut self) {
        self.close();
    }
}

fn create_listener(name: &str) -> Option<Listener> {
    let build = || -> io::Result<Listener> {
        let options = ListenerOptions::new().name(local_name(name)?);
        #[cfg(windows)]
        let options = winsec::restrict_to_current_user(options);
        let listener = options.create_sync()?;
        listener.set_nonblocking(ListenerNonblockingMode::Accept)?;
        Ok(listener)
    };
    let created = match build() {
        // An instance that crashed leaves its socket file behind on Unix. We hold the lock (or
        // nobody answered the probe), so it is stale.
        Err(err) if cfg!(unix) && err.kind() == io::ErrorKind::AddrInUse => {
            log::debug!("listen({name}) failed ({err}); removing a stale socket");
            remove_stale_socket(name);
            build()
        }
        other => other,
    };
    match created {
        Ok(listener) => {
            restrict_socket_file(name);
            log::debug!("Listening for commands on {name}");
            Some(listener)
        }
        Err(err) => {
            log::warn!("Cannot create the instance socket {name}: {err}");
            None
        }
    }
}

#[cfg(unix)]
fn remove_stale_socket(name: &str) {
    let _ = fs::remove_file(socket_path(name));
}

#[cfg(not(unix))]
fn remove_stale_socket(_name: &str) {}

/// Only this user may connect (Qt's `UserAccessOption`); the pipe's access rules do the same on
/// Windows.
#[cfg(unix)]
fn restrict_socket_file(name: &str) {
    use std::os::unix::fs::PermissionsExt;

    let path = socket_path(name);
    if let Err(err) = fs::set_permissions(&path, fs::Permissions::from_mode(0o700)) {
        log::debug!("Cannot restrict {}: {err}", path.display());
    }
}

#[cfg(not(unix))]
fn restrict_socket_file(_name: &str) {}

fn accept_loop(listener: &Listener, shared: &Arc<Shared>) {
    while !shared.stop.load(Ordering::SeqCst) {
        let stream = match listener.accept() {
            Ok(stream) => stream,
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(ACCEPT_INTERVAL);
                continue;
            }
            Err(err) => {
                log::debug!("accept failed: {err}");
                thread::sleep(Duration::from_millis(50));
                continue;
            }
        };
        if !(shared.client_trusted)(&stream) {
            // Checked before a byte is read: pause, privacy-on or quit from another account
            // would switch off the walk-away lock.
            let previous = shared.refused_clients.fetch_add(1, Ordering::SeqCst);
            log::log!(
                refusal_level(previous),
                "Refused a command connection from another user account"
            );
            continue;
        }
        let connection = thread::Builder::new().name("ipc-connection".into()).spawn({
            let shared = Arc::clone(shared);
            move || serve(stream, &shared)
        });
        match connection {
            Ok(handle) => {
                let mut connections = shared.connections.lock();
                connections.retain(|c| !c.is_finished());
                connections.push(handle);
            }
            Err(err) => log::warn!("Cannot start a command connection thread: {err}"),
        }
    }
}

/// Loud once, quiet afterwards: a client retrying in a loop must not flood the log.
fn refusal_level(previous_refusals: usize) -> log::Level {
    if previous_refusals > 0 {
        log::Level::Debug
    } else {
        log::Level::Warn
    }
}

/// One client connection: buffers the request line and sends a single reply.
fn serve(stream: Stream, shared: &Shared) {
    let deadline = Instant::now() + shared.idle_timeout;
    if start_polling(&stream).is_err() {
        return;
    }
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 512];
    let line = loop {
        if shared.stop.load(Ordering::SeqCst) {
            return;
        }
        match read_ready(&stream, &mut chunk) {
            Ok(0) => return,
            Ok(n) => {
                buffer.extend_from_slice(&chunk[..n]);
                // One request per connection; anything after the first line is ignored.
                if let Some(newline) = buffer.iter().position(|b| *b == b'\n') {
                    break buffer[..newline].to_vec();
                }
                if buffer.len() > MAX_REQUEST_BYTES {
                    return respond(&stream, "error: request too long");
                }
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    log::debug!("Closing an idle command connection");
                    return respond(&stream, "error: no complete request received");
                }
                thread::sleep(POLL_INTERVAL);
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => return,
        }
    };
    let Ok(text) = std::str::from_utf8(&line) else {
        return respond(&stream, "error: the request is not valid UTF-8");
    };
    let command = text.trim().to_lowercase();
    if !COMMANDS.contains(&command.as_str()) {
        let mut valid = COMMANDS.to_vec();
        valid.sort_unstable();
        return respond(
            &stream,
            &format!(
                "error: unknown command '{command}' (valid: {})",
                valid.join(", ")
            ),
        );
    }
    log::info!("Received command {command:?}");
    let (reply, answers) = crossbeam_channel::bounded(1);
    let event = Event::CommandReceived {
        command,
        reply: Reply(reply),
    };
    if shared.events.send(event).is_err() {
        return respond(&stream, "error: no handler");
    }
    match await_reply(&answers, deadline, shared) {
        Some(text) => respond(&stream, &text),
        None if shared.stop.load(Ordering::SeqCst) => {}
        None => respond(&stream, "error: the command was not answered in time"),
    }
}

/// The owner's answer, or `None` when the idle time ran out, the owner dropped its [`Reply`]
/// without answering, or the server is closing.
fn await_reply(answers: &Receiver<String>, deadline: Instant, shared: &Shared) -> Option<String> {
    loop {
        let wait = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(20));
        match answers.recv_timeout(wait) {
            Ok(text) => return Some(text),
            Err(RecvTimeoutError::Disconnected) => return Some("error: no handler".into()),
            Err(RecvTimeoutError::Timeout) => {
                // A reply queued just before close (e.g. to `quit`) still goes out.
                if shared.stop.load(Ordering::SeqCst) || Instant::now() >= deadline {
                    return answers.try_recv().ok();
                }
            }
        }
    }
}

/// Write `text` as the one-line reply and close the connection.
fn respond(stream: &Stream, text: &str) {
    let line = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut line = line.lines().collect::<Vec<_>>().join(" ");
    if line.is_empty() {
        line = "ok".into();
    }
    line.push('\n');
    if stream.set_nonblocking(false).is_err() {
        return;
    }
    let mut writer = stream;
    if writer
        .write_all(line.as_bytes())
        .and_then(|()| writer.flush())
        .is_err()
    {
        return;
    }
    // Unread request bytes would make closing reset the connection and could lose the reply.
    let _ = start_polling(stream);
    let mut sink = [0u8; 512];
    while matches!(read_ready(stream, &mut sink), Ok(n) if n > 0) {}
}

/// Make [`read_ready`] on `stream` return instead of waiting for data.
fn start_polling(stream: &Stream) -> io::Result<()> {
    // Windows: interprocess reports a non-blocking pipe read with nothing to read as the end of
    // the stream, so the pipe stays blocking and `read_ready` looks before it reads.
    if cfg!(windows) {
        Ok(())
    } else {
        stream.set_nonblocking(true)
    }
}

/// Read what has arrived, or fail with `WouldBlock` when nothing has (after [`start_polling`]).
fn read_ready(stream: &Stream, buf: &mut [u8]) -> io::Result<usize> {
    #[cfg(windows)]
    let buf = match winsec::bytes_waiting(stream)? {
        0 => return Err(io::ErrorKind::WouldBlock.into()),
        waiting => {
            let len = buf.len().min(waiting as usize);
            &mut buf[..len]
        }
    };
    (&*stream).read(buf)
}

// ---------------------------------------------------------------------- client
/// Connect to `name`; send `request` and read until a newline, close or timeout.
///
/// With `request = None` only the connection is tested. Returns `(connected, data)`.
fn exchange(
    name: &str,
    request: Option<&[u8]>,
    timeout_ms: u64,
    server_trusted: &dyn Fn(&Stream, &str) -> bool,
) -> (bool, Vec<u8>) {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms.max(1));
    let Some(stream) = connect(name, deadline) else {
        return (false, Vec::new());
    };
    // Checked before a single byte is sent: an impostor learns nothing and, on Windows, cannot
    // impersonate us (that needs data read from the pipe).
    if !server_trusted(&stream, name) {
        log::warn!("Ignoring the instance socket {name}: another user account owns it");
        return (false, Vec::new());
    }
    let mut buffer = Vec::new();
    let Some(request) = request else {
        return (true, buffer);
    };
    let mut writer = &stream;
    if writer
        .write_all(request)
        .and_then(|()| writer.flush())
        .is_err()
        || start_polling(&stream).is_err()
    {
        return (true, buffer);
    }
    let mut chunk = [0u8; 4096];
    while Instant::now() < deadline {
        match read_ready(&stream, &mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buffer.extend_from_slice(&chunk[..n]);
                if buffer.contains(&b'\n') || buffer.len() > MAX_REPLY_BYTES {
                    break;
                }
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => thread::sleep(POLL_INTERVAL),
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    (true, buffer)
}

fn connect(name: &str, deadline: Instant) -> Option<Stream> {
    let local = local_name(name).ok()?;
    loop {
        match Stream::connect(local.clone()) {
            Ok(stream) => return Some(stream),
            // Windows: every pipe instance is in use for a moment while the server makes the next.
            Err(err)
                if cfg!(windows)
                    && err.raw_os_error() == Some(ERROR_PIPE_BUSY)
                    && Instant::now() < deadline =>
            {
                thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

// ------------------------------------------------------------ server identity
/// False when the server that answered on our socket name runs as another user.
///
/// The name is predictable and, on Windows, pipe names are machine-wide, so another local account
/// could create it first: to keep this app from starting (and its walk-away lock from running),
/// or to read our requests. `true` when the owner cannot be determined (nothing to go on).
fn server_is_trusted(stream: &Stream, name: &str) -> bool {
    let verdict = if cfg!(windows) {
        pipe_peer_is_current_user(stream)
    } else {
        Ok(socket_file_is_current_user(&socket_file(name)))
    };
    trusted(verdict, "the owner of the instance socket")
}

/// False when the process that connected to our server runs as another user.
///
/// Only Windows needs this (see the module docs): a pipe name another account created first keeps
/// that account's access rules, so its owner could connect to our instances and pause tracking,
/// switch privacy mode on or quit. Elevated processes of the same user pass. `true` when the
/// client cannot be identified, like [`server_is_trusted`], and always on Unix, where the private
/// socket directory is the access check.
fn client_is_trusted(stream: &Stream) -> bool {
    if !cfg!(windows) {
        return true;
    }
    trusted(pipe_peer_is_current_user(stream), "the command client")
}

/// `false` only for a definite "another account"; an error or unknown owner is trusted.
fn trusted(verdict: io::Result<Option<bool>>, what: &str) -> bool {
    match verdict {
        Ok(verdict) => verdict != Some(false),
        Err(err) => {
            log::debug!("Could not identify {what}: {err}");
            true
        }
    }
}

#[cfg(unix)]
fn socket_file(name: &str) -> String {
    socket_path(name).to_string_lossy().into_owned()
}

#[cfg(not(unix))]
fn socket_file(name: &str) -> String {
    name.to_string()
}

/// Whether the process on the other end of the pipe runs as this user (Windows).
///
/// The pid is recorded by the pipe file system when the client connects, so the client cannot
/// choose it. A client that exited meanwhile cannot be opened and counts as another account,
/// which only drops a dead connection.
fn pipe_peer_is_current_user(stream: &Stream) -> io::Result<Option<bool>> {
    let Some(pid) = stream.peer_creds()?.pid() else {
        return Ok(None);
    };
    #[allow(clippy::unnecessary_cast)] // i32 on Unix, u32 on Windows
    let pid = pid as u32;
    if pid == std::process::id() {
        return Ok(Some(true));
    }
    Ok(winsec::process_is_current_user(pid))
}

/// Whether the Unix socket file at `path` belongs to this user.
///
/// The socket file is created by the process that listens on it, so its owner is the server's
/// user. (Another account can neither replace our file nor create one owned by us.)
fn socket_file_is_current_user(path: &str) -> Option<bool> {
    #[cfg(unix)]
    {
        // SAFETY: getuid has no preconditions and cannot fail.
        socket_file_owned_by(path, Some(unsafe { libc::getuid() }))
    }
    #[cfg(not(unix))]
    {
        socket_file_owned_by(path, None)
    }
}

fn socket_file_owned_by(path: &str, uid: Option<u32>) -> Option<bool> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        let uid = uid?;
        if path.is_empty() {
            return None;
        }
        Some(fs::metadata(path).ok()?.uid() == uid)
    }
    #[cfg(not(unix))]
    {
        let _ = (path, uid);
        None
    }
}

// ------------------------------------------------------------------- Windows
#[cfg(not(windows))]
mod winsec {
    //! The Windows identity checks do not exist elsewhere.

    pub fn allow_foreground_handoff() {}

    pub fn process_is_current_user(_pid: u32) -> Option<bool> {
        None
    }

    pub fn pipe_name_taken(_name: &str, _grace_s: f64) -> bool {
        false
    }
}

#[cfg(windows)]
mod winsec {
    //! Windows: byte-range lock, account checks through process tokens, pipe name squatting,
    //! peeking into pipes and the foreground hand-off.

    use std::fs::File;
    use std::io;
    use std::os::windows::io::{AsHandle, AsRawHandle};
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    use interprocess::local_socket::{ListenerOptions, Stream};
    use interprocess::os::windows::local_socket::ListenerOptionsExt;
    use interprocess::os::windows::security_descriptor::{
        AsSecurityDescriptorExt, BorrowedSecurityDescriptor, SecurityDescriptor,
    };
    use windows::Win32::Foundation::{
        CloseHandle, ERROR_LOCK_VIOLATION, GetLastError, HANDLE, HLOCAL, LocalFree,
    };
    use windows::Win32::Security::Authorization::{
        ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
    };
    use windows::Win32::Security::{
        EqualSid, GetTokenInformation, PSECURITY_DESCRIPTOR, PSID, TOKEN_QUERY, TOKEN_USER,
        TokenUser,
    };
    use windows::Win32::Storage::FileSystem::{LockFile, UnlockFile};
    use windows::Win32::System::Pipes::{PeekNamedPipe, WaitNamedPipeW};
    use windows::Win32::System::Threading::{
        GetCurrentProcess, OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows::Win32::UI::WindowsAndMessaging::{ASFW_ANY, AllowSetForegroundWindow};
    use windows::core::{PCWSTR, PWSTR};

    use super::{PIPE_NAME_POLL, PIPE_PREFIX};

    /// `WaitNamedPipeW` results (`GetLastError`) that tell whether a pipe name exists.
    const ERROR_FILE_NOT_FOUND: u32 = 2;
    const ERROR_SEM_TIMEOUT: u32 = 121; // the name exists, but no instance is waiting for a client
    const ERROR_PIPE_BUSY: u32 = 231;

    pub fn allow_foreground_handoff() {
        // SAFETY: plain Win32 call without pointers.
        if let Err(err) = unsafe { AllowSetForegroundWindow(ASFW_ANY) } {
            log::debug!("AllowSetForegroundWindow failed: {err}");
        }
    }

    fn handle_of(file: &File) -> HANDLE {
        HANDLE(file.as_raw_handle())
    }

    /// The byte-range lock on the first byte (locking beyond the end of the empty file is
    /// allowed): the same range `msvcrt.locking` locks, so Python and Rust copies exclude each
    /// other.
    pub fn lock_first_byte(file: &File) -> io::Result<bool> {
        // SAFETY: the handle is open for the duration of the call.
        match unsafe { LockFile(handle_of(file), 0, 0, 1, 0) } {
            Ok(()) => Ok(true),
            // A range somebody else locked. windows-rs errors carry an HRESULT, which converting
            // to io::Error keeps as the raw code, so compare before converting.
            Err(err) if err.code() == ERROR_LOCK_VIOLATION.to_hresult() => Ok(false),
            Err(err) => Err(io::Error::from(err)),
        }
    }

    /// How many bytes wait to be read on `stream`'s pipe.
    pub fn bytes_waiting(stream: &Stream) -> io::Result<u32> {
        let Stream::NamedPipe(pipe) = stream;
        let mut waiting = 0u32;
        // SAFETY: the handle is open for the duration of the call and `waiting` outlives it.
        unsafe {
            PeekNamedPipe(
                HANDLE(pipe.as_handle().as_raw_handle()),
                None,
                0,
                None,
                Some(&mut waiting),
                None,
            )
        }
        .map_err(io::Error::from)?;
        Ok(waiting)
    }

    pub fn unlock_first_byte(file: &File) -> io::Result<()> {
        // SAFETY: the handle is open for the duration of the call.
        unsafe { UnlockFile(handle_of(file), 0, 0, 1, 0) }.map_err(io::Error::from)
    }

    /// A `TOKEN_USER` structure; the buffer is `u64`s to keep it aligned for the SID pointer.
    pub struct TokenUserBuf(Vec<u64>);

    impl TokenUserBuf {
        fn sid(&self) -> PSID {
            // SAFETY: the buffer holds a TOKEN_USER filled in by GetTokenInformation.
            unsafe { (*self.0.as_ptr().cast::<TOKEN_USER>()).User.Sid }
        }
    }

    /// The `TOKEN_USER` of `process`, or `None`.
    fn token_user(process: HANDLE) -> Option<TokenUserBuf> {
        let mut token = HANDLE::default();
        // SAFETY: `token` outlives the call; the handle is closed below.
        unsafe { OpenProcessToken(process, TOKEN_QUERY, &mut token) }.ok()?;
        let read = || {
            let mut size = 0u32;
            // SAFETY: asks for the size only.
            let _ = unsafe { GetTokenInformation(token, TokenUser, None, 0, &mut size) };
            if size == 0 {
                return None;
            }
            let mut buffer = vec![0u64; (size as usize).div_ceil(8)];
            // SAFETY: the buffer is at least `size` bytes.
            unsafe {
                GetTokenInformation(
                    token,
                    TokenUser,
                    Some(buffer.as_mut_ptr().cast()),
                    size,
                    &mut size,
                )
            }
            .ok()?;
            Some(TokenUserBuf(buffer))
        };
        let result = read();
        // SAFETY: `token` was opened above.
        let _ = unsafe { CloseHandle(token) };
        result
    }

    /// `TOKEN_USER` of this process (cached; it cannot change).
    fn own_token_user() -> Option<&'static TokenUserBuf> {
        static OWN: OnceLock<Option<TokenUserBuf>> = OnceLock::new();
        // SAFETY: GetCurrentProcess returns a pseudo handle.
        OWN.get_or_init(|| token_user(unsafe { GetCurrentProcess() }))
            .as_ref()
    }

    /// Whether process `pid` runs under this process's user account.
    ///
    /// `Some(false)` also when the process cannot be opened: processes of the same user always
    /// grant limited query access, so that means another account.
    pub fn process_is_current_user(pid: u32) -> Option<bool> {
        let own = own_token_user()?;
        // SAFETY: plain Win32 call; the handle is closed below.
        let Ok(process) = (unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) })
        else {
            return Some(false);
        };
        let theirs = token_user(process);
        // SAFETY: `process` was opened above.
        let _ = unsafe { CloseHandle(process) };
        let Some(theirs) = theirs else {
            return Some(false);
        };
        // SAFETY: both SIDs live in their buffers.
        Some(unsafe { EqualSid(own.sid(), theirs.sid()) }.is_ok())
    }

    /// Whether a named pipe called `name` already exists.
    ///
    /// `WaitNamedPipeW` only looks the name up; unlike opening the pipe it neither connects nor
    /// uses up an instance. A name that exists is watched for `grace_s`, since one whose last
    /// server just exited can linger while a client still holds it. Any answer other than
    /// "exists" or "not found" counts as free: refusing to listen would cost `ctl` for nothing.
    pub fn pipe_name_taken(name: &str, grace_s: f64) -> bool {
        let path = if name.starts_with("\\\\") {
            name.to_string()
        } else {
            format!("{PIPE_PREFIX}{name}")
        };
        let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
        let deadline = Instant::now() + Duration::from_secs_f64(grace_s);
        loop {
            // SAFETY: `wide` is NUL-terminated and outlives the call.
            let exists = if unsafe { WaitNamedPipeW(PCWSTR(wide.as_ptr()), 1) }.as_bool() {
                true
            } else {
                // SAFETY: no preconditions.
                let error = unsafe { GetLastError() }.0;
                if !matches!(error, ERROR_SEM_TIMEOUT | ERROR_PIPE_BUSY)
                    && error != ERROR_FILE_NOT_FOUND
                {
                    log::debug!("WaitNamedPipe({path}) failed with error {error}");
                }
                matches!(error, ERROR_SEM_TIMEOUT | ERROR_PIPE_BUSY)
            };
            if !exists {
                return false;
            }
            if Instant::now() >= deadline {
                return true;
            }
            std::thread::sleep(PIPE_NAME_POLL);
        }
    }

    /// Qt's `UserAccessOption`: only this user's account may open the pipe.
    pub fn restrict_to_current_user(options: ListenerOptions<'_>) -> ListenerOptions<'_> {
        match user_only_descriptor() {
            Ok(descriptor) => options.security_descriptor(descriptor),
            Err(err) => {
                log::warn!("Cannot restrict the command pipe to this user: {err}");
                options
            }
        }
    }

    fn user_only_descriptor() -> io::Result<SecurityDescriptor> {
        let own = own_token_user().ok_or_else(|| io::Error::other("no process token"))?;
        let mut text = PWSTR::null();
        // SAFETY: `text` receives a LocalAlloc'd string that is freed below.
        unsafe { ConvertSidToStringSidW(own.sid(), &mut text) }.map_err(io::Error::from)?;
        // SAFETY: the call succeeded, so `text` is a valid NUL-terminated string.
        let sid = unsafe { text.to_string() };
        // SAFETY: allocated by ConvertSidToStringSidW.
        let _ = unsafe { LocalFree(Some(HLOCAL(text.0.cast()))) };
        let sddl: Vec<u16> = format!("D:(A;;GA;;;{})", sid.map_err(io::Error::other)?)
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let mut descriptor = PSECURITY_DESCRIPTOR::default();
        // SAFETY: `sddl` is NUL-terminated; `descriptor` receives a LocalAlloc'd block.
        unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                PCWSTR(sddl.as_ptr()),
                1, // SDDL_REVISION_1
                &mut descriptor,
                None,
            )
        }
        .map_err(io::Error::from)?;
        // SAFETY: a valid self-relative descriptor; the clone owns its own memory.
        let owned = unsafe { BorrowedSecurityDescriptor::from_ptr(descriptor.0) }.to_owned_sd();
        // SAFETY: allocated by the conversion above and no longer referenced.
        let _ = unsafe { LocalFree(Some(HLOCAL(descriptor.0))) };
        owned
    }
}

#[cfg(test)]
#[path = "ipc_tests.rs"]
mod tests;
