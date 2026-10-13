//! Tests of the single-instance socket and command channel (tests/test_ipc.py).
//!
//! The server runs on its own threads, so unlike in the Python tests the client needs no event
//! loop to serve it. Every test uses [`env`], so the socket name (derived from the config
//! directory) is unique. The Python server's synchronous `handler` argument became the owner of
//! the event channel: [`server_with`] starts a thread that answers each event with a closure.
//!
//! Not ported:
//! * `test_commands_match_the_controller`: the controller is not ported yet (there is no
//!   `engine::controller::COMMANDS` to compare with).
//! * `test_private_temp_dir_needs_posix_users`: it deletes `os.getuid` to simulate a system
//!   without users; the Unix and non-Unix code are separate `cfg` branches here.
//! * The `_pipe_server_is_current_user(0)` / `_pipe_client_is_current_user(0)` lines of
//!   `test_windows_process_owner_check` and the `_client_is_trusted(object())` line of
//!   `test_windows_checks_are_inert_elsewhere`: the pipe checks take a connected stream here,
//!   not a raw handle; the inert case is checked with a real connection instead.
//! * `test_instance_lock_on_macos_lives_next_to_the_socket` patched `sys.platform`; the same
//!   assertions run on `lock_path` with an explicit platform.
//! * The log levels of the refusals in `test_client_of_another_account_is_refused`: the records
//!   are written by the server's threads, which the per-thread log capture does not see; the rule
//!   is checked on `refusal_level`.
//! * The exact warning text with the `ctl` command in `test_pipe_name_created_by_someone_else_is_not_joined`
//!   (`cli_command_text` cannot be replaced here); the rest of the message is checked.

use std::sync::atomic::{AtomicBool, AtomicUsize};

use crate::testing::{AppDirs, app_dirs, capture_logs, lock_paths};

use super::*;

/// The `app_dirs` fixture; also removes the default instance lock file, which (unlike the
/// config directory) can live in the shared temporary directory.
struct Env {
    dirs: AppDirs,
}

impl Env {
    fn path(&self) -> &Path {
        self.dirs.path()
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = fs::remove_file(InstanceLock::new(None).path());
    }
}

fn env() -> Env {
    Env { dirs: app_dirs() }
}

/// A server whose owner answers every command with `handler(command)`.
fn server_with(handler: impl Fn(&str) -> String + Send + 'static) -> InstanceServer {
    let (events, received) = crossbeam_channel::unbounded();
    thread::spawn(move || {
        for event in received {
            let Event::CommandReceived { command, reply } = event;
            reply.send(handler(&command));
        }
    });
    InstanceServer::new(events)
}

fn ok_server() -> InstanceServer {
    server_with(|_| "ok".into())
}

/// A handler that records each command and answers `reply`.
fn recorder(
    calls: &Arc<Mutex<Vec<String>>>,
    reply: &'static str,
) -> impl Fn(&str) -> String + use<> {
    let calls = Arc::clone(calls);
    move |command| {
        calls.lock().push(command.to_string());
        reply.to_string()
    }
}

fn send(command: &str) -> Option<String> {
    send_command(command, DEFAULT_TIMEOUT_MS, 0).unwrap()
}

fn probe_unrestricted(name: &str, request: &[u8]) -> (bool, Vec<u8>) {
    exchange(name, Some(request), 3000, &server_is_trusted)
}

#[test]
fn test_round_trip() {
    let _env = env();
    let received = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&received);
    let mut server = server_with(move |command| {
        log.lock().push(command.to_string());
        if command == "status" {
            r#"{"state":"tracking"}"#.into()
        } else {
            "ok".into()
        }
    });
    assert!(server.listen());
    assert!(server.is_listening());
    assert!(is_running(PROBE_TIMEOUT_MS));
    assert_eq!(send("status").as_deref(), Some(r#"{"state":"tracking"}"#));
    // Commands are normalised (case, surrounding blanks) before they are sent.
    assert_eq!(send("  Pause ").as_deref(), Some("ok"));
    assert_eq!(*received.lock(), ["status", "pause"]);
}

#[test]
fn test_no_instance() {
    let _env = env();
    assert!(!is_running(PROBE_TIMEOUT_MS));
    assert_eq!(send_command("status", 200, 0).unwrap(), None);
}

#[test]
fn test_invalid_commands_are_rejected_locally() {
    let _env = env();
    for bad in ["", "   ", "pause\nquit", "a\rb"] {
        let err = send_command(bad, DEFAULT_TIMEOUT_MS, 0).unwrap_err();
        assert!(err.to_string().contains("invalid command"), "{bad:?}");
    }
}

#[test]
fn test_unknown_command_is_refused_by_the_server() {
    let _env = env();
    let handled = Arc::new(Mutex::new(Vec::new()));
    let mut server = server_with(recorder(&handled, "ok"));
    assert!(server.listen());
    let reply = send("format-disk").unwrap();
    assert!(
        reply.starts_with("error: unknown command 'format-disk'"),
        "{reply}"
    );
    assert!(reply.contains("privacy-toggle")); // lists the valid commands
    assert!(handled.lock().is_empty());
}

#[test]
fn test_handler_errors_become_error_replies() {
    let _env = env();
    let mut server = server_with(|_| "error: boom".into());
    assert!(server.listen());
    assert_eq!(send("pause").as_deref(), Some("error: boom"));
}

#[test]
fn test_server_without_an_owner_answers_no_handler() {
    let _env = env();
    let (events, received) = crossbeam_channel::unbounded();
    drop(received);
    let mut server = InstanceServer::new(events);
    assert!(server.listen());
    assert_eq!(send("pause").as_deref(), Some("error: no handler"));
}

#[test]
fn test_multi_line_replies_are_flattened() {
    let _env = env();
    let mut server = server_with(|_| "first\nsecond".into());
    assert!(server.listen());
    assert_eq!(send("status").as_deref(), Some("first second"));
}

#[test]
fn test_slot_can_reply_asynchronously() {
    let _env = env();
    let (events, received) = crossbeam_channel::unbounded();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    thread::spawn(move || {
        for event in received {
            let Event::CommandReceived { command, reply } = event;
            log.lock().push(command.clone());
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                reply.send(format!("later:{command}"));
            });
        }
    });
    let mut server = InstanceServer::new(events);
    assert!(server.listen());
    assert_eq!(send("toggle").as_deref(), Some("later:toggle"));
    assert_eq!(*seen.lock(), ["toggle"]);
}

#[test]
fn test_slot_reply_takes_precedence_over_handler() {
    // There is no separate handler any more; what remains of the rule is that the first reply
    // counts and later ones are ignored.
    let _env = env();
    let (events, received) = crossbeam_channel::unbounded();
    thread::spawn(move || {
        for event in received {
            let Event::CommandReceived { reply, .. } = event;
            reply.send("from slot");
            reply.send("from handler");
        }
    });
    let mut server = InstanceServer::new(events);
    assert!(server.listen());
    assert_eq!(send("show").as_deref(), Some("from slot"));
}

#[test]
fn test_second_server_detects_the_first() {
    let _env = env();
    let mut first = ok_server();
    assert!(first.listen());
    assert!(first.listen()); // idempotent
    let mut second = ok_server();
    assert!(!second.listen());
    assert!(second.another_instance_running());
    assert!(!second.is_listening());
}

#[test]
fn test_close_stops_listening_and_is_idempotent() {
    let _env = env();
    let mut server = ok_server();
    assert!(server.listen());
    server.close();
    server.close();
    assert!(!server.is_listening());
    assert!(!is_running(PROBE_TIMEOUT_MS));
    assert_eq!(send_command("status", 200, 0).unwrap(), None);
    // The name is free again for a new instance.
    let mut again = ok_server();
    assert!(again.listen());
    assert_eq!(send("status").as_deref(), Some("ok"));
}

#[test]
fn test_idle_client_is_disconnected() {
    let _env = env();
    let mut server = ok_server().with_idle_timeout(Duration::from_millis(50));
    assert!(server.listen());
    // Connect and send nothing: the server gives up after the idle timeout.
    let (connected, data) = probe_unrestricted(server.name(), b"");
    assert!(connected);
    assert_eq!(
        String::from_utf8(data).unwrap().trim(),
        "error: no complete request received"
    );
}

#[test]
fn test_overlong_request_is_refused() {
    let _env = env();
    let mut server = ok_server();
    assert!(server.listen());
    let (connected, data) = probe_unrestricted(server.name(), &vec![b'x'; MAX_REQUEST_BYTES + 10]);
    assert!(connected);
    assert_eq!(
        String::from_utf8(data).unwrap().trim(),
        "error: request too long"
    );
}

#[test]
fn test_invalid_utf8_is_refused() {
    let _env = env();
    let mut server = ok_server();
    assert!(server.listen());
    let (connected, data) = probe_unrestricted(server.name(), b"\xff\xfe\n");
    assert!(connected);
    assert_eq!(
        String::from_utf8(data).unwrap().trim(),
        "error: the request is not valid UTF-8"
    );
}

#[test]
fn test_explicit_server_name() {
    let _env = env();
    let name = server_name() + "-explicit";
    let mut server = ok_server().with_name(name.clone());
    assert_eq!(server.name(), name);
    assert!(server.listen());
    let (connected, data) = probe_unrestricted(&name, b"status\n");
    assert!(connected);
    assert_eq!(data, b"ok\n");
    // The default name is still free.
    assert!(!is_running(PROBE_TIMEOUT_MS));
    server.close();
    let _ = fs::remove_file(InstanceLock::new(Some(&name)).path());
}

#[test]
fn test_server_name_depends_on_the_config_dir() {
    let _guard = lock_paths();
    let tmp = tempfile::tempdir().unwrap();
    paths::set_base_override(Some(&tmp.path().join("a")));
    let first = server_name();
    paths::set_base_override(Some(&tmp.path().join("b")));
    let second = server_name();
    paths::set_base_override(None);
    assert_ne!(first, second);
    for name in [first, second] {
        let path = Path::new(&name);
        let base = path.file_name().unwrap().to_string_lossy();
        assert!(base.starts_with("eye-tracker-"), "{name}");
        // A bare name (the system chooses the place) or, on Linux, a socket path in a private
        // directory, unless that is too long.
        if path.parent().is_some_and(|p| !p.as_os_str().is_empty()) {
            assert_eq!(Platform::CURRENT, Platform::Other);
            assert!(path.is_absolute());
        }
    }
}

#[test]
fn test_server_name_uses_the_runtime_dir_on_linux() {
    let _env = env();
    let tmp = tempfile::tempdir().unwrap();
    let runtime = tmp.path().join("run");
    fs::create_dir(&runtime).unwrap();
    let runtime = runtime.to_str().unwrap();
    let name = paths::ipc_name();
    let full = Path::new(runtime).join(&name).to_str().unwrap().to_string();
    let linux =
        |runtime: &str, max: usize| server_name_for(Platform::Other, &name, runtime, || None, max);
    // The temporary directory's own length must not decide the outcome.
    assert_eq!(linux(runtime, full.len()), full);

    // Too long for a Unix socket path, missing or relative: the system's default location.
    assert_eq!(linux(runtime, full.len() - 1), name);
    assert_eq!(linux("relative/dir", 10_000), name);
    assert_eq!(linux("", 10_000), name);
}

#[test]
fn test_server_name_without_runtime_dir_uses_a_private_temp_dir() {
    // Never the shared /tmp, where another account could create the socket first.
    let _env = env();
    let tmp = tempfile::tempdir().unwrap();
    let private = tmp.path().join("private").to_str().unwrap().to_string();
    let name = paths::ipc_name();
    let got = server_name_for(
        Platform::Other,
        &name,
        "",
        || Some(private.clone()),
        10_000, // the temporary directory may be long
    );
    assert_eq!(got, Path::new(&private).join(&name).to_str().unwrap());
}

#[cfg(unix)]
#[test]
fn test_private_temp_dir_is_created_for_this_user_only() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().unwrap();
    let uid = unsafe { libc::getuid() };
    let path = private_temp_dir_in(tmp.path()).unwrap();
    assert_eq!(
        path,
        tmp.path()
            .join(format!("eye-tracker-{uid}"))
            .to_str()
            .unwrap()
    );
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(
        private_temp_dir_in(tmp.path()).as_deref(),
        Some(path.as_str())
    ); // reused
}

#[cfg(unix)]
#[test]
fn test_private_temp_dir_refuses_a_directory_others_can_write() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().unwrap();
    let uid = unsafe { libc::getuid() };
    let shared = tmp.path().join(format!("eye-tracker-{uid}"));
    fs::create_dir(&shared).unwrap();
    fs::set_permissions(&shared, fs::Permissions::from_mode(0o777)).unwrap();
    assert_eq!(private_temp_dir_in(tmp.path()), None);
    fs::remove_dir(&shared).unwrap();
    std::os::unix::fs::symlink(tmp.path(), &shared).unwrap(); // a symlink planted by someone else
    assert_eq!(private_temp_dir_in(tmp.path()), None);
}

#[test]
fn test_server_name_ignores_the_runtime_dir_elsewhere() {
    let _env = env();
    let tmp = tempfile::tempdir().unwrap();
    let runtime = tmp.path().to_str().unwrap();
    let name = paths::ipc_name();
    for platform in [Platform::Windows, Platform::MacOs] {
        assert_eq!(
            server_name_for(platform, &name, runtime, || None, 10_000),
            name
        );
    }
}

#[test]
fn test_send_command_hands_over_the_foreground_right() {
    // Every request lets the running instance raise its window (Windows focus rules).
    let _env = env();
    let calls = AtomicUsize::new(0);
    let hand_over = || {
        calls.fetch_add(1, Ordering::SeqCst);
    };
    let hooks = ClientHooks {
        before_attempt: &hand_over,
        server_trusted: &server_is_trusted,
    };
    // nothing is listening
    assert_eq!(send_command_with("status", 200, 0, &hooks).unwrap(), None);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

// ------------------------------------------------------------------ instance lock
#[test]
fn test_instance_lock_is_exclusive() {
    let env = env();
    let (first, second) = (InstanceLock::new(None), InstanceLock::new(None));
    let name = server_name();
    let file_name = format!(
        "{}.lock",
        Path::new(&name).file_name().unwrap().to_string_lossy()
    );
    assert_eq!(
        first.path().file_name().unwrap().to_string_lossy(),
        file_name
    );
    let parent = first.path().parent().unwrap().to_path_buf();
    if Path::new(&name).is_absolute() {
        // Linux: next to the socket
        assert_eq!(Some(parent.as_path()), Path::new(&name).parent());
    } else if Platform::CURRENT == Platform::MacOs {
        // next to the socket in the private, local $TMPDIR
        assert_eq!(parent, std::env::temp_dir());
    } else {
        // Windows: in the (local) config directory
        assert_eq!(parent, paths::config_dir());
        assert_eq!(parent, env.path().canonicalize().unwrap());
    }
    assert!(first.acquire());
    assert!(first.acquire()); // idempotent
    assert!(first.is_held());
    assert!(first.enforced());
    assert!(!second.acquire());
    assert!(!second.is_held());
    first.release();
    first.release(); // idempotent
    assert!(!first.is_held());
    assert!(second.acquire());
    second.release();
    assert!(first.path().exists()); // kept: deleting it would race with other launches
}

#[test]
fn test_instance_lock_next_to_a_socket_path() {
    let tmp = tempfile::tempdir().unwrap();
    let socket_path = tmp.path().join("run").join("eye-tracker-abc");
    let socket_path = socket_path.to_str().unwrap();
    let lock = InstanceLock::new(Some(socket_path));
    assert_eq!(lock.path(), PathBuf::from(format!("{socket_path}.lock")));
    assert!(lock.acquire());
    assert!(lock.path().is_file());
    lock.release();
}

#[test]
fn test_instance_lock_on_macos_lives_next_to_the_socket() {
    // In $TMPDIR (local, per user), not in a home directory that may be on a network share.
    let temp = Path::new("/tmp/T");
    let config = Path::new("/config");
    assert_eq!(
        lock_path("eye-tracker-abc", Platform::MacOs, temp, config),
        temp.join("eye-tracker-abc.lock")
    );
    assert_eq!(
        lock_path("eye-tracker-abc", Platform::Windows, temp, config),
        config.join("eye-tracker-abc.lock")
    );
}

fn locking_unsupported(_file: &File) -> io::Result<bool> {
    Err(io::Error::other("No locks available"))
}

#[test]
fn test_instance_lock_without_file_locking_is_not_enforced() {
    let _env = env();
    let logs = capture_logs();
    let lock = InstanceLock::new(None).with_locker(locking_unsupported);
    assert!(lock.acquire());
    assert!(!lock.enforced());
    assert!(!lock.is_held());
    let warnings = logs.messages_at_least(log::Level::Warn);
    assert!(
        warnings
            .iter()
            .any(|m| m.contains("relying on the socket alone"))
    );
}

#[test]
fn test_instance_lock_that_cannot_be_created_is_not_enforced() {
    let tmp = tempfile::tempdir().unwrap();
    let blocker = tmp.path().join("file");
    fs::write(&blocker, "").unwrap();
    // its "directory" is a file
    let lock = InstanceLock::new(Some(blocker.join("sock").to_str().unwrap()));
    assert!(lock.acquire());
    assert!(!lock.enforced());
    assert!(!lock.is_held());
}

const LOCK_HOLDER_DIR: &str = "EYE_TRACKER_TEST_LOCK_HOLDER";

/// Not a test of its own: the process that `test_instance_lock_between_processes_and_after_a_crash`
/// starts (this test binary, run for just this test) takes the lock and holds it until killed.
#[test]
fn helper_lock_holder() {
    let Ok(dir) = std::env::var(LOCK_HOLDER_DIR) else {
        return;
    };
    paths::set_base_override(Some(Path::new(&dir)));
    let lock = InstanceLock::new(None);
    println!("{}", if lock.acquire() { "held" } else { "busy" });
    let _ = io::stdin().read_to_end(&mut Vec::new());
}

#[test]
fn test_instance_lock_between_processes_and_after_a_crash() {
    // Of two processes only one gets the lock; a killed holder never leaves it stale.
    use std::io::BufRead;
    use std::process::{Command, Stdio};

    let env = env();
    let mut holder = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "ipc::tests::helper_lock_holder", "--nocapture"])
        .env(LOCK_HOLDER_DIR, env.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let lock = InstanceLock::new(None);
    let stdout = holder.stdout.take().unwrap();
    let state = io::BufReader::new(stdout)
        .lines()
        .map_while(Result::ok)
        .map(|l| l.trim().to_string())
        .find(|l| l == "held" || l == "busy");
    let contended = lock.acquire();
    holder.kill().unwrap();
    holder.wait().unwrap();
    assert_eq!(state.as_deref(), Some("held"));
    assert!(!contended);
    // The OS releases the lock of a dead process (on Windows possibly a moment later).
    let deadline = Instant::now() + Duration::from_secs(5);
    while !lock.acquire() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    assert!(lock.is_held());
    lock.release();
}

// ----------------------------------------------------------- server and the lock
#[test]
fn test_lock_holder_that_is_not_listening_yet_blocks_a_second_server() {
    // The race of two launches: probing the socket alone would let both listen.
    let _env = env();
    let starting = InstanceLock::new(None);
    assert!(starting.acquire());
    assert!(!is_running(PROBE_TIMEOUT_MS)); // nobody answers yet...
    let mut server = ok_server();
    assert!(!server.listen()); // ...but the lock decides
    assert!(server.another_instance_running());
    assert!(!server.is_listening());
    starting.release();
}

#[test]
fn test_server_uses_a_lock_taken_at_startup() {
    let _env = env();
    let lock = InstanceLock::new(None);
    assert!(lock.acquire());
    let mut server = ok_server().with_lock(lock.clone());
    assert!(server.lock().is_some());
    assert!(server.listen());
    assert_eq!(send("status").as_deref(), Some("ok"));
    server.close();
    assert!(!lock.is_held());
    assert!(!is_running(PROBE_TIMEOUT_MS));
}

#[test]
fn test_server_takes_the_lock_itself_and_releases_it() {
    let _env = env();
    let mut server = ok_server();
    assert!(server.lock().is_none());
    assert!(server.listen());
    let lock = server.lock().unwrap().clone();
    assert!(lock.is_held());
    server.close();
    assert!(!lock.is_held());
}

#[test]
fn test_without_file_locks_the_socket_probe_decides() {
    let _env = env();
    let unlocked = || InstanceLock::new(None).with_locker(locking_unsupported);
    let mut first = ok_server().with_lock(unlocked());
    assert!(first.listen());
    let mut second = ok_server().with_lock(unlocked());
    assert!(!second.listen());
    assert!(second.another_instance_running());
}

// ------------------------------------------------------- waiting for an instance
#[test]
fn test_send_command_waits_for_an_instance_that_is_starting() {
    let _env = env();
    let mut server = server_with(|command| format!("hello {command}"));
    let starting = thread::spawn(move || {
        thread::sleep(Duration::from_millis(150));
        assert!(server.listen());
        server
    });
    let started = Instant::now();
    assert_eq!(
        send_command("show", DEFAULT_TIMEOUT_MS, 5000)
            .unwrap()
            .as_deref(),
        Some("hello show")
    );
    assert!(started.elapsed() >= Duration::from_millis(100));
    drop(starting.join().unwrap());
}

#[test]
fn test_send_command_gives_up_after_the_wait() {
    let _env = env();
    let started = Instant::now();
    assert_eq!(send_command("show", DEFAULT_TIMEOUT_MS, 300).unwrap(), None);
    let elapsed = started.elapsed();
    assert!(
        Duration::from_millis(250) <= elapsed && elapsed < Duration::from_secs(3),
        "{elapsed:?}"
    );
}

#[test]
fn test_send_command_does_not_resend_to_a_silent_server() {
    let _env = env();
    let (events, received) = crossbeam_channel::unbounded();
    let mut server = InstanceServer::new(events); // nobody replies
    assert!(server.listen());
    assert_eq!(send_command("show", 200, 2000).unwrap(), None);
    let seen: Vec<String> = received
        .try_iter()
        .map(|Event::CommandReceived { command, .. }| command)
        .collect();
    assert_eq!(seen, ["show"]);
}

// ----------------------------------------------------------------- server identity
#[test]
fn test_server_of_another_account_is_ignored() {
    // A squatter on our predictable name gets no request and is not "already running".
    let _env = env();
    let logs = capture_logs();
    let handled = Arc::new(Mutex::new(Vec::new()));
    let mut impostor = server_with(recorder(&handled, "ok"));
    assert!(impostor.listen());
    let distrust = |_: &Stream, _: &str| false;
    let hooks = ClientHooks {
        before_attempt: &|| {},
        server_trusted: &distrust,
    };
    assert_eq!(send_command_with("show", 300, 0, &hooks).unwrap(), None);
    assert!(!exchange(&server_name(), None, PROBE_TIMEOUT_MS, &distrust).0);
    let warnings = logs.messages_at_least(log::Level::Warn);
    assert!(warnings.iter().any(|m| m.contains("another user account")));
    thread::sleep(Duration::from_millis(50));
    assert!(handled.lock().is_empty());
}

#[test]
fn test_trust_check_accepts_our_own_server() {
    let _env = env();
    let verdicts = Mutex::new(Vec::new());
    let spy = |stream: &Stream, name: &str| {
        let verdict = server_is_trusted(stream, name);
        verdicts.lock().push(verdict);
        verdict
    };
    let hooks = ClientHooks {
        before_attempt: &|| {},
        server_trusted: &spy,
    };
    let mut server = ok_server();
    assert!(server.listen());
    assert_eq!(
        send_command_with("status", DEFAULT_TIMEOUT_MS, 0, &hooks)
            .unwrap()
            .as_deref(),
        Some("ok")
    );
    assert_eq!(*verdicts.lock(), [true]);
}

#[test]
fn test_trust_check_fails_open_when_the_owner_is_unknown() {
    assert!(trusted(Err(io::Error::other("no descriptor")), "the owner"));
    assert!(trusted(Ok(None), "the owner"));
    assert!(trusted(Ok(Some(true)), "the owner"));
    assert!(!trusted(Ok(Some(false)), "the owner"));
}

#[cfg(unix)]
#[test]
fn test_socket_file_owner_check() {
    use std::os::unix::fs::MetadataExt;

    let tmp = tempfile::tempdir().unwrap();
    let socket_file = tmp.path().join("sock");
    fs::write(&socket_file, "").unwrap();
    let path = socket_file.to_str().unwrap();
    let owner = fs::metadata(&socket_file).unwrap().uid();
    assert_eq!(socket_file_owned_by(path, Some(owner)), Some(true));
    assert_eq!(socket_file_owned_by(path, Some(owner + 1)), Some(false));
    assert_eq!(
        socket_file_owned_by(tmp.path().join("missing").to_str().unwrap(), Some(owner)),
        None
    );
    assert_eq!(socket_file_owned_by("", Some(owner)), None);
    assert_eq!(socket_file_owned_by(path, None), None);
    // and the real user id is the owner of what this process creates
    assert_eq!(socket_file_is_current_user(path), Some(true));
}

#[cfg(windows)]
#[test]
fn test_windows_process_owner_check() {
    assert_eq!(
        winsec::process_is_current_user(std::process::id()),
        Some(true)
    );
    assert_eq!(winsec::process_is_current_user(4), Some(false)); // the System process
}

#[cfg(not(windows))]
#[test]
fn test_windows_checks_are_inert_elsewhere() {
    let _env = env();
    assert_eq!(winsec::process_is_current_user(1), None);
    assert!(!winsec::pipe_name_taken("anything", PIPE_NAME_GRACE_S));
    let mut server = ok_server();
    assert!(server.listen());
    let stream = connect(server.name(), Instant::now() + Duration::from_secs(1)).unwrap();
    assert!(client_is_trusted(&stream));
}

// ------------------------------------------------------------- client identity
#[test]
fn test_client_of_another_account_is_refused() {
    // Another account reaching our pipe (a squatted name) cannot pause or quit us.
    let _env = env();
    let handled = Arc::new(Mutex::new(Vec::new()));
    let allowed = Arc::new(AtomicBool::new(false));
    let gate = Arc::clone(&allowed);
    let mut server = server_with(recorder(&handled, "ok"))
        .with_client_trust(move |_| gate.load(Ordering::SeqCst));
    assert!(server.listen());
    assert_eq!(send_command("privacy-on", 300, 0).unwrap(), None);
    assert_eq!(send_command("quit", 300, 0).unwrap(), None);
    thread::sleep(Duration::from_millis(50));
    assert!(handled.lock().is_empty());
    // Loud once, quiet afterwards: a client retrying in a loop must not flood the log.
    assert_eq!(refusal_level(0), log::Level::Warn);
    assert_eq!(refusal_level(1), log::Level::Debug);
    assert_eq!(refusal_level(2), log::Level::Debug);
    allowed.store(true, Ordering::SeqCst);
    assert_eq!(send("status").as_deref(), Some("ok")); // our own clients still get through
}

#[test]
fn test_client_check_accepts_our_own_client() {
    let _env = env();
    let verdicts = Arc::new(Mutex::new(Vec::new()));
    let spy = Arc::clone(&verdicts);
    let mut server = ok_server().with_client_trust(move |stream| {
        let verdict = client_is_trusted(stream);
        spy.lock().push(verdict);
        verdict
    });
    assert!(server.listen());
    assert_eq!(send("status").as_deref(), Some("ok"));
    assert_eq!(*verdicts.lock(), [true]);
}

#[test]
fn test_client_check_fails_open_when_the_client_is_unknown() {
    assert!(trusted(
        Err(io::Error::other("no descriptor")),
        "the command client"
    ));
}

// ------------------------------------------------------- squatted pipe names
/// A named pipe created with this process's default security (Windows).
#[cfg(windows)]
struct SquatterPipe(Option<isize>); // the raw handle value, so the pipe can move to a thread

#[cfg(windows)]
impl SquatterPipe {
    fn new(name: &str) -> Self {
        use windows::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows::Win32::Storage::FileSystem::PIPE_ACCESS_DUPLEX;
        use windows::Win32::System::Pipes::{CreateNamedPipeW, PIPE_TYPE_BYTE, PIPE_WAIT};
        use windows::core::PCWSTR;

        let wide: Vec<u16> = format!("{PIPE_PREFIX}{name}")
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: `wide` is NUL-terminated; the default security is requested with `None`.
        let handle = unsafe {
            CreateNamedPipeW(
                PCWSTR(wide.as_ptr()),
                PIPE_ACCESS_DUPLEX,
                PIPE_TYPE_BYTE | PIPE_WAIT,
                255, // PIPE_UNLIMITED_INSTANCES
                4096,
                4096,
                0,
                None,
            )
        };
        assert_ne!(handle, INVALID_HANDLE_VALUE, "CreateNamedPipeW failed");
        SquatterPipe(Some(handle.0 as isize))
    }

    fn close(&mut self) {
        if let Some(handle) = self.0.take() {
            // SAFETY: the handle was created by `new` and is closed once.
            let handle = windows::Win32::Foundation::HANDLE(handle as *mut std::ffi::c_void);
            let _ = unsafe { windows::Win32::Foundation::CloseHandle(handle) };
        }
    }
}

#[cfg(windows)]
impl Drop for SquatterPipe {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(windows)]
#[test]
fn test_pipe_name_created_by_someone_else_is_not_joined() {
    // Joining would give our pipe instances the squatter's access rules.
    let _env = env();
    let logs = capture_logs();
    let mut squatter = SquatterPipe::new(&server_name());
    let mut server = ok_server().with_pipe_name_grace(0.2);
    assert!(!server.listen());
    assert!(!server.another_instance_running()); // we still are the instance ...
    assert!(server.lock().unwrap().is_held()); // ... and keep the lock, just without a command channel
    assert!(!server.is_listening());
    let warnings = logs.messages_at_least(log::Level::Warn).join("\n");
    assert!(warnings.contains("already exists"), "{warnings}");
    assert!(
        warnings.contains("and desktop shortcuts cannot reach"),
        "{warnings}"
    );
    squatter.close();
    assert!(server.listen()); // the name is ours once the squatter is gone
    assert_eq!(send("status").as_deref(), Some("ok"));
}

#[cfg(windows)]
#[test]
fn test_pipe_name_check_waits_for_a_lingering_pipe() {
    let name = format!(
        "eye-tracker-test-{}-{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    );
    assert!(!winsec::pipe_name_taken(&name, PIPE_NAME_GRACE_S));
    let mut squatter = SquatterPipe::new(&name);
    let closer = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        squatter.close();
    });
    let started = Instant::now();
    // Gone within the grace period: an exiting instance's pipe, not a squatter.
    assert!(!winsec::pipe_name_taken(&name, PIPE_NAME_GRACE_S));
    assert!(started.elapsed() < Duration::from_secs_f64(PIPE_NAME_GRACE_S));
    closer.join().unwrap();
    let mut squatter = SquatterPipe::new(&name);
    assert!(winsec::pipe_name_taken(&name, 0.0));
    assert!(winsec::pipe_name_taken(
        &format!("{PIPE_PREFIX}{name}"),
        0.0
    )); // full names work too
    squatter.close();
}
