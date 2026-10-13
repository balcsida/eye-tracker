//! Tests for `autostart` on all three OS code paths (ports of tests/test_autostart.py).
//!
//! The OS is selected through the `Host` the functions run on; the registry is an in-memory
//! dictionary and the LaunchAgent / .desktop locations point into a temporary directory, so the
//! real `HKCU\...\Run` key and the real `launchctl` are never touched.
//!
//! Not ported:
//!
//! * `TestLaunchCommand::test_source_on_windows_prefers_pythonw`,
//!   `test_source_on_windows_without_pythonw`, `test_source_elsewhere_uses_sys_executable`,
//!   `test_appimage_env_ignored_for_source_runs`: there are no source runs; the Rust binary is
//!   always the packaged one.
//! * `test_frozen_cli_registers_the_windowed_executable`,
//!   `test_frozen_cli_without_gui_sibling_keeps_itself`,
//!   `test_installer_command_on_the_path_registers_the_windowed_executable`,
//!   `test_linux_bundle_has_one_executable_for_both_roles`, `test_appimage`,
//!   `test_missing_appimage_file_is_ignored`: choosing the executable is `cli::app_command`'s
//!   job and is tested in cli_tests.rs.
//! * `TestProfiles::test_the_cli_parses_the_login_command`: needs the clap parser of cli.rs,
//!   which is not ported yet.
//! * `TestRefresh::test_source_run_does_not_take_over_a_working_entry`,
//!   `test_source_run_repairs_an_entry_of_a_deleted_environment`: no source runs (see above).
//! * The relative-path half of `test_launch_command_carries_the_profile`: it changes the
//!   process's working directory, which parallel tests share.
//! * `test_default_path_honours_xdg_config_home`, `test_location_and_default_path`: the
//!   environment is not modified; the same checks run on the pure `linux_desktop_path_in` and
//!   `mac_plist_path_in`.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use super::*;
use crate::testing::capture_logs;

// Generous: the test machine may be loaded.
const SLOW: Duration = Duration::from_secs(60);
const LAUNCHD_DOMAIN: &str = "gui/501";

/// launchd's per-user database of switched-off services, seen through `launchctl`.
#[derive(Default)]
struct FakeLaunchd {
    /// label -> state word as `print-disabled` shows it ("disabled", "enabled", "true" ...)
    states: RefCell<BTreeMap<String, String>>,
    calls: RefCell<Vec<Vec<String>>>,
    /// launchctl fails (missing, times out ...)
    broken: Cell<bool>,
    /// "enable" does not help (a switch-off it cannot override)
    sticky: Cell<bool>,
}

impl FakeLaunchd {
    fn call(&self, args: &[&str]) -> Option<String> {
        self.calls
            .borrow_mut()
            .push(args.iter().map(|a| (*a).to_string()).collect());
        if self.broken.get() {
            return None;
        }
        if args == ["print-disabled", LAUNCHD_DOMAIN] {
            let services: Vec<String> = self
                .states
                .borrow()
                .iter()
                .map(|(label, state)| format!("\t\"{label}\" => {state}"))
                .collect();
            // Recent macOS adds a section mapping labels to app identifiers.
            let association = format!("\t\"{APP_ID}\" => {APP_ID}");
            let mut lines = vec!["disabled services = {".to_string()];
            lines.extend(services);
            lines.extend([
                "}".into(),
                String::new(),
                "login item associations = {".into(),
            ]);
            lines.extend([association, "}".into(), String::new()]);
            return Some(lines.join("\n"));
        }
        if args.len() == 2 && args[0] == "enable" {
            let (domain, label) = args[1].rsplit_once('/').expect("domain/label");
            assert_eq!(domain, LAUNCHD_DOMAIN);
            if !self.sticky.get() {
                self.states
                    .borrow_mut()
                    .insert(label.to_string(), "enabled".to_string());
            }
            return Some(String::new());
        }
        panic!("unexpected launchctl call: {args:?}");
    }

    fn set(&self, label: &str, state: &str) {
        self.states
            .borrow_mut()
            .insert(label.to_string(), state.to_string());
    }

    fn enables(&self) -> Vec<Vec<String>> {
        self.calls
            .borrow()
            .iter()
            .filter(|call| call[0] == "enable")
            .cloned()
            .collect()
    }
}

type RegistryMap = HashMap<(String, String), RegValue>;

/// In-memory HKCU: `{(key_path, value_name): data}`.
#[derive(Clone, Default)]
struct FakeRegistry(Rc<RefCell<RegistryMap>>);

impl Registry for FakeRegistry {
    fn read(&self, key_path: &str, name: &str) -> io::Result<Option<RegValue>> {
        Ok(self
            .0
            .borrow()
            .get(&(key_path.to_string(), name.to_string()))
            .cloned())
    }

    fn write_str(&self, key_path: &str, name: &str, value: &str) -> io::Result<()> {
        self.0.borrow_mut().insert(
            (key_path.to_string(), name.to_string()),
            RegValue::Str(value.to_string()),
        );
        Ok(())
    }

    fn delete(&self, key_path: &str, name: &str) -> io::Result<()> {
        self.0
            .borrow_mut()
            .remove(&(key_path.to_string(), name.to_string()));
        Ok(())
    }
}

impl FakeRegistry {
    fn get(&self, key_path: &str, name: &str) -> Option<RegValue> {
        self.0
            .borrow()
            .get(&(key_path.to_string(), name.to_string()))
            .cloned()
    }

    fn put(&self, key_path: &str, name: &str, value: RegValue) {
        self.0
            .borrow_mut()
            .insert((key_path.to_string(), name.to_string()), value);
    }

    fn put_str(&self, key_path: &str, name: &str, value: &str) {
        self.put(key_path, name, RegValue::Str(value.to_string()));
    }

    fn run_value(&self) -> Vec<String> {
        match self.get(RUN_KEY, RUN_VALUE_NAME) {
            Some(RegValue::Str(value)) => win_split(&value),
            other => panic!("no Run value: {other:?}"),
        }
    }
}

/// A registry whose every access is refused.
struct DeniedRegistry;

impl Registry for DeniedRegistry {
    fn read(&self, _: &str, _: &str) -> io::Result<Option<RegValue>> {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "access denied",
        ))
    }

    fn write_str(&self, _: &str, _: &str, _: &str) -> io::Result<()> {
        self.read("", "").map(|_| ())
    }

    fn delete(&self, _: &str, _: &str) -> io::Result<()> {
        self.read("", "").map(|_| ())
    }
}

/// Everything redirected into a temporary directory (the `isolate` fixture), with a frozen
/// install ("C:/Program Files/Eye Tracker"-like) as the running copy.
struct Fixture {
    dir: tempfile::TempDir,
    host: Host,
    registry: FakeRegistry,
    launchd: Rc<FakeLaunchd>,
    /// What `cli::app_command` gives.
    command: Rc<RefCell<Vec<String>>>,
    /// The windowed executable of the fake install.
    app: PathBuf,
}

impl Fixture {
    fn new(system: System) -> Fixture {
        let dir = tempfile::tempdir().expect("temporary directory");
        let app_dir = dir.path().join("Program Files").join("Eye Tracker");
        fs::create_dir_all(&app_dir).unwrap();
        let app = app_dir.join("EyeTracker.exe");
        fs::write(&app, "").unwrap();
        fs::write(app_dir.join("eye-tracker-cli.exe"), "").unwrap();
        let registry = FakeRegistry::default();
        let launchd = Rc::new(FakeLaunchd::default());
        let command = Rc::new(RefCell::new(vec![path_text(&app)]));
        let host = Host {
            system,
            registry: Box::new(registry.clone()),
            mac_plist_path: dir
                .path()
                .join("LaunchAgents")
                .join(format!("{APP_ID}.plist")),
            linux_desktop_path: dir
                .path()
                .join("autostart")
                .join(format!("{APP_SLUG}.desktop")),
            launchctl: {
                let launchd = launchd.clone();
                Box::new(move |args| launchd.call(args))
            },
            launchd_domain: Some(LAUNCHD_DOMAIN.to_string()),
            app_command: {
                let command = command.clone();
                Box::new(move || command.borrow().clone())
            },
            active_profile: None,
            on_path: Box::new(|_| false),
            is_read_only_volume: Box::new(|_| false),
            write_text: Box::new(write_text),
        };
        Fixture {
            dir,
            host,
            registry,
            launchd,
            command,
            app,
        }
    }

    fn set_command(&self, command: &[&str]) {
        *self.command.borrow_mut() = command.iter().map(|a| (*a).to_string()).collect();
    }

    fn app_text(&self) -> String {
        path_text(&self.app)
    }

    /// A `--config-dir` profile directory.
    fn portable(&self) -> PathBuf {
        let directory = self.dir.path().join("Portable Data");
        fs::create_dir_all(&directory).unwrap();
        directory.canonicalize().unwrap()
    }

    fn write_agent(&self, arguments: &[&str]) {
        let items: String = arguments
            .iter()
            .map(|a| format!("<string>{}</string>", xml_escape(a)))
            .collect();
        self.write_agent_text(&format!(
            "<plist><dict><key>Label</key><string>{APP_ID}</string>\
<key>ProgramArguments</key><array>{items}</array></dict></plist>"
        ));
    }

    fn write_agent_text(&self, text: &str) {
        let path = &self.host.mac_plist_path;
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn write_desktop_entry(&self, command: &[&str], hidden: bool) {
        let command: Vec<String> = command.iter().map(|a| (*a).to_string()).collect();
        let path = &self.host.linux_desktop_path;
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut text = desktop_entry(&command);
        if hidden {
            // What GNOME Tweaks / KDE do to switch an entry off.
            text = text.replace("Hidden=false", "Hidden=true");
        }
        fs::write(path, text).unwrap();
    }

    fn load_agent(&self) -> BTreeMap<String, Plist> {
        let text = fs::read_to_string(&self.host.mac_plist_path).unwrap();
        match parse_plist(&text).unwrap() {
            Plist::Dict(agent) => agent,
            other => panic!("not a dict: {other:?}"),
        }
    }
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|a| (*a).to_string()).collect()
}

fn plist_strings(items: &[&str]) -> Plist {
    Plist::Array(
        items
            .iter()
            .map(|a| Plist::String((*a).to_string()))
            .collect(),
    )
}

/// Reference parser for an `Exec` value (Desktop Entry spec 1.5).
fn desktop_exec_split_reference(value: &str) -> Vec<String> {
    // 1. key-file string unescaping
    let mut unescaped = String::new();
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            unescaped.push(match chars.next().unwrap() {
                's' => ' ',
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\' => '\\',
                other => panic!("bad escape \\{other}"),
            });
        } else {
            unescaped.push(c);
        }
    }
    // 2. argument splitting with double-quote rules, 3. %% -> %
    let text: Vec<char> = unescaped.chars().collect();
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut started = false;
    let mut i = 0;
    while i < text.len() {
        let c = text[i];
        if in_quotes {
            if c == '\\' && i + 1 < text.len() && "\"`$\\".contains(text[i + 1]) {
                current.push(text[i + 1]);
                i += 1;
            } else if c == '"' {
                in_quotes = false;
            } else {
                current.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
            started = true;
        } else if c == ' ' {
            if started || !current.is_empty() {
                args.push(std::mem::take(&mut current));
                started = false;
            }
        } else {
            current.push(c);
        }
        i += 1;
    }
    if started || !current.is_empty() {
        args.push(current);
    }
    args.iter().map(|a| a.replace("%%", "%")).collect()
}

// ------------------------------------------------------------ launch command
mod launch_command {
    use super::*;

    #[test]
    fn test_frozen() {
        let fx = Fixture::new(System::Windows);
        let app = fx.app_text();
        assert_eq!(
            fx.host.launch_command(true, None),
            [app.as_str(), "--background"]
        );
        assert_eq!(fx.host.launch_command(false, None), [app.as_str()]);
    }

    #[test]
    fn test_the_running_profile_of_app_command_is_replaced() {
        let fx = Fixture::new(System::Linux);
        let portable = fx.portable();
        let other = path_text(&fx.dir.path().join("elsewhere"));
        fx.set_command(&[&fx.app_text(), "--config-dir", &other]);
        assert_eq!(
            fx.host.launch_command(false, None),
            [fx.app_text().as_str()],
            "the default profile has no --config-dir"
        );
        assert_eq!(
            fx.host.launch_command(false, Some(&portable)),
            [fx.app_text(), "--config-dir".into(), path_text(&portable)]
        );
    }
}

// -------------------------------------------------------------------- Windows
mod windows_registry {
    use super::*;

    fn fx() -> Fixture {
        Fixture::new(System::Windows)
    }

    #[test]
    fn test_enable_writes_quoted_run_value() {
        let fx = fx();
        assert!(fx.host.status(None) != Status::Enabled);
        fx.host.enable(true, None).unwrap();
        let Some(RegValue::Str(value)) = fx.registry.get(RUN_KEY, "EyeTracker") else {
            panic!("no Run value");
        };
        assert_eq!(value, list2cmdline(&[fx.app_text(), "--background".into()]));
        assert!(value.starts_with(&format!("\"{}\"", fx.app_text()))); // path contains spaces
        assert_eq!(fx.host.status(None), Status::Enabled);
    }

    #[test]
    fn test_enable_without_background() {
        let fx = fx();
        fx.host.enable(false, None).unwrap();
        assert_eq!(
            fx.registry.get(RUN_KEY, "EyeTracker"),
            Some(RegValue::Str(format!("\"{}\"", fx.app_text())))
        );
    }

    #[test]
    fn test_task_manager_disabled_flag() {
        for (flag, enabled) in [
            (vec![3u8, 0, 0, 0], false),
            (vec![2, 0, 0, 0], true),
            (vec![6, 0], true),
        ] {
            let fx = fx();
            fx.registry.put_str(
                RUN_KEY,
                "EyeTracker",
                &format!("\"{}\" --background", fx.app_text()),
            );
            fx.registry
                .put(STARTUP_APPROVED_KEY, "EyeTracker", RegValue::Bytes(flag));
            let expected = if enabled {
                Status::Enabled
            } else {
                Status::Disabled
            };
            assert_eq!(fx.host.status(None), expected);
        }
    }

    #[test]
    fn test_enable_clears_task_manager_disabled_flag() {
        let fx = fx();
        fx.registry.put_str(RUN_KEY, "EyeTracker", "old.exe");
        let mut flag = vec![3u8];
        flag.extend([0; 11]);
        fx.registry
            .put(STARTUP_APPROVED_KEY, "EyeTracker", RegValue::Bytes(flag));
        fx.host.enable(true, None).unwrap();
        assert_eq!(fx.registry.get(STARTUP_APPROVED_KEY, "EyeTracker"), None);
        assert_eq!(fx.host.status(None), Status::Enabled);
    }

    #[test]
    fn test_blank_value_is_not_enabled() {
        let fx = fx();
        fx.registry.put_str(RUN_KEY, "EyeTracker", "   ");
        assert_eq!(fx.host.status(None), Status::Disabled);
    }

    #[test]
    fn test_disable_removes_everything() {
        let fx = fx();
        fx.host.enable(true, None).unwrap();
        fx.registry
            .put(STARTUP_APPROVED_KEY, "EyeTracker", RegValue::Bytes(vec![3]));
        fx.registry.put_str(RUN_KEY, "OtherApp", "other.exe");
        fx.host.disable(None).unwrap();
        let expected: RegistryMap = HashMap::from([(
            (RUN_KEY.to_string(), "OtherApp".to_string()),
            RegValue::Str("other.exe".into()),
        )]);
        assert_eq!(*fx.registry.0.borrow(), expected);
        assert_eq!(fx.host.status(None), Status::Disabled);
        fx.host.disable(None).unwrap(); // idempotent
    }

    #[test]
    fn test_registry_errors() {
        let mut fx = fx();
        fx.host.registry = Box::new(DeniedRegistry);
        assert_eq!(fx.host.status(None), Status::Disabled);
        let err = fx.host.enable(true, None).unwrap_err();
        assert!(err.0.contains("access denied"), "{err}");
        // The entry cannot be read, so disable() goes on to remove it, and fails.
        let err = fx.host.disable(None).unwrap_err();
        assert!(err.0.contains("disable"), "{err}");
    }

    #[test]
    fn test_location() {
        assert_eq!(
            fx().host.location().as_deref(),
            Some(r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run\EyeTracker")
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn test_real_registry_helpers_refuse_off_windows() {
        assert!(RealRegistry.read(RUN_KEY, "EyeTracker").is_err());
        assert!(RealRegistry.write_str(RUN_KEY, "EyeTracker", "x").is_err());
        assert!(RealRegistry.delete(RUN_KEY, "EyeTracker").is_err());
    }
}

// ---------------------------------------------------------------------- macOS
mod mac_os {
    use super::*;

    fn fx() -> Fixture {
        Fixture::new(System::MacOs)
    }

    #[test]
    fn test_enable_writes_launch_agent() {
        let fx = fx();
        fx.host.enable(true, None).unwrap();
        let expected = BTreeMap::from([
            ("Label".to_string(), Plist::String(APP_ID.to_string())),
            (
                "ProgramArguments".to_string(),
                plist_strings(&[&fx.app_text(), "--background"]),
            ),
            ("RunAtLoad".to_string(), Plist::Bool(true)),
            (
                "ProcessType".to_string(),
                Plist::String("Interactive".to_string()),
            ),
            (
                "LimitLoadToSessionType".to_string(),
                Plist::String("Aqua".to_string()),
            ),
        ]);
        assert_eq!(fx.load_agent(), expected);
        assert_eq!(fx.host.status(None), Status::Enabled);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&fx.host.mac_plist_path)
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o644);
        }
    }

    /// The bytes `plistlib.dumps(agent, fmt=FMT_XML, sort_keys=True)` gave.
    #[test]
    fn test_plist_text_is_what_plistlib_wrote() {
        let text = mac_plist(&strings(&["/A & B/x", "--background"]));
        assert_eq!(
            text,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \
\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
<plist version=\"1.0\">\n<dict>\n\
\t<key>Label</key>\n\t<string>io.github.bugraskl.eyetracker</string>\n\
\t<key>LimitLoadToSessionType</key>\n\t<string>Aqua</string>\n\
\t<key>ProcessType</key>\n\t<string>Interactive</string>\n\
\t<key>ProgramArguments</key>\n\t<array>\n\
\t\t<string>/A &amp; B/x</string>\n\t\t<string>--background</string>\n\t</array>\n\
\t<key>RunAtLoad</key>\n\t<true/>\n</dict>\n</plist>\n"
        );
        let Plist::Dict(agent) = parse_plist(&text).unwrap() else {
            panic!("not a dict");
        };
        assert_eq!(
            agent["ProgramArguments"],
            plist_strings(&["/A & B/x", "--background"])
        );
    }

    #[test]
    fn test_enable_overwrites() {
        let fx = fx();
        fx.host.enable(true, None).unwrap();
        fx.host.enable(false, None).unwrap();
        assert_eq!(
            fx.load_agent()["ProgramArguments"],
            plist_strings(&[&fx.app_text()])
        );
    }

    #[test]
    fn test_disabled_or_corrupt_agents_are_not_enabled() {
        let fx = fx();
        fx.write_agent_text(&format!(
            "<plist><dict><key>Label</key><string>{APP_ID}</string>\
<key>ProgramArguments</key><array><string>x</string></array>\
<key>Disabled</key><true/></dict></plist>"
        ));
        assert_eq!(fx.host.status(None), Status::Disabled);
        fx.write_agent_text("<plist><dict><key>Label</oops>");
        assert_eq!(fx.host.status(None), Status::Disabled);
        fx.write_agent(&[]);
        assert_eq!(fx.host.status(None), Status::Disabled);
    }

    #[test]
    fn test_disable() {
        let fx = fx();
        fx.host.enable(true, None).unwrap();
        fx.host.disable(None).unwrap();
        assert!(!fx.host.mac_plist_path.exists());
        assert_eq!(fx.host.status(None), Status::Disabled);
        fx.host.disable(None).unwrap();
    }

    #[test]
    fn test_location_and_default_path() {
        let fx = fx();
        assert_eq!(
            fx.host.location(),
            Some(fx.host.mac_plist_path.display().to_string())
        );
        let home = Path::new("/Users/me");
        assert_eq!(
            mac_plist_path_in(home),
            home.join("Library")
                .join("LaunchAgents")
                .join(format!("{APP_ID}.plist"))
        );
    }

    #[test]
    fn test_switch_off_recorded_by_launchd_is_reported() {
        // `launchctl disable gui/$UID/<label>` leaves the plist as it is.
        for state in ["disabled", "true", "Disabled"] {
            let fx = fx();
            fx.host.enable(true, None).unwrap();
            assert!(fx.launchd.enables().is_empty()); // nothing to clear: launchd left alone
            fx.launchd.set(APP_ID, state);
            assert_eq!(fx.host.status(None), Status::Disabled);
            assert_eq!(
                fx.host.registered_command(),
                Some(vec![fx.app_text(), "--background".into()])
            );
            // A switch-off is not undone behind the user's back.
            assert!(!fx.host.refresh(None));
        }
    }

    #[test]
    fn test_other_services_and_enabled_records_do_not_count() {
        let fx = fx();
        fx.host.enable(true, None).unwrap();
        fx.launchd.set("com.example.other", "disabled");
        fx.launchd.set(APP_ID, "enabled");
        assert_eq!(fx.host.status(None), Status::Enabled);
    }

    #[test]
    fn test_enable_clears_the_launchd_switch_off() {
        let fx = fx();
        fx.host.enable(true, None).unwrap();
        fx.launchd.set(APP_ID, "disabled");
        fx.host.enable(true, None).unwrap(); // the checkbox turned on again
        assert_eq!(
            fx.launchd.enables(),
            [vec![
                "enable".to_string(),
                format!("{LAUNCHD_DOMAIN}/{APP_ID}")
            ]]
        );
        assert_eq!(fx.host.status(None), Status::Enabled);
    }

    #[test]
    fn test_enable_reports_a_switch_off_it_cannot_clear() {
        let fx = fx();
        fx.launchd.set(APP_ID, "disabled");
        fx.launchd.sticky.set(true);
        let err = fx.host.enable(true, None).unwrap_err();
        assert!(err.0.contains("Login Items"), "{err}");
        assert!(fx.host.mac_plist_path.exists()); // written; launchd has the last word
        assert_eq!(fx.host.status(None), Status::Disabled);
    }

    #[test]
    fn test_unreadable_launchd_state_trusts_the_plist() {
        let fx = fx();
        fx.host.enable(true, None).unwrap();
        fx.launchd.broken.set(true);
        assert_eq!(fx.host.status(None), Status::Enabled);
    }

    #[test]
    fn test_launchd_is_only_asked_about_an_existing_agent() {
        let fx = fx();
        assert_eq!(fx.host.status(None), Status::Disabled);
        assert!(fx.launchd.calls.borrow().is_empty());
    }

    #[test]
    fn test_print_disabled_parsing() {
        let cases = [
            (
                format!("disabled services = {{\n\t\"{APP_ID}\" => disabled\n}}\n"),
                true,
            ),
            // macOS 10.x
            (
                format!("disabled services = {{\n\t\"{APP_ID}\" => true\n}}\n"),
                true,
            ),
            (
                format!("disabled services = {{\n\t\"{APP_ID}\" => enabled\n}}\n"),
                false,
            ),
            (
                format!("disabled services = {{\n\t\"{APP_ID}\" => false\n}}\n"),
                false,
            ),
            (
                format!("disabled services = {{\n\t\"{APP_ID}.helper\" => disabled\n}}\n"),
                false,
            ),
            (
                format!("login item associations = {{\n\t\"{APP_ID}\" => {APP_ID}\n}}\n"),
                false,
            ),
            (String::new(), false),
            ("garbage => disabled".to_string(), false),
        ];
        for (listing, expected) in cases {
            assert_eq!(
                launchd_lists_disabled(&listing, APP_ID),
                expected,
                "{listing}"
            );
        }
    }

    /// Its error handling, with stand-in programs (the real launchctl is never run).
    #[test]
    fn test_real_launchctl_runner() {
        let missing = Path::new("/nonexistent-dir/missing");
        assert_eq!(
            run_program(
                &path_text(missing),
                &["print-disabled", LAUNCHD_DOMAIN],
                SLOW
            ),
            None
        );
        #[cfg(unix)]
        {
            assert_eq!(
                run_program("/bin/sh", &["-c", "echo listing"], SLOW).as_deref(),
                Some("listing\n")
            );
            assert_eq!(run_program("/bin/sh", &["-c", "exit 3"], SLOW), None);
        }
    }

    #[test]
    fn test_launchd_domain_is_the_users_gui_session() {
        #[cfg(unix)]
        {
            // SAFETY: `getuid` has no preconditions.
            let uid = unsafe { libc::getuid() };
            assert_eq!(launchd_domain(), Some(format!("gui/{uid}")));
        }
        #[cfg(not(unix))]
        assert_eq!(launchd_domain(), None);
    }
}

// ---------------------------------------------------------------------- Linux
mod linux {
    use super::*;

    fn fx() -> Fixture {
        Fixture::new(System::Linux)
    }

    #[test]
    fn test_enable_writes_desktop_entry() {
        let fx = fx();
        let image = fx.dir.path().join("My Apps").join("Eye Tracker.AppImage");
        fs::create_dir_all(image.parent().unwrap()).unwrap();
        fs::write(&image, "").unwrap();
        fx.set_command(&[&path_text(&image)]);

        fx.host.enable(true, None).unwrap();
        let path = &fx.host.linux_desktop_path;
        let text = fs::read_to_string(path).unwrap();
        let entry = parse_desktop_entry(&text);
        assert!(text.starts_with("[Desktop Entry]\n"));
        assert_eq!(entry["Type"], "Application");
        assert_eq!(entry["Name"], APP_NAME);
        assert_eq!(entry["Icon"], APP_SLUG);
        assert_eq!(entry["Terminal"], "false");
        assert_eq!(entry["Hidden"], "false");
        assert_eq!(entry["NoDisplay"], "false");
        assert_eq!(entry["X-GNOME-Autostart-enabled"], "true");
        assert_eq!(
            desktop_exec_split_reference(&entry["Exec"]),
            [path_text(&image), "--background".to_string()]
        );
        assert_eq!(fx.host.status(None), Status::Enabled);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o644
            );
        }
    }

    #[test]
    fn test_entries_switched_off_by_the_desktop() {
        for line in [
            "Hidden=true",
            "X-GNOME-Autostart-enabled=false",
            "Hidden = True",
        ] {
            let fx = fx();
            let path = &fx.host.linux_desktop_path;
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(
                path,
                format!("[Desktop Entry]\nType=Application\nExec=eye-tracker\n{line}\n"),
            )
            .unwrap();
            assert_eq!(fx.host.status(None), Status::Disabled, "{line}");
        }
    }

    #[test]
    fn test_entry_without_exec_is_not_enabled() {
        let fx = fx();
        let path = &fx.host.linux_desktop_path;
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "[Desktop Entry]\nType=Application\n").unwrap();
        assert_eq!(fx.host.status(None), Status::Disabled);
    }

    #[test]
    fn test_disable() {
        let fx = fx();
        fx.host.enable(true, None).unwrap();
        assert_eq!(fx.host.status(None), Status::Enabled);
        fx.host.disable(None).unwrap();
        assert!(!fx.host.linux_desktop_path.exists());
        fx.host.disable(None).unwrap();
    }

    #[test]
    fn test_location() {
        let fx = fx();
        assert_eq!(
            fx.host.location(),
            Some(fx.host.linux_desktop_path.display().to_string())
        );
    }

    #[test]
    fn test_default_path_honours_xdg_config_home() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("cfg");
        let home = Path::new("/home/me");
        assert_eq!(
            linux_desktop_path_in(&path_text(&cfg), home),
            cfg.join("autostart").join("eye-tracker.desktop")
        );
        // Invalid per spec, so ignored.
        let default = home
            .join(".config")
            .join("autostart")
            .join("eye-tracker.desktop");
        assert_eq!(linux_desktop_path_in("relative/dir", home), default);
        assert_eq!(linux_desktop_path_in("", home), default);
    }
}

mod desktop_exec_quoting {
    use super::*;

    #[test]
    fn test_argument_quoting() {
        let cases = [
            ("/usr/bin/python3", "/usr/bin/python3"),
            ("--background", "--background"),
            ("/opt/My App/run", "\"/opt/My App/run\""),
            ("say \"hi\"", r#""say \"hi\"""#),
            ("/opt/a$b", r#""/opt/a\$b""#),
            ("back`tick", r#""back\`tick""#),
            (r"C:\x", r#""C:\\x""#),
            ("100%", "100%%"),
            ("", "\"\""),
        ];
        for (arg, expected) in cases {
            assert_eq!(desktop_exec_arg(arg), expected, "{arg}");
        }
    }

    #[test]
    fn test_string_escaping_is_applied_on_top() {
        // Spec example: a literal "$" inside quotes is written as \\$ in the file.
        assert_eq!(desktop_exec(&strings(&["/opt/a$b"])), "\"/opt/a\\\\$b\"");
    }

    #[test]
    fn test_round_trip() {
        let commands = [
            strings(&["/usr/bin/python3", "-m", "eye_tracker", "--background"]),
            strings(&["/home/me/My Apps/Eye Tracker.AppImage", "--background"]),
            strings(&["/opt/we$ird `name`/app \"q\" 50%", "a\\b", "x;y|z"]),
        ];
        for command in commands {
            assert_eq!(
                desktop_exec_split_reference(&desktop_exec(&command)),
                command
            );
        }
    }

    #[test]
    fn test_parse_desktop_entry_only_reads_main_group() {
        let text = "# comment\n[Desktop Entry]\nName = Eye Tracker\nExec=a\n\
[Desktop Action x]\nExec=b\nHidden=true\n";
        let entry = parse_desktop_entry(text);
        let expected = HashMap::from([
            ("Name".to_string(), "Eye Tracker".to_string()),
            ("Exec".to_string(), "a".to_string()),
        ]);
        assert_eq!(entry, expected);
    }
}

// ---------------------------------------------------------------- unsupported
#[test]
fn test_unsupported_everything_degrades() {
    let fx = Fixture::new(System::Unsupported);
    assert_eq!(fx.host.system, System::Unsupported);
    assert_eq!(fx.host.status(None), Status::Disabled);
    assert_eq!(fx.host.registered_command(), None);
    assert!(!fx.host.refresh(None));
    assert_eq!(fx.host.location(), None);
    let err = fx.host.enable(true, None).unwrap_err();
    assert!(err.0.contains("not supported"), "{err}");
    fx.host.disable(None).unwrap(); // no-op
}

// ------------------------------------------------------------------- profiles
mod profiles {
    //! `--config-dir` survives into the login entry (portable mode).
    use super::*;

    #[test]
    fn test_launch_command_carries_the_profile() {
        let fx = Fixture::new(System::Windows);
        let portable = fx.portable();
        assert_eq!(
            fx.host.launch_command(true, Some(&portable)),
            [
                fx.app_text(),
                "--config-dir".into(),
                path_text(&portable),
                "--background".into()
            ]
        );
        assert_eq!(
            fx.host.launch_command(false, Some(&portable))[1..],
            ["--config-dir".to_string(), path_text(&portable)]
        );
    }

    #[test]
    fn test_default_is_the_running_profile() {
        let mut fx = Fixture::new(System::Windows);
        let portable = fx.portable();
        assert_eq!(
            fx.host.launch_command(true, None),
            [fx.app_text(), "--background".to_string()]
        );
        fx.host.active_profile = Some(portable.clone());
        assert_eq!(
            fx.host.launch_command(true, None),
            fx.host.launch_command(true, Some(&portable))
        );
    }

    #[test]
    fn test_windows_entry_is_per_profile() {
        let fx = Fixture::new(System::Windows);
        let portable = fx.portable();
        fx.host.enable(true, Some(&portable)).unwrap();
        assert_eq!(
            fx.registry.run_value(),
            fx.host.launch_command(true, Some(&portable))
        );
        assert_eq!(fx.host.status(Some(&portable)), Status::Enabled);
        // The installed (default-profile) app does not claim the portable entry...
        assert_eq!(fx.host.status(None), Status::OtherProfile);
        // ...and does not remove it either.
        fx.host.disable(None).unwrap();
        assert_eq!(fx.host.status(Some(&portable)), Status::Enabled);
        // Enabling it there takes the single login entry over (one tracker per camera).
        fx.host.enable(true, None).unwrap();
        assert_eq!(
            fx.registry.run_value(),
            [fx.app_text(), "--background".to_string()]
        );
        assert_eq!(fx.host.status(Some(&portable)), Status::OtherProfile);
        fx.host.disable(None).unwrap();
        assert!(fx.registry.0.borrow().is_empty());
    }

    /// The tray/Settings/wizard calls need no argument in a `--config-dir` process.
    #[test]
    fn test_running_portable_instance_uses_its_profile() {
        let mut fx = Fixture::new(System::Windows);
        let portable = fx.portable();
        fx.host.active_profile = Some(portable.clone());
        fx.host.enable(true, None).unwrap();
        assert_eq!(
            fx.registry.run_value()[1..],
            [
                "--config-dir".to_string(),
                path_text(&portable),
                "--background".into()
            ]
        );
        assert_eq!(fx.host.status(None), Status::Enabled);
        fx.host.disable(None).unwrap();
        assert_eq!(fx.host.status(None), Status::Disabled);
    }

    #[test]
    fn test_windows_profile_comparison_ignores_case() {
        let fx = Fixture::new(System::Windows);
        let portable = fx.portable();
        let command = [
            fx.app_text(),
            "--config-dir".into(),
            path_text(&portable).to_uppercase(),
            "--background".into(),
        ];
        fx.registry
            .put_str(RUN_KEY, "EyeTracker", &list2cmdline(&command));
        assert_eq!(fx.host.status(Some(&portable)), Status::Enabled);
    }

    #[test]
    fn test_equals_form_is_understood() {
        let fx = Fixture::new(System::Windows);
        let portable = fx.portable();
        let command = [
            fx.app_text(),
            format!("--config-dir={}", portable.display()),
        ];
        fx.registry
            .put_str(RUN_KEY, "EyeTracker", &list2cmdline(&command));
        assert_eq!(fx.host.status(Some(&portable)), Status::Enabled);
        assert_eq!(fx.host.status(None), Status::OtherProfile);
    }

    #[test]
    fn test_macos_and_linux_entries_carry_the_profile() {
        let mut fx = Fixture::new(System::MacOs);
        let portable = fx.portable();
        fx.host.enable(true, Some(&portable)).unwrap();
        assert_eq!(
            fx.load_agent()["ProgramArguments"],
            Plist::Array(
                fx.host
                    .launch_command(true, Some(&portable))
                    .into_iter()
                    .map(Plist::String)
                    .collect()
            )
        );
        assert_eq!(fx.host.status(Some(&portable)), Status::Enabled);
        assert_eq!(fx.host.status(None), Status::OtherProfile);

        fx.host.system = System::Linux;
        fx.host.enable(true, Some(&portable)).unwrap();
        assert_eq!(
            fx.host.registered_command(),
            Some(fx.host.launch_command(true, Some(&portable)))
        );
        assert_eq!(fx.host.status(Some(&portable)), Status::Enabled);
        assert_eq!(fx.host.status(None), Status::OtherProfile);
    }
}

// --------------------------------------------------------------- stale entries
const TRANSLOCATED: &str = "/private/var/folders/ab/xyz/T/AppTranslocation/0A1B-2C3D/d/\
Eye Tracker.app/Contents/MacOS/Eye Tracker";
const ON_DMG: &str = "/Volumes/Eye Tracker/Eye Tracker.app/Contents/MacOS/Eye Tracker";

mod stale_entries {
    use super::*;

    #[test]
    fn test_windows_entry_for_a_missing_program() {
        let fx = Fixture::new(System::Windows);
        let gone = fx.dir.path().join("old portable").join("EyeTracker.exe");
        fx.registry.put_str(
            RUN_KEY,
            "EyeTracker",
            &format!("\"{}\" --background", gone.display()),
        );
        assert_eq!(fx.host.status(None), Status::Stale);
        assert_ne!(fx.host.status(None), Status::Enabled);
        assert_eq!(
            fx.host.registered_command(),
            Some(vec![path_text(&gone), "--background".into()])
        );
    }

    #[test]
    fn test_program_found_on_path() {
        let mut fx = Fixture::new(System::Windows);
        fx.registry
            .put_str(RUN_KEY, "EyeTracker", "eye-tracker-gui --background");
        fx.host.on_path = Box::new(|_| true);
        assert_eq!(fx.host.status(None), Status::Enabled);
        fx.host.on_path = Box::new(|_| false);
        assert_eq!(fx.host.status(None), Status::Stale);
    }

    #[test]
    fn test_linux_entry_for_a_replaced_appimage() {
        let fx = Fixture::new(System::Linux);
        let gone = fx.dir.path().join("EyeTracker-0.1.0-linux-x86_64.AppImage");
        fx.write_desktop_entry(&[&path_text(&gone)], false);
        assert_eq!(fx.host.status(None), Status::Stale);
    }

    #[test]
    fn test_macos_entry_in_a_temporary_location() {
        let mut fx = Fixture::new(System::MacOs);
        fx.write_agent(&[TRANSLOCATED, "--background"]);
        assert_eq!(fx.host.status(None), Status::Stale);
        fx.host.is_read_only_volume = Box::new(|_| true);
        fx.write_agent(&[ON_DMG, "--background"]);
        assert_eq!(fx.host.status(None), Status::Stale);
    }

    #[test]
    fn test_macos_refuses_to_register_a_temporary_copy() {
        for executable in [TRANSLOCATED, ON_DMG] {
            let mut fx = Fixture::new(System::MacOs);
            fx.set_command(&[executable]);
            fx.host.is_read_only_volume = Box::new(|_| true);
            let err = fx.host.enable(true, None).unwrap_err();
            assert!(err.0.contains("Applications folder"), "{err}");
            assert!(!fx.host.mac_plist_path.exists());
        }
    }

    #[test]
    fn test_macos_app_on_a_writable_external_volume_is_fine() {
        let mut fx = Fixture::new(System::MacOs);
        fx.set_command(&[ON_DMG]);
        fx.host.is_read_only_volume = Box::new(|_| false);
        fx.host.enable(true, None).unwrap();
        assert_eq!(
            fx.host.registered_command(),
            Some(vec![ON_DMG.to_string(), "--background".into()])
        );
    }

    #[test]
    fn test_transient_locations_only_exist_on_macos() {
        let fx = Fixture::new(System::Linux);
        assert!(!fx.host.is_transient_location(TRANSLOCATED));
    }

    #[test]
    fn test_read_only_volume_check() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_read_only_volume(&path_text(dir.path()))); // writable (or no statvfs)
        assert!(!is_read_only_volume(&path_text(
            &dir.path().join("missing")
        )));
    }
}

// --------------------------------------------------------------------- refresh
/// The app calls `refresh()` at startup to follow a moved or upgraded copy.
mod refresh {
    use super::*;

    /// Running from EyeTracker-0.2.0 as an AppImage build.
    fn appimage() -> (Fixture, PathBuf) {
        let fx = Fixture::new(System::Linux);
        let image = fx
            .dir
            .path()
            .join("Apps")
            .join("EyeTracker-0.2.0-linux-x86_64.AppImage");
        fs::create_dir_all(image.parent().unwrap()).unwrap();
        fs::write(&image, "").unwrap();
        fx.set_command(&[&path_text(&image)]);
        (fx, image)
    }

    #[test]
    fn test_replaced_appimage_is_followed() {
        let (fx, image) = appimage();
        let old = image.with_file_name("EyeTracker-0.1.0-linux-x86_64.AppImage"); // deleted
        fx.write_desktop_entry(&[&path_text(&old), "--background"], false);
        assert!(fx.host.refresh(None));
        assert_eq!(
            fx.host.registered_command(),
            Some(vec![path_text(&image), "--background".into()])
        );
        assert_eq!(fx.host.status(None), Status::Enabled);
    }

    #[test]
    fn test_packaged_build_replaces_an_older_copy_that_still_exists() {
        let (fx, image) = appimage();
        let old = image.with_file_name("EyeTracker-0.1.0-linux-x86_64.AppImage");
        fs::write(&old, "").unwrap();
        fx.write_desktop_entry(&[&path_text(&old), "--background"], false);
        assert!(fx.host.refresh(None));
        assert_eq!(
            fx.host.registered_command(),
            Some(vec![path_text(&image), "--background".into()])
        );
    }

    #[test]
    fn test_background_choice_and_profile_are_kept() {
        let (fx, image) = appimage();
        let portable = fx.portable();
        let old = image.with_file_name("gone.AppImage");
        fx.write_desktop_entry(
            &[&path_text(&old), "--config-dir", &path_text(&portable)],
            false,
        );
        assert!(!fx.host.refresh(None)); // another profile's entry: not ours to move
        assert!(fx.host.refresh(Some(&portable)));
        assert_eq!(
            fx.host.registered_command(),
            Some(vec![
                path_text(&image),
                "--config-dir".into(),
                path_text(&portable)
            ])
        );
    }

    #[test]
    fn test_up_to_date_entry_is_not_rewritten() {
        let (fx, _image) = appimage();
        fx.host.enable(true, None).unwrap();
        let before = fs::metadata(&fx.host.linux_desktop_path)
            .unwrap()
            .modified()
            .unwrap();
        assert!(!fx.host.refresh(None));
        let after = fs::metadata(&fx.host.linux_desktop_path)
            .unwrap()
            .modified()
            .unwrap();
        assert_eq!(after, before);
    }

    #[test]
    fn test_switched_off_entry_is_left_alone() {
        let (fx, image) = appimage();
        let old = path_text(&image.with_file_name("gone.AppImage"));
        fx.write_desktop_entry(&[&old], true);
        assert_eq!(fx.host.status(None), Status::Disabled);
        assert!(!fx.host.refresh(None));
        assert_eq!(fx.host.registered_command(), Some(vec![old]));
    }

    #[test]
    fn test_no_entry_is_not_created() {
        let (fx, _image) = appimage();
        assert!(!fx.host.refresh(None));
        assert!(!fx.host.linux_desktop_path.exists());
    }

    #[test]
    fn test_windows_installer_entry_matches_despite_case() {
        let fx = Fixture::new(System::Windows);
        let installer_value = format!("\"{}\" --background", fx.app_text().to_uppercase());
        fx.registry.put_str(RUN_KEY, "EyeTracker", &installer_value);
        let mut approved = vec![2u8];
        approved.extend([0; 11]);
        fx.registry.put(
            STARTUP_APPROVED_KEY,
            "EyeTracker",
            RegValue::Bytes(approved),
        );
        assert!(!fx.host.refresh(None));
        assert_eq!(
            fx.registry.get(RUN_KEY, "EyeTracker"),
            Some(RegValue::Str(installer_value))
        );
        // The Task Manager state is untouched as well.
        assert!(
            fx.registry
                .get(STARTUP_APPROVED_KEY, "EyeTracker")
                .is_some()
        );
    }

    #[test]
    fn test_task_manager_disabled_entry_is_not_revived() {
        let fx = Fixture::new(System::Windows);
        let gone = fx.dir.path().join("gone.exe");
        fx.registry
            .put_str(RUN_KEY, "EyeTracker", &format!("\"{}\"", gone.display()));
        let mut approved = vec![3u8];
        approved.extend([0; 11]);
        fx.registry.put(
            STARTUP_APPROVED_KEY,
            "EyeTracker",
            RegValue::Bytes(approved.clone()),
        );
        assert!(!fx.host.refresh(None));
        assert_eq!(
            fx.registry.get(STARTUP_APPROVED_KEY, "EyeTracker"),
            Some(RegValue::Bytes(approved))
        );
    }

    #[test]
    fn test_macos_entry_made_from_the_disk_image_is_moved_to_applications() {
        let fx = Fixture::new(System::MacOs);
        fx.write_agent(&[TRANSLOCATED, "--background"]);
        assert!(fx.host.refresh(None));
        assert_eq!(
            fx.host.registered_command(),
            Some(vec![fx.app_text(), "--background".into()])
        );
    }

    #[test]
    fn test_never_points_the_entry_at_a_temporary_copy() {
        let fx = Fixture::new(System::MacOs);
        fx.set_command(&[TRANSLOCATED]);
        let gone = path_text(
            &fx.dir
                .path()
                .join("Eye Tracker.app")
                .join("Contents")
                .join("MacOS")
                .join("Eye Tracker"),
        );
        fx.write_agent(&[&gone, "--background"]);
        assert!(!fx.host.refresh(None));
        assert_eq!(
            fx.host.registered_command(),
            Some(vec![gone, "--background".into()])
        );
    }

    #[test]
    fn test_failures_are_logged_not_raised() {
        let (mut fx, image) = appimage();
        fx.write_desktop_entry(&[&path_text(&image.with_file_name("gone.AppImage"))], false);
        fx.host.write_text = Box::new(|_, _| {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "read-only home",
            ))
        });
        let logs = capture_logs();
        assert!(!fx.host.refresh(None));
        let warnings = logs.messages_at_least(log::Level::Warn);
        assert!(
            warnings.iter().any(|m| m.contains("Could not update")),
            "{warnings:?}"
        );
    }
}

// ----------------------------------------------------------- command parsing
fn windows_commands() -> Vec<Vec<String>> {
    vec![
        strings(&[
            r"C:\Program Files\Eye Tracker\EyeTracker.exe",
            "--background",
        ]),
        strings(&[
            r"C:\Python\pythonw.exe",
            "-m",
            "eye_tracker",
            "--config-dir",
            r"D:\My Data\ET",
        ]),
        strings(&[r"C:\x\app.exe", "--config-dir", "D:\\trailing slash\\"]),
        strings(&[
            r"C:\x\app.exe",
            "say \"hi\"",
            "back\\\\slashes",
            "",
            "tab\there",
            "a\\\"b",
        ]),
        strings(&[r"C:\no-spaces\app.exe"]),
    ]
}

mod command_parsing {
    use super::*;

    #[test]
    fn test_windows_round_trip() {
        for command in windows_commands() {
            assert_eq!(win_split(&list2cmdline(&command)), command);
        }
    }

    #[test]
    fn test_windows_rules() {
        let cases: [(&str, Vec<String>); 6] = [
            (
                "\"C:\\a b\\x.exe\" --background",
                strings(&["C:\\a b\\x.exe", "--background"]),
            ),
            ("  x.exe   a\tb  ", strings(&["x.exe", "a", "b"])),
            ("x.exe \"a\"\"b\"", strings(&["x.exe", "a\"b"])),
            ("x.exe a\\\\\"b c\"", strings(&["x.exe", "a\\b c"])),
            ("\"C:\\unterminated", strings(&["C:\\unterminated"])),
            ("", Vec::new()),
        ];
        for (line, expected) in cases {
            assert_eq!(win_split(line), expected, "{line:?}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn test_windows_split_agrees_with_command_line_to_argv() {
        use windows::Win32::Foundation::{HLOCAL, LocalFree};
        use windows::Win32::UI::Shell::CommandLineToArgvW;
        use windows::core::HSTRING;

        for command in windows_commands() {
            let line = list2cmdline(&command);
            let wide = HSTRING::from(line.as_str());
            let mut count = 0;
            // SAFETY: `wide` is NUL-terminated; the returned block holds `count` valid
            // NUL-terminated strings and is freed once, after they were copied.
            let reference: Vec<String> = unsafe {
                let argv = CommandLineToArgvW(windows::core::PCWSTR(wide.as_ptr()), &mut count);
                let parsed = std::slice::from_raw_parts(argv, count as usize)
                    .iter()
                    .map(|arg| arg.to_string().unwrap())
                    .collect();
                LocalFree(Some(HLOCAL(argv.cast())));
                parsed
            };
            assert_eq!(win_split(&line), reference, "{line}");
        }
    }

    #[test]
    fn test_desktop_exec_round_trip() {
        let commands = [
            strings(&["/usr/bin/python3", "-m", "eye_tracker", "--background"]),
            strings(&[
                "/home/me/My Apps/Eye Tracker.AppImage",
                "--config-dir",
                "/home/me/ET data",
            ]),
            strings(&[
                "/opt/we$ird `name`/app \"q\" 50%",
                "a\\b",
                "x;y|z",
                "tab\there",
            ]),
        ];
        for command in commands {
            let value = desktop_exec(&command);
            assert_eq!(desktop_exec_split(&value), command);
            assert_eq!(
                desktop_exec_split(&value),
                desktop_exec_split_reference(&value)
            );
        }
    }

    #[test]
    fn test_desktop_exec_split_is_lenient_with_foreign_entries() {
        assert_eq!(
            desktop_exec_split("sh -c \"echo \\\"hi\\\"\""),
            ["sh", "-c", "echo \"hi\""]
        );
        assert_eq!(desktop_exec_split("app %U"), ["app", "%U"]);
    }

    #[test]
    fn test_profile_helpers() {
        let command = strings(&[
            "app",
            "--config-dir",
            "/a",
            "--background",
            "--config-dir=/b",
        ]);
        assert_eq!(command_profile(&command).as_deref(), Some("/a"));
        assert_eq!(without_profile(&command), ["--background"]);
        // no value
        assert_eq!(command_profile(&strings(&["app", "--config-dir"])), None);
        assert_eq!(
            command_profile(&strings(&["--config-dir", "/program/is/first"])),
            None
        );
    }
}

#[test]
fn test_system_detection() {
    for (os, expected) in [
        ("windows", System::Windows),
        ("macos", System::MacOs),
        ("linux", System::Linux),
        ("freebsd", System::Unsupported),
    ] {
        assert_eq!(system_for(os), expected, "{os}");
    }
}

#[test]
fn test_system_defaults_to_running_os() {
    assert_eq!(Host::real().system, system_for(std::env::consts::OS));
    assert_eq!(is_supported(), Host::real().system != System::Unsupported);
}

#[test]
fn test_status_values() {
    let values = [
        (Status::Enabled, "enabled"),
        (Status::Disabled, "disabled"),
        (Status::Stale, "stale"),
        (Status::OtherProfile, "other-profile"),
    ];
    for (status, text) in values {
        assert_eq!(status.as_str(), text);
    }
}
