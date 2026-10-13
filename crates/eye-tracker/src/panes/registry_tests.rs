// Ported from tests/test_panes_registry.py (the pane provider registry and its deny-list).
//
// Changed: the Python tests replaced `registry._optional_provider` with monkeypatch; the optional
// providers are created by the function passed to `default_providers_with` instead, and
// test_default_providers_follow_the_settings uses it too, so it can see Windows Terminal on every
// platform (the real factory only has it on Windows).

use super::*;
use crate::testing::panes_fakes::{FakeRunner, RecordingProvider, current_thread_name, two_panes};
use crate::types::{Rect, WindowHandle, WindowRef};

use super::super::worker::{PaneEvent, PaneWorker, PaneWorkerOptions};

fn identity(process: &str, app_id: &str) -> AppIdentity {
    AppIdentity {
        process: process.into(),
        app_id: app_id.into(),
    }
}

fn electron_and_chromium() -> Vec<AppIdentity> {
    vec![
        identity("code", "Chrome_WidgetWin_1"),
        identity("cursor", "Chrome_WidgetWin_1"),
        identity("claude", "Chrome_WidgetWin_1"),
        identity("chatgpt", ""),
        identity("chrome", "Chrome_WidgetWin_1"),
        identity("msedge", "Chrome_WidgetWin_1"),
        identity("firefox", "MozillaWindowClass"),
        identity("slack", "Chrome_WidgetWin_1"),
        identity("teams", ""),
        identity("ms-teams", "TeamsWebView"),
        identity("discord", "Chrome_WidgetWin_1"),
        identity("obsidian", "obsidian"),
        identity("electron", ""),
        // An unknown Electron app on Windows: caught by its window class alone.
        identity("someapp", "Chrome_WidgetWin_1"),
        // macOS bundle ids and X11 process names.
        identity("code helper", "com.microsoft.VSCode"),
        identity("google chrome", "com.google.Chrome"),
    ]
}

fn boxed(providers: &[&RecordingProvider]) -> Vec<Box<dyn PaneProvider>> {
    providers
        .iter()
        .map(|p| Box::new((*p).clone()) as Box<dyn PaneProvider>)
        .collect()
}

fn names(providers: Vec<&mut dyn PaneProvider>) -> Vec<String> {
    providers.iter().map(|p| p.name().to_string()).collect()
}

#[test]
fn test_denied_apps_never_reach_a_provider() {
    for app in electron_and_chromium() {
        let eager =
            RecordingProvider::fake().snapshot(|window| Some(two_panes(window.handle.clone())));
        let mut registry = PaneRegistry::new(boxed(&[&eager]));
        assert!(is_denied(&app, false), "{}|{}", app.process, app.app_id);
        assert!(registry.providers_for(Some(&app)).is_empty());
        let worker = PaneWorker::new(
            registry,
            PaneWorkerOptions {
                synchronous: true,
                ..PaneWorkerOptions::default()
            },
        );
        let window = WindowRef {
            handle: WindowHandle::Hwnd(1),
            pid: Some(2),
            rect: None,
        };
        worker.request_detect(&window, &app);
        assert_eq!(eager.calls(), vec![]); // not even applies()
        let events = worker.drain_events();
        let [PaneEvent::Detected(result)] = &events[..] else {
            panic!("expected one detection result, got {events:?}");
        };
        assert_eq!(result.snapshot, None);
    }
}

#[test]
fn test_the_plan_lists_are_all_denied() {
    let required = [
        "code", "cursor", "claude", "chatgpt", "chrome", "msedge", "firefox", "slack", "teams",
        "ms-teams", "discord", "obsidian", "electron",
    ];
    for name in required {
        assert!(DENIED_PROCESSES.contains(&name), "{name}");
    }
    assert!(DENIED_APP_IDS.contains(&"Chrome_WidgetWin_1"));
}

#[test]
fn test_terminals_are_not_denied_and_providers_are_asked_in_order() {
    let first = RecordingProvider::new("first").applies(false);
    let second = RecordingProvider::new("second");
    let third = RecordingProvider::new("third");
    let mut registry = PaneRegistry::new(boxed(&[&first, &second, &third]));
    let app = identity("windowsterminal", "CASCADIA_HOSTING_WINDOW_CLASS");
    assert!(!is_denied(&app, false));
    assert_eq!(
        names(registry.providers_for(Some(&app))),
        vec!["second", "third"]
    );
    assert_eq!(
        registry.provider("third").map(|p| p.name().to_string()),
        Some("third".into())
    );
    assert!(registry.provider("nope").is_none());
    assert!(registry.providers_for(None).is_empty());
}

/// A provider whose `applies` panics (the Python test raised `RuntimeError`).
struct BrokenApplies;

impl PaneProvider for BrokenApplies {
    fn name(&self) -> &str {
        "broken"
    }

    fn applies(&mut self, _app: &AppIdentity) -> bool {
        panic!("boom")
    }

    fn detect(
        &mut self,
        _window: &WindowRef,
        _app: &AppIdentity,
    ) -> Result<Option<super::super::types::PaneSnapshot>, super::super::types::PaneError> {
        Ok(None)
    }

    fn focus(
        &mut self,
        _window: &WindowRef,
        _pane: &super::super::types::Pane,
    ) -> Result<bool, super::super::types::PaneError> {
        Ok(false)
    }
}

#[test]
fn test_a_provider_failing_in_applies_is_skipped() {
    let good = RecordingProvider::new("good");
    let mut providers: Vec<Box<dyn PaneProvider>> = vec![Box::new(BrokenApplies)];
    providers.extend(boxed(&[&good]));
    let mut registry = PaneRegistry::new(providers);
    assert_eq!(
        names(registry.providers_for(Some(&identity("kitty", "kitty")))),
        vec!["good"]
    );
}

fn area() -> ClientRect {
    Arc::new(|_| Some(Rect::new(0, 0, 100, 100)))
}

/// The optional providers as recording fakes named after their slot.
fn fake_optional(
    name: &str,
    _client_rect: &ClientRect,
    _runner: &Arc<dyn CommandRunner>,
) -> Option<Box<dyn PaneProvider>> {
    Some(Box::new(RecordingProvider::new(name)))
}

#[test]
fn test_default_providers_follow_the_settings() {
    let mut settings = PaneSettings::default();

    let provider_names = |settings: &PaneSettings, system: &str| -> Vec<String> {
        default_providers_with(
            settings,
            area(),
            Some(Arc::new(FakeRunner::default())),
            system,
            &fake_optional,
        )
        .iter()
        .map(|p| p.name().to_string())
        .collect()
    };

    // Windows Terminal is asked through UI Automation: on Windows only.
    assert_eq!(
        provider_names(&settings, "win32"),
        vec!["wezterm", "windows_terminal", "tmux"]
    );
    assert_eq!(provider_names(&settings, "linux"), vec!["wezterm", "tmux"]);
    assert_eq!(provider_names(&settings, "darwin"), vec!["wezterm", "tmux"]);
    settings.windows_terminal = false;
    assert_eq!(provider_names(&settings, "win32"), vec!["wezterm", "tmux"]);
    settings.wezterm = false;
    assert_eq!(provider_names(&settings, "win32"), vec!["tmux"]);
    settings.tmux = false;
    assert!(provider_names(&settings, "win32").is_empty());
}

#[test]
fn test_an_available_optional_provider_takes_its_slot() {
    let created = std::cell::RefCell::new(Vec::new());
    let optional = |name: &str,
                    client_rect: &ClientRect,
                    runner: &Arc<dyn CommandRunner>|
     -> Option<Box<dyn PaneProvider>> {
        created.borrow_mut().push(name.to_string());
        fake_optional(name, client_rect, runner)
    };
    let mut settings = PaneSettings::default();
    let provider_names = |settings: &PaneSettings| -> Vec<String> {
        default_providers_with(settings, area(), None, "win32", &optional)
            .iter()
            .map(|p| p.name().to_string())
            .collect()
    };
    assert_eq!(
        provider_names(&settings),
        vec!["wezterm", "windows_terminal", "tmux"]
    );
    assert_eq!(*created.borrow(), vec!["windows_terminal"]);
    settings.windows_terminal = false;
    assert!(!provider_names(&settings).contains(&"windows_terminal".to_string()));
}

#[test]
fn test_missing_optional_module_is_skipped() {
    let runner: Arc<dyn CommandRunner> = Arc::new(FakeRunner::default());
    assert!(optional_provider("does_not_exist", &area(), &runner).is_none());
}

#[test]
fn test_default_providers_on_this_system_have_no_optional_slots_off_windows() {
    // New: `default_providers` itself (the real factory) on a system without the optional
    // providers.
    let settings = PaneSettings::default();
    let found = default_providers(
        &settings,
        area(),
        Some(Arc::new(FakeRunner::default())),
        "linux",
    );
    let found: Vec<&str> = found.iter().map(|p| p.name()).collect();
    assert_eq!(found, vec!["wezterm", "tmux"]);
}

#[test]
fn test_desktop_apps_skip_the_deny_list_and_reach_their_provider_alone() {
    // New: the registry half of the opt-in that the Python chromium_apps tests cover.
    let claude = identity("claude", "Chrome_WidgetWin_1");
    assert!(is_denied(&claude, false));
    assert!(!is_denied(&claude, true));
    assert!(is_denied(&identity("code", "Chrome_WidgetWin_1"), true));

    let desktop = RecordingProvider::new("desktop_apps");
    let terminal = RecordingProvider::new("tmux");
    let mut registry = PaneRegistry::new(boxed(&[&desktop, &terminal])).with_desktop_apps(true);
    assert_eq!(
        names(registry.providers_for(Some(&claude))),
        vec!["desktop_apps"]
    );
    let kitty = identity("kitty", "kitty");
    assert_eq!(names(registry.providers_for(Some(&kitty))), vec!["tmux"]);
    let mut off = PaneRegistry::new(boxed(&[&desktop, &terminal]));
    assert!(off.providers_for(Some(&claude)).is_empty());
}

#[test]
fn test_close_reaches_every_provider() {
    // New: registry.close() on the calling thread (see test_a_failing_close_is_contained in
    // worker_tests.rs for the failure case).
    let a = RecordingProvider::new("a");
    let b = RecordingProvider::new("b");
    let mut registry = PaneRegistry::new(boxed(&[&a, &b]));
    registry.close();
    assert_eq!(a.closed_on(), vec![current_thread_name()]);
    assert_eq!(b.closed_on(), vec![current_thread_name()]);
}
