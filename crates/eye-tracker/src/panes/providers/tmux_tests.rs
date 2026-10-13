// Ported from tests/test_panes_tmux.py (the tmux pane provider with recorded CLI output).

use std::collections::HashMap;

use super::*;
use crate::testing::panes_fakes::{Answer, FakeRunner};
use crate::types::WindowHandle;

use super::super::super::types::PaneId;
use super::super::command::CommandResult;

const TERMINAL_PID: u32 = 4000;
const CLIENT_PID: u32 = 4100;
const AREA: Rect = Rect::new(100, 50, 1600, 900);

fn window() -> WindowRef {
    window_with_pid(TERMINAL_PID)
}

fn window_with_pid(pid: u32) -> WindowRef {
    WindowRef {
        handle: WindowHandle::Hwnd(0x3012A),
        pid: Some(pid),
        rect: Some(Rect::new(92, 42, 1616, 916)),
    }
}

fn app() -> AppIdentity {
    AppIdentity {
        process: "windowsterminal".into(),
        app_id: "CASCADIA_HOSTING_WINDOW_CLASS".into(),
    }
}

fn app_named(process: &str, app_id: &str) -> AppIdentity {
    AppIdentity {
        process: process.into(),
        app_id: app_id.into(),
    }
}

/// `tmux list-clients -F CLIENT_FORMAT`: two terminals attached to one server.
fn clients() -> String {
    format!(
        "{CLIENT_PID}\t1727791200\t@1\t200\t50\tbottom\n5100\t1727791100\t@4\t120\t40\tbottom\n"
    )
}

fn first_client() -> String {
    clients().lines().next().unwrap().to_string()
}

/// `tmux list-panes -t @1 -F PANE_FORMAT`: one pane left, two stacked on the right, a 49-row
/// window under a one-row status line.
const PANES: &str =
    "%0 0 0 100 49 1 200 49 0\n%1 101 0 99 24 0 200 49 0\n%2 101 25 99 24 0 200 49 0\n";

/// The keyword arguments of the Python `provider` helper.
struct Setup {
    which: Option<HashMap<String, String>>,
    parents: Option<HashMap<u32, Vec<u32>>>,
    descendants: HashMap<u32, HashSet<String>>,
    system: &'static str,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            which: None,
            parents: None,
            descendants: HashMap::new(),
            system: "linux",
        }
    }
}

fn provider(runner: &FakeRunner, setup: Setup) -> TmuxProvider {
    let tools = setup
        .which
        .unwrap_or_else(|| HashMap::from([("tmux".to_string(), "/usr/bin/tmux".to_string())]));
    let tree = setup.parents.unwrap_or_else(|| {
        HashMap::from([
            (CLIENT_PID, vec![3900, TERMINAL_PID, 1]),
            (5100, vec![5000, 1]),
        ])
    });
    let below = setup.descendants;
    let mut p = TmuxProvider::new(
        Arc::new(|w: &WindowRef| (w.handle == window().handle).then_some(AREA)),
        Arc::new(runner.clone()),
    );
    p.which = Box::new(move |name| tools.get(name).cloned());
    p.parents = Box::new(move |pid| tree.get(&pid).cloned().unwrap_or_default());
    p.descendants = Box::new(move |pid| below.get(&pid).cloned().unwrap_or_default());
    p.system = setup.system.into();
    p.clock = Box::new(|| 0.0);
    p
}

fn native_runner() -> FakeRunner {
    FakeRunner::new(&[
        (&["list-clients"], clients().into()),
        (&["list-panes"], PANES.into()),
    ])
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

// ------------------------------------------------------------------ parsing
#[test]
fn test_parse_clients_skips_malformed_lines() {
    let found = parse_clients(&(clients() + "garbage\n12\tx\t@1\t1\t1\tbottom\n"));
    let summary: Vec<_> = found
        .iter()
        .map(|c| (c.pid, c.window_id.as_str(), c.width, c.height))
        .collect();
    assert_eq!(
        summary,
        vec![(CLIENT_PID, "@1", 200, 50), (5100, "@4", 120, 40)]
    );
    assert!(!found[0].status_top);
}

#[test]
fn test_parse_panes_and_unreadable_output() {
    let panes = parse_panes(PANES).unwrap();
    let summary: Vec<_> = panes
        .iter()
        .map(|p| {
            (
                p.pane_id.as_str(),
                p.left,
                p.top,
                p.width,
                p.height,
                p.active,
            )
        })
        .collect();
    assert_eq!(
        summary,
        vec![
            ("%0", 0, 0, 100, 49, true),
            ("%1", 101, 0, 99, 24, false),
            ("%2", 101, 25, 99, 24, false),
        ]
    );
    assert_eq!(parse_panes("").unwrap(), vec![]);
    // an older tmux without some of the formats
    assert!(parse_panes("%0 0 0 100\n").is_err());
}

#[test]
fn test_layout_maps_cells_onto_the_content_area() {
    let client = parse_clients(&first_client()).remove(0);
    let panes = layout_panes(&parse_panes(PANES).unwrap(), &client, &AREA, "tmux");
    // 200 columns over 1600 px: 8 px per column; 50 rows over 900 px: 18 px per row.
    assert_eq!(
        panes[0],
        Pane::new("%0", Rect::new(100, 50, 800, 882), true, "tmux")
    );
    assert_eq!(
        panes[1],
        Pane::new("%1", Rect::new(908, 50, 792, 432), false, "tmux")
    );
    assert_eq!(
        panes[2],
        Pane::new("%2", Rect::new(908, 500, 792, 432), false, "tmux")
    );
}

#[test]
fn test_layout_with_the_status_line_on_top() {
    let client = parse_clients("1\t0\t@1\t200\t50\ttop\n").remove(0);
    let panes = layout_panes(&parse_panes(PANES).unwrap(), &client, &AREA, "tmux");
    assert_eq!(panes[0].rect.y, 50 + 18); // below the status line
}

// ------------------------------------------------------------------ provider
#[test]
fn test_applies_to_terminals_with_tmux_only() {
    let mut p = provider(&FakeRunner::default(), Setup::default());
    assert!(p.applies(&app()));
    assert!(p.applies(&app_named("kitty", "kitty")));
    assert!(!p.applies(&app_named("notepad", "Notepad")));
    let no_tmux = Setup {
        which: Some(HashMap::new()),
        ..Setup::default()
    };
    // no tmux anywhere
    assert!(!provider(&FakeRunner::default(), no_tmux).applies(&app()));
}

#[test]
fn test_detects_the_client_running_in_the_window() {
    let runner = native_runner();
    let snapshot = provider(&runner, Setup::default())
        .detect(&window(), &app())
        .unwrap()
        .unwrap();
    assert_eq!(snapshot.window_handle, window().handle);
    let ids: Vec<PaneId> = snapshot.panes.iter().map(|p| p.id.clone()).collect();
    assert_eq!(ids, vec!["%0".into(), "%1".into(), "%2".into()]);
    assert_eq!(snapshot.focused().unwrap().id, PaneId::from("%0"));
    assert_eq!(
        runner.commands(),
        vec![
            strings(&["list-clients", "-F", CLIENT_FORMAT]),
            // the window of *this* client
            strings(&["list-panes", "-t", "@1", "-F", PANE_FORMAT]),
        ]
    );
    assert!(runner.calls().iter().all(|call| call[0] == "/usr/bin/tmux"));
}

#[test]
fn test_no_snapshot_without_a_matching_client_or_server() {
    let other = window_with_pid(999);
    assert_eq!(
        provider(&native_runner(), Setup::default()).detect(&other, &app()),
        Ok(None)
    );
    let no_server = FakeRunner::new(&[(
        &["list-clients"],
        Answer::Result(CommandResult::new(1, "", "no server running")),
    )]);
    assert_eq!(
        provider(&no_server, Setup::default()).detect(&window(), &app()),
        Ok(None)
    );
    let unknown_area = WindowRef {
        handle: WindowHandle::Hwnd(1),
        pid: Some(TERMINAL_PID),
        rect: None,
    };
    assert_eq!(
        provider(&native_runner(), Setup::default()).detect(&unknown_area, &app()),
        Ok(None)
    );
}

#[test]
fn test_single_or_zoomed_pane_is_no_snapshot() {
    let single = FakeRunner::new(&[
        (&["list-clients"], clients().into()),
        (&["list-panes"], "%0 0 0 200 49 1 200 49 0\n".into()),
    ]);
    assert_eq!(
        provider(&single, Setup::default()).detect(&window(), &app()),
        Ok(None)
    );
    let zoomed = FakeRunner::new(&[
        (&["list-clients"], clients().into()),
        (&["list-panes"], PANES.replace(" 0\n", " 1\n").into()),
    ]);
    assert_eq!(
        provider(&zoomed, Setup::default()).detect(&window(), &app()),
        Ok(None)
    );
}

#[test]
fn test_focus_selects_the_pane_on_the_same_server() {
    let runner = native_runner();
    let mut p = provider(&runner, Setup::default());
    let pane = Pane::new("%2", Rect::new(0, 0, 1, 1), false, "tmux");
    // nothing detected for this window yet
    assert_eq!(p.focus(&window(), &pane), Ok(false));
    assert!(p.detect(&window(), &app()).unwrap().is_some());
    runner.answer(&["select-pane"], "");
    assert_eq!(p.focus(&window(), &pane), Ok(true));
    let expected = strings(&["/usr/bin/tmux", "select-pane", "-t", "%2"]);
    assert_eq!(runner.calls().last().unwrap(), &expected);
    // Never anything but a pane id after -t.
    let injected = Pane::new("; kill-server", Rect::new(0, 0, 1, 1), false, "tmux");
    assert_eq!(p.focus(&window(), &injected), Ok(false));
    assert_eq!(runner.calls().last().unwrap(), &expected);
}

#[test]
fn test_wsl_server_is_used_only_for_windows_running_wsl() {
    let runner = FakeRunner::new(&[
        (&["-e", "tmux", "list-clients"], first_client().into()),
        (&["-e", "tmux", "list-panes"], PANES.into()),
    ]);
    let wsl = "C:\\Windows\\System32\\wsl.exe";
    let wsl_setup = || Setup {
        which: Some(HashMap::from([("wsl".to_string(), wsl.to_string())])),
        parents: Some(HashMap::new()),
        descendants: HashMap::from([(
            TERMINAL_PID,
            ["openconsole", "wsl", "bash"].map(String::from).into(),
        )]),
        system: "win32",
    };
    let mut p = provider(&runner, wsl_setup());
    assert!(p.applies(&app()));
    let snapshot = p.detect(&window(), &app()).unwrap().unwrap();
    assert_eq!(
        runner.calls()[0],
        strings(&[wsl, "-e", "tmux", "list-clients", "-F", CLIENT_FORMAT])
    );
    runner.answer(&["-e", "tmux", "select-pane"], "");
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(true));
    assert_eq!(
        runner.calls().last().unwrap(),
        &strings(&[wsl, "-e", "tmux", "select-pane", "-t", "%1"])
    );

    // A window without WSL in its process tree never reaches the WSL server.
    let plain = Setup {
        descendants: HashMap::new(),
        ..wsl_setup()
    };
    let mut plain = provider(&FakeRunner::default(), plain);
    assert_eq!(plain.detect(&window(), &app()), Ok(None));
    // WSL is a Windows thing.
    let linux = Setup {
        system: "linux",
        ..wsl_setup()
    };
    assert!(!provider(&FakeRunner::default(), linux).applies(&app()));
}

#[test]
fn test_two_wsl_clients_are_ambiguous() {
    let runner = FakeRunner::new(&[(&["-e", "tmux", "list-clients"], clients().into())]);
    let mut p = provider(
        &runner,
        Setup {
            which: Some(HashMap::from([("wsl".to_string(), "wsl.exe".to_string())])),
            parents: Some(HashMap::new()),
            descendants: HashMap::from([(TERMINAL_PID, ["wslhost"].map(String::from).into())]),
            system: "win32",
        },
    );
    assert_eq!(p.detect(&window(), &app()), Ok(None));
    assert!(
        runner
            .calls()
            .iter()
            .all(|call| !call.iter().any(|a| a == "list-panes"))
    );
}

#[test]
fn test_runner_errors_propagate_as_pane_errors() {
    let runner = FakeRunner::new(&[(
        &["list-clients"],
        Answer::Error(PaneError::new("tmux did not answer within 1 s")),
    )]);
    assert!(
        provider(&runner, Setup::default())
            .detect(&window(), &app())
            .is_err()
    );
}

#[test]
fn test_terminal_list_covers_the_common_emulators() {
    for name in [
        "windowsterminal",
        "wezterm-gui",
        "kitty",
        "iterm2",
        "ghostty",
    ] {
        assert!(TERMINAL_PROCESSES.contains(&name), "{name}");
    }
}
