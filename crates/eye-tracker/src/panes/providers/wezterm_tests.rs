// Ported from tests/test_panes_wezterm.py (the WezTerm pane provider with recorded CLI output).
//
// Not ported: the `not hasattr(panes[0], "title")` / `"cwd"` assertions of
// test_parse_panes_reads_only_geometry; `WezPane` has no such fields, which the compiler checks.

use serde_json::json;

use super::*;
use crate::testing::panes_fakes::{Answer, FakeRunner};
use crate::types::WindowHandle;

use super::super::command::CommandResult;

const GUI_PID: u32 = 4000;
const AREA: Rect = Rect::new(100, 50, 1600, 900);
const CLI: &str = "C:\\Program Files\\WezTerm\\wezterm.exe";
const LIST: [&str; 5] = ["cli", "--no-auto-start", "list", "--format", "json"];
const CLIENTS_CMD: [&str; 5] = ["cli", "--no-auto-start", "list-clients", "--format", "json"];

fn window() -> WindowRef {
    WindowRef {
        handle: WindowHandle::Hwnd(0x5A0C),
        pid: Some(GUI_PID),
        rect: Some(Rect::new(92, 18, 1616, 940)),
    }
}

fn app() -> AppIdentity {
    AppIdentity {
        process: "wezterm-gui".into(),
        app_id: "org.wezfurlong.wezterm".into(),
    }
}

/// The keyword arguments of the Python `pane_json` helper.
struct PaneJson {
    tab_id: i64,
    window_id: i64,
    cols: i64,
    rows: i64,
    left_col: i64,
    top_row: i64,
    is_active: bool,
    is_zoomed: bool,
}

impl PaneJson {
    fn new(cols: i64) -> Self {
        Self {
            tab_id: 0,
            window_id: 0,
            cols,
            rows: 52,
            left_col: 0,
            top_row: 0,
            is_active: false,
            is_zoomed: false,
        }
    }
}

/// One entry of `wezterm cli list --format json` (8x16 px cells).
fn pane_json(pane_id: i64, p: PaneJson) -> serde_json::Value {
    json!({
        "window_id": p.window_id,
        "tab_id": p.tab_id,
        "pane_id": pane_id,
        "workspace": "default",
        "size": {
            "rows": p.rows,
            "cols": p.cols,
            "pixel_width": p.cols * 8,
            "pixel_height": p.rows * 16,
            "dpi": 96,
        },
        "title": "vim notes.md",
        "cwd": "file:///C:/Users/me/secret-project/",
        "cursor_x": 3,
        "cursor_y": 10,
        "cursor_shape": "Default",
        "cursor_visibility": "Visible",
        "left_col": p.left_col,
        "top_row": p.top_row,
        "tab_title": "",
        "window_title": "vim notes.md",
        "is_active": p.is_active,
        "is_zoomed": p.is_zoomed,
        "tty_name": null,
    })
}

/// Tab 0: two panes side by side (a one-column divider); tab 1: a single pane.
fn listed() -> String {
    json!([
        pane_json(
            0,
            PaneJson {
                is_active: true,
                ..PaneJson::new(99)
            }
        ),
        pane_json(
            1,
            PaneJson {
                left_col: 100,
                ..PaneJson::new(100)
            }
        ),
        pane_json(
            2,
            PaneJson {
                tab_id: 1,
                is_active: true,
                ..PaneJson::new(200)
            }
        ),
    ])
    .to_string()
}

fn clients() -> String {
    json!([{
        "username": "me",
        "hostname": "desk",
        "pid": GUI_PID,
        "connection_elapsed": {"secs": 812, "nanos": 0},
        "idle_time": {"secs": 0, "nanos": 250000000},
        "workspace": "default",
        "focused_pane_id": 1,
    }])
    .to_string()
}

fn provider(runner: &FakeRunner, on_path: bool) -> WezTermProvider {
    let mut p = WezTermProvider::new(
        Arc::new(|w: &WindowRef| (w.handle == window().handle).then_some(AREA)),
        Arc::new(runner.clone()),
    );
    p.which = Box::new(move |name| {
        if name == "wezterm" {
            on_path.then(|| CLI.to_string())
        } else {
            name.ends_with("wezterm.exe").then(|| name.to_string())
        }
    });
    p.executable =
        Box::new(|pid| (pid == GUI_PID).then(|| "C:\\Apps\\WezTerm\\wezterm-gui.exe".to_string()));
    p.system = "win32".into();
    p
}

fn runner(listed: &str, clients: &str) -> FakeRunner {
    FakeRunner::new(&[(&LIST, listed.into()), (&CLIENTS_CMD, clients.into())])
}

fn default_runner() -> FakeRunner {
    runner(&listed(), &clients())
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn ids(snapshot: &PaneSnapshot) -> Vec<PaneId> {
    snapshot.panes.iter().map(|p| p.id.clone()).collect()
}

// ------------------------------------------------------------------ parsing
#[test]
fn test_parse_panes_reads_only_geometry() {
    let panes = parse_panes(&listed()).unwrap();
    let summary: Vec<_> = panes
        .iter()
        .map(|p| (p.pane_id, p.tab_id, p.cols, p.left_col, p.active))
        .collect();
    assert_eq!(
        summary,
        vec![
            (0, 0, 99, 0, true),
            (1, 0, 100, 100, false),
            (2, 1, 200, 0, true)
        ]
    );
    assert!(parse_panes("not json").is_err());
    assert!(parse_panes(&json!([{"pane_id": 1}]).to_string()).is_err());
    assert_eq!(parse_panes("[]").unwrap(), vec![]);
}

#[test]
fn test_parse_clients() {
    let found = parse_clients(&clients()).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(
        (found[0].pid, found[0].focused_pane_id),
        (i64::from(GUI_PID), Some(1))
    );
    approx::assert_relative_eq!(found[0].idle_s, 0.25, max_relative = 1e-6);
    assert_eq!(parse_clients("[{\"pid\": \"x\"}, 5]").unwrap(), vec![]);
}

#[test]
fn test_layout_uses_the_real_cell_size() {
    let tab: Vec<WezPane> = parse_panes(&listed())
        .unwrap()
        .into_iter()
        .filter(|p| p.tab_id == 0)
        .collect();
    let laid_out = layout_panes(&tab, &AREA, "wezterm");
    // The grid is 1600 x 832 px; the 68 px above it are the tab bar.
    assert_eq!(
        laid_out[0],
        Pane::new(0, Rect::new(100, 118, 792, 832), true, "wezterm")
    );
    assert_eq!(
        laid_out[1],
        Pane::new(1, Rect::new(900, 118, 800, 832), false, "wezterm")
    );
}

#[test]
fn test_layout_on_a_retina_display() {
    let tab: Vec<WezPane> = parse_panes(&listed())
        .unwrap()
        .into_iter()
        .filter(|p| p.tab_id == 0)
        .collect();
    let laid_out = layout_panes(&tab, &Rect::new(0, 25, 800, 450), "wezterm"); // points, half the pixels
    assert_eq!(laid_out[0].rect, Rect::new(0, 59, 396, 416));
    assert_eq!(laid_out[1].rect, Rect::new(400, 59, 400, 416));
}

// ------------------------------------------------------------------ provider
#[test]
fn test_applies_to_the_wezterm_gui_only() {
    let mut p = provider(&default_runner(), true);
    assert!(p.applies(&app()));
    assert!(!p.applies(&AppIdentity {
        process: "windowsterminal".into(),
        app_id: String::new()
    }));
}

#[test]
fn test_detects_the_tab_with_the_guis_focused_pane() {
    let r = default_runner();
    let snapshot = provider(&r, true)
        .detect(&window(), &app())
        .unwrap()
        .unwrap();
    assert_eq!(ids(&snapshot), vec![PaneId::Int(0), PaneId::Int(1)]);
    // The GUI focuses pane 1, whatever is_active says.
    assert_eq!(snapshot.focused().unwrap().id, PaneId::Int(1));
    assert_eq!(r.commands(), vec![strings(&LIST), strings(&CLIENTS_CMD)]);
    assert_eq!(r.calls()[0][0], CLI);
}

#[test]
fn test_without_focus_information_only_an_unambiguous_tab_is_used() {
    let two_tabs = runner(&listed(), "[]");
    assert_eq!(
        provider(&two_tabs, true).detect(&window(), &app()),
        Ok(None)
    ); // two tabs
    let one_tab = json!([
        pane_json(
            0,
            PaneJson {
                is_active: true,
                ..PaneJson::new(99)
            }
        ),
        pane_json(
            1,
            PaneJson {
                left_col: 100,
                ..PaneJson::new(100)
            }
        ),
    ])
    .to_string();
    let snapshot = provider(&runner(&one_tab, "[]"), true)
        .detect(&window(), &app())
        .unwrap()
        .unwrap();
    assert_eq!(snapshot.focused().unwrap().id, PaneId::Int(0));
}

#[test]
fn test_single_pane_zoom_or_no_gui_is_no_snapshot() {
    let only_tab_one = json!([{"pid": GUI_PID, "focused_pane_id": 2}]).to_string();
    // tab 1: one pane
    assert_eq!(
        provider(&runner(&listed(), &only_tab_one), true).detect(&window(), &app()),
        Ok(None)
    );
    let zoomed = json!([
        pane_json(
            0,
            PaneJson {
                is_zoomed: true,
                is_active: true,
                ..PaneJson::new(99)
            }
        ),
        pane_json(
            1,
            PaneJson {
                left_col: 100,
                ..PaneJson::new(100)
            }
        ),
    ])
    .to_string();
    assert_eq!(
        provider(&runner(&zoomed, "[]"), true).detect(&window(), &app()),
        Ok(None)
    );
    let down = FakeRunner::new(&[
        (
            &LIST,
            Answer::Result(CommandResult::new(1, "", "failed to connect")),
        ),
        (&CLIENTS_CMD, "[]".into()),
    ]);
    assert_eq!(provider(&down, true).detect(&window(), &app()), Ok(None));
    // nothing else once the GUI does not answer
    assert_eq!(down.commands(), vec![strings(&LIST)]);
}

#[test]
fn test_cli_found_beside_the_gui_when_not_on_path() {
    let r = default_runner();
    assert!(
        provider(&r, false)
            .detect(&window(), &app())
            .unwrap()
            .is_some()
    );
    assert_eq!(r.calls()[0][0], "C:\\Apps\\WezTerm\\wezterm.exe");
}

#[test]
fn test_focus_activates_the_pane() {
    let r = default_runner();
    let mut p = provider(&r, true);
    let target = Pane::new(0, Rect::new(0, 0, 1, 1), false, "wezterm");
    // not detected yet: no CLI known
    assert_eq!(p.focus(&window(), &target), Ok(false));
    assert!(p.detect(&window(), &app()).unwrap().is_some());
    r.answer(&["cli", "--no-auto-start", "activate-pane"], "");
    assert_eq!(p.focus(&window(), &target), Ok(true));
    assert_eq!(
        r.calls().last().unwrap(),
        &strings(&[
            CLI,
            "cli",
            "--no-auto-start",
            "activate-pane",
            "--pane-id",
            "0"
        ])
    );
    let injected = Pane::new("0; rm", Rect::new(0, 0, 1, 1), false, "wezterm");
    assert_eq!(p.focus(&window(), &injected), Ok(false));
}
