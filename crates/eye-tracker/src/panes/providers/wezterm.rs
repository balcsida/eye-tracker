//! WezTerm panes, through `wezterm cli` (never starting a mux server).
//!
//! `wezterm cli --no-auto-start list --format json` lists every pane with its
//! cell position and size, including pixel sizes; `list-clients` tells which
//! pane the GUI has focused, and so which window and tab the user sees. The panes
//! of that tab are laid out over the window's content area with the real cell
//! size in pixels (scaled down on Retina displays, where WezTerm counts device
//! pixels and the desktop counts points). Focus moves with `activate-pane`.
//! Only the needed fields are read; titles and working directories are ignored.
//!
//! The padding around the grid is assumed equal on the left, right and bottom, and
//! everything else above the grid is taken to be the tab bar (WezTerm's default
//! layout). A tab bar at the bottom shifts the panes by its height.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::types::{AppIdentity, Rect, WindowRef};

use super::super::types::{ClientRect, Pane, PaneError, PaneId, PaneProvider, PaneSnapshot};
use super::command::{
    COMMAND_TIMEOUT_S, CommandRunner, Which, WindowKey, executable_of, which, window_key,
};

pub const GUI_PROCESSES: [&str; 1] = ["wezterm-gui"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WezPane {
    pub window_id: i64,
    pub tab_id: i64,
    pub pane_id: i64,
    pub rows: i64,
    pub cols: i64,
    pub pixel_width: i64,
    pub pixel_height: i64,
    pub left_col: i64,
    pub top_row: i64,
    pub active: bool,
    pub zoomed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WezClient {
    pub pid: i64,
    pub idle_s: f64,
    pub focused_pane_id: Option<i64>,
}

/// An integer field (a boolean or a float is not one).
fn int(value: Option<&Value>) -> Option<i64> {
    value?.as_i64()
}

/// Python's truthiness of a JSON value, for `bool(item.get(key, False))`.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

fn parse_pane(item: &Value) -> Option<WezPane> {
    let size = item.get("size")?;
    Some(WezPane {
        window_id: int(item.get("window_id"))?,
        tab_id: int(item.get("tab_id"))?,
        pane_id: int(item.get("pane_id"))?,
        rows: int(size.get("rows"))?,
        cols: int(size.get("cols"))?,
        pixel_width: int(size.get("pixel_width"))?,
        pixel_height: int(size.get("pixel_height"))?,
        left_col: int(item.get("left_col"))?,
        top_row: int(item.get("top_row"))?,
        active: truthy(item.get("is_active")),
        zoomed: truthy(item.get("is_zoomed")),
    })
}

/// `wezterm cli list --format json` output. Fails with [`PaneError`] if unreadable.
pub fn parse_panes(text: &str) -> Result<Vec<WezPane>, PaneError> {
    let text = if text.is_empty() { "[]" } else { text };
    let data: Value = serde_json::from_str(text)
        .map_err(|_| PaneError::new("wezterm cli list output is not JSON"))?;
    let Value::Array(data) = data else {
        return Err(PaneError::new("wezterm cli list output is not a list"));
    };
    let panes: Vec<WezPane> = data
        .iter()
        .filter_map(parse_pane)
        .filter(|p| p.rows.min(p.cols).min(p.pixel_width).min(p.pixel_height) > 0)
        .collect();
    if !data.is_empty() && panes.is_empty() {
        return Err(PaneError::new("wezterm cli list output could not be read"));
    }
    Ok(panes)
}

fn parse_client(item: &Value) -> Option<WezClient> {
    let idle = item.get("idle_time").filter(|v| truthy(Some(v)));
    let seconds = |key: &str| -> Option<f64> {
        match idle {
            None => Some(0.0),
            Some(idle) => match idle.get(key) {
                None => Some(0.0),
                Some(value) => value.as_f64(),
            },
        }
    };
    let focused = match item.get("focused_pane_id") {
        None | Some(Value::Null) => None,
        other => Some(int(other)?),
    };
    Some(WezClient {
        pid: int(item.get("pid"))?,
        idle_s: seconds("secs")? + seconds("nanos")? / 1e9,
        focused_pane_id: focused,
    })
}

/// `wezterm cli list-clients --format json` output (unreadable entries skipped).
pub fn parse_clients(text: &str) -> Result<Vec<WezClient>, PaneError> {
    let text = if text.is_empty() { "[]" } else { text };
    let data: Value = serde_json::from_str(text)
        .map_err(|_| PaneError::new("wezterm cli list-clients output is not JSON"))?;
    let Value::Array(data) = data else {
        return Ok(Vec::new());
    };
    Ok(data.iter().filter_map(parse_client).collect())
}

/// Place the panes of one tab on the window's content area `area`.
pub fn layout_panes(panes: &[WezPane], area: &Rect, provider: &str) -> Vec<Pane> {
    let first = &panes[0];
    let cell_w = first.pixel_width as f64 / first.cols as f64;
    let cell_h = first.pixel_height as f64 / first.rows as f64;
    let grid_w = panes.iter().map(|p| p.left_col + p.cols).max().unwrap_or(0) as f64 * cell_w;
    let grid_h = panes.iter().map(|p| p.top_row + p.rows).max().unwrap_or(0) as f64 * cell_h;
    let area_w = f64::from(area.w);
    // Device pixels against desktop points (macOS Retina): a whole-number ratio.
    let ratio = if grid_w > area_w * 1.02 {
        (grid_w / area_w).round_ties_even().max(1.0)
    } else {
        1.0
    };
    let scale = 1.0 / ratio;
    let pad_x = ((area_w - grid_w * scale) / 2.0).max(0.0);
    let pad_top = (f64::from(area.h) - grid_h * scale - pad_x).max(0.0);
    panes
        .iter()
        .map(|p| {
            let x0 = f64::from(area.x) + pad_x + p.left_col as f64 * cell_w * scale;
            let y0 = f64::from(area.y) + pad_top + p.top_row as f64 * cell_h * scale;
            let x1 = x0 + p.cols as f64 * cell_w * scale;
            let y1 = y0 + p.rows as f64 * cell_h * scale;
            let round = |v: f64| v.round_ties_even() as i32;
            let rect = Rect::new(
                round(x0),
                round(y0),
                (round(x1) - round(x0)).max(1),
                (round(y1) - round(y0)).max(1),
            );
            Pane::new(p.pane_id, rect, p.active, provider)
        })
        .collect()
}

/// `path`'s directory joined with `name`, in the path syntax of `windows` or POSIX.
fn sibling(path: &str, name: &str, windows: bool) -> String {
    let cut = if windows {
        path.rfind(['\\', '/'])
    } else {
        path.rfind('/')
    };
    format!("{}{name}", cut.map_or("", |i| &path[..=i]))
}

/// Panes of the tab a WezTerm GUI window shows.
pub struct WezTermProvider {
    client_rect: ClientRect,
    run: Arc<dyn CommandRunner>,
    which: Which,
    executable: Box<dyn Fn(u32) -> Option<String> + Send>,
    /// Python's `sys.platform` values ("win32", "linux", "darwin").
    system: String,
    timeout: f64,
    path_checked: bool,
    on_path: Option<String>,
    cli: HashMap<WindowKey, String>,
}

impl WezTermProvider {
    pub const NAME: &'static str = "wezterm";

    /// A provider that asks the real system (`which`, the process list).
    pub fn new(client_rect: ClientRect, runner: Arc<dyn CommandRunner>) -> Self {
        Self {
            client_rect,
            run: runner,
            which: Box::new(which),
            executable: Box::new(executable_of),
            system: crate::panes::registry::current_system().to_string(),
            timeout: COMMAND_TIMEOUT_S,
            path_checked: false,
            on_path: None,
            cli: HashMap::new(),
        }
    }

    /// The `wezterm` program: on PATH, else beside the GUI's executable.
    fn cli_for(&mut self, window: &WindowRef) -> Option<String> {
        let key = window_key(window);
        if let Some(cli) = self.cli.get(&key) {
            return Some(cli.clone());
        }
        if !self.path_checked {
            self.path_checked = true;
            self.on_path = (self.which)("wezterm");
        }
        let mut cli = self.on_path.clone();
        if cli.is_none()
            && let Some(pid) = window.pid
            && let Some(gui) = (self.executable)(pid).filter(|g| !g.is_empty())
        {
            let windows = self.system.starts_with("win");
            let name = if windows { "wezterm.exe" } else { "wezterm" };
            cli = (self.which)(&sibling(&gui, name, windows));
        }
        let cli = cli?;
        if self.cli.len() > 64 {
            self.cli.clear();
        }
        self.cli.insert(key, cli.clone());
        Some(cli)
    }

    fn focused_pane(clients: &[WezClient], window: &WindowRef) -> Option<i64> {
        let known: Vec<&WezClient> = clients
            .iter()
            .filter(|c| c.focused_pane_id.is_some())
            .collect();
        let own: Vec<&WezClient> = known
            .iter()
            .copied()
            .filter(|c| window.pid.is_some_and(|pid| c.pid == i64::from(pid)))
            .collect();
        let pool = if own.is_empty() { known } else { own };
        // The most recently used client; the first of equals, like Python's min().
        pool.iter()
            .fold(None::<&&WezClient>, |best, c| match best {
                Some(b) if b.idle_s <= c.idle_s => Some(b),
                _ => Some(c),
            })
            .and_then(|c| c.focused_pane_id)
    }

    fn visible_tab(panes: &[WezPane], focused: Option<i64>) -> Vec<WezPane> {
        let key = match focused {
            Some(focused) => {
                let Some(owner) = panes.iter().find(|p| p.pane_id == focused) else {
                    return Vec::new();
                };
                (owner.window_id, owner.tab_id)
            }
            None => {
                let mut tabs: Vec<(i64, i64)> =
                    panes.iter().map(|p| (p.window_id, p.tab_id)).collect();
                tabs.sort_unstable();
                tabs.dedup();
                match tabs[..] {
                    [only] => only,
                    _ => return Vec::new(), // which tab is on screen cannot be told
                }
            }
        };
        panes
            .iter()
            .filter(|p| (p.window_id, p.tab_id) == key)
            .cloned()
            .collect()
    }
}

impl PaneProvider for WezTermProvider {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn applies(&mut self, app: &AppIdentity) -> bool {
        GUI_PROCESSES.contains(&app.process.as_str())
    }

    fn detect(
        &mut self,
        window: &WindowRef,
        _app: &AppIdentity,
    ) -> Result<Option<PaneSnapshot>, PaneError> {
        let Some(area) = (self.client_rect)(window).filter(|a| a.w > 0 && a.h > 0) else {
            return Ok(None);
        };
        let Some(cli) = self.cli_for(window) else {
            return Ok(None);
        };
        let argv = |verb: &str| -> Vec<String> {
            [&cli, "cli", "--no-auto-start", verb, "--format", "json"]
                .map(String::from)
                .to_vec()
        };
        let listed = self.run.run(&argv("list"), self.timeout)?;
        if !listed.ok() {
            return Ok(None); // no GUI or mux to talk to
        }
        let panes = parse_panes(&listed.stdout)?;
        let clients = self.run.run(&argv("list-clients"), self.timeout)?;
        let clients = if clients.ok() {
            parse_clients(&clients.stdout)?
        } else {
            Vec::new()
        };
        let focused = Self::focused_pane(&clients, window);
        let mut tab = Self::visible_tab(&panes, focused);
        if tab.len() < 2 || tab.iter().any(|p| p.zoomed) {
            return Ok(None);
        }
        if let Some(focused) = focused {
            // The GUI's focus is the truth; is_active is per tab.
            for p in &mut tab {
                p.active = p.pane_id == focused;
            }
        }
        Ok(Some(PaneSnapshot {
            window_handle: window.handle.clone(),
            panes: layout_panes(&tab, &area, Self::NAME),
            taken_at: 0.0,
        }))
    }

    fn focus(&mut self, window: &WindowRef, pane: &Pane) -> Result<bool, PaneError> {
        let (Some(cli), PaneId::Int(pane_id)) = (self.cli.get(&window_key(window)), &pane.id)
        else {
            return Ok(false);
        };
        let argv = [
            cli.as_str(),
            "cli",
            "--no-auto-start",
            "activate-pane",
            "--pane-id",
            &pane_id.to_string(),
        ]
        .map(String::from);
        Ok(self.run.run(&argv, self.timeout)?.ok())
    }
}

#[cfg(test)]
#[path = "wezterm_tests.rs"]
mod tests;
