//! tmux panes, through the `tmux` command (also inside WSL on Windows).
//!
//! Detection asks the tmux server which clients are attached (`list-clients`)
//! and picks the one running in the focused terminal window: a client whose
//! process descends from the window's process. Inside WSL the Linux process ids
//! cannot be matched to Windows ones, so a WSL server is only used when the
//! window's process tree contains WSL and exactly one client is attached. The
//! panes of that client's current window (`list-panes`) are then laid out over
//! the terminal's content area in proportion to their cell positions. Focus
//! moves with `select-pane`. Only the default server socket is used; titles, commands
//! and working directories are never asked for.
//!
//! The cell grid is assumed to fill the content area; a terminal that draws its
//! own tab bar or header bar there shifts the panes by that bar's height, which
//! the precision gate in the decider absorbs for panes of a useful size.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::types::{AppIdentity, Rect, WindowRef};

use super::super::types::{ClientRect, Pane, PaneError, PaneProvider, PaneSnapshot};
use super::command::{
    COMMAND_TIMEOUT_S, CommandRunner, Which, WindowKey, descendant_names, parent_pids, which,
    window_key,
};

/// Terminal emulators whose windows may run a tmux client (process names).
pub const TERMINAL_PROCESSES: [&str; 13] = [
    "windowsterminal",
    "wezterm-gui",
    "alacritty",
    "kitty",
    "gnome-terminal-server",
    "konsole",
    "xterm",
    "iterm2",
    "terminal",
    "foot",
    "ghostty",
    "tilix",
    "xfce4-terminal",
];

/// Processes that show a terminal window runs WSL.
pub const WSL_PROCESSES: [&str; 2] = ["wsl", "wslhost"];

pub const CLIENT_FORMAT: &str = "#{client_pid}\t#{client_activity}\t#{window_id}\t#{client_width}\t#{client_height}\t#{status-position}";
pub const PANE_FORMAT: &str = "#{pane_id} #{pane_left} #{pane_top} #{pane_width} #{pane_height} #{pane_active} #{window_width} #{window_height} #{window_zoomed_flag}";

/// How long a window's "runs WSL" answer is reused (walking a process tree is slow).
pub const WSL_CHECK_TTL_S: f64 = 10.0;

/// `prefix` followed by digits only, like the regexes `^%\d+$` and `^@\d+$`.
fn is_id(text: &str, prefix: char) -> bool {
    text.strip_prefix(prefix)
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

fn is_pane_id(text: &str) -> bool {
    is_id(text, '%')
}

fn is_window_id(text: &str) -> bool {
    is_id(text, '@')
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TmuxClient {
    pub pid: u32,
    pub activity: i64,
    pub window_id: String,
    pub width: i32,
    pub height: i32,
    pub status_top: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TmuxPane {
    pub pane_id: String,
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
    pub active: bool,
    pub window_width: i32,
    pub window_height: i32,
    pub zoomed: bool,
}

fn parse_client(line: &str) -> Option<TmuxClient> {
    let parts: Vec<&str> = line.split('\t').collect();
    if parts.len() != 6 || !is_window_id(parts[2]) {
        return None;
    }
    let activity = if parts[1].is_empty() {
        0
    } else {
        parts[1].trim().parse().ok()?
    };
    Some(TmuxClient {
        pid: parts[0].trim().parse().ok()?,
        activity,
        window_id: parts[2].to_string(),
        width: parts[3].trim().parse().ok()?,
        height: parts[4].trim().parse().ok()?,
        status_top: parts[5].trim() == "top",
    })
}

/// `list-clients -F CLIENT_FORMAT` output; malformed lines are skipped.
pub fn parse_clients(text: &str) -> Vec<TmuxClient> {
    text.lines().filter_map(parse_client).collect()
}

fn parse_pane(line: &str) -> Option<TmuxPane> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 9 || !is_pane_id(parts[0]) {
        return None;
    }
    let numbers: Vec<i32> = parts[1..]
        .iter()
        .map(|v| v.parse().ok())
        .collect::<Option<_>>()?;
    let [left, top, width, height, active, ww, wh, zoomed] = numbers[..] else {
        return None;
    };
    if width <= 0 || height <= 0 || ww <= 0 || wh <= 0 {
        return None;
    }
    Some(TmuxPane {
        pane_id: parts[0].to_string(),
        left,
        top,
        width,
        height,
        active: active == 1,
        window_width: ww,
        window_height: wh,
        zoomed: zoomed == 1,
    })
}

/// `list-panes -F PANE_FORMAT` output. Fails with [`PaneError`] when the output is not empty but
/// no line can be read (another tmux format).
pub fn parse_panes(text: &str) -> Result<Vec<TmuxPane>, PaneError> {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let panes: Vec<TmuxPane> = lines.iter().filter_map(|l| parse_pane(l)).collect();
    if !lines.is_empty() && panes.is_empty() {
        return Err(PaneError::new("tmux list-panes output could not be read"));
    }
    Ok(panes)
}

/// Map pane cells onto the terminal's content area `area`.
pub fn layout_panes(
    panes: &[TmuxPane],
    client: &TmuxClient,
    area: &Rect,
    provider: &str,
) -> Vec<Pane> {
    let first = &panes[0];
    let cols = client.width.max(first.window_width);
    let rows = client.height.max(first.window_height);
    let top_rows = if client.status_top {
        rows - first.window_height
    } else {
        0
    };
    let cw = f64::from(area.w) / f64::from(cols);
    let ch = f64::from(area.h) / f64::from(rows);

    let span = |start: i32, size: i32, origin: i32, cell: f64| -> (i32, i32) {
        let a = origin + (f64::from(start) * cell).round_ties_even() as i32;
        let b = origin + (f64::from(start + size) * cell).round_ties_even() as i32;
        (a, (b - a).max(1))
    };

    panes
        .iter()
        .map(|p| {
            let (x, w) = span(p.left, p.width, area.x, cw);
            let (y, h) = span(top_rows + p.top, p.height, area.y, ch);
            Pane::new(
                p.pane_id.as_str(),
                Rect::new(x, y, w, h),
                p.active,
                provider,
            )
        })
        .collect()
}

/// Panes of a tmux client running in the focused terminal window.
pub struct TmuxProvider {
    client_rect: ClientRect,
    run: Arc<dyn CommandRunner>,
    which: Which,
    parents: Box<dyn Fn(u32) -> Vec<u32> + Send>,
    descendants: Box<dyn Fn(u32) -> HashSet<String> + Send>,
    /// Python's `sys.platform` values ("win32", "linux", "darwin").
    system: String,
    clock: Box<dyn Fn() -> f64 + Send>,
    timeout: f64,
    tools: HashMap<String, Option<String>>,
    wsl_windows: HashMap<u32, (f64, bool)>,
    /// Command prefix that reached the server of each window's panes.
    prefixes: HashMap<WindowKey, Vec<String>>,
}

impl TmuxProvider {
    pub const NAME: &'static str = "tmux";

    /// A provider that asks the real system (`which`, the process list, the monotonic clock).
    pub fn new(client_rect: ClientRect, runner: Arc<dyn CommandRunner>) -> Self {
        Self {
            client_rect,
            run: runner,
            which: Box::new(which),
            parents: Box::new(parent_pids),
            descendants: Box::new(descendant_names),
            system: crate::panes::registry::current_system().to_string(),
            clock: Box::new(crate::panes::worker::monotonic),
            timeout: COMMAND_TIMEOUT_S,
            tools: HashMap::new(),
            wsl_windows: HashMap::new(),
            prefixes: HashMap::new(),
        }
    }

    fn tool(&mut self, name: &str) -> Option<String> {
        if !self.tools.contains_key(name) {
            let found = (self.which)(name);
            self.tools.insert(name.to_string(), found);
        }
        self.tools[name].clone()
    }

    fn wsl_tool(&mut self) -> Option<String> {
        if !self.system.starts_with("win") {
            return None;
        }
        self.tool("wsl")
    }

    fn client_for(
        &mut self,
        window: &WindowRef,
    ) -> Result<Option<(Vec<String>, TmuxClient)>, PaneError> {
        let tmux = self.tool("tmux");
        if let (Some(tmux), Some(pid)) = (tmux, window.pid) {
            let prefix = vec![tmux];
            let clients = self.list_clients(&prefix)?;
            let mine: Vec<&TmuxClient> = clients
                .iter()
                .filter(|c| (self.parents)(c.pid).contains(&pid))
                .collect();
            // The most recently active one; the first of equals, like Python's max().
            let best = mine
                .iter()
                .fold(None::<&&TmuxClient>, |best, c| match best {
                    Some(b) if b.activity >= c.activity => Some(b),
                    _ => Some(c),
                });
            if let Some(best) = best {
                return Ok(Some((prefix, (*best).clone())));
            }
        }
        let wsl = self.wsl_tool();
        if let (Some(wsl), Some(pid)) = (wsl, window.pid)
            && self.runs_wsl(pid)
        {
            let prefix = vec![wsl, "-e".to_string(), "tmux".to_string()];
            let mut clients = self.list_clients(&prefix)?;
            if clients.len() == 1 {
                return Ok(Some((prefix, clients.remove(0))));
            }
            if !clients.is_empty() {
                log::debug!(
                    "{} tmux clients in WSL; cannot tell which is this window",
                    clients.len()
                );
            }
        }
        Ok(None)
    }

    fn list_clients(&self, prefix: &[String]) -> Result<Vec<TmuxClient>, PaneError> {
        let mut argv = prefix.to_vec();
        argv.extend(["list-clients".into(), "-F".into(), CLIENT_FORMAT.into()]);
        let result = self.run.run(&argv, self.timeout)?;
        if !result.ok() {
            return Ok(Vec::new()); // no server running
        }
        Ok(parse_clients(&result.stdout))
    }

    fn runs_wsl(&mut self, pid: u32) -> bool {
        let now = (self.clock)();
        if let Some((at, runs)) = self.wsl_windows.get(&pid)
            && now - at < WSL_CHECK_TTL_S
        {
            return *runs;
        }
        let runs = (self.descendants)(pid)
            .iter()
            .any(|name| WSL_PROCESSES.contains(&name.as_str()));
        if self.wsl_windows.len() > 32 {
            self.wsl_windows.clear();
        }
        self.wsl_windows.insert(pid, (now, runs));
        runs
    }
}

impl PaneProvider for TmuxProvider {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn applies(&mut self, app: &AppIdentity) -> bool {
        if !TERMINAL_PROCESSES.contains(&app.process.as_str()) {
            return false;
        }
        self.tool("tmux").is_some() || self.wsl_tool().is_some()
    }

    fn detect(
        &mut self,
        window: &WindowRef,
        _app: &AppIdentity,
    ) -> Result<Option<PaneSnapshot>, PaneError> {
        let Some(area) = (self.client_rect)(window).filter(|a| a.w > 0 && a.h > 0) else {
            return Ok(None);
        };
        let Some((prefix, client)) = self.client_for(window)? else {
            self.prefixes.remove(&window_key(window));
            return Ok(None);
        };
        let mut argv = prefix.clone();
        argv.extend([
            "list-panes".into(),
            "-t".into(),
            client.window_id.clone(),
            "-F".into(),
            PANE_FORMAT.into(),
        ]);
        let result = self.run.run(&argv, self.timeout)?;
        if !result.ok() {
            return Ok(None); // the window closed between the two commands
        }
        let panes = parse_panes(&result.stdout)?;
        if panes.len() < 2 || panes.iter().any(|p| p.zoomed) {
            return Ok(None); // nothing to choose between (a zoomed pane fills the window)
        }
        if self.prefixes.len() > 64 {
            self.prefixes.clear();
        }
        self.prefixes.insert(window_key(window), prefix);
        Ok(Some(PaneSnapshot {
            window_handle: window.handle.clone(),
            panes: layout_panes(&panes, &client, &area, Self::NAME),
            taken_at: 0.0,
        }))
    }

    fn focus(&mut self, window: &WindowRef, pane: &Pane) -> Result<bool, PaneError> {
        let pane_id = pane.id.to_string();
        let Some(prefix) = self.prefixes.get(&window_key(window)) else {
            return Ok(false);
        };
        if !is_pane_id(&pane_id) {
            return Ok(false);
        }
        let mut argv = prefix.clone();
        argv.extend(["select-pane".into(), "-t".into(), pane_id]);
        Ok(self.run.run(&argv, self.timeout)?.ok())
    }
}

#[cfg(test)]
#[path = "tmux_tests.rs"]
mod tests;
