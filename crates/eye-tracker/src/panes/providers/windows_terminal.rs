//! Windows Terminal panes, through UI Automation (Windows only).
//!
//! Windows Terminal has no command-line interface that lists or focuses panes,
//! but every pane is a `TermControl` element in its UI Automation tree. Looking
//! up the `TermControl` descendants of the window gives each pane's bounding
//! rectangle (physical pixels in global coordinates, like the rest of the app on
//! Windows, which is per-monitor DPI aware) and whether it has the keyboard focus;
//! `SetFocus` on another one makes it the active pane. Nothing is typed or
//! clicked, and neither names nor text of any element are read.
//!
//! Only the tab on screen has `TermControl` elements in the tree; off-screen
//! ones (and empty rectangles) are ignored all the same.
//!
//! Pane ids are UI Automation runtime ids (`GetRuntimeId`), which stay the same
//! for as long as the pane exists, so the pane decider can follow a pane from one
//! snapshot to the next. If a runtime id cannot be read, the pane's rectangle
//! stands in for it ([`PaneId::Rect`]): stable while the layout is.
//!
//! Both calls run only while the window is in the foreground: Windows Terminal
//! reports the keyboard focus only then, and `SetFocus` on a background window
//! could bring it to the front. The controller only asks about the foreground
//! window anyway, but the pane worker answers a little later.
//!
//! One provider serves one window: when Windows Terminal shows two or more panes
//! they are used and tmux running inside one of them is not looked at; with a
//! single Windows Terminal pane the tmux provider gets its turn (see
//! `registry`). Following tmux panes nested inside Windows Terminal panes is
//! left for later.

use std::collections::HashSet;

use crate::platform::win_uia::UiAutomation;
use crate::platform::win_uia::{UiaElement, UiaError};
use crate::types::{AppIdentity, Rect, WindowRef};

use super::super::types::{ClientRect, Pane, PaneError, PaneId, PaneProvider, PaneSnapshot};

/// Process names (see `platform::base::process_basename`) of Windows Terminal.
pub const TERMINAL_PROCESSES: [&str; 1] = ["windowsterminal"];
/// Class of Windows Terminal's top-level windows.
pub const WINDOW_CLASS: &str = "CASCADIA_HOSTING_WINDOW_CLASS";
/// UI Automation class name of one terminal pane.
pub const PANE_CLASS: &str = "TermControl";

/// The UI Automation calls the provider makes ([`UiAutomation`](crate::platform::win_uia::UiAutomation)).
///
/// The tests answer them with a fake, so the provider runs on every OS.
pub trait UiaClient: Send {
    fn find(&self, hwnd: isize, class_name: &str) -> Result<Vec<UiaElement>, UiaError>;

    /// Give the keyboard focus to the first descendant `matches` accepts; `guard` is asked right
    /// before and refuses by returning `false`.
    fn focus(
        &self,
        hwnd: isize,
        class_name: &str,
        matches: &dyn Fn(&UiaElement) -> bool,
        guard: Option<&dyn Fn() -> Result<bool, UiaError>>,
    ) -> Result<bool, UiaError>;

    fn foreground_window(&self) -> Result<Option<isize>, UiaError>;

    fn close(&self);
}

/// The pane id of a `TermControl`: its runtime id, else its rectangle.
pub fn pane_id(element: &UiaElement) -> Option<PaneId> {
    if let Some(runtime_id) = element.runtime_id.as_ref().filter(|ids| !ids.is_empty()) {
        return Some(PaneId::Runtime(runtime_id.clone()));
    }
    element.rect.map(PaneId::Rect)
}

/// `rect` cut to `bounds`; `None` if nothing is left.
pub(super) fn clip(rect: Rect, bounds: Option<Rect>) -> Option<Rect> {
    let Some(bounds) = bounds else {
        return Some(rect);
    };
    let (x0, y0) = (rect.x.max(bounds.x), rect.y.max(bounds.y));
    let (x1, y1) = (
        rect.right().min(bounds.right()),
        rect.bottom().min(bounds.bottom()),
    );
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(Rect::new(x0, y0, x1 - x0, y1 - y0))
}

/// The window handle as an HWND, if it is one.
pub(super) fn hwnd(window: &WindowRef) -> Option<isize> {
    match window.handle {
        crate::types::WindowHandle::Hwnd(handle) if handle > 0 => Some(handle),
        _ => None,
    }
}

/// `PaneError` for a failed UI Automation call.
pub(super) fn pane_error(error: UiaError) -> PaneError {
    PaneError::new(format!("UI Automation: {error}"))
}

/// Panes of the tab a Windows Terminal window shows.
pub struct WindowsTerminalProvider {
    client_rect: ClientRect,
    client: Option<Box<dyn UiaClient>>,
}

impl WindowsTerminalProvider {
    pub const NAME: &'static str = "windows_terminal";

    /// Creating the provider touches nothing: UI Automation is used only on the worker thread,
    /// when asked.
    pub fn new(client_rect: ClientRect) -> Self {
        Self {
            client_rect,
            client: None,
        }
    }

    pub fn with_client(mut self, client: Box<dyn UiaClient>) -> Self {
        self.client = Some(client);
        self
    }

    /// The UI Automation client, created on first use (on the worker thread).
    fn uia(&mut self) -> &dyn UiaClient {
        let client = self.client.get_or_insert_with(|| {
            let client: Box<dyn UiaClient> = Box::new(UiAutomation::new());
            client
        });
        &**client
    }
}

impl PaneProvider for WindowsTerminalProvider {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn applies(&mut self, app: &AppIdentity) -> bool {
        TERMINAL_PROCESSES.contains(&app.process.trim().to_lowercase().as_str())
            && app.app_id == WINDOW_CLASS
    }

    fn detect(
        &mut self,
        window: &WindowRef,
        _app: &AppIdentity,
    ) -> Result<Option<PaneSnapshot>, PaneError> {
        let Some(handle) = hwnd(window) else {
            return Ok(None);
        };
        let client_rect = self.client_rect.clone();
        let client = self.uia();
        let elements = (|| -> Result<Option<Vec<UiaElement>>, UiaError> {
            if client.foreground_window()? != Some(handle) {
                return Ok(None);
            }
            client.find(handle, PANE_CLASS).map(Some)
        })()
        .map_err(pane_error)?;
        let Some(elements) = elements else {
            return Ok(None);
        };
        let bounds = client_rect(window).or(window.rect);
        let mut panes: Vec<Pane> = Vec::new();
        let mut seen: HashSet<PaneId> = HashSet::new();
        for element in &elements {
            let Some(element_rect) = element.rect.filter(|_| !element.offscreen) else {
                continue;
            };
            let (Some(rect), Some(ident)) = (clip(element_rect, bounds), pane_id(element)) else {
                continue;
            };
            if !seen.insert(ident.clone()) {
                continue;
            }
            panes.push(Pane::new(
                ident,
                rect,
                element.has_keyboard_focus,
                Self::NAME,
            ));
        }
        if panes.len() < 2 {
            return Ok(None);
        }
        panes.sort_by_key(|p| (p.rect.y, p.rect.x));
        Ok(Some(PaneSnapshot {
            window_handle: window.handle.clone(),
            panes,
            taken_at: 0.0,
        }))
    }

    fn focus(&mut self, window: &WindowRef, pane: &Pane) -> Result<bool, PaneError> {
        let Some(handle) = hwnd(window) else {
            return Ok(false);
        };
        if pane.provider != Self::NAME {
            return Ok(false);
        }
        let client = self.uia();
        let matches = |element: &UiaElement| {
            !element.offscreen && pane_id(element).as_ref() == Some(&pane.id)
        };
        // Asked again right before SetFocus: the walk takes a moment.
        let still_in_front =
            || -> Result<bool, UiaError> { Ok(client.foreground_window()? == Some(handle)) };
        (|| -> Result<bool, UiaError> {
            if client.foreground_window()? != Some(handle) {
                return Ok(false); // SetFocus could bring a background window to the front
            }
            client.focus(handle, PANE_CLASS, &matches, Some(&still_in_front))
        })()
        .map_err(pane_error)
    }

    /// Release the UI Automation client (on the pane worker's thread).
    fn close(&mut self) {
        if let Some(client) = self.client.take() {
            client.close();
        }
    }
}

/// The provider for `registry::default_providers`.
#[cfg(windows)]
pub fn create(
    client_rect: ClientRect,
    _runner: std::sync::Arc<dyn super::command::CommandRunner>,
) -> Option<Box<dyn PaneProvider>> {
    Some(Box::new(WindowsTerminalProvider::new(client_rect)))
}

#[cfg(test)]
#[path = "windows_terminal_tests.rs"]
mod tests;
