//! Value types shared by the split-pane providers, worker, decider and controller.

use std::sync::Arc;

use thiserror::Error;

use crate::types::{AppIdentity, Rect, WindowHandle, WindowRef};

/// A provider could not talk to its terminal (timeout, unreadable output, ...).
///
/// Counts as a failure of the provider; "this window has no panes of mine" is
/// not an error but `Ok(None)` from [`PaneProvider::detect`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct PaneError(pub String);

impl PaneError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// A provider's own identifier of a pane: tmux `"%3"`, a WezTerm pane id, a UI Automation
/// runtime id (the Python ids were any hashable value).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PaneId {
    Str(String),
    Int(i64),
    /// A UI Automation runtime id.
    Runtime(Vec<i32>),
    /// The pane's rectangle, when a runtime id cannot be read (the Python `("rect", x, y, w, h)`).
    Rect(Rect),
}

impl From<&str> for PaneId {
    fn from(value: &str) -> Self {
        Self::Str(value.to_string())
    }
}

impl From<String> for PaneId {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

impl From<i64> for PaneId {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl std::fmt::Display for PaneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Str(s) => f.write_str(s),
            Self::Int(i) => write!(f, "{i}"),
            Self::Runtime(ids) => write!(f, "{ids:?}"),
            Self::Rect(r) => write!(f, "rect({}, {}, {}, {})", r.x, r.y, r.w, r.h),
        }
    }
}

/// One split pane of a terminal window.
///
/// `id` is the provider's own identifier (tmux `"%3"`, a WezTerm pane id); `rect` is where the
/// pane is on the screen, in the global coordinates of [`crate::types`]. Titles, commands and
/// working directories are never part of it.
#[derive(Debug, Clone, PartialEq)]
pub struct Pane {
    pub id: PaneId,
    pub rect: Rect,
    pub focused: bool,
    pub provider: String,
}

impl Pane {
    pub fn new(
        id: impl Into<PaneId>,
        rect: Rect,
        focused: bool,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            rect,
            focused,
            provider: provider.into(),
        }
    }
}

/// The panes of one window as a provider saw them at `taken_at` (monotonic s).
#[derive(Debug, Clone, PartialEq)]
pub struct PaneSnapshot {
    pub window_handle: WindowHandle,
    pub panes: Vec<Pane>,
    pub taken_at: f64,
}

impl PaneSnapshot {
    /// The pane with keyboard focus (`None` if the provider could not tell).
    pub fn focused(&self) -> Option<&Pane> {
        self.panes.iter().find(|p| p.focused)
    }

    pub fn provider(&self) -> Option<&str> {
        self.panes.first().map(|p| p.provider.as_str())
    }

    pub fn pane(&self, pane_id: &PaneId) -> Option<&Pane> {
        self.panes.iter().find(|p| p.id == *pane_id)
    }

    /// A copy in which `pane_id` has the focus (after we gave it focus).
    pub fn with_focus(&self, pane_id: &PaneId) -> PaneSnapshot {
        let panes = self
            .panes
            .iter()
            .map(|p| Pane::new(p.id.clone(), p.rect, p.id == *pane_id, p.provider.clone()))
            .collect();
        PaneSnapshot {
            window_handle: self.window_handle.clone(),
            panes,
            taken_at: self.taken_at,
        }
    }
}

/// `PlatformServices::window_client_rect`: the content area of a window, onto which providers
/// map their panes.
pub type ClientRect = Arc<dyn Fn(&WindowRef) -> Option<Rect> + Send + Sync>;

/// Finds and focuses the split panes of one kind of terminal.
///
/// Providers run on the pane worker's thread (`worker`), never on the GUI thread, and may block
/// for at most their command timeout. A provider that holds system resources releases them in
/// [`close`](Self::close); the worker calls it on its thread when the provider is replaced or the
/// worker ends. They must never synthesise input: they talk to the terminal or multiplexer
/// through its own command-line interface or an accessibility API (UI Automation).
pub trait PaneProvider: Send {
    /// Short name (`"tmux"`) used in logs, `status` and the trace.
    fn name(&self) -> &str;

    /// Whether windows of `app` may have panes of this kind. Must be cheap.
    fn applies(&mut self, app: &AppIdentity) -> bool;

    /// The panes of the window, `None` if it has fewer than two of this kind (then the next
    /// provider may look: a single WezTerm pane can run tmux).
    ///
    /// Fails with [`PaneError`] when the terminal could not be asked.
    fn detect(
        &mut self,
        window: &WindowRef,
        app: &AppIdentity,
    ) -> Result<Option<PaneSnapshot>, PaneError>;

    /// Give `pane` the keyboard focus inside its window; `false` if that failed.
    fn focus(&mut self, window: &WindowRef, pane: &Pane) -> Result<bool, PaneError>;

    /// Release system resources (nothing to do by default).
    fn close(&mut self) {}
}
