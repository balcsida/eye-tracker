//! Side-by-side sessions of desktop apps built on Chromium, through UI Automation.
//!
//! Opt-in (`panes.desktop_apps`) and Windows only. Electron and Chromium
//! applications are on the registry's hard deny-list (`registry`) because
//! asking them about their contents makes them build an accessibility tree, which
//! costs them CPU and memory. With the setting on, exactly the apps that have an
//! [`AppProfile`] here (process name *and* window class) skip the deny-list
//! and are asked by this provider, and by no other. Every other Chromium window
//! (VS Code, Cursor, browsers, apps without a profile) stays denied.
//!
//! # Profiles
//!
//! Class names are the HTML class attribute, split on whitespace and matched token
//! by token (never as substrings). A profile finds its panes in one of two ways:
//!
//! * by a **pane token**: every `Group` whose class has the token is a pane. The
//!   Claude desktop app shows two chat sessions side by side, each a group with the
//!   token `dframe-pane`.
//! * by a **container prefix**: the direct `Group` children of the first group
//!   with a class token starting with the prefix are the panes (CSS-module classes
//!   carry a build hash, so only the prefix is stable). The ChatGPT desktop app
//!   (which also hosts Codex) shows the main conversation and a side chat or side
//!   panel as the children of `_MainContentSurface_<hash>`.
//!
//! Either way a pane must be at least [`AppProfile::min_pane`] pixels in both
//! directions, which leaves out small overlays. Each session has a message box: an
//! `Edit` whose class has the token `ProseMirror`. Names are never used (they
//! are localised, and they are what the user wrote).
//!
//! # The walk (`detect`)
//!
//! The control view of the window is walked step by step (first child, next
//! sibling), never as a whole: a full walk reads ~630 elements of the Claude
//! window in ~0.7 s, ~420 of the ChatGPT window in ~0.33 s.
//!
//! 1. From the window element, depth first through the native views (any control
//!    type except `Document`, at most [`NATIVE_DEPTH`] levels) to the
//!    `Document` whose automation id is `RootWebArea`: the web content. Other
//!    documents are not entered. A window may hold more than one such document
//!    (the Claude app does, the first one nearly empty): they are tried in tree
//!    order until one has panes.
//! 2. From there through `Group` elements only, at most [`WEB_DEPTH`] levels
//!    below the document. A group whose class has the profile's pane token is a
//!    pane: it is not entered. A group with a token starting with the profile's
//!    container prefix is the container: its children (one step each) are the
//!    panes and the walk ends there. A group with one of the profile's skip tokens
//!    (the sidebar) is not entered. Once a group's children included panes, the
//!    walk ends after that group.
//!
//! Every step of the walk (a first-child or next-sibling call) counts against
//! [`NODE_BUDGET`]; when it is used up the walk stops with what it found. On
//! trees shaped like the measured ones, a detection takes about 25 steps and 35
//! property reads for Claude, 15 and 30 for ChatGPT (a focus a few more). Per
//! element only the control type is read, plus the class name of groups and
//! editors and the automation id of documents; of the panes their rectangle,
//! visibility and runtime id. Names, values and text are never read.
//!
//! Panes are the pane groups that are not off-screen and whose rectangle, cut to
//! the window, is at least [`AppProfile::min_pane`] large, at least two of
//! them. The focused pane is the one containing the element with the keyboard
//! focus (`GetFocusedElement`): its centre must lie in the pane and it must not
//! be larger than the pane, so the whole page having the focus means no pane has
//! it. The focused element counts
//! only if it belongs to the app's process. Pane ids are runtime ids, else the
//! pane's rectangle (as for Windows Terminal).
//!
//! Chromium builds its accessibility tree only once a client asks: the first walk
//! often finds only the native views, or the web content without panes. That is
//! "no panes" (`None`), not a failure; the next refresh finds them.
//!
//! # Focus
//!
//! The pane is found again by the same walk, then its message box by a walk of
//! the pane's subtree from the last child backwards (the message box is at the
//! bottom, after a conversation of any length), at most [`COMPOSER_DEPTH`]
//! levels and [`COMPOSER_BUDGET`] elements, not entering `Edit` elements.
//! The first `Edit` with the profile's composer token is the message box; if
//! there is none, the first keyboard-focusable `Edit` seen. `SetFocus` on it
//! moves the keyboard focus into that session. Nothing is typed or clicked.
//!
//! Both calls run only while the window is in the foreground (like Windows
//! Terminal): `SetFocus` on a background window could bring it to the front.
use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::platform::win_uia::{
    UIA_DOCUMENT_CONTROL_TYPE_ID, UIA_EDIT_CONTROL_TYPE_ID, UIA_GROUP_CONTROL_TYPE_ID,
    UiAutomation, UiaError, UiaNode,
};
use crate::types::{AppIdentity, Rect, WindowRef};

use super::super::types::{ClientRect, Pane, PaneError, PaneId, PaneProvider, PaneSnapshot};
use super::windows_terminal::{clip, hwnd, pane_error};

/// Name of the provider (`Pane::provider`, logs, `status`).
pub const PROVIDER_NAME: &str = "desktop_apps";
/// Automation id of the document holding a Chromium window's web content.
pub const ROOT_WEB_AREA: &str = "RootWebArea";
/// Most elements one detection steps to (the window walk of `focus` too).
pub const NODE_BUDGET: usize = 400;
/// Most elements the search for a message box steps to, inside one pane.
pub const COMPOSER_BUDGET: usize = 400;
/// Deepest level below the window searched for the web content document.
pub const NATIVE_DEPTH: usize = 14;
/// Deepest level below the web content document searched for panes.
pub const WEB_DEPTH: usize = 8;
/// Deepest level below a pane searched for its message box.
pub const COMPOSER_DEPTH: usize = 24;

/// The smallest pane (width, height) in pixels unless a profile says otherwise.
pub const DEFAULT_MIN_PANE: (i32, i32) = (200, 200);

/// An [`AppProfile`] with neither or both of `pane_token` and `container_prefix`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("an AppProfile needs exactly one of pane_token, container_prefix")]
pub struct InvalidProfile;

/// How to find the side-by-side sessions of one desktop app.
///
/// Exactly one of `pane_token` and `container_prefix` is set ([`validated`](Self::validated)
/// checks it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppProfile {
    /// Short name used in logs.
    pub name: &'static str,
    /// Process names (`platform::base::process_basename`) of the app.
    pub processes: &'static [&'static str],
    /// Class of the app's top-level windows.
    pub window_class: &'static str,
    /// Class token of the session's message box (an `Edit`).
    pub composer_token: &'static str,
    /// Class token of a `Group` that is one session pane.
    pub pane_token: Option<&'static str>,
    /// Prefix of a class token of the `Group` whose `Group` children are the panes.
    pub container_prefix: Option<&'static str>,
    /// Class tokens of groups never entered while looking for panes.
    pub skip_tokens: &'static [&'static str],
    /// Smallest pane (width, height) in pixels; smaller children are overlays.
    pub min_pane: (i32, i32),
}

impl AppProfile {
    /// `self` if it finds its panes in exactly one way.
    pub fn validated(self) -> Result<Self, InvalidProfile> {
        if self.pane_token.is_some() == self.container_prefix.is_some() {
            return Err(InvalidProfile);
        }
        Ok(self)
    }

    pub fn matches(&self, app: &AppIdentity) -> bool {
        self.processes
            .contains(&app.process.trim().to_lowercase().as_str())
            && app.app_id == self.window_class
    }

    /// Whether a group with these class tokens holds the panes.
    pub fn is_container(&self, tokens: &HashSet<&str>) -> bool {
        match self.container_prefix {
            Some(prefix) if !prefix.is_empty() => tokens.iter().any(|t| t.starts_with(prefix)),
            _ => false,
        }
    }

    pub fn large_enough(&self, rect: Rect) -> bool {
        rect.w >= self.min_pane.0 && rect.h >= self.min_pane.1
    }
}

/// The Claude desktop app (Windows).
pub const CLAUDE: AppProfile = AppProfile {
    name: "claude",
    processes: &["claude"],
    window_class: "Chrome_WidgetWin_1",
    composer_token: "ProseMirror",
    pane_token: Some("dframe-pane"),
    container_prefix: None,
    skip_tokens: &["dframe-sidebar"],
    min_pane: DEFAULT_MIN_PANE,
};

/// The ChatGPT desktop app (Windows), which also hosts Codex.
pub const CHATGPT: AppProfile = AppProfile {
    name: "chatgpt",
    processes: &["chatgpt"],
    window_class: "Chrome_WidgetWin_1",
    composer_token: "ProseMirror",
    pane_token: None,
    container_prefix: Some("_MainContentSurface_"),
    skip_tokens: &["app-shell-left-panel"],
    min_pane: DEFAULT_MIN_PANE,
};

/// Every app with a profile; only these ever skip the registry's deny-list.
pub const PROFILES: [&AppProfile; 2] = [&CLAUDE, &CHATGPT];

/// The profile of `app` (process name and window class), if it has one.
pub fn profile_for(app: Option<&AppIdentity>) -> Option<&'static AppProfile> {
    let app = app?;
    PROFILES.into_iter().find(|p| p.matches(app))
}

/// The whitespace-separated tokens of an element's class name.
pub fn class_tokens(class_name: &str) -> HashSet<&str> {
    class_name.split_whitespace().collect()
}

/// A walk of one window's control view ([`ControlView`](crate::platform::win_uia)).
///
/// Nodes are opaque; they stay valid until [`close`](Self::close).
pub trait ElementView {
    /// The window element.
    fn root(&self) -> Option<UiaNode>;

    fn first_child(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError>;

    fn last_child(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError>;

    fn next_sibling(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError>;

    fn previous_sibling(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError>;

    /// The element with the keyboard focus, on the whole desktop.
    fn focused(&mut self) -> Result<Option<UiaNode>, UiaError>;

    fn control_type(&self, node: UiaNode) -> Result<i32, UiaError>;

    fn process_id(&self, node: UiaNode) -> Result<u32, UiaError>;

    fn class_name(&self, node: UiaNode) -> Result<String, UiaError>;

    fn automation_id(&self, node: UiaNode) -> Result<String, UiaError>;

    fn is_keyboard_focusable(&self, node: UiaNode) -> Result<bool, UiaError>;

    fn is_offscreen(&self, node: UiaNode) -> Result<bool, UiaError>;

    /// Bounding rectangle in physical pixels, global coordinates; `None` if empty.
    fn rect(&self, node: UiaNode) -> Result<Option<Rect>, UiaError>;

    fn runtime_id(&self, node: UiaNode) -> Result<Option<Vec<i32>>, UiaError>;

    /// Give `node` the keyboard focus.
    fn set_focus(&self, node: UiaNode) -> Result<(), UiaError>;

    fn close(&mut self);
}

/// The UI Automation calls the provider makes ([`UiAutomation`]).
///
/// The tests answer them with a fake, so the provider runs on every OS.
pub trait UiaTreeClient: Send {
    fn control_view(&self, hwnd: isize) -> Result<Box<dyn ElementView>, UiaError>;

    fn foreground_window(&self) -> Result<Option<isize>, UiaError>;

    fn close(&self);
}

/// Counts the elements a walk steps to.
pub struct Budget {
    left: usize,
    pub exhausted: bool,
}

impl Budget {
    pub fn new(limit: usize) -> Self {
        Self {
            left: limit,
            exhausted: false,
        }
    }

    fn take(&mut self) -> bool {
        if self.left == 0 {
            self.exhausted = true;
            return false;
        }
        self.left -= 1;
        true
    }
}

/// What the callbacks of the walks get to see and say: `true` stops the walk.
type Visit<'a> =
    dyn FnMut(&mut dyn ElementView, &mut Budget, UiaNode) -> Result<bool, UiaError> + 'a;

/// Calls `visit` for the children of `node` in the control view, one step each, while the
/// budget lasts; `true` if `visit` stopped the walk.
fn each_child(
    view: &mut dyn ElementView,
    node: UiaNode,
    budget: &mut Budget,
    backwards: bool,
    visit: &mut Visit,
) -> Result<bool, UiaError> {
    if !budget.take() {
        return Ok(false);
    }
    let mut child = if backwards {
        view.last_child(node)?
    } else {
        view.first_child(node)?
    };
    while let Some(current) = child {
        if visit(view, budget, current)? {
            return Ok(true);
        }
        if !budget.take() {
            return Ok(false);
        }
        child = if backwards {
            view.previous_sibling(current)?
        } else {
            view.next_sibling(current)?
        };
    }
    Ok(false)
}

/// Calls `found` with the `RootWebArea` documents below `root`, in tree order, one at a time,
/// until it returns `true`.
///
/// A window can hold several (the Claude app has an empty one in front of the
/// one with the sessions), so the caller walks on only while it has found no panes.
pub fn find_documents(
    view: &mut dyn ElementView,
    root: UiaNode,
    budget: &mut Budget,
    found: &mut Visit,
) -> Result<bool, UiaError> {
    fn visit(
        view: &mut dyn ElementView,
        node: UiaNode,
        depth: usize,
        budget: &mut Budget,
        found: &mut Visit,
    ) -> Result<bool, UiaError> {
        each_child(view, node, budget, false, &mut |view, budget, child| {
            if view.control_type(child)? == UIA_DOCUMENT_CONTROL_TYPE_ID {
                if view.automation_id(child)? == ROOT_WEB_AREA {
                    return found(view, budget, child);
                }
                return Ok(false); // another document's web content is not ours to walk
            }
            if depth < NATIVE_DEPTH {
                return visit(view, child, depth + 1, budget, found);
            }
            Ok(false)
        })
    }

    visit(view, root, 1, budget, found)
}

/// The pane groups below the web content `document`, in tree order.
pub fn find_panes(
    view: &mut dyn ElementView,
    document: UiaNode,
    profile: &AppProfile,
    budget: &mut Budget,
) -> Result<Vec<UiaNode>, UiaError> {
    fn visit(
        view: &mut dyn ElementView,
        node: UiaNode,
        depth: usize,
        profile: &AppProfile,
        budget: &mut Budget,
        panes: &mut Vec<UiaNode>,
    ) -> Result<bool, UiaError> {
        let mut here: Vec<UiaNode> = Vec::new();
        let stopped = each_child(view, node, budget, false, &mut |view, budget, child| {
            if view.control_type(child)? != UIA_GROUP_CONTROL_TYPE_ID {
                return Ok(false);
            }
            let class_name = view.class_name(child)?;
            let tokens = class_tokens(&class_name);
            if profile.pane_token.is_some_and(|t| tokens.contains(t)) {
                here.push(child); // a pane: its contents are not walked
                return Ok(false);
            }
            if profile.is_container(&tokens) {
                // Its group children are the panes; their contents are not walked.
                each_child(view, child, budget, false, &mut |view, _, c| {
                    if view.control_type(c)? == UIA_GROUP_CONTROL_TYPE_ID {
                        panes.push(c);
                    }
                    Ok(false)
                })?;
                return Ok(true);
            }
            if tokens.iter().any(|t| profile.skip_tokens.contains(t)) || depth >= WEB_DEPTH {
                return Ok(false);
            }
            visit(view, child, depth + 1, profile, budget, panes)
        })?;
        if stopped {
            return Ok(true);
        }
        if !here.is_empty() {
            panes.extend(here);
            return Ok(true);
        }
        Ok(false)
    }

    let mut panes = Vec::new();
    visit(view, document, 1, profile, budget, &mut panes)?;
    Ok(panes)
}

/// The message box of `pane`: the first `Edit` with the composer token
/// seen from the bottom up, else the first keyboard-focusable `Edit`.
pub fn find_composer(
    view: &mut dyn ElementView,
    pane: UiaNode,
    profile: &AppProfile,
    budget: &mut Budget,
) -> Result<Option<UiaNode>, UiaError> {
    fn visit(
        view: &mut dyn ElementView,
        node: UiaNode,
        depth: usize,
        profile: &AppProfile,
        budget: &mut Budget,
        fallback: &mut Option<UiaNode>,
    ) -> Result<Option<UiaNode>, UiaError> {
        let mut found = None;
        each_child(view, node, budget, true, &mut |view, budget, child| {
            if view.control_type(child)? == UIA_EDIT_CONTROL_TYPE_ID {
                let class_name = view.class_name(child)?;
                if class_tokens(&class_name).contains(profile.composer_token) {
                    found = Some(child);
                    return Ok(true);
                }
                if fallback.is_none() && view.is_keyboard_focusable(child)? {
                    *fallback = Some(child);
                }
                return Ok(false); // an editor's contents are text
            }
            if depth < COMPOSER_DEPTH {
                found = visit(view, child, depth + 1, profile, budget, fallback)?;
                return Ok(found.is_some());
            }
            Ok(false)
        })?;
        Ok(found)
    }

    let mut fallback = None;
    let found = visit(view, pane, 1, profile, budget, &mut fallback)?;
    Ok(found.or(fallback))
}

/// A pane's id: its runtime id, else its (unclipped) rectangle.
pub fn pane_id(runtime_id: Option<&[i32]>, rect: Option<Rect>) -> Option<PaneId> {
    match runtime_id {
        Some(ids) if !ids.is_empty() => Some(PaneId::Runtime(ids.to_vec())),
        _ => rect.map(PaneId::Rect),
    }
}

/// Whether `inner` belongs to `outer`: centre inside, and not larger.
fn within(inner: Rect, outer: Rect) -> bool {
    let (cx, cy) = inner.center();
    outer.contains(cx, cy) && inner.w <= outer.w && inner.h <= outer.h
}

/// The panes of a window and the rectangle of the focused element.
type Detected = (Vec<Found>, Option<Rect>);

struct Found {
    node: UiaNode,
    id: PaneId,
    rect: Rect,
}

/// Side-by-side sessions of the desktop apps in [`PROFILES`].
pub struct ChromiumAppProvider {
    client_rect: ClientRect,
    client: Option<Box<dyn UiaTreeClient>>,
    profiles: Vec<&'static AppProfile>,
    /// The profile each window was detected with (`focus` gets no app).
    window_profiles: HashMap<(isize, Option<u32>), &'static AppProfile>,
}

impl ChromiumAppProvider {
    /// Creating the provider touches nothing: UI Automation is used only on the worker thread,
    /// when asked.
    pub fn new(client_rect: ClientRect) -> Self {
        Self {
            client_rect,
            client: None,
            profiles: PROFILES.to_vec(),
            window_profiles: HashMap::new(),
        }
    }

    pub fn with_client(mut self, client: Box<dyn UiaTreeClient>) -> Self {
        self.client = Some(client);
        self
    }

    pub fn with_profiles(mut self, profiles: Vec<&'static AppProfile>) -> Self {
        self.profiles = profiles;
        self
    }

    fn profile(&self, app: &AppIdentity) -> Option<&'static AppProfile> {
        self.profiles.iter().copied().find(|p| p.matches(app))
    }

    /// The UI Automation client, created on first use (on the worker thread).
    fn uia(&mut self) -> &dyn UiaTreeClient {
        let client = self.client.get_or_insert_with(|| {
            let client: Box<dyn UiaTreeClient> = Box::new(UiAutomation::new());
            client
        });
        &**client
    }
}

/// The visible panes of the window (any number), in tree order.
fn panes(
    view: &mut dyn ElementView,
    client_rect: &ClientRect,
    window: &WindowRef,
    profile: &AppProfile,
) -> Result<Vec<Found>, UiaError> {
    let Some(root) = view.root() else {
        return Ok(Vec::new());
    };
    let mut budget = Budget::new(NODE_BUDGET);
    let mut nodes: Vec<UiaNode> = Vec::new();
    let mut saw_document = false;
    find_documents(view, root, &mut budget, &mut |view, budget, document| {
        saw_document = true;
        nodes = find_panes(view, document, profile, budget)?;
        Ok(!nodes.is_empty())
    })?;
    if !saw_document {
        log::debug!("No web content in the {} window (yet)", profile.name);
        return Ok(Vec::new());
    }
    if budget.exhausted {
        log::debug!(
            "Walk of the {} window stopped after {NODE_BUDGET} elements",
            profile.name
        );
    }
    let bounds = client_rect(window).or(window.rect);
    let mut found = Vec::new();
    let mut seen: HashSet<PaneId> = HashSet::new();
    for node in nodes {
        if view.is_offscreen(node)? {
            continue;
        }
        let Some(rect) = view.rect(node)? else {
            continue;
        };
        let Some(clipped) = clip(rect, bounds).filter(|c| profile.large_enough(*c)) else {
            continue;
        };
        let Some(id) = pane_id(view.runtime_id(node)?.as_deref(), Some(rect)) else {
            continue;
        };
        if !seen.insert(id.clone()) {
            continue;
        }
        found.push(Found {
            node,
            id,
            rect: clipped,
        });
    }
    Ok(found)
}

/// Rectangle of the element with the keyboard focus, if it is the app's.
fn focused_rect(view: &mut dyn ElementView, window: &WindowRef) -> Result<Option<Rect>, UiaError> {
    let Some(element) = view.focused()? else {
        return Ok(None);
    };
    let mut expected = window.pid;
    if expected.is_none()
        && let Some(root) = view.root()
    {
        expected = Some(view.process_id(root)?);
    }
    match expected {
        Some(expected) if expected != 0 && view.process_id(element)? == expected => {
            view.rect(element)
        }
        _ => Ok(None),
    }
}

impl PaneProvider for ChromiumAppProvider {
    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn applies(&mut self, app: &AppIdentity) -> bool {
        self.profile(app).is_some()
    }

    fn detect(
        &mut self,
        window: &WindowRef,
        app: &AppIdentity,
    ) -> Result<Option<PaneSnapshot>, PaneError> {
        let (Some(handle), Some(profile)) = (hwnd(window), self.profile(app)) else {
            return Ok(None);
        };
        let client_rect = self.client_rect.clone();
        let client = self.uia();
        let detected = (|| -> Result<Option<Detected>, UiaError> {
            if client.foreground_window()? != Some(handle) {
                return Ok(None);
            }
            let mut view = client.control_view(handle)?;
            let result = (|| {
                let found = panes(view.as_mut(), &client_rect, window, profile)?;
                if found.len() < 2 {
                    return Ok(None);
                }
                let focus_rect = focused_rect(view.as_mut(), window)?;
                Ok(Some((found, focus_rect)))
            })();
            view.close();
            result
        })()
        .map_err(pane_error)?;
        let Some((found, focus_rect)) = detected else {
            return Ok(None);
        };
        self.window_profiles = HashMap::from([((handle, window.pid), profile)]);
        let mut panes: Vec<Pane> = found
            .into_iter()
            .map(|f| {
                let focused = focus_rect.is_some_and(|r| within(r, f.rect));
                Pane::new(f.id, f.rect, focused, PROVIDER_NAME)
            })
            .collect();
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
        if pane.provider != PROVIDER_NAME {
            return Ok(false);
        }
        let Some(profile) = self.window_profiles.get(&(handle, window.pid)).copied() else {
            return Ok(false);
        };
        let client_rect = self.client_rect.clone();
        let client = self.uia();
        (|| -> Result<bool, UiaError> {
            if client.foreground_window()? != Some(handle) {
                return Ok(false); // SetFocus could bring a background window to the front
            }
            let mut view = client.control_view(handle)?;
            let result = (|| {
                let found = panes(view.as_mut(), &client_rect, window, profile)?;
                let Some(target) = found.iter().find(|f| f.id == pane.id) else {
                    return Ok(false);
                };
                let mut budget = Budget::new(COMPOSER_BUDGET);
                let Some(composer) =
                    find_composer(view.as_mut(), target.node, profile, &mut budget)?
                else {
                    log::debug!("No message box found in the {} pane", profile.name);
                    return Ok(false);
                };
                if client.foreground_window()? != Some(handle) {
                    return Ok(false); // another window came to the front during the walk
                }
                view.set_focus(composer)?;
                Ok(true)
            })();
            view.close();
            result
        })()
        .map_err(pane_error)
    }

    /// Release the UI Automation client (on the pane worker's thread).
    fn close(&mut self) {
        self.window_profiles.clear();
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
    Some(Box::new(ChromiumAppProvider::new(client_rect)))
}

#[cfg(test)]
#[path = "chromium_apps_tests.rs"]
mod tests;
