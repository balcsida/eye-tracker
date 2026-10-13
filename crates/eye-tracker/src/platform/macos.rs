//! macOS integration through AppKit, Quartz and the Accessibility API.
//!
//! # Design
//!
//! * **Coordinates.** Quartz events, the Accessibility (AX) API and
//!   `CGWindowListCopyWindowInfo` all use global display points with a top-left origin, which
//!   are the coordinates this code base uses on macOS (see [`crate::types`]), so nothing is
//!   converted.
//! * **Permissions.** Idle time, lock state, cursor warping and the on-screen window list need
//!   no permission. Window-level focus needs Accessibility; without it the platform still works
//!   at *application* level (the [`WindowHandle::Ax`] element is `0`, rectangles come from the
//!   window list), so focus-follows-gaze degrades instead of failing.
//! * **Permissions after an update.** Release builds are signed ad hoc, so every build has a new
//!   code identity. macOS keeps showing the old build as allowed under Privacy & Security ›
//!   Accessibility, yet `AXIsProcessTrusted()` is `false` for the new one: the entry must be
//!   removed with "−" and the app added again. [`MacPlatform::accessibility_status`] reports
//!   this case as `"stale"` (it remembers which build was last trusted) so the UI can say so.
//!   `eye-tracker-cli`, the bundle's second executable, identifies the build by the app's
//!   executable as well, but its own trust belongs to the terminal that started it: it never
//!   records a grant and reports a missing one as unknown.
//! * **Hung apps.** Accessibility calls block until the target app answers. An app that does not
//!   answer in time is left alone for a few seconds (window list and application-level handles
//!   only), so polling it cannot stall the UI.
//! * **App Nap.** A menu-bar app without visible windows is a candidate for App Nap, which
//!   throttles its timers by seconds. [`PlatformServices::set_accessory_app`] therefore also opts out
//!   of it ([`MacPlatform::set_background_activity`]).
//! * **Frameworks.** Every framework call goes through a small trait ([`Quartz`], [`AppKit`],
//!   [`Accessibility`], [`CoreFoundation`], [`Foundation`], [`CameraAuthorization`], [`Host`])
//!   with the real implementation in [`system`]. A platform whose framework is `None` degrades
//!   to `None`/`false`, as the Python did when pyobjc or a symbol was missing, and the tests
//!   plug in fakes.
//!
//! [`WindowHandle::Ax`] is `{ pid, element }`: `element` is the address of the process's
//! `AXUIElement` (`0` for an application-level handle). The real implementation keeps every
//! element it hands out retained for the life of the process (equal elements share one entry), so
//! the address stays valid for as long as a [`WindowRef`] can hold it.

use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use serde_json::json;

use super::base::PlatformServices;
use crate::types::{AppIdentity, Rect, WindowHandle, WindowRef};

pub const LOGIN_FRAMEWORK: &str =
    "/System/Library/PrivateFrameworks/login.framework/Versions/Current/login";
pub const AVFOUNDATION_FRAMEWORK: &str = "/System/Library/Frameworks/AVFoundation.framework";

const PERMISSION_URLS: [(&str, &str); 2] = [
    (
        "accessibility",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    ),
    (
        "camera",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
    ),
];

// Framework constants.
const CG_EVENT_SOURCE_STATE_COMBINED: i32 = 0; // kCGEventSourceStateCombinedSessionState
const CG_ANY_INPUT_EVENT_TYPE: u32 = 0xFFFF_FFFF; // kCGAnyInputEventType, i.e. (CGEventType)~0
const CG_EVENT_KEY_DOWN: u32 = 10; // kCGEventKeyDown
const CG_WINDOW_LIST_ON_SCREEN_ONLY: u32 = 1 << 0;
const CG_WINDOW_LIST_EXCLUDE_DESKTOP: u32 = 1 << 4;
const CG_NULL_WINDOW_ID: u32 = 0;
const AX_SUCCESS: i32 = 0;
/// `kAXErrorCannotComplete`: the app did not answer in time.
const AX_ERROR_CANNOT_COMPLETE: i32 = -25204;
const NS_ACTIVATE_IGNORING_OTHER_APPS: u64 = 1 << 1;
const NS_ACTIVATION_POLICY_ACCESSORY: i64 = 1;
/// `NSActivityUserInitiatedAllowingIdleSystemSleep`: no App Nap and no timer coalescing, while
/// the system and the displays may still sleep when idle.
const NS_ACTIVITY_USER_INITIATED_ALLOWING_IDLE_SYSTEM_SLEEP: u64 = 0x00EF_FFFF;
const ACTIVITY_REASON: &str = "Eye tracking follows the camera in real time";
const AV_MEDIA_TYPE_VIDEO: &str = "vide"; // value of AVMediaTypeVideo

/// Default AX messaging timeout is 6 s; a hung app must not freeze our UI thread.
const AX_TIMEOUT_S: f64 = 0.5;
/// Tighter timeout for the frontmost app's focused window, which is polled.
const AX_APP_TIMEOUT_S: f64 = 0.25;
/// How long an app that did not answer an AX request is left alone.
const AX_UNRESPONSIVE_S: f64 = 5.0;
const TRUST_TTL_S: f64 = 5.0;
const TOOL_TIMEOUT_S: f64 = 5.0;
/// An AX frame and a window-list frame this close (points) are the same window.
const FRAME_TOLERANCE: i32 = 2;
/// Remembers which build Accessibility was granted to (see `accessibility_status`).
const ACCESSIBILITY_STATE_FILE: &str = "macos-accessibility.json";
/// Name suffix of the bundle's command-line executable (`eye-tracker-cli`).
const CLI_SUFFIX: &str = "-cli";

// ---------------------------------------------------------------------------
// the frameworks (tests replace these)
// ---------------------------------------------------------------------------

/// An Accessibility element (`AXUIElementRef`) as the address the backend gave out; `0` is none.
type Element = usize;

/// What `CGSessionCopyCurrentDictionary()` says about the session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct SessionInfo {
    /// `CGSSessionScreenIsLocked`
    screen_locked: bool,
    /// `kCGSSessionOnConsoleKey`, when present.
    on_console: Option<bool>,
}

/// One entry of `CGWindowListCopyWindowInfo`; missing keys have their documented defaults.
#[derive(Debug, Clone, PartialEq)]
struct CgWindow {
    /// `kCGWindowLayer` (default 0).
    layer: i64,
    /// `kCGWindowAlpha` (default 1).
    alpha: f64,
    /// `kCGWindowOwnerPID` (default 0).
    owner_pid: i64,
    /// `kCGWindowBounds` as `(X, Y, Width, Height)`.
    bounds: Option<(f64, f64, f64, f64)>,
}

/// Quartz (CoreGraphics): input idle time, session state, cursor, window list.
trait Quartz: Send + Sync {
    /// `CGEventSourceSecondsSinceLastEventType`; `None` when it cannot be asked.
    fn seconds_since_last_event(&self, state: i32, event_type: u32) -> Option<f64>;
    /// `CGSessionCopyCurrentDictionary`; `None` without a window-server session (e.g. ssh).
    fn session_info(&self) -> Option<SessionInfo>;
    /// `CGWarpMouseCursorPosition`: the `CGError`.
    fn warp_cursor(&self, x: f64, y: f64) -> i32;
    /// `CGAssociateMouseAndMouseCursorPosition`: the `CGError`.
    fn associate_mouse_and_cursor(&self, connected: bool) -> i32;
    /// Height of the main display (`CGDisplayBounds(CGMainDisplayID())`), in points.
    fn main_display_height(&self) -> Option<f64>;
    /// `CGWindowListCopyWindowInfo`, front to back.
    fn window_list(&self, options: u32, relative_to: u32) -> Option<Vec<CgWindow>>;
}

/// A running application as `NSRunningApplication` describes it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RunningApp {
    terminated: bool,
    hidden: bool,
    /// `""` when it has none.
    bundle_id: String,
}

/// AppKit: the frontmost app, running applications, activation.
trait AppKit: Send + Sync {
    /// Process id of `NSWorkspace.frontmostApplication`.
    fn frontmost_pid(&self) -> Option<i32>;
    /// `NSRunningApplication.runningApplication(withProcessIdentifier:)`.
    fn running_application(&self, pid: i32) -> Option<RunningApp>;
    /// `activate(options:)` of the running application with that pid.
    fn activate(&self, pid: i32, options: u64) -> bool;
    /// `NSApplication.shared.setActivationPolicy`; `false` off the main thread.
    fn set_activation_policy(&self, policy: i64) -> bool;
    /// `NSEvent.mouseLocation`: Cocoa points, origin at the bottom left of the main display.
    fn mouse_location(&self) -> Option<(f64, f64)>;
}

/// The two numbers of an unpacked `AXValue`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum AxStruct {
    Point(f64, f64),
    Size(f64, f64),
}

/// An `AXValue` holding a `CGPoint` or `CGSize`.
#[derive(Debug, Clone, PartialEq)]
struct AxValue {
    /// What `AXValueGetValue` returned; `None` when it could not unpack the value.
    unpacked: Option<AxStruct>,
    /// The value's description, which always carries the numbers (`{value = x:24.0 y:38.0 …}`).
    /// Only filled in when `unpacked` is `None`.
    description: String,
}

/// An attribute value, by what the callers need of it.
#[derive(Debug, Clone, PartialEq)]
enum AxAttr {
    Bool(bool),
    Text(String),
    Element(Element),
    Elements(Vec<Element>),
    Value(AxValue),
    Other,
}

/// ApplicationServices: the Accessibility API. Methods return the `AXError` with the value.
trait Accessibility: Send + Sync {
    fn create_application(&self, pid: i32) -> Element;
    fn create_system_wide(&self) -> Element;
    fn set_messaging_timeout(&self, element: Element, seconds: f64) -> i32;
    /// `AXUIElementCopyAttributeValue`; the value is `None` on error.
    fn copy_attribute(&self, element: Element, name: &str) -> (i32, Option<AxAttr>);
    fn set_bool_attribute(&self, element: Element, name: &str, value: bool) -> i32;
    fn perform_action(&self, element: Element, action: &str) -> i32;
    /// `AXUIElementCopyElementAtPosition` on the system-wide element.
    fn element_at_position(&self, system: Element, x: f64, y: f64) -> (i32, Option<Element>);
    fn pid_of(&self, element: Element) -> (i32, Option<i32>);
    /// `AXIsProcessTrusted`
    fn is_process_trusted(&self) -> bool;
    /// `AXIsProcessTrustedWithOptions` with `AXTrustedCheckOptionPrompt` set: shows the prompt.
    fn is_process_trusted_prompting(&self) -> bool;
}

/// CoreFoundation: `CFEqual` on two elements the Accessibility backend handed out.
trait CoreFoundation: Send + Sync {
    fn equal(&self, a: Element, b: Element) -> bool;
}

/// What [`Foundation::begin_activity`] returns; the activity ends when it is passed to
/// `end_activity`.
type ActivityToken = Box<dyn Any + Send + Sync>;

/// Foundation: `NSProcessInfo` activities (App Nap).
trait Foundation: Send + Sync {
    /// `beginActivity(options:reason:)`; `None` when it failed.
    fn begin_activity(&self, options: u64, reason: &str) -> Option<ActivityToken>;
    fn end_activity(&self, token: ActivityToken);
}

/// AVFoundation: the camera permission (`AVCaptureDevice`).
trait CameraAuthorization: Send + Sync {
    /// `authorizationStatus(for:)`: 0 not asked, 1 restricted, 2 denied, 3 authorized.
    fn authorization_status(&self, media_type: &str) -> Option<i64>;
    /// `requestAccess(for:completionHandler:)`; the answer is logged.
    fn request_access(&self, media_type: &str);
}

/// The process and operating system: tools, child processes, the login framework, identity.
trait Host: Send + Sync {
    /// Whether this is macOS at all (tests flip it to exercise the macOS-only paths).
    fn is_macos(&self) -> bool;
    /// Absolute path of a system tool (GUI apps may start with a minimal PATH).
    fn tool(&self, name: &str) -> Option<String>;
    /// Run `argv` (no shell) to the end within `timeout` seconds; its exit code.
    fn run(&self, argv: &[String], timeout: f64) -> Result<i32, String>;
    /// Start `argv` without waiting for it; the child is still reaped.
    fn spawn(&self, argv: &[String]) -> Result<(), String>;
    /// Load `framework` and call `SACLockScreenImmediate`; its status.
    fn lock_screen_immediate(&self, framework: &str) -> Result<i32, String>;
    /// Major macOS version (e.g. 14); 0 when unknown.
    fn macos_major_version(&self) -> u32;
    fn own_pid(&self) -> i32;
    /// `PlatformServices::app_process_name`.
    fn process_name(&self, pid: i64) -> Option<String>;
    /// [`build_fingerprint`]
    fn build_fingerprint(&self) -> Option<String>;
    /// [`runs_helper_executable`]
    fn runs_helper_executable(&self) -> bool;
}

// ---------------------------------------------------------------------------
// the packaged build
// ---------------------------------------------------------------------------

/// The main executable of the app bundle `executable` belongs to.
///
/// The bundle holds two programs side by side in `<App>.app/Contents/MacOS`: the app itself,
/// named by `CFBundleExecutable` in `Contents/Info.plist`, and the `eye-tracker-cli` command.
/// `None` when `executable` is not inside a bundle or the plist cannot be read.
fn bundle_main_executable(executable: &Path) -> Option<PathBuf> {
    let macos_dir = executable.parent()?;
    let contents = macos_dir.parent()?;
    if macos_dir.file_name()? != "MacOS" || contents.file_name()? != "Contents" {
        return None;
    }
    let name = match system::bundle_executable_name(&contents.join("Info.plist")) {
        Ok(name) => name,
        Err(err) => {
            log::debug!("Cannot read the bundle's Info.plist: {err}");
            return None;
        }
    };
    let valid = !name.is_empty() && Path::new(&name).file_name().is_some_and(|n| n == &*name);
    valid.then(|| macos_dir.join(name))
}

/// `(this process's executable, the app's main executable)` for a process running `executable`.
fn packaged_executables_for(executable: &Path) -> Option<(PathBuf, PathBuf)> {
    let executable = executable.canonicalize().ok()?;
    let main = bundle_main_executable(&executable).unwrap_or_else(|| executable.clone());
    Some((executable, main))
}

/// `(this process's executable, the app's main executable)`.
fn packaged_executables() -> Option<(PathBuf, PathBuf)> {
    packaged_executables_for(&std::env::current_exe().ok()?)
}

/// Identity of the build whose executables are `executables`; `None` when they are unknown.
///
/// An ad-hoc signed app's Accessibility grant is tied to its code hash, which changes with every
/// build - and so do the version, size and modification time of the bundle's main executable.
/// That is the app's executable even in `eye-tracker-cli`: both programs must name the same
/// build.
fn build_fingerprint_of(executables: Option<&(PathBuf, PathBuf)>) -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    let (_, main) = executables?;
    let stat = std::fs::metadata(main).ok()?;
    let mtime_ns = i128::from(stat.mtime()) * 1_000_000_000 + i128::from(stat.mtime_nsec());
    Some(format!("{}:{}:{}", crate::VERSION, stat.len(), mtime_ns))
}

/// Identity of this packaged build; `None` when it cannot be read.
fn build_fingerprint() -> Option<String> {
    build_fingerprint_of(packaged_executables().as_ref())
}

/// Whether the process running `executables` is a packaged build's `eye-tracker-cli`, not the
/// app.
///
/// macOS attributes the privacy permissions of such a process to whatever started it (the
/// terminal, for a command typed there), so `AXIsProcessTrusted()` says nothing about the app's
/// own grant.
fn runs_helper_executable_of(executables: Option<&(PathBuf, PathBuf)>) -> bool {
    let Some((executable, main)) = executables else {
        return false;
    };
    executable != main
        || executable
            .file_name()
            .is_some_and(|n| n.to_string_lossy().ends_with(CLI_SUFFIX))
}

fn runs_helper_executable() -> bool {
    runs_helper_executable_of(packaged_executables().as_ref())
}

// ---------------------------------------------------------------------------
// pure helpers (unit-tested with fakes)
// ---------------------------------------------------------------------------

/// `(pid, element)` of a macOS window handle; `None` for any other handle or a bad pid.
fn split_handle(handle: &WindowHandle) -> Option<(i32, Element)> {
    match handle {
        WindowHandle::Ax { pid, element } if *pid > 0 => Some((*pid, *element)),
        _ => None,
    }
}

/// Interpret `CGSessionCopyCurrentDictionary()`.
fn session_locked_from(info: Option<&SessionInfo>) -> Option<bool> {
    // No window-server session (e.g. ssh).
    let info = info?;
    if info.screen_locked {
        return Some(true);
    }
    // Fast user switching: another user owns the screen. Treated as locked so the camera is
    // released while nobody can see our session.
    Some(info.on_console == Some(false))
}

/// Parse an AXValue description such as `{value = x:24.0 y:38.0 type = …}`: the two numbers
/// after `x:`/`y:` (`size == false`) or `w:`/`h:`.
fn pair_from_repr(text: &str, size: bool) -> Option<(f64, f64)> {
    let (first, second) = if size { ("w:", "h:") } else { ("x:", "y:") };
    text.match_indices(first)
        .find_map(|(at, _)| pair_after(&text[at + first.len()..], second))
}

/// `<number> <second> <number>` at the start of `text`, blanks allowed (and required before
/// `second`).
fn pair_after(text: &str, second: &str) -> Option<(f64, f64)> {
    let (a, rest) = number_prefix(text.trim_start())?;
    let after = rest.trim_start();
    if after.len() == rest.len() {
        return None;
    }
    let (b, _) = number_prefix(after.strip_prefix(second)?.trim_start())?;
    Some((a, b))
}

/// A decimal number (`-?[0-9.]+`) at the start of `text`, and the rest.
fn number_prefix(text: &str) -> Option<(f64, &str)> {
    let digits = text.strip_prefix('-').unwrap_or(text);
    let len = digits
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(digits.len());
    let end = text.len() - digits.len() + len;
    if len == 0 {
        return None;
    }
    Some((text[..end].parse().ok()?, &text[end..]))
}

fn rect_from(origin: Option<(f64, f64)>, size: Option<(f64, f64)>) -> Option<Rect> {
    let ((x, y), (w, h)) = (origin?, size?);
    if ![x, y, w, h].iter().all(|v| v.is_finite()) {
        return None;
    }
    let [x, y, w, h] = [x, y, w, h].map(|v| v.round_ties_even() as i32);
    (w > 0 && h > 0).then(|| Rect::new(x, y, w, h))
}

fn cg_bounds(info: &CgWindow) -> Option<Rect> {
    let (x, y, w, h) = info.bounds?;
    rect_from(Some((x, y)), Some((w, h)))
}

/// `(pid, bounds)` of ordinary app windows, front to back.
///
/// Layer 0 is the normal window level; the menu bar, Dock, status items and our own
/// always-on-top helper windows live on other layers or belong to us.
fn cg_user_windows(windows: &[CgWindow], own_pid: i32) -> impl Iterator<Item = (i32, Rect)> + '_ {
    windows.iter().filter_map(move |info| {
        if info.layer != 0 || info.alpha <= 0.0 {
            return None;
        }
        let pid = i32::try_from(info.owner_pid).ok()?;
        if pid <= 0 || pid == own_pid {
            return None;
        }
        let rect = cg_bounds(info)?;
        (rect.w >= 2 && rect.h >= 2).then_some((pid, rect))
    })
}

fn cg_window_at(windows: &[CgWindow], x: f64, y: f64, own_pid: i32) -> Option<(i32, Rect)> {
    cg_user_windows(windows, own_pid).find(|(_, rect)| rect.contains(x, y))
}

fn cg_front_rect(windows: &[CgWindow], pid: i32, own_pid: i32) -> Option<Rect> {
    cg_user_windows(windows, own_pid).find_map(|(owner, rect)| (owner == pid).then_some(rect))
}

fn frames_match(a: &Rect, b: &Rect, tolerance: i32) -> bool {
    (a.x - b.x).abs() <= tolerance
        && (a.y - b.y).abs() <= tolerance
        && (a.w - b.w).abs() <= tolerance
        && (a.h - b.h).abs() <= tolerance
}

/// `(AXError, value)` of an attribute; the value is `None` on error.
fn ax_attr_err(ax: &dyn Accessibility, element: Element, name: &str) -> (i32, Option<AxAttr>) {
    let (err, value) = ax.copy_attribute(element, name);
    (err, if err == AX_SUCCESS { value } else { None })
}

fn ax_attr(ax: &dyn Accessibility, element: Element, name: &str) -> Option<AxAttr> {
    ax_attr_err(ax, element, name).1
}

fn ax_text(ax: &dyn Accessibility, element: Element, name: &str) -> Option<String> {
    match ax_attr(ax, element, name) {
        Some(AxAttr::Text(text)) => Some(text),
        _ => None,
    }
}

fn ax_element(ax: &dyn Accessibility, element: Element, name: &str) -> Option<Element> {
    match ax_attr(ax, element, name) {
        Some(AxAttr::Element(found)) => Some(found),
        _ => None,
    }
}

fn ax_set(ax: &dyn Accessibility, element: Element, name: &str, value: bool) -> bool {
    ax.set_bool_attribute(element, name, value) == AX_SUCCESS
}

fn ax_perform(ax: &dyn Accessibility, element: Element, action: &str) -> bool {
    ax.perform_action(element, action) == AX_SUCCESS
}

/// Unpack an AXValue holding a CGPoint (`size == false`) or a CGSize.
fn ax_pair(value: Option<&AxAttr>, size: bool) -> Option<(f64, f64)> {
    let Some(AxAttr::Value(value)) = value else {
        return None;
    };
    match (value.unpacked, size) {
        (Some(AxStruct::Size(w, h)), true) => return Some((w, h)),
        (Some(AxStruct::Point(x, y)), false) => return Some((x, y)),
        _ => {}
    }
    // Some builds cannot unpack an AXValue; its description always carries the numbers.
    pair_from_repr(&value.description, size)
}

/// `(AXError, frame)` of a window; the error lets callers spot hung apps.
fn ax_rect_checked(ax: Option<&dyn Accessibility>, element: Element) -> (i32, Option<Rect>) {
    let Some(ax) = ax.filter(|_| element != 0) else {
        return (AX_SUCCESS, None);
    };
    let (err, position) = ax_attr_err(ax, element, "AXPosition");
    if err != AX_SUCCESS {
        // Closed, or its app is hung: asking for AXSize would wait all over again.
        return (err, None);
    }
    let (err, size) = ax_attr_err(ax, element, "AXSize");
    if err != AX_SUCCESS {
        return (err, None);
    }
    let origin = ax_pair(position.as_ref(), false);
    let dimensions = ax_pair(size.as_ref(), true);
    (AX_SUCCESS, rect_from(origin, dimensions))
}

fn ax_rect(ax: Option<&dyn Accessibility>, element: Element) -> Option<Rect> {
    ax_rect_checked(ax, element).1
}

/// The window that contains an arbitrary UI element.
fn ax_window_of(ax: &dyn Accessibility, element: Element) -> Option<Element> {
    if ax_text(ax, element, "AXRole").as_deref() == Some("AXWindow") {
        return Some(element);
    }
    if let Some(window) = ax_element(ax, element, "AXWindow") {
        return Some(window);
    }
    let mut current = element;
    for _ in 0..32 {
        // bounded: a broken hierarchy must not loop forever
        current = ax_element(ax, current, "AXParent")?;
        match ax_text(ax, current, "AXRole").as_deref() {
            Some("AXWindow") => return Some(current),
            Some("AXApplication") => return None,
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// platform
// ---------------------------------------------------------------------------

/// Which build Accessibility was last granted to, read from the state file once.
#[derive(Default)]
struct TrustedBuild {
    value: Option<String>,
    loaded: bool,
}

/// macOS 12+ implementation of [`PlatformServices`].
///
/// `clock` and `state_path` exist for tests; production code uses the defaults (`state_path`:
/// the file remembering which build was trusted).
pub struct MacPlatform {
    clock: Box<dyn Fn() -> f64 + Send + Sync>,
    host: Arc<dyn Host>,
    quartz: Option<Arc<dyn Quartz>>,
    appkit: Option<Arc<dyn AppKit>>,
    ax: Option<Arc<dyn Accessibility>>,
    core_foundation: Option<Arc<dyn CoreFoundation>>,
    foundation: Option<Arc<dyn Foundation>>,
    camera: Option<Arc<dyn CameraAuthorization>>,
    ax_ready: Mutex<bool>,
    trust_cache: Mutex<Option<(f64, bool)>>,
    /// pid -> clock time until which AX requests to that app are skipped.
    ax_unresponsive: Mutex<HashMap<i32, f64>>,
    /// The `NSProcessInfo` activity token (App Nap opt-out); the activity ends with it.
    activity: Mutex<Option<ActivityToken>>,
    state_path: Mutex<Option<PathBuf>>,
    /// Last build known to be trusted (cached).
    trusted_build: Mutex<TrustedBuild>,
    /// See [`Self::fingerprint`].
    build_id: OnceLock<Option<String>>,
    /// See [`Self::is_helper`].
    helper: OnceLock<bool>,
    warned_untrusted: AtomicBool,
}

impl Default for MacPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl MacPlatform {
    pub fn new() -> Self {
        let mut platform = Self::with_host(
            Box::new(crate::panes::worker::monotonic),
            None,
            Arc::new(system::SystemHost),
        );
        system::install(&mut platform);
        platform
    }

    /// A platform whose frameworks are all missing, so every call degrades.
    fn with_host(
        clock: Box<dyn Fn() -> f64 + Send + Sync>,
        state_path: Option<PathBuf>,
        host: Arc<dyn Host>,
    ) -> Self {
        Self {
            clock,
            host,
            quartz: None,
            appkit: None,
            ax: None,
            core_foundation: None,
            foundation: None,
            camera: None,
            ax_ready: Mutex::new(false),
            trust_cache: Mutex::new(None),
            ax_unresponsive: Mutex::new(HashMap::new()),
            activity: Mutex::new(None),
            state_path: Mutex::new(state_path),
            trusted_build: Mutex::new(TrustedBuild::default()),
            build_id: OnceLock::new(),
            helper: OnceLock::new(),
            warned_untrusted: AtomicBool::new(false),
        }
    }

    /// The Accessibility API, with a short global AX messaging timeout set once.
    fn ax(&self) -> Option<&dyn Accessibility> {
        let ax = self.ax.as_deref()?;
        let mut ready = self.ax_ready.lock();
        if !*ready {
            *ready = true;
            // Setting the timeout on the system-wide element applies globally.
            let err = ax.set_messaging_timeout(ax.create_system_wide(), AX_TIMEOUT_S);
            if err != AX_SUCCESS {
                log::debug!("AXUIElementSetMessagingTimeout failed: {err}");
            }
        }
        Some(ax)
    }

    fn ax_trusted(&self) -> bool {
        let now = (self.clock)();
        if let Some((at, trusted)) = *self.trust_cache.lock()
            && now - at < TRUST_TTL_S
        {
            return trusted;
        }
        let trusted = self.accessibility_permission().unwrap_or(false);
        *self.trust_cache.lock() = Some((now, trusted));
        trusted
    }

    /// `false` while an app that recently timed out an AX request is left alone.
    fn ax_responsive(&self, pid: i32) -> bool {
        let mut unresponsive = self.ax_unresponsive.lock();
        let Some(until) = unresponsive.get(&pid).copied() else {
            return true;
        };
        if (self.clock)() >= until {
            unresponsive.remove(&pid);
            return true;
        }
        false
    }

    fn note_ax_error(&self, pid: i32, err: i32) {
        if err == AX_ERROR_CANNOT_COMPLETE {
            let mut unresponsive = self.ax_unresponsive.lock();
            if !unresponsive.contains_key(&pid) {
                log::debug!("App {pid} does not answer Accessibility requests; skipping it");
            }
            unresponsive.insert(pid, (self.clock)() + AX_UNRESPONSIVE_S);
        }
    }

    /// Run a system tool to the end; whether it exited with 0.
    fn run_ok(&self, argv: &[&str]) -> bool {
        let Some(exe) = self.host.tool(argv[0]) else {
            return false;
        };
        let mut command = vec![exe];
        command.extend(argv[1..].iter().map(|a| (*a).to_string()));
        match self.host.run(&command, TOOL_TIMEOUT_S) {
            Ok(code) => {
                if code != 0 {
                    log::debug!("{} exited with {code}", argv[0]);
                }
                code == 0
            }
            Err(err) => {
                log::debug!("{} failed: {err}", argv[0]);
                false
            }
        }
    }

    fn lock_screen_immediate(&self) -> bool {
        match self.host.lock_screen_immediate(LOGIN_FRAMEWORK) {
            Ok(0) => true,
            Ok(status) => {
                log::debug!("SACLockScreenImmediate returned {status}");
                false
            }
            Err(err) => {
                log::debug!("SACLockScreenImmediate unavailable: {err}");
                false
            }
        }
    }

    fn seconds_since(&self, event_type: u32) -> Option<f64> {
        let value = self
            .quartz
            .as_deref()?
            .seconds_since_last_event(CG_EVENT_SOURCE_STATE_COMBINED, event_type)?;
        (value.is_finite() && value >= 0.0).then_some(value)
    }

    /// The window of `pid` with keyboard focus, `0` when there is none or it cannot be asked.
    fn focused_ax_window(&self, pid: i32) -> Element {
        let Some(ax) = self.ax() else {
            return 0;
        };
        let app = ax.create_application(pid);
        // Polled twice a second: a busy app must not hold the UI thread for the full global
        // timeout on every poll.
        ax.set_messaging_timeout(app, AX_APP_TIMEOUT_S);
        let (err, window) = ax_attr_err(ax, app, "AXFocusedWindow");
        self.note_ax_error(pid, err);
        match window {
            Some(AxAttr::Element(window)) => window,
            _ => 0,
        }
    }

    /// On-screen windows, front to back; `None` when the list is unavailable.
    fn cg_windows(&self) -> Option<Vec<CgWindow>> {
        let quartz = self.quartz.as_deref()?;
        quartz.window_list(
            CG_WINDOW_LIST_ON_SCREEN_ONLY | CG_WINDOW_LIST_EXCLUDE_DESKTOP,
            CG_NULL_WINDOW_ID,
        )
    }

    fn cg_front_rect(&self, pid: i32) -> Option<Rect> {
        let windows = self.cg_windows().filter(|w| !w.is_empty())?;
        cg_front_rect(&windows, pid, self.host.own_pid())
    }

    /// The app's AX window under the point, preferring the one matching `hint`; `0` for none.
    fn ax_window_containing(&self, pid: i32, x: i32, y: i32, hint: Rect) -> Element {
        let Some(ax) = self.ax() else {
            return 0;
        };
        let app = ax.create_application(pid);
        let (err, windows) = ax_attr_err(ax, app, "AXWindows");
        self.note_ax_error(pid, err);
        let windows = match windows {
            Some(AxAttr::Elements(windows)) => windows,
            _ => Vec::new(),
        };
        let mut fallback = 0;
        for window in windows {
            // AXWindows is ordered front to back
            if matches!(ax_attr(ax, window, "AXMinimized"), Some(AxAttr::Bool(true))) {
                continue;
            }
            let Some(rect) = ax_rect(Some(ax), window) else {
                continue;
            };
            if !rect.contains(f64::from(x), f64::from(y)) {
                continue;
            }
            if rect == hint {
                return window;
            }
            if fallback == 0 {
                fallback = window;
            }
        }
        fallback
    }

    /// Hit-test through Accessibility (used when the window list is unavailable).
    fn ax_window_at(&self, x: i32, y: i32) -> Option<WindowRef> {
        let ax = self.ax().filter(|_| self.ax_trusted())?;
        let system = ax.create_system_wide();
        let (err, element) = ax.element_at_position(system, f64::from(x), f64::from(y));
        let element = element.filter(|_| err == AX_SUCCESS)?;
        let window = ax_window_of(ax, element)?;
        let (err, pid) = ax.pid_of(window);
        let pid = pid.filter(|pid| err == AX_SUCCESS && *pid != 0)?;
        if pid == self.host.own_pid() {
            return None;
        }
        Some(WindowRef {
            handle: WindowHandle::Ax {
                pid,
                element: window,
            },
            pid: u32::try_from(pid).ok(),
            rect: ax_rect(Some(ax), window),
        })
    }

    /// Whether `pid` shows a window (with about this frame) on a visible Space.
    fn on_screen(&self, pid: i32, frame: Option<Rect>) -> bool {
        let Some(windows) = self.cg_windows() else {
            return true; // cannot tell: keep the previous behaviour
        };
        cg_user_windows(&windows, self.host.own_pid()).any(|(owner, bounds)| {
            owner == pid
                && frame
                    .as_ref()
                    .is_none_or(|f| frames_match(&bounds, f, FRAME_TOLERANCE))
        })
    }

    /// This build's [`build_fingerprint`], computed once per process.
    fn fingerprint(&self) -> Option<String> {
        self.build_id
            .get_or_init(|| self.host.build_fingerprint())
            .clone()
    }

    /// Whether this process is `eye-tracker-cli` (cached [`runs_helper_executable`]).
    ///
    /// macOS attributes its Accessibility trust to the process that started it - usually the
    /// terminal - so a trusted terminal must not be recorded as a trusted app build, and an
    /// untrusted one says nothing about the app.
    fn is_helper(&self) -> bool {
        *self
            .helper
            .get_or_init(|| self.host.runs_helper_executable())
    }

    /// Whether *this process* may use the Accessibility API (`AXIsProcessTrusted`).
    fn accessibility_permission(&self) -> Option<bool> {
        let trusted = self.ax()?.is_process_trusted();
        if trusted && !self.is_helper() {
            self.remember_trusted_build();
        }
        Some(trusted)
    }

    fn warn_untrusted_once(&self) {
        if self.warned_untrusted.swap(true, Ordering::SeqCst) {
            return;
        }
        if self.accessibility_status() == "stale" {
            log::warn!(
                "Accessibility access was granted to an earlier version of Eye Tracker and \
                 does not apply to this one: keyboard focus cannot follow your gaze. In System \
                 Settings › Privacy & Security › Accessibility, remove Eye Tracker with '−' \
                 and add it again."
            );
        } else {
            log::warn!(
                "Accessibility permission is missing: keyboard focus cannot follow your gaze \
                 (System Settings › Privacy & Security › Accessibility)."
            );
        }
    }

    fn state_file(&self) -> PathBuf {
        self.state_path
            .lock()
            .get_or_insert_with(|| crate::paths::data_dir().join(ACCESSIBILITY_STATE_FILE))
            .clone()
    }

    fn load_trusted_build(&self) -> Option<String> {
        let mut trusted = self.trusted_build.lock();
        if !trusted.loaded {
            trusted.loaded = true;
            let data = std::fs::read_to_string(self.state_file())
                .map_err(|err| err.to_string())
                .and_then(|text| {
                    serde_json::from_str::<serde_json::Value>(&text).map_err(|err| err.to_string())
                });
            let data = match data {
                Ok(data) => Some(data),
                Err(err) => {
                    log::debug!("No record of an earlier Accessibility grant: {err}");
                    None
                }
            };
            trusted.value = data
                .as_ref()
                .and_then(|d| d.get("trusted_build"))
                .and_then(|v| v.as_str())
                .map(str::to_string);
        }
        trusted.value.clone()
    }

    /// Record this build as trusted, so a later update can recognise a stale grant.
    fn remember_trusted_build(&self) {
        let Some(current) = self.fingerprint() else {
            return;
        };
        if self.load_trusted_build().as_deref() == Some(&current) {
            return;
        }
        let path = self.state_file();
        let write = || -> std::io::Result<()> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut name = path.file_name().unwrap_or_default().to_os_string();
            name.push(".tmp");
            let partial = path.with_file_name(name);
            std::fs::write(&partial, json!({ "trusted_build": current }).to_string())?;
            std::fs::rename(&partial, &path)
        };
        if let Err(err) = write() {
            log::debug!("Could not record the Accessibility grant: {err}");
            return;
        }
        self.trusted_build.lock().value = Some(current);
    }

    fn camera_permission(&self) -> Option<bool> {
        let status = self
            .camera
            .as_deref()?
            .authorization_status(AV_MEDIA_TYPE_VIDEO)?;
        // restricted, denied, authorized (0 = not asked)
        match status {
            1 | 2 => Some(false),
            3 => Some(true),
            _ => None,
        }
    }
}

impl PlatformServices for MacPlatform {
    fn name(&self) -> &'static str {
        "macos"
    }

    // -------------------------------------------------------------- lifecycle
    /// Nothing to do: the UI toolkit already uses point coordinates, like Quartz and AX.
    ///
    /// The Dock icon is hidden by `LSUIElement` in the app bundle and, when running from
    /// source, by [`PlatformServices::set_accessory_app`] after the application exists.
    fn prepare_process(&self) {}

    /// Run as a menu-bar ("accessory") app: no Dock icon, no app menu, no App Nap.
    ///
    /// A menu-bar app without visible windows is exactly what App Nap throttles, so this also
    /// calls [`set_background_activity`](PlatformServices::set_background_activity). Must be
    /// called on the main thread after the UI toolkit created the application. Returns whether
    /// the activation policy was set.
    fn set_accessory_app(&self) -> bool {
        self.set_background_activity(true);
        let Some(appkit) = self.appkit.as_deref() else {
            return false;
        };
        appkit.set_activation_policy(NS_ACTIVATION_POLICY_ACCESSORY)
    }

    /// Opt out of App Nap while `active` (`true`), or allow it again.
    ///
    /// Without this, macOS naps a window-less menu-bar app: its timers (the controller tick,
    /// the walk-away countdown) and the camera loop get delayed by up to seconds. The activity
    /// is "user initiated, allowing idle system sleep": the Mac and its displays still sleep
    /// when idle. The controller may end it while tracking is paused to save power. Returns
    /// whether the requested state is in effect.
    fn set_background_activity(&self, active: bool) -> bool {
        let mut activity = self.activity.lock();
        if active == activity.is_some() {
            return true;
        }
        let Some(foundation) = self.foundation.as_deref() else {
            return false;
        };
        if active {
            // The token must stay referenced: the activity ends with it.
            *activity = foundation.begin_activity(
                NS_ACTIVITY_USER_INITIATED_ALLOWING_IDLE_SYSTEM_SLEEP,
                ACTIVITY_REASON,
            );
            if activity.is_some() {
                log::debug!("App Nap disabled");
                return true;
            }
            log::debug!("NSProcessInfo activity change failed");
            return false;
        }
        if let Some(token) = activity.take() {
            foundation.end_activity(token);
        }
        log::debug!("App Nap allowed again");
        true
    }

    fn capabilities(&self) -> HashMap<&'static str, bool> {
        let quartz = self.quartz.is_some();
        let focus = self.appkit.is_some() && self.ax.is_some();
        let macos = self.host.is_macos();
        HashMap::from([
            ("lock", macos),
            ("display_off", macos && self.host.tool("pmset").is_some()),
            (
                "wake_display",
                macos && self.host.tool("caffeinate").is_some(),
            ),
            ("input_idle", quartz),
            ("key_idle", quartz),
            ("session_locked", quartz),
            ("focus", focus),
            ("cursor", true),
            ("camera_in_use", false),
            ("hotkeys", macos),
            ("panes", focus),
        ])
    }

    // ---------------------------------------------------------- session/power
    /// Lock immediately (like Ctrl+Cmd+Q); fall back to display sleep.
    fn lock_screen(&self) -> bool {
        if !self.host.is_macos() {
            return false;
        }
        // `SACLockScreenImmediate` from the private login framework.
        if self.lock_screen_immediate() {
            log::info!("Screen locked");
            return true;
        }
        if self.run_ok(&["pmset", "displaysleepnow"]) {
            log::info!(
                "Screen lock API unavailable; displays put to sleep instead (this locks \
                 when 'Require password immediately after sleep' is enabled)"
            );
            return true;
        }
        log::warn!("Could not lock the screen");
        false
    }

    fn display_off(&self) -> bool {
        self.host.is_macos() && self.run_ok(&["pmset", "displaysleepnow"])
    }

    /// Declare user activity for 2 s (`caffeinate -u`), which wakes the displays.
    fn wake_display(&self) -> bool {
        if !self.host.is_macos() {
            return false;
        }
        let Some(exe) = self.host.tool("caffeinate") else {
            return false;
        };
        let argv = [exe, "-u".to_string(), "-t".to_string(), "2".to_string()];
        // Non-blocking for the caller, but the child is still reaped.
        match self.host.spawn(&argv) {
            Ok(()) => true,
            Err(err) => {
                log::debug!("caffeinate failed: {err}");
                false
            }
        }
    }

    fn is_session_locked(&self) -> Option<bool> {
        let quartz = self.quartz.as_deref()?;
        session_locked_from(quartz.session_info().as_ref())
    }

    // ------------------------------------------------------------------ input
    fn seconds_since_input(&self) -> Option<f64> {
        self.seconds_since(CG_ANY_INPUT_EVENT_TYPE)
    }

    /// Time since the last key press; only a timestamp is read, never the key.
    fn seconds_since_key_input(&self) -> Option<f64> {
        self.seconds_since(CG_EVENT_KEY_DOWN)
    }

    /// Quartz reports the pointer everywhere, over every app's windows.
    fn cursor_position_reliable(&self) -> bool {
        true
    }

    fn move_cursor(&self, x: i32, y: i32) -> Option<bool> {
        // Missing Quartz: the pointer cannot be moved.
        let quartz = self.quartz.as_deref()?;
        let err = quartz.warp_cursor(f64::from(x), f64::from(y));
        // A warp suppresses local mouse events for ~0.25 s; re-associating the mouse ends that,
        // so the user can take over immediately.
        quartz.associate_mouse_and_cursor(true);
        Some(err == 0)
    }

    /// `NSEvent.mouseLocation` in the top-left-origin points the rest of this module uses.
    fn cursor_position(&self) -> Option<(i32, i32)> {
        let (x, y) = self.appkit.as_deref()?.mouse_location()?;
        let height = self.quartz.as_deref()?.main_display_height()?;
        // Cocoa's y grows upwards from the main display's bottom edge; the pixel row at its
        // top edge is y = height - 1 in top-left coordinates.
        Some((x.floor() as i32, (height - y).floor() as i32))
    }

    // ---------------------------------------------------------------- windows
    fn foreground_window(&self) -> Option<WindowRef> {
        let appkit = self.appkit.as_deref()?;
        let pid = appkit.frontmost_pid().unwrap_or(0);
        if pid <= 0 || pid == self.host.own_pid() {
            return None;
        }
        let mut window = 0;
        if self.ax_trusted() && self.ax_responsive(pid) {
            window = self.focused_ax_window(pid);
        }
        let mut rect = None;
        if window != 0 {
            let (err, found) = ax_rect_checked(self.ax(), window);
            self.note_ax_error(pid, err);
            rect = found;
        }
        if rect.is_none() {
            rect = self.cg_front_rect(pid);
        }
        Some(WindowRef {
            handle: WindowHandle::Ax {
                pid,
                element: window,
            },
            pid: u32::try_from(pid).ok(),
            rect,
        })
    }

    /// Frontmost ordinary window under a point (excluding our own).
    ///
    /// The on-screen window list is consulted first: it needs no permission, skips our
    /// click-through overlay and cannot hang on a busy app. The AX element is then looked up for
    /// window-level focus when permitted.
    fn window_at(&self, x: i32, y: i32) -> Option<WindowRef> {
        let Some(windows) = self.cg_windows() else {
            return self.ax_window_at(x, y);
        };
        let (pid, mut rect) =
            cg_window_at(&windows, f64::from(x), f64::from(y), self.host.own_pid())?;
        let mut window = 0;
        if self.ax_trusted() && self.ax_responsive(pid) {
            window = self.ax_window_containing(pid, x, y, rect);
        }
        if window != 0 {
            rect = ax_rect(self.ax(), window).unwrap_or(rect);
        }
        Some(WindowRef {
            handle: WindowHandle::Ax {
                pid,
                element: window,
            },
            pid: u32::try_from(pid).ok(),
            rect: Some(rect),
        })
    }

    /// Bring the app forward and raise the window (no synthetic click).
    ///
    /// `activateWithOptions` alone is no longer enough on macOS 14+ (cooperative activation
    /// ignores requests from inactive apps while still reporting success), so the Accessibility
    /// route (`AXFrontmost` + `AXRaise`) is used when permitted. Without Accessibility on macOS
    /// 14+ the request is still made but reported as failed: it has no effect for a menu-bar
    /// app.
    fn activate_window(&self, window_ref: &WindowRef) -> bool {
        let Some((pid, window)) = split_handle(&window_ref.handle) else {
            return false;
        };
        let trusted = self.ax_trusted();
        if !trusted {
            self.warn_untrusted_once();
        }
        // Every AX call to a hung app would block for the full timeout.
        let mut ax = if trusted && self.ax_responsive(pid) {
            self.ax()
        } else {
            None
        };
        if let Some(ax_api) = ax.filter(|_| window != 0) {
            let (err, minimized) = ax_attr_err(ax_api, window, "AXMinimized");
            self.note_ax_error(pid, err);
            if matches!(minimized, Some(AxAttr::Bool(true))) {
                return false; // never un-minimise behind the user's back
            }
            if err == AX_ERROR_CANNOT_COMPLETE {
                ax = None;
            } else {
                ax_set(ax_api, window, "AXMain", true);
            }
        }
        let mut activated = false;
        if let Some(appkit) = self.appkit.as_deref()
            && appkit
                .running_application(pid)
                .is_some_and(|a| !a.terminated)
        {
            activated = appkit.activate(pid, NS_ACTIVATE_IGNORING_OTHER_APPS);
        }
        if let Some(ax) = ax {
            let app_element = ax.create_application(pid);
            activated = ax_set(ax, app_element, "AXFrontmost", true) || activated;
            if window != 0 {
                ax_perform(ax, window, "AXRaise");
            }
        }
        if !trusted && self.host.macos_major_version() >= 14 {
            return false;
        }
        activated
    }

    /// Running, not hidden or minimised, and on a Space that is shown right now.
    ///
    /// Accessibility keeps answering for windows on other Spaces, other Stage Manager stages or
    /// behind another app's full-screen Space, with frames inside the same display. Activating
    /// such a window would make macOS switch that display back to its Space, so the on-screen
    /// window list decides. Windows of an app that does not answer AX requests are not valid
    /// targets either, for now.
    fn is_window_valid(&self, window_ref: &WindowRef) -> bool {
        let (Some((pid, window)), Some(appkit)) =
            (split_handle(&window_ref.handle), self.appkit.as_deref())
        else {
            return false;
        };
        match appkit.running_application(pid) {
            Some(app) if !app.terminated && !app.hidden => {}
            _ => return false,
        }
        if window == 0 {
            return self.on_screen(pid, None);
        }
        if !self.ax_responsive(pid) {
            return false;
        }
        let Some(ax) = self.ax() else {
            return false;
        };
        let (err, minimized) = ax_attr_err(ax, window, "AXMinimized");
        self.note_ax_error(pid, err);
        if matches!(minimized, Some(AxAttr::Bool(true))) || err == AX_ERROR_CANNOT_COMPLETE {
            return false;
        }
        // A closed window's element answers every query with an error.
        let (err, rect) = ax_rect_checked(Some(ax), window);
        self.note_ax_error(pid, err);
        match rect {
            Some(rect) => self.on_screen(pid, Some(rect)),
            None => false,
        }
    }

    fn window_rect(&self, window_ref: &WindowRef) -> Option<Rect> {
        let (pid, window) = split_handle(&window_ref.handle)?;
        if window != 0 && self.ax_responsive(pid) {
            let (err, rect) = ax_rect_checked(self.ax(), window);
            self.note_ax_error(pid, err);
            if err != AX_ERROR_CANNOT_COMPLETE {
                return rect;
            }
        }
        // Application-level handle, or a hung app: the window list still knows where its front
        // window is.
        self.cg_front_rect(pid)
    }

    /// The window frame: Accessibility has no portable content-area attribute, and the title
    /// bar's height depends on the app (none in full screen).
    fn window_client_rect(&self, window_ref: &WindowRef) -> Option<Rect> {
        self.window_rect(window_ref)
    }

    fn window_app(&self, window_ref: &WindowRef) -> Option<AppIdentity> {
        let pid = split_handle(&window_ref.handle)
            .map(|(pid, _)| pid)
            .or(window_ref.pid.and_then(|p| i32::try_from(p).ok()))?;
        if pid <= 0 {
            return None;
        }
        let process = self.app_process_name(Some(i64::from(pid)))?;
        let app_id = self
            .appkit
            .as_deref()
            .and_then(|appkit| appkit.running_application(pid))
            .map(|app| app.bundle_id)
            .unwrap_or_default();
        Some(AppIdentity { process, app_id })
    }

    fn app_process_name(&self, pid: Option<i64>) -> Option<String> {
        self.host.process_name(pid?)
    }

    fn same_window(&self, a: Option<&WindowRef>, b: Option<&WindowRef>) -> bool {
        let (Some(a), Some(b)) = (a, b) else {
            return false;
        };
        let (Some((pid_a, win_a)), Some((pid_b, win_b))) =
            (split_handle(&a.handle), split_handle(&b.handle))
        else {
            return false;
        };
        if pid_a != pid_b {
            return false;
        }
        match (win_a, win_b) {
            (0, 0) => true, // application-level handles (no Accessibility permission)
            (0, _) | (_, 0) => false,
            _ => match self.core_foundation.as_deref() {
                Some(cf) => cf.equal(win_a, win_b),
                None => win_a == win_b,
            },
        }
    }

    // ------------------------------------------------------------ permissions
    /// Camera and Accessibility permission (`None`: unknown or not decided).
    ///
    /// In `eye-tracker-cli` Accessibility is `None` whatever macOS says: it judged whatever
    /// started the command (see [`MacPlatform::is_helper`]), usually the terminal, not the app,
    /// so `doctor` must neither call the app's grant missing nor report a trusted terminal's
    /// grant as the app's.
    fn permissions(&self) -> HashMap<&'static str, Option<bool>> {
        let accessibility = if self.is_helper() {
            None
        } else {
            self.accessibility_permission()
        };
        HashMap::from([
            ("camera", self.camera_permission()),
            ("accessibility", accessibility),
        ])
    }

    /// Accessibility permission, in more detail than [`permissions`](Self::permissions).
    ///
    /// * `"granted"`: keyboard focus can follow the gaze.
    /// * `"missing"`: never granted (or revoked); `request_permission` asks.
    /// * `"stale"`: not trusted, but a *previous build* of the app was. Release builds are
    ///   signed ad hoc, so each update has a new code identity; System Settings keeps showing
    ///   the old entry as enabled while it no longer applies. The fix: remove Eye Tracker from
    ///   Privacy & Security › Accessibility with "−", then add the app again.
    /// * `"unknown"`: the Accessibility API is unavailable, or this is `eye-tracker-cli`,
    ///   whose answer from macOS belongs to whatever started it (usually the terminal), not to
    ///   the app; see [`MacPlatform::is_helper`].
    ///
    /// Only builds with a readable executable can be told apart this way.
    fn accessibility_status(&self) -> &'static str {
        if self.is_helper() {
            return "unknown";
        }
        let Some(trusted) = self.accessibility_permission() else {
            return "unknown";
        };
        *self.trust_cache.lock() = Some(((self.clock)(), trusted));
        if trusted {
            return "granted";
        }
        let Some(current) = self.fingerprint() else {
            return "missing";
        };
        match self.load_trusted_build() {
            Some(previous) if previous != current => "stale",
            _ => "missing",
        }
    }

    fn request_permission(&self, name: &str) {
        match name {
            "accessibility" => {
                let Some(ax) = self.ax() else {
                    return;
                };
                ax.is_process_trusted_prompting();
                *self.trust_cache.lock() = None;
            }
            "camera" => {
                if let Some(camera) = self.camera.as_deref() {
                    camera.request_access(AV_MEDIA_TYPE_VIDEO);
                }
            }
            _ => {}
        }
    }

    fn open_permission_settings(&self, name: &str) -> bool {
        let Some((_, url)) = PERMISSION_URLS.iter().find(|(n, _)| *n == name) else {
            return false;
        };
        self.host.is_macos() && self.run_ok(&["open", url])
    }
}

/// The real frameworks: Quartz and the Accessibility API through their C functions (declared
/// here, with the `core-foundation` types), AppKit and Foundation through `objc2`, AVFoundation
/// through the Objective-C runtime.
mod system {
    use std::collections::HashMap;
    use std::ffi::{CString, c_void};
    use std::path::Path;
    use std::sync::Arc;

    use core_foundation::array::{CFArrayGetCount, CFArrayGetTypeID, CFArrayGetValueAtIndex};
    use core_foundation::base::{
        CFCopyDescription, CFEqual, CFGetTypeID, CFHash, CFType, CFTypeRef, TCFType,
    };
    use core_foundation::boolean::{CFBoolean, CFBooleanGetTypeID};
    use core_foundation::data::CFData;
    use core_foundation::dictionary::{
        CFDictionary, CFDictionaryGetTypeID, CFDictionaryGetValue, CFDictionaryRef,
    };
    use core_foundation::number::CFNumber;
    use core_foundation::propertylist::{create_with_data, kCFPropertyListImmutable};
    use core_foundation::string::{CFString, CFStringGetTypeID, CFStringRef};
    use objc2::encode::{Encode, Encoding};
    use objc2::msg_send;
    use objc2::rc::Retained;
    use objc2::runtime::{AnyClass, Bool, ProtocolObject};
    use objc2_app_kit::{
        NSApplication, NSApplicationActivationOptions, NSApplicationActivationPolicy, NSEvent,
        NSRunningApplication, NSWorkspace,
    };
    use objc2_foundation::{
        MainThreadMarker, NSActivityOptions, NSBundle, NSObjectProtocol, NSProcessInfo, NSString,
    };
    use parking_lot::Mutex;

    use super::{
        AVFOUNDATION_FRAMEWORK, Accessibility, ActivityToken, AppKit, AxAttr, AxStruct, AxValue,
        CameraAuthorization, CgWindow, CoreFoundation, Element, Foundation, Host, MacPlatform,
        Quartz, RunningApp, SessionInfo,
    };
    use crate::panes::providers::command::run_command;
    use crate::platform::base::{GenericPlatform, PlatformServices, which};

    /// `kAXErrorInvalidUIElement`
    const AX_ERROR_INVALID_UI_ELEMENT: i32 = -25202;
    /// `kAXErrorFailure`
    const AX_ERROR_FAILURE: i32 = -25200;
    const AX_VALUE_CGPOINT: u32 = 1; // kAXValueCGPointType
    const AX_VALUE_CGSIZE: u32 = 2; // kAXValueCGSizeType

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct CgPoint {
        x: f64,
        y: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct CgSize {
        width: f64,
        height: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct CgRect {
        origin: CgPoint,
        size: CgSize,
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGEventSourceSecondsSinceLastEventType(state: i32, event_type: u32) -> f64;
        fn CGSessionCopyCurrentDictionary() -> CFDictionaryRef;
        fn CGWarpMouseCursorPosition(position: CgPoint) -> i32;
        fn CGAssociateMouseAndMouseCursorPosition(connected: i32) -> i32;
        fn CGWindowListCopyWindowInfo(option: u32, relative_to_window: u32) -> CFTypeRef;
        fn CGMainDisplayID() -> u32;
        fn CGDisplayBounds(display: u32) -> CgRect;
    }

    #[link(name = "ApplicationServices", kind = "framework")]
    unsafe extern "C" {
        fn AXUIElementCreateApplication(pid: i32) -> CFTypeRef;
        fn AXUIElementCreateSystemWide() -> CFTypeRef;
        fn AXUIElementGetTypeID() -> usize;
        fn AXUIElementSetMessagingTimeout(element: CFTypeRef, timeout_in_seconds: f32) -> i32;
        fn AXUIElementCopyAttributeValue(
            element: CFTypeRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> i32;
        fn AXUIElementSetAttributeValue(
            element: CFTypeRef,
            attribute: CFStringRef,
            value: CFTypeRef,
        ) -> i32;
        fn AXUIElementPerformAction(element: CFTypeRef, action: CFStringRef) -> i32;
        fn AXUIElementCopyElementAtPosition(
            application: CFTypeRef,
            x: f32,
            y: f32,
            element: *mut CFTypeRef,
        ) -> i32;
        fn AXUIElementGetPid(element: CFTypeRef, pid: *mut i32) -> i32;
        fn AXValueGetTypeID() -> usize;
        fn AXValueGetType(value: CFTypeRef) -> u32;
        fn AXValueGetValue(value: CFTypeRef, kind: u32, out: *mut c_void) -> u8;
        fn AXIsProcessTrusted() -> u8;
        fn AXIsProcessTrustedWithOptions(options: CFDictionaryRef) -> u8;
        static kAXTrustedCheckOptionPrompt: CFStringRef;
    }

    unsafe extern "C" {
        /// The class of statically allocated blocks.
        static _NSConcreteGlobalBlock: c_void;
    }

    /// Put the real frameworks into `platform`.
    pub(super) fn install(platform: &mut MacPlatform) {
        let registry = Arc::new(ElementRegistry::default());
        platform.quartz = Some(Arc::new(SystemQuartz));
        platform.appkit = Some(Arc::new(SystemAppKit));
        platform.ax = Some(Arc::new(SystemAccessibility {
            registry: registry.clone(),
        }));
        platform.core_foundation = Some(Arc::new(SystemCoreFoundation { registry }));
        platform.foundation = Some(Arc::new(SystemFoundation));
        platform.camera = Some(Arc::new(SystemCamera));
    }

    // ------------------------------------------------------------------ helpers
    /// The value of `key` in a `CFDictionary`.
    fn dict_value(dict: CFDictionaryRef, key: &str) -> Option<CFType> {
        let key = CFString::new(key);
        // SAFETY: `dict` is a live CFDictionary and the key a live CFString.
        let value = unsafe { CFDictionaryGetValue(dict, key.as_concrete_TypeRef().cast()) };
        // SAFETY: the value is owned by the dictionary; the get rule retains it.
        (!value.is_null()).then(|| unsafe { CFType::wrap_under_get_rule(value) })
    }

    fn is_type(value: &CFType, type_id: usize) -> bool {
        // SAFETY: `value` is a live CF object.
        unsafe { CFGetTypeID(value.as_CFTypeRef()) == type_id }
    }

    fn number_of(value: Option<CFType>) -> Option<CFNumber> {
        value?.downcast::<CFNumber>()
    }

    /// Python's `bool()` of a CFBoolean/CFNumber property.
    fn truthy(value: &CFType) -> bool {
        if let Some(flag) = value.downcast::<CFBoolean>() {
            return flag.into();
        }
        if let Some(number) = value.downcast::<CFNumber>() {
            return number.to_i64().is_none_or(|n| n != 0);
        }
        true
    }

    /// An object retained for the life of the process.
    struct Held(CFType);

    // SAFETY: Core Foundation objects are reference counted atomically; an AXUIElement is
    // immutable and its functions may be called from any thread.
    unsafe impl Send for Held {}

    /// Every element handed out, by address. They stay retained, so a [`WindowRef`] can keep an
    /// address; equal elements (the same window asked twice) share one entry, so the table grows
    /// with the number of distinct windows seen, not with the number of queries.
    ///
    /// [`WindowRef`]: crate::types::WindowRef
    #[derive(Default)]
    struct ElementRegistry {
        inner: Mutex<RegistryInner>,
    }

    #[derive(Default)]
    struct RegistryInner {
        by_address: HashMap<usize, Held>,
        /// `CFHash` -> addresses
        by_hash: HashMap<usize, Vec<usize>>,
    }

    impl ElementRegistry {
        /// The address standing for `element`.
        fn intern(&self, element: CFType) -> Element {
            let address = element.as_CFTypeRef() as usize;
            // SAFETY: `element` is a live CF object.
            let hash = unsafe { CFHash(element.as_CFTypeRef()) };
            let mut inner = self.inner.lock();
            let known = inner.by_hash.get(&hash).and_then(|addresses| {
                addresses
                    .iter()
                    .copied()
                    .find(|a| *a == address || inner.by_address[a].0 == element)
            });
            if let Some(known) = known {
                return known;
            }
            inner.by_address.insert(address, Held(element));
            inner.by_hash.entry(hash).or_default().push(address);
            address
        }

        /// The element at `address` (retained for the caller); `None` for an address that was
        /// not handed out.
        fn get(&self, address: Element) -> Option<CFType> {
            self.inner
                .lock()
                .by_address
                .get(&address)
                .map(|held| held.0.clone())
        }
    }

    // ------------------------------------------------------------------ Quartz
    pub(super) struct SystemQuartz;

    impl Quartz for SystemQuartz {
        fn seconds_since_last_event(&self, state: i32, event_type: u32) -> Option<f64> {
            // SAFETY: plain value arguments.
            Some(unsafe { CGEventSourceSecondsSinceLastEventType(state, event_type) })
        }

        fn session_info(&self) -> Option<SessionInfo> {
            // SAFETY: no arguments; the copy rule gives us ownership of the result.
            let raw = unsafe { CGSessionCopyCurrentDictionary() };
            if raw.is_null() {
                return None;
            }
            // SAFETY: owned reference from a "Copy" function.
            let dict = unsafe { CFType::wrap_under_create_rule(raw.cast()) };
            let flag = |key: &str| dict_value(dict.as_CFTypeRef().cast(), key).map(|v| truthy(&v));
            Some(SessionInfo {
                screen_locked: flag("CGSSessionScreenIsLocked").unwrap_or(false),
                on_console: flag("kCGSSessionOnConsoleKey"),
            })
        }

        fn warp_cursor(&self, x: f64, y: f64) -> i32 {
            // SAFETY: plain value argument.
            unsafe { CGWarpMouseCursorPosition(CgPoint { x, y }) }
        }

        fn associate_mouse_and_cursor(&self, connected: bool) -> i32 {
            // SAFETY: plain value argument.
            unsafe { CGAssociateMouseAndMouseCursorPosition(i32::from(connected)) }
        }

        fn main_display_height(&self) -> Option<f64> {
            // SAFETY: no arguments / a plain value argument; both work on any thread.
            let bounds = unsafe { CGDisplayBounds(CGMainDisplayID()) };
            (bounds.size.height > 0.0).then_some(bounds.size.height)
        }

        fn window_list(&self, options: u32, relative_to: u32) -> Option<Vec<CgWindow>> {
            // SAFETY: plain value arguments; the copy rule gives us ownership of the result.
            let raw = unsafe { CGWindowListCopyWindowInfo(options, relative_to) };
            if raw.is_null() {
                return None;
            }
            // SAFETY: owned reference from a "Copy" function.
            let array = unsafe { CFType::wrap_under_create_rule(raw) };
            // SAFETY: `raw` is a live CFArray (what CGWindowListCopyWindowInfo returns).
            let count = unsafe { CFArrayGetCount(raw.cast()) };
            let mut windows = Vec::with_capacity(count.max(0) as usize);
            for index in 0..count {
                // SAFETY: `index` is below the count of the live array.
                let item = unsafe { CFArrayGetValueAtIndex(raw.cast(), index) };
                if item.is_null() {
                    continue;
                }
                windows.push(cg_window(item.cast()));
            }
            drop(array);
            Some(windows)
        }
    }

    /// One `CFDictionary` of the window list.
    fn cg_window(info: CFDictionaryRef) -> CgWindow {
        let integer = |key: &str| number_of(dict_value(info, key)).and_then(|n| n.to_i64());
        let bounds = dict_value(info, "kCGWindowBounds")
            .filter(|b| is_type(b, unsafe { CFDictionaryGetTypeID() }))
            .and_then(|b| {
                let dict: CFDictionaryRef = b.as_CFTypeRef().cast();
                let value = |key: &str| number_of(dict_value(dict, key)).and_then(|n| n.to_f64());
                Some((value("X")?, value("Y")?, value("Width")?, value("Height")?))
            });
        CgWindow {
            layer: integer("kCGWindowLayer").unwrap_or(0),
            alpha: number_of(dict_value(info, "kCGWindowAlpha"))
                .and_then(|n| n.to_f64())
                .unwrap_or(1.0),
            owner_pid: integer("kCGWindowOwnerPID").unwrap_or(0),
            bounds,
        }
    }

    // ------------------------------------------------------------------ AppKit
    pub(super) struct SystemAppKit;

    impl AppKit for SystemAppKit {
        fn frontmost_pid(&self) -> Option<i32> {
            let app = NSWorkspace::sharedWorkspace().frontmostApplication()?;
            Some(app.processIdentifier())
        }

        fn running_application(&self, pid: i32) -> Option<RunningApp> {
            let app = NSRunningApplication::runningApplicationWithProcessIdentifier(pid)?;
            Some(RunningApp {
                terminated: app.isTerminated(),
                hidden: app.isHidden(),
                bundle_id: app
                    .bundleIdentifier()
                    .map(|id| id.to_string())
                    .unwrap_or_default(),
            })
        }

        fn activate(&self, pid: i32, options: u64) -> bool {
            let Some(app) = NSRunningApplication::runningApplicationWithProcessIdentifier(pid)
            else {
                return false;
            };
            app.activateWithOptions(NSApplicationActivationOptions(options as usize))
        }

        fn set_activation_policy(&self, policy: i64) -> bool {
            let Some(mtm) = MainThreadMarker::new() else {
                log::debug!("setActivationPolicy needs the main thread");
                return false;
            };
            NSApplication::sharedApplication(mtm)
                .setActivationPolicy(NSApplicationActivationPolicy(policy as isize))
        }

        fn mouse_location(&self) -> Option<(f64, f64)> {
            let point = NSEvent::mouseLocation();
            Some((point.x, point.y))
        }
    }

    // ------------------------------------------------------------ Accessibility
    pub(super) struct SystemAccessibility {
        registry: Arc<ElementRegistry>,
    }

    #[cfg(test)]
    impl SystemAccessibility {
        pub(super) fn new() -> Self {
            Self {
                registry: Arc::new(ElementRegistry::default()),
            }
        }
    }

    impl SystemAccessibility {
        /// `raw` (owned) as the address of an interned element.
        fn adopt(&self, raw: CFTypeRef) -> Element {
            if raw.is_null() {
                return 0;
            }
            // SAFETY: an owned reference from a "Create"/"Copy" function.
            self.registry
                .intern(unsafe { CFType::wrap_under_create_rule(raw) })
        }

        fn classify(&self, value: CFType) -> AxAttr {
            // SAFETY: the type-id functions take no arguments.
            let (boolean, string, array, ax_element, ax_value) = unsafe {
                (
                    CFBooleanGetTypeID(),
                    CFStringGetTypeID(),
                    CFArrayGetTypeID(),
                    AXUIElementGetTypeID(),
                    AXValueGetTypeID(),
                )
            };
            // SAFETY: `value` is a live CF object.
            let type_id = unsafe { CFGetTypeID(value.as_CFTypeRef()) };
            if type_id == boolean {
                AxAttr::Bool(truthy(&value))
            } else if type_id == string {
                match value.downcast::<CFString>() {
                    Some(text) => AxAttr::Text(text.to_string()),
                    None => AxAttr::Other,
                }
            } else if type_id == ax_element {
                AxAttr::Element(self.registry.intern(value))
            } else if type_id == array {
                let raw: CFTypeRef = value.as_CFTypeRef();
                // SAFETY: `raw` is a live CFArray.
                let count = unsafe { CFArrayGetCount(raw.cast()) };
                let elements = (0..count)
                    .filter_map(|index| {
                        // SAFETY: `index` is below the count of the live array.
                        let item = unsafe { CFArrayGetValueAtIndex(raw.cast(), index) };
                        // SAFETY: owned by the array; the get rule retains it.
                        let item = (!item.is_null())
                            .then(|| unsafe { CFType::wrap_under_get_rule(item) })?;
                        is_type(&item, ax_element).then(|| self.registry.intern(item))
                    })
                    .collect();
                AxAttr::Elements(elements)
            } else if type_id == ax_value {
                AxAttr::Value(ax_value_of(&value))
            } else {
                AxAttr::Other
            }
        }
    }

    /// What can be told of an `AXValue`: its CGPoint/CGSize, or else its description.
    fn ax_value_of(value: &CFType) -> AxValue {
        let raw = value.as_CFTypeRef();
        // SAFETY: `raw` is a live AXValue; the out pointers match the requested types.
        let unpacked = unsafe {
            match AXValueGetType(raw) {
                AX_VALUE_CGPOINT => {
                    let mut point = CgPoint::default();
                    (AXValueGetValue(raw, AX_VALUE_CGPOINT, (&raw mut point).cast()) != 0)
                        .then_some(AxStruct::Point(point.x, point.y))
                }
                AX_VALUE_CGSIZE => {
                    let mut size = CgSize::default();
                    (AXValueGetValue(raw, AX_VALUE_CGSIZE, (&raw mut size).cast()) != 0)
                        .then_some(AxStruct::Size(size.width, size.height))
                }
                _ => None,
            }
        };
        let description = if unpacked.is_some() {
            String::new()
        } else {
            // SAFETY: `raw` is a live CF object; the copy rule gives us the string.
            let text = unsafe { CFCopyDescription(raw) };
            if text.is_null() {
                String::new()
            } else {
                // SAFETY: owned reference from a "Copy" function.
                unsafe { CFString::wrap_under_create_rule(text) }.to_string()
            }
        };
        AxValue {
            unpacked,
            description,
        }
    }

    impl Accessibility for SystemAccessibility {
        fn create_application(&self, pid: i32) -> Element {
            // SAFETY: plain value argument.
            self.adopt(unsafe { AXUIElementCreateApplication(pid) })
        }

        fn create_system_wide(&self) -> Element {
            // SAFETY: no arguments.
            self.adopt(unsafe { AXUIElementCreateSystemWide() })
        }

        fn set_messaging_timeout(&self, element: Element, seconds: f64) -> i32 {
            let Some(element) = self.registry.get(element) else {
                return AX_ERROR_INVALID_UI_ELEMENT;
            };
            // SAFETY: `element` is a live AXUIElement.
            unsafe { AXUIElementSetMessagingTimeout(element.as_CFTypeRef(), seconds as f32) }
        }

        fn copy_attribute(&self, element: Element, name: &str) -> (i32, Option<AxAttr>) {
            let Some(element) = self.registry.get(element) else {
                return (AX_ERROR_INVALID_UI_ELEMENT, None);
            };
            let name = CFString::new(name);
            let mut raw: CFTypeRef = std::ptr::null();
            // SAFETY: live element and attribute name; `raw` is an out pointer.
            let err = unsafe {
                AXUIElementCopyAttributeValue(
                    element.as_CFTypeRef(),
                    name.as_concrete_TypeRef(),
                    &mut raw,
                )
            };
            if err != 0 || raw.is_null() {
                return (err, None);
            }
            // SAFETY: an owned reference from a "Copy" function.
            let value = unsafe { CFType::wrap_under_create_rule(raw) };
            (err, Some(self.classify(value)))
        }

        fn set_bool_attribute(&self, element: Element, name: &str, value: bool) -> i32 {
            let Some(element) = self.registry.get(element) else {
                return AX_ERROR_INVALID_UI_ELEMENT;
            };
            let name = CFString::new(name);
            let value = CFBoolean::from(value);
            // SAFETY: live element, attribute name and value.
            unsafe {
                AXUIElementSetAttributeValue(
                    element.as_CFTypeRef(),
                    name.as_concrete_TypeRef(),
                    value.as_CFTypeRef(),
                )
            }
        }

        fn perform_action(&self, element: Element, action: &str) -> i32 {
            let Some(element) = self.registry.get(element) else {
                return AX_ERROR_INVALID_UI_ELEMENT;
            };
            let action = CFString::new(action);
            // SAFETY: live element and action name.
            unsafe {
                AXUIElementPerformAction(element.as_CFTypeRef(), action.as_concrete_TypeRef())
            }
        }

        fn element_at_position(&self, system: Element, x: f64, y: f64) -> (i32, Option<Element>) {
            let Some(system) = self.registry.get(system) else {
                return (AX_ERROR_INVALID_UI_ELEMENT, None);
            };
            let mut raw: CFTypeRef = std::ptr::null();
            // SAFETY: live element; `raw` is an out pointer.
            let err = unsafe {
                AXUIElementCopyElementAtPosition(
                    system.as_CFTypeRef(),
                    x as f32,
                    y as f32,
                    &mut raw,
                )
            };
            if err != 0 {
                return (err, None);
            }
            match self.adopt(raw) {
                0 => (AX_ERROR_FAILURE, None),
                element => (err, Some(element)),
            }
        }

        fn pid_of(&self, element: Element) -> (i32, Option<i32>) {
            let Some(element) = self.registry.get(element) else {
                return (AX_ERROR_INVALID_UI_ELEMENT, None);
            };
            let mut pid = 0;
            // SAFETY: live element; `pid` is an out pointer.
            let err = unsafe { AXUIElementGetPid(element.as_CFTypeRef(), &mut pid) };
            (err, (err == 0).then_some(pid))
        }

        fn is_process_trusted(&self) -> bool {
            // SAFETY: no arguments.
            unsafe { AXIsProcessTrusted() != 0 }
        }

        fn is_process_trusted_prompting(&self) -> bool {
            // SAFETY: a constant owned by the framework.
            let key = unsafe { CFString::wrap_under_get_rule(kAXTrustedCheckOptionPrompt) };
            let options = CFDictionary::from_CFType_pairs(&[(key, CFBoolean::true_value())]);
            // SAFETY: a live dictionary.
            unsafe { AXIsProcessTrustedWithOptions(options.as_concrete_TypeRef()) != 0 }
        }
    }

    struct SystemCoreFoundation {
        registry: Arc<ElementRegistry>,
    }

    impl CoreFoundation for SystemCoreFoundation {
        fn equal(&self, a: Element, b: Element) -> bool {
            let (Some(a), Some(b)) = (self.registry.get(a), self.registry.get(b)) else {
                return false;
            };
            // SAFETY: two live CF objects.
            unsafe { CFEqual(a.as_CFTypeRef(), b.as_CFTypeRef()) != 0 }
        }
    }

    // -------------------------------------------------------------- Foundation
    struct SystemFoundation;

    /// The `NSProcessInfo` activity object.
    struct Activity(Retained<ProtocolObject<dyn NSObjectProtocol>>);

    // SAFETY: the object is only held and handed back to `endActivity:`, which `NSProcessInfo`
    // documents as thread safe.
    unsafe impl Send for Activity {}
    // SAFETY: as above; nothing is read through a shared reference.
    unsafe impl Sync for Activity {}

    impl Foundation for SystemFoundation {
        fn begin_activity(&self, options: u64, reason: &str) -> Option<ActivityToken> {
            let info = NSProcessInfo::processInfo();
            let activity = info.beginActivityWithOptions_reason(
                NSActivityOptions(options),
                &NSString::from_str(reason),
            );
            Some(Box::new(Activity(activity)))
        }

        fn end_activity(&self, token: ActivityToken) {
            match token.downcast::<Activity>() {
                // SAFETY: the activity was begun by `beginActivity` of this process info and is
                // ended once, as the token is consumed.
                Ok(activity) => unsafe { NSProcessInfo::processInfo().endActivity(&activity.0) },
                Err(_) => log::debug!("endActivity: not one of our activity tokens"),
            }
        }
    }

    // ------------------------------------------------------------ AVFoundation
    pub(super) struct SystemCamera;

    /// `AVCaptureDevice`, loading AVFoundation first if this process has not.
    fn capture_device_class() -> Option<&'static AnyClass> {
        if let Some(class) = AnyClass::get(c"AVCaptureDevice") {
            return Some(class);
        }
        let bundle = NSBundle::bundleWithPath(&NSString::from_str(AVFOUNDATION_FRAMEWORK))?;
        // SAFETY: loading a system framework.
        if !unsafe { bundle.load() } {
            log::debug!("AVCaptureDevice unavailable: AVFoundation did not load");
            return None;
        }
        let class = AnyClass::get(c"AVCaptureDevice");
        if class.is_none() {
            log::debug!("AVCaptureDevice unavailable: no such class");
        }
        class
    }

    /// A block without captured state, in static memory (`BLOCK_IS_GLOBAL`): nothing to copy or
    /// dispose, valid forever.
    #[repr(C)]
    pub(super) struct GlobalBlock {
        isa: *const c_void,
        pub(super) flags: i32,
        reserved: i32,
        pub(super) invoke: unsafe extern "C" fn(*const GlobalBlock, Bool),
        descriptor: *const BlockDescriptor,
    }

    #[repr(C)]
    struct BlockDescriptor {
        reserved: usize,
        size: usize,
    }

    // SAFETY: immutable data only.
    unsafe impl Sync for GlobalBlock {}
    // SAFETY: immutable data only.
    unsafe impl Sync for BlockDescriptor {}

    /// `void (^)(BOOL granted)`: logs the answer.
    unsafe extern "C" fn camera_access_answered(_block: *const GlobalBlock, granted: Bool) {
        log::info!("Camera access granted: {}", granted.as_bool());
    }

    static BLOCK_DESCRIPTOR: BlockDescriptor = BlockDescriptor {
        reserved: 0,
        size: std::mem::size_of::<GlobalBlock>(),
    };

    pub(super) static CAMERA_ACCESS_BLOCK: GlobalBlock = GlobalBlock {
        isa: (&raw const _NSConcreteGlobalBlock).cast(),
        flags: 1 << 28, // BLOCK_IS_GLOBAL
        reserved: 0,
        invoke: camera_access_answered,
        descriptor: &BLOCK_DESCRIPTOR,
    };

    /// A block pointer as a message argument.
    #[repr(transparent)]
    struct BlockArgument(*const GlobalBlock);

    // SAFETY: a pointer to a block is encoded `@?`.
    unsafe impl Encode for BlockArgument {
        const ENCODING: Encoding = Encoding::Block;
    }

    impl CameraAuthorization for SystemCamera {
        fn authorization_status(&self, media_type: &str) -> Option<i64> {
            let class = capture_device_class()?;
            let media_type = NSString::from_str(media_type);
            // SAFETY: `+[AVCaptureDevice authorizationStatusForMediaType:]` takes an NSString
            // and returns an NSInteger.
            let status: isize =
                unsafe { msg_send![class, authorizationStatusForMediaType: &*media_type] };
            Some(status as i64)
        }

        fn request_access(&self, media_type: &str) {
            let Some(class) = capture_device_class() else {
                return;
            };
            let media_type = NSString::from_str(media_type);
            let handler = BlockArgument(&raw const CAMERA_ACCESS_BLOCK);
            // SAFETY: `+[AVCaptureDevice requestAccessForMediaType:completionHandler:]` takes an
            // NSString and a `void (^)(BOOL)` block, which the static block is.
            let _: () = unsafe {
                msg_send![class, requestAccessForMediaType: &*media_type, completionHandler: handler]
            };
        }
    }

    // -------------------------------------------------------------------- host
    pub(super) struct SystemHost;

    impl Host for SystemHost {
        fn is_macos(&self) -> bool {
            true
        }

        fn tool(&self, name: &str) -> Option<String> {
            if let Some(found) = which(name) {
                return Some(found.to_string_lossy().into_owned());
            }
            ["/usr/bin", "/usr/sbin", "/bin"]
                .iter()
                .map(|directory| format!("{directory}/{name}"))
                .find(|candidate| Path::new(candidate).exists())
        }

        fn run(&self, argv: &[String], timeout: f64) -> Result<i32, String> {
            run_command(argv, timeout)
                .map(|result| result.returncode)
                .map_err(|err| err.to_string())
        }

        fn spawn(&self, argv: &[String]) -> Result<(), String> {
            use std::process::{Command, Stdio};
            let (program, args) = argv.split_first().ok_or("no command to run")?;
            let mut child = Command::new(program)
                .args(args)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|err| err.to_string())?;
            let reaper = std::thread::Builder::new()
                .name("caffeinate-reaper".into())
                .spawn(move || {
                    let _ = child.wait();
                });
            if let Err(err) = reaper {
                log::debug!("caffeinate reaper not started: {err}");
            }
            Ok(())
        }

        fn lock_screen_immediate(&self, framework: &str) -> Result<i32, String> {
            let path = CString::new(framework).map_err(|err| err.to_string())?;
            // SAFETY: a NUL-terminated path.
            let library = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_LAZY) };
            if library.is_null() {
                return Err(format!("{framework} is not loadable"));
            }
            // SAFETY: a live handle and a NUL-terminated symbol name.
            let symbol = unsafe { libc::dlsym(library, c"SACLockScreenImmediate".as_ptr()) };
            let result = if symbol.is_null() {
                Err("no SACLockScreenImmediate".into())
            } else {
                // SAFETY: `int SACLockScreenImmediate(void)`
                let lock: unsafe extern "C" fn() -> i32 = unsafe { std::mem::transmute(symbol) };
                // SAFETY: as above.
                Ok(unsafe { lock() })
            };
            // SAFETY: the handle from `dlopen` above; `symbol` is not used afterwards.
            unsafe { libc::dlclose(library) };
            result
        }

        fn macos_major_version(&self) -> u32 {
            let version = NSProcessInfo::processInfo().operatingSystemVersion();
            u32::try_from(version.majorVersion).unwrap_or(0)
        }

        fn own_pid(&self) -> i32 {
            i32::try_from(std::process::id()).unwrap_or(0)
        }

        fn process_name(&self, pid: i64) -> Option<String> {
            GenericPlatform.app_process_name(Some(pid))
        }

        fn build_fingerprint(&self) -> Option<String> {
            super::build_fingerprint()
        }

        fn runs_helper_executable(&self) -> bool {
            super::runs_helper_executable()
        }
    }

    /// `CFBundleExecutable` of an `Info.plist` (XML or binary).
    pub(super) fn bundle_executable_name(plist: &Path) -> Result<String, String> {
        let bytes = std::fs::read(plist).map_err(|err| err.to_string())?;
        let (parsed, _format) =
            create_with_data(CFData::from_buffer(&bytes), kCFPropertyListImmutable)
                .map_err(|err| err.description().to_string())?;
        // SAFETY: an owned reference from a "Create" function.
        let parsed = unsafe { CFType::wrap_under_create_rule(parsed) };
        // SAFETY: the type-id functions take no arguments.
        if !is_type(&parsed, unsafe { CFDictionaryGetTypeID() }) {
            return Err("not a dictionary".into());
        }
        let name = dict_value(parsed.as_CFTypeRef().cast(), "CFBundleExecutable")
            .and_then(|v| v.downcast::<CFString>())
            .ok_or("no CFBundleExecutable")?;
        Ok(name.to_string())
    }
}

#[cfg(test)]
#[path = "macos_tests.rs"]
mod tests;
