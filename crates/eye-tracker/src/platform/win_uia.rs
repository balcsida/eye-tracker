//! A minimal UI Automation client.
//!
//! Only what the pane providers need is used. For Windows Terminal: find the
//! descendants of a window with a given class name, read four properties of each
//! (runtime id, bounding rectangle, keyboard focus, off-screen) and give one of
//! them the keyboard focus. For desktop apps built on Chromium:
//! [`ControlView`], a step-by-step walk of the control view of one window
//! (first/last child, next/previous sibling), the focused element, and a few
//! properties per element (control type, class name, automation id, keyboard
//! focusability, process id). Names, values and text of elements are never read.
//! The COM interfaces come from the `windows` crate (`IUIAutomation` and friends);
//! the Python version called them through hand-counted vtable slots.
//!
//! # Timeouts
//!
//! `CUIAutomation8` hands out `IUIAutomation2`, whose connection and
//! transaction timeouts are set to [`CONNECTION_TIMEOUT_MS`] and
//! [`TRANSACTION_TIMEOUT_MS`] (UIA's defaults are 2 s and 20 s), so a
//! terminal that hangs makes a call fail quickly instead of blocking the pane
//! worker. Without `CUIAutomation8` the plain `CUIAutomation` is used with
//! UIA's defaults; the pane worker's own call timeout then still counts slow
//! calls as failures.
//!
//! # Threads
//!
//! COM is initialised as multithreaded (MTA) on each calling thread, once per
//! thread, the first time a [`UiAutomation`] is used there; it is never
//! uninitialised (the pane worker's thread lives for the session). A thread
//! already initialised as single-threaded (the GUI thread) cannot be used:
//! [`ComError`] is returned. One automation object is created per
//! [`UiAutomation`] and shared by its MTA threads; [`UiAutomation::close`]
//! releases it.
//!
//! Every interface pointer obtained is released when its wrapper is dropped, also
//! when a call fails; a failed HRESULT is a [`ComError`].
//!
//! The module builds on every platform: nothing Windows-specific is used until
//! a method is called, and then off Windows [`UiaError::Unavailable`] is returned.

use thiserror::Error;

use crate::panes::providers::chromium_apps::{ElementView, UiaTreeClient};
use crate::panes::providers::windows_terminal::UiaClient;
use crate::types::Rect;

/// Connection timeout set on `IUIAutomation2` (milliseconds).
pub const CONNECTION_TIMEOUT_MS: u32 = 200;
/// Transaction timeout set on `IUIAutomation2` (milliseconds).
pub const TRANSACTION_TIMEOUT_MS: u32 = 500;

/// Control type ids (`UIA_...ControlTypeId` in `UIAutomationClient.h`).
pub const UIA_EDIT_CONTROL_TYPE_ID: i32 = 50004;
pub const UIA_GROUP_CONTROL_TYPE_ID: i32 = 50026;
pub const UIA_DOCUMENT_CONTROL_TYPE_ID: i32 = 50030;
pub const UIA_PANE_CONTROL_TYPE_ID: i32 = 50033;

/// A COM call returned a failed HRESULT (`hresult` is unsigned).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{what} failed (HRESULT 0x{hresult:08X})")]
pub struct ComError {
    pub what: String,
    pub hresult: u32,
}

impl ComError {
    /// `hresult` may be signed, as COM returns it.
    pub fn new(what: &str, hresult: i64) -> Self {
        Self {
            what: what.to_string(),
            hresult: (hresult & 0xFFFF_FFFF) as u32,
        }
    }
}

/// Why a UI Automation call failed (the Python `OSError`, of which `ComError` was a subclass).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum UiaError {
    #[error(transparent)]
    Com(#[from] ComError),
    /// Not Windows.
    #[error("UI Automation is only available on Windows")]
    Unavailable,
}

/// What is read of one UI Automation element (no names, no text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiaElement {
    /// `GetRuntimeId`: unique among the elements on the desktop while the
    /// element exists; `None` if it could not be read.
    pub runtime_id: Option<Vec<i32>>,
    /// Bounding rectangle in physical pixels, global coordinates; `None` if empty.
    pub rect: Option<Rect>,
    pub has_keyboard_focus: bool,
    pub offscreen: bool,
}

/// An element handed out by a [`ControlView`] (valid until it closes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UiaNode(pub usize);

/// Calls `set_focus` on the first element `each` visits that `matches` accepts.
///
/// `each` calls the visitor it is given for every candidate until the visitor returns `true`.
/// `guard` is asked right before `set_focus`: when it returns `false` nothing is focused.
fn focus_first<E>(
    each: impl FnOnce(&mut dyn FnMut(&E, UiaElement) -> Result<bool, UiaError>) -> Result<(), UiaError>,
    set_focus: impl Fn(&E) -> Result<(), UiaError>,
    matches: &dyn Fn(&UiaElement) -> bool,
    guard: Option<&dyn Fn() -> Result<bool, UiaError>>,
) -> Result<bool, UiaError> {
    let mut done = false;
    each(&mut |element, info| {
        if !matches(&info) {
            return Ok(false);
        }
        if let Some(guard) = guard
            && !guard()?
        {
            return Ok(true); // stop: the window lost the foreground meanwhile
        }
        set_focus(element)?;
        done = true;
        Ok(true)
    })?;
    Ok(done)
}

// ------------------------------------------------------------------ client
/// A UI Automation client (see the module documentation for threads).
///
/// Creating one loads nothing: the automation object is created on first use,
/// on the calling thread.
pub struct UiAutomation {
    #[cfg_attr(not(windows), allow(dead_code))]
    connection_timeout_ms: u32,
    #[cfg_attr(not(windows), allow(dead_code))]
    transaction_timeout_ms: u32,
    #[cfg(windows)]
    state: parking_lot::Mutex<com::State>,
}

impl Default for UiAutomation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(windows)]
type Element = windows::Win32::UI::Accessibility::IUIAutomationElement;
#[cfg(not(windows))]
type Element = ();

impl UiAutomation {
    pub fn new() -> Self {
        Self::with_timeouts(CONNECTION_TIMEOUT_MS, TRANSACTION_TIMEOUT_MS)
    }

    pub fn with_timeouts(connection_timeout_ms: u32, transaction_timeout_ms: u32) -> Self {
        Self {
            connection_timeout_ms,
            transaction_timeout_ms,
            #[cfg(windows)]
            state: parking_lot::Mutex::new(com::State::default()),
        }
    }

    /// The descendants of window `hwnd` whose class name is `class_name`.
    pub fn find(&self, hwnd: isize, class_name: &str) -> Result<Vec<UiaElement>, UiaError> {
        let mut found = Vec::new();
        self.each(hwnd, class_name, &mut |_, info| {
            found.push(info);
            Ok(false)
        })?;
        Ok(found)
    }

    /// Give the keyboard focus to the first such descendant `matches` accepts.
    ///
    /// `guard` is asked right before `SetFocus` (after the walk that found
    /// the element): when it returns `false` nothing is focused. Callers pass a
    /// check that the window is still in the foreground, since `SetFocus` on
    /// a background window could bring it to the front.
    ///
    /// `false` when no element matches or the guard refuses; [`ComError`]
    /// when `SetFocus` fails.
    pub fn focus(
        &self,
        hwnd: isize,
        class_name: &str,
        matches: &dyn Fn(&UiaElement) -> bool,
        guard: Option<&dyn Fn() -> Result<bool, UiaError>>,
    ) -> Result<bool, UiaError> {
        focus_first(
            |visit| self.each(hwnd, class_name, visit),
            set_focus,
            matches,
            guard,
        )
    }

    /// The top-level window in the foreground (`None` if there is none).
    pub fn foreground_window(&self) -> Result<Option<isize>, UiaError> {
        #[cfg(windows)]
        {
            Ok(com::foreground_window())
        }
        #[cfg(not(windows))]
        {
            Err(UiaError::Unavailable)
        }
    }

    /// (connection, transaction) timeouts in ms; `None` without `IUIAutomation2`.
    pub fn timeouts(&self) -> Result<Option<(u32, u32)>, UiaError> {
        #[cfg(windows)]
        {
            self.com_timeouts()
        }
        #[cfg(not(windows))]
        {
            Err(UiaError::Unavailable)
        }
    }

    /// Release the automation object and its conditions (call on an MTA thread).
    pub fn close(&self) {
        #[cfg(windows)]
        {
            let state = std::mem::take(&mut *self.state.lock());
            drop(state);
        }
    }

    /// Call `visit(element, info)` for each matching descendant until it
    /// returns `true`.
    #[cfg_attr(not(windows), allow(clippy::unused_self))]
    fn each(
        &self,
        hwnd: isize,
        class_name: &str,
        visit: &mut dyn FnMut(&Element, UiaElement) -> Result<bool, UiaError>,
    ) -> Result<(), UiaError> {
        #[cfg(windows)]
        {
            self.com_each(hwnd, class_name, visit)
        }
        #[cfg(not(windows))]
        {
            let _ = (hwnd, class_name, visit);
            Err(UiaError::Unavailable)
        }
    }
}

fn set_focus(element: &Element) -> Result<(), UiaError> {
    #[cfg(windows)]
    {
        com::set_focus(element)
    }
    #[cfg(not(windows))]
    {
        let _ = element;
        Err(UiaError::Unavailable)
    }
}

impl UiaClient for UiAutomation {
    fn find(&self, hwnd: isize, class_name: &str) -> Result<Vec<UiaElement>, UiaError> {
        UiAutomation::find(self, hwnd, class_name)
    }

    fn focus(
        &self,
        hwnd: isize,
        class_name: &str,
        matches: &dyn Fn(&UiaElement) -> bool,
        guard: Option<&dyn Fn() -> Result<bool, UiaError>>,
    ) -> Result<bool, UiaError> {
        UiAutomation::focus(self, hwnd, class_name, matches, guard)
    }

    fn foreground_window(&self) -> Result<Option<isize>, UiaError> {
        UiAutomation::foreground_window(self)
    }

    fn close(&self) {
        UiAutomation::close(self);
    }
}

impl UiaTreeClient for UiAutomation {
    fn control_view(&self, hwnd: isize) -> Result<Box<dyn ElementView>, UiaError> {
        #[cfg(windows)]
        {
            Ok(Box::new(self.control_view(hwnd)?))
        }
        #[cfg(not(windows))]
        {
            let _ = hwnd;
            Err(UiaError::Unavailable)
        }
    }

    fn foreground_window(&self) -> Result<Option<isize>, UiaError> {
        UiAutomation::foreground_window(self)
    }

    fn close(&self) {
        UiAutomation::close(self);
    }
}

// ------------------------------------------------------------ the real COM
#[cfg(windows)]
pub use com::ControlView;

#[cfg(windows)]
mod com {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::ffi::c_void;

    use windows::Win32::Foundation::{HWND, RPC_E_CHANGED_MODE};
    use windows::Win32::System::Com::{
        CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, SAFEARRAY,
    };
    use windows::Win32::System::Ole::{
        SafeArrayAccessData, SafeArrayDestroy, SafeArrayGetDim, SafeArrayGetElemsize,
        SafeArrayGetLBound, SafeArrayGetUBound, SafeArrayGetVartype, SafeArrayUnaccessData,
    };
    use windows::Win32::System::Variant::{VARIANT, VT_I4};
    use windows::Win32::UI::Accessibility::{
        CUIAutomation, CUIAutomation8, IUIAutomation, IUIAutomation2, IUIAutomationCondition,
        IUIAutomationTreeWalker, TreeScope_Descendants, UIA_ClassNamePropertyId,
    };
    use windows::Win32::UI::WindowsAndMessaging::{GA_ROOT, GetAncestor, GetForegroundWindow};
    use windows::core::BSTR;

    use super::{ComError, Element, Rect, UiAutomation, UiaElement, UiaError, UiaNode};
    use crate::panes::providers::chromium_apps::ElementView;

    /// Longest runtime id accepted (real ones have 2-4 parts).
    const MAX_RUNTIME_ID: usize = 32;
    /// Longest class name or automation id read (UTF-16 units); the rest is cut off.
    const MAX_BSTR: usize = 4096;
    const E_POINTER: i64 = 0x8000_4003;

    /// What a [`UiAutomation`] holds once used. The COM objects are free-threaded (created in
    /// the MTA), so they may be used and released from any MTA thread.
    #[derive(Default)]
    pub(super) struct State {
        uia: Option<IUIAutomation>,
        /// Set when `IUIAutomation2` is there and its timeouts were applied.
        timeouts: Option<IUIAutomation2>,
        conditions: HashMap<String, IUIAutomationCondition>,
    }

    // SAFETY: see the type's documentation.
    unsafe impl Send for State {}

    thread_local! {
        static COM_READY: Cell<bool> = const { Cell::new(false) };
    }

    fn com<T>(result: windows::core::Result<T>, what: &str) -> Result<T, UiaError> {
        result.map_err(|e| ComError::new(what, i64::from(e.code().0)).into())
    }

    /// Like [`com`], for a method whose out interface pointer may be null (an empty `Error`
    /// with a success code, which is `None` here).
    fn optional<T>(result: windows::core::Result<T>, what: &str) -> Result<Option<T>, UiaError> {
        match result {
            Ok(value) => Ok(Some(value)),
            Err(e) if e.code().is_ok() => Ok(None),
            Err(e) => Err(ComError::new(what, i64::from(e.code().0)).into()),
        }
    }

    fn init_thread() -> Result<(), UiaError> {
        if COM_READY.get() {
            return Ok(());
        }
        // SAFETY: plain call; the result is checked.
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if hr == RPC_E_CHANGED_MODE {
            return Err(ComError::new(
                "CoInitializeEx (thread is single-threaded)",
                i64::from(hr.0),
            )
            .into());
        }
        // S_OK, or S_FALSE when already initialised
        if hr.is_err() {
            return Err(ComError::new("CoInitializeEx", i64::from(hr.0)).into());
        }
        COM_READY.set(true);
        Ok(())
    }

    pub(super) fn foreground_window() -> Option<isize> {
        // SAFETY: plain calls without pointers.
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd.0.is_null() {
                return None;
            }
            let root = GetAncestor(hwnd, GA_ROOT);
            Some(if root.0.is_null() { hwnd } else { root }.0 as isize)
        }
    }

    pub(super) fn set_focus(element: &Element) -> Result<(), UiaError> {
        // SAFETY: `element` is a live interface.
        com(
            unsafe { element.SetFocus() },
            "IUIAutomationElement::SetFocus",
        )
    }

    fn hwnd_of(hwnd: isize) -> HWND {
        HWND(hwnd as *mut c_void)
    }

    pub(super) fn bstr_text(bstr: &BSTR) -> String {
        let wide: &[u16] = bstr;
        String::from_utf16_lossy(&wide[..wide.len().min(MAX_BSTR)])
    }

    /// Bounding rectangle in physical pixels, global coordinates; `None` if empty.
    fn read_rect(element: &Element) -> Result<Option<Rect>, UiaError> {
        // SAFETY: `element` is a live interface.
        let rect = com(
            unsafe { element.CurrentBoundingRectangle() },
            "get_CurrentBoundingRectangle",
        )?;
        let (width, height) = (rect.right - rect.left, rect.bottom - rect.top);
        Ok((width > 0 && height > 0).then(|| Rect::new(rect.left, rect.top, width, height)))
    }

    fn read_runtime_id(element: &Element) -> Result<Option<Vec<i32>>, UiaError> {
        // SAFETY: `element` is a live interface; the array it returns is consumed below.
        let Ok(psa) = (unsafe { element.GetRuntimeId() }) else {
            return Ok(None);
        };
        // SAFETY: `psa` was just handed out by `GetRuntimeId`, which transfers its ownership.
        Ok(unsafe { read_int_safearray(psa) }?.filter(|ids| !ids.is_empty()))
    }

    /// Destroys the `SAFEARRAY` when dropped.
    struct SafeArray(*mut SAFEARRAY);

    impl Drop for SafeArray {
        fn drop(&mut self) {
            // SAFETY: `read_int_safearray` owns the array.
            let _ = unsafe { SafeArrayDestroy(self.0) };
        }
    }

    /// The values of a one-dimensional `SAFEARRAY` of 32-bit integers, which is
    /// destroyed. `None` if it is anything else.
    ///
    /// # Safety
    /// `psa` is null or a valid `SAFEARRAY` that the caller owns and does not use afterwards.
    pub(crate) unsafe fn read_int_safearray(
        psa: *mut SAFEARRAY,
    ) -> Result<Option<Vec<i32>>, UiaError> {
        if psa.is_null() {
            return Ok(None);
        }
        let array = SafeArray(psa);
        // SAFETY: `psa` is valid (see above) for every call below.
        unsafe {
            if SafeArrayGetDim(array.0) != 1 {
                return Ok(None);
            }
            match SafeArrayGetVartype(array.0) {
                Ok(vt) => {
                    if vt != VT_I4 {
                        return Ok(None);
                    }
                }
                Err(_) => {
                    if SafeArrayGetElemsize(array.0) as usize != size_of::<i32>() {
                        return Ok(None);
                    }
                }
            }
            let lower = com(SafeArrayGetLBound(array.0, 1), "SafeArrayGetLBound")?;
            let upper = com(SafeArrayGetUBound(array.0, 1), "SafeArrayGetUBound")?;
            let count = i64::from(upper) - i64::from(lower) + 1;
            if count <= 0 || count > MAX_RUNTIME_ID as i64 {
                return Ok(None);
            }
            let mut data: *mut c_void = std::ptr::null_mut();
            com(
                SafeArrayAccessData(array.0, &mut data),
                "SafeArrayAccessData",
            )?;
            let values = if data.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(data as *const i32, count as usize).to_vec())
            };
            let _ = SafeArrayUnaccessData(array.0);
            Ok(values)
        }
    }

    fn describe(element: &Element) -> Result<UiaElement, UiaError> {
        Ok(UiaElement {
            runtime_id: read_runtime_id(element)?,
            rect: read_rect(element)?,
            // SAFETY: `element` is a live interface.
            has_keyboard_focus: com(
                unsafe { element.CurrentHasKeyboardFocus() },
                "get_CurrentHasKeyboardFocus",
            )?
            .as_bool(),
            // SAFETY: as above.
            offscreen: com(
                unsafe { element.CurrentIsOffscreen() },
                "get_CurrentIsOffscreen",
            )?
            .as_bool(),
        })
    }

    impl UiAutomation {
        /// The automation object, created on first use (on the calling thread).
        fn automation(&self) -> Result<IUIAutomation, UiaError> {
            init_thread()?;
            let mut state = self.state.lock();
            if let Some(uia) = &state.uia {
                return Ok(uia.clone());
            }
            self.create(&mut state)
        }

        fn create(&self, state: &mut State) -> Result<IUIAutomation, UiaError> {
            // SAFETY: plain COM activation; the result is checked.
            match unsafe {
                CoCreateInstance::<_, IUIAutomation2>(&CUIAutomation8, None, CLSCTX_INPROC_SERVER)
            } {
                Ok(uia2) => {
                    let uia: IUIAutomation = uia2.clone().into();
                    state.timeouts = self.set_timeouts(&uia2).then_some(uia2);
                    state.uia = Some(uia.clone());
                    return Ok(uia);
                }
                Err(e) => log::debug!(
                    "CUIAutomation8 not available (0x{:08X}); no UIA timeouts",
                    e.code().0 as u32
                ),
            }
            // SAFETY: as above.
            let uia = com(
                unsafe {
                    CoCreateInstance::<_, IUIAutomation>(&CUIAutomation, None, CLSCTX_INPROC_SERVER)
                },
                "CoCreateInstance(CUIAutomation)",
            )?;
            state.timeouts = None;
            state.uia = Some(uia.clone());
            Ok(uia)
        }

        fn set_timeouts(&self, uia: &IUIAutomation2) -> bool {
            // SAFETY: `uia` is a live interface.
            let result = unsafe {
                com(
                    uia.SetConnectionTimeout(self.connection_timeout_ms),
                    "put_ConnectionTimeout",
                )
                .and_then(|()| {
                    com(
                        uia.SetTransactionTimeout(self.transaction_timeout_ms),
                        "put_TransactionTimeout",
                    )
                })
            };
            if let Err(e) = result {
                log::debug!("Could not set the UI Automation timeouts: {e}");
                return false;
            }
            true
        }

        pub(super) fn com_timeouts(&self) -> Result<Option<(u32, u32)>, UiaError> {
            self.automation()?;
            let Some(uia2) = self.state.lock().timeouts.clone() else {
                return Ok(None);
            };
            // SAFETY: `uia2` is a live interface.
            let connection = com(unsafe { uia2.ConnectionTimeout() }, "get_ConnectionTimeout")?;
            // SAFETY: as above.
            let transaction = com(
                unsafe { uia2.TransactionTimeout() },
                "get_TransactionTimeout",
            )?;
            Ok(Some((connection, transaction)))
        }

        /// A cached `ClassName == class_name` property condition.
        fn condition(
            &self,
            uia: &IUIAutomation,
            class_name: &str,
        ) -> Result<IUIAutomationCondition, UiaError> {
            if let Some(cached) = self.state.lock().conditions.get(class_name) {
                return Ok(cached.clone());
            }
            let value = VARIANT::from(class_name);
            // SAFETY: `uia` is a live interface; the condition keeps its own copy of the value.
            let condition = optional(
                unsafe { uia.CreatePropertyCondition(UIA_ClassNamePropertyId, &value) },
                "IUIAutomation::CreatePropertyCondition",
            )?
            .ok_or_else(|| ComError::new("IUIAutomation::CreatePropertyCondition", E_POINTER))?;
            let mut state = self.state.lock();
            // Another thread may have been quicker; keep its condition.
            Ok(state
                .conditions
                .entry(class_name.to_string())
                .or_insert(condition)
                .clone())
        }

        /// Every pointer is released before returning or failing.
        pub(super) fn com_each(
            &self,
            hwnd: isize,
            class_name: &str,
            visit: &mut dyn FnMut(&Element, UiaElement) -> Result<bool, UiaError>,
        ) -> Result<(), UiaError> {
            let uia = self.automation()?;
            let condition = self.condition(&uia, class_name)?;
            // SAFETY: `uia` is a live interface.
            let Some(root) = optional(
                unsafe { uia.ElementFromHandle(hwnd_of(hwnd)) },
                "IUIAutomation::ElementFromHandle",
            )?
            else {
                return Ok(());
            };
            // SAFETY: `root` is a live interface.
            let Some(array) = optional(
                unsafe { root.FindAll(TreeScope_Descendants, &condition) },
                "IUIAutomationElement::FindAll",
            )?
            else {
                return Ok(());
            };
            // SAFETY: `array` is a live interface.
            let length = com(
                unsafe { array.Length() },
                "IUIAutomationElementArray::get_Length",
            )?;
            for i in 0..length.max(0) {
                // SAFETY: as above.
                let Some(element) = optional(
                    unsafe { array.GetElement(i) },
                    "IUIAutomationElementArray::GetElement",
                )?
                else {
                    continue;
                };
                if visit(&element, describe(&element)?)? {
                    return Ok(());
                }
            }
            Ok(())
        }

        /// A walk of the control view of window `hwnd`; close it when done (it also closes when
        /// dropped).
        pub fn control_view(&self, hwnd: isize) -> Result<ControlView, UiaError> {
            ControlView::new(self.automation()?, hwnd)
        }
    }

    // -------------------------------------------------------------- tree walk
    /// Step through the control view of one window, one element at a time.
    ///
    /// Every element handed out is held until [`close`](ElementView::close) (or the drop)
    /// releases it, together with the tree walker; nothing is released earlier, so a caller
    /// bounds what it holds by bounding its walk. Only the properties below are read: never
    /// names, values or text.
    pub struct ControlView {
        uia: IUIAutomation,
        walker: Option<IUIAutomationTreeWalker>,
        held: Vec<Element>,
        root: Option<UiaNode>,
        closed: bool,
    }

    impl ControlView {
        fn new(uia: IUIAutomation, hwnd: isize) -> Result<Self, UiaError> {
            // SAFETY: `uia` is a live interface.
            let walker = optional(
                unsafe { uia.ControlViewWalker() },
                "IUIAutomation::get_ControlViewWalker",
            )?
            .ok_or_else(|| ComError::new("IUIAutomation::get_ControlViewWalker", E_POINTER))?;
            // SAFETY: as above.
            let root = optional(
                unsafe { uia.ElementFromHandle(hwnd_of(hwnd)) },
                "IUIAutomation::ElementFromHandle",
            )?;
            let mut view = Self {
                uia,
                walker: Some(walker),
                held: Vec::new(),
                root: None,
                closed: false,
            };
            view.root = view.hold(root);
            Ok(view)
        }

        /// How many elements are held (released by `close`).
        pub fn held(&self) -> usize {
            self.held.len()
        }

        fn hold(&mut self, element: Option<Element>) -> Option<UiaNode> {
            let element = element?;
            self.held.push(element);
            Some(UiaNode(self.held.len() - 1))
        }

        fn closed_error() -> UiaError {
            ComError::new("ControlView (closed)", E_POINTER).into()
        }

        fn element(&self, node: UiaNode) -> Result<&Element, UiaError> {
            if self.closed {
                return Err(Self::closed_error());
            }
            self.held
                .get(node.0)
                .ok_or_else(|| ComError::new("ControlView (unknown element)", E_POINTER).into())
        }

        fn step(
            &mut self,
            node: UiaNode,
            what: &str,
            walk: impl FnOnce(&IUIAutomationTreeWalker, &Element) -> windows::core::Result<Element>,
        ) -> Result<Option<UiaNode>, UiaError> {
            let element = self.element(node)?;
            let walker = self.walker.as_ref().ok_or_else(Self::closed_error)?;
            let next = optional(
                walk(walker, element),
                &format!("IUIAutomationTreeWalker::{what}"),
            )?;
            Ok(self.hold(next))
        }
    }

    impl Drop for ControlView {
        fn drop(&mut self) {
            ElementView::close(self);
        }
    }

    impl ElementView for ControlView {
        fn root(&self) -> Option<UiaNode> {
            self.root
        }

        fn first_child(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError> {
            // SAFETY (all four steps): live walker and element.
            self.step(node, "GetFirstChildElement", |w, e| unsafe {
                w.GetFirstChildElement(e)
            })
        }

        fn last_child(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError> {
            self.step(node, "GetLastChildElement", |w, e| unsafe {
                w.GetLastChildElement(e)
            })
        }

        fn next_sibling(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError> {
            self.step(node, "GetNextSiblingElement", |w, e| unsafe {
                w.GetNextSiblingElement(e)
            })
        }

        fn previous_sibling(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError> {
            self.step(node, "GetPreviousSiblingElement", |w, e| unsafe {
                w.GetPreviousSiblingElement(e)
            })
        }

        fn focused(&mut self) -> Result<Option<UiaNode>, UiaError> {
            if self.closed {
                return Err(Self::closed_error());
            }
            // SAFETY: `uia` is a live interface.
            let element = optional(
                unsafe { self.uia.GetFocusedElement() },
                "IUIAutomation::GetFocusedElement",
            )?;
            Ok(self.hold(element))
        }

        fn control_type(&self, node: UiaNode) -> Result<i32, UiaError> {
            let element = self.element(node)?;
            // SAFETY: `element` is a live interface (all property reads below).
            Ok(com(
                unsafe { element.CurrentControlType() },
                "get_CurrentControlType",
            )?
            .0)
        }

        fn process_id(&self, node: UiaNode) -> Result<u32, UiaError> {
            let element = self.element(node)?;
            Ok(com(
                unsafe { element.CurrentProcessId() },
                "get_CurrentProcessId",
            )? as u32)
        }

        fn class_name(&self, node: UiaNode) -> Result<String, UiaError> {
            let element = self.element(node)?;
            Ok(bstr_text(&com(
                unsafe { element.CurrentClassName() },
                "get_CurrentClassName",
            )?))
        }

        fn automation_id(&self, node: UiaNode) -> Result<String, UiaError> {
            let element = self.element(node)?;
            Ok(bstr_text(&com(
                unsafe { element.CurrentAutomationId() },
                "get_CurrentAutomationId",
            )?))
        }

        fn is_keyboard_focusable(&self, node: UiaNode) -> Result<bool, UiaError> {
            let element = self.element(node)?;
            Ok(com(
                unsafe { element.CurrentIsKeyboardFocusable() },
                "get_CurrentIsKeyboardFocusable",
            )?
            .as_bool())
        }

        fn is_offscreen(&self, node: UiaNode) -> Result<bool, UiaError> {
            let element = self.element(node)?;
            Ok(com(
                unsafe { element.CurrentIsOffscreen() },
                "get_CurrentIsOffscreen",
            )?
            .as_bool())
        }

        fn rect(&self, node: UiaNode) -> Result<Option<Rect>, UiaError> {
            read_rect(self.element(node)?)
        }

        fn runtime_id(&self, node: UiaNode) -> Result<Option<Vec<i32>>, UiaError> {
            read_runtime_id(self.element(node)?)
        }

        fn set_focus(&self, node: UiaNode) -> Result<(), UiaError> {
            set_focus(self.element(node)?)
        }

        /// Release every element handed out and the walker (idempotent).
        fn close(&mut self) {
            self.closed = true;
            while self.held.pop().is_some() {}
            self.walker = None;
        }
    }
}

#[cfg(test)]
#[path = "win_uia_tests.rs"]
mod tests;
