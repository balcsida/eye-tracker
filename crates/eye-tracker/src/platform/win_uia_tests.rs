// Ported from the binding tests of tests/test_panes_windows_terminal.py and
// tests/test_panes_chromium_apps.py (the UI Automation client in platform/win_uia.py).
//
// Not ported (the Python tests were about the hand-written ctypes layer, which the `windows`
// crate's generated interfaces replace):
// - test_the_vtable_slots_match_uiautomationclient_h: the crate's vtables are generated from the
//   Windows metadata; only the control type ids remain to be checked.
// - test_control_view_releases_everything_it_handed_out and
//   test_control_view_releases_the_walker_when_creating_fails: they faked `_out_ptr` and
//   `_release`. Interface pointers are released when their wrapper is dropped, which the
//   compiler guarantees; `ControlView` keeps its elements in a `Vec` that `close` empties.
// - test_smoke_reads_and_frees_a_bstr: a `BSTR` frees itself; its truncation to 4096 units is
//   `bstr_text`, checked below on Windows.
// - test_binding_refuses_other_platforms and test_control_view_refuses_other_platforms
//   (monkeypatched `sys.platform`): run on the platforms that are not Windows.
// - test_binding_imports_everywhere_and_loads_nothing_until_used: `UiAutomation::new` is plain
//   data; checked below with `ComError`.

use std::cell::RefCell;

use super::*;

fn element(id: i32, focus: bool) -> UiaElement {
    UiaElement {
        runtime_id: Some(vec![42, id]),
        rect: Some(Rect::new(8, 80, 948, 952)),
        has_keyboard_focus: focus,
        offscreen: false,
    }
}

#[test]
fn test_binding_focus_asks_the_guard_right_before_set_focus() {
    // `UiAutomation::focus` with the COM layer replaced: SetFocus only if the guard agrees.
    let calls: RefCell<Vec<String>> = RefCell::new(Vec::new());
    let left = element(1, true);
    let right = element(2, false);
    let candidates = [(11, left), (12, right.clone())];

    let each = |visit: &mut dyn FnMut(&i32, UiaElement) -> Result<bool, UiaError>| -> Result<(), UiaError> {
        for (ptr, info) in &candidates {
            calls.borrow_mut().push(format!("visit {ptr}"));
            if visit(ptr, info.clone())? {
                return Ok(());
            }
        }
        Ok(())
    };
    let set_focus = |ptr: &i32| -> Result<(), UiaError> {
        calls.borrow_mut().push(format!("SetFocus {ptr}"));
        Ok(())
    };
    let is_right = |element: &UiaElement| *element == right;

    let refuse = || -> Result<bool, UiaError> {
        calls.borrow_mut().push("guard".to_string());
        Ok(false)
    };
    assert_eq!(
        focus_first(each, set_focus, &is_right, Some(&refuse)),
        Ok(false)
    );
    assert_eq!(*calls.borrow(), vec!["visit 11", "visit 12", "guard"]); // no SetFocus
    calls.borrow_mut().clear();
    let agree = || -> Result<bool, UiaError> { Ok(true) };
    assert_eq!(
        focus_first(each, set_focus, &is_right, Some(&agree)),
        Ok(true)
    );
    assert_eq!(*calls.borrow(), vec!["visit 11", "visit 12", "SetFocus 12"]);
}

#[test]
fn test_binding_focus_without_a_guard_and_without_a_match() {
    // New: the other two branches of the walk.
    let candidates = [(11, element(1, true))];
    let each = |visit: &mut dyn FnMut(&i32, UiaElement) -> Result<bool, UiaError>| -> Result<(), UiaError> {
        for (ptr, info) in &candidates {
            if visit(ptr, info.clone())? {
                break;
            }
        }
        Ok(())
    };
    let focused = RefCell::new(Vec::new());
    let set_focus = |ptr: &i32| -> Result<(), UiaError> {
        focused.borrow_mut().push(*ptr);
        Ok(())
    };
    assert_eq!(focus_first(each, set_focus, &|_| false, None), Ok(false));
    assert!(focused.borrow().is_empty());
    assert_eq!(focus_first(each, set_focus, &|_| true, None), Ok(true));
    assert_eq!(*focused.borrow(), vec![11]);
    // A SetFocus that fails is the caller's error.
    let failing = |_: &i32| -> Result<(), UiaError> {
        Err(ComError::new("IUIAutomationElement::SetFocus", 0x8000_4005).into())
    };
    assert_eq!(
        focus_first(each, failing, &|_| true, None),
        Err(UiaError::Com(ComError::new(
            "IUIAutomationElement::SetFocus",
            0x8000_4005
        )))
    );
}

#[test]
fn test_binding_creates_nothing_and_com_errors_are_unsigned() {
    let client = UiAutomation::new();
    client.close(); // nothing to release
    let error = ComError::new("FindAll", -2_146_233_083); // 0x80131505 as a signed HRESULT
    assert_eq!(error.hresult, 0x8013_1505);
    assert!(error.to_string().contains("0x80131505"), "{error}");
    assert!(UiaError::from(error).to_string().contains("FindAll failed"));
}

#[test]
fn test_the_control_type_ids_match_uiautomationclient_h() {
    assert_eq!(
        (
            UIA_EDIT_CONTROL_TYPE_ID,
            UIA_GROUP_CONTROL_TYPE_ID,
            UIA_DOCUMENT_CONTROL_TYPE_ID,
            UIA_PANE_CONTROL_TYPE_ID
        ),
        (50004, 50026, 50030, 50033)
    );
}

#[cfg(not(windows))]
#[test]
fn test_binding_refuses_other_platforms() {
    const HWND: isize = 0x40A2C;
    const PANE_CLASS: &str = "TermControl";
    let message = "only available on Windows";
    let error = UiAutomation::new().find(HWND, PANE_CLASS).unwrap_err();
    assert!(error.to_string().contains(message), "{error}");
    let error = UiAutomation::new().foreground_window().unwrap_err();
    assert!(error.to_string().contains(message), "{error}");
    let client = UiAutomation::new();
    let error = UiaTreeClient::control_view(&client, HWND)
        .err()
        .expect("an error");
    assert!(error.to_string().contains(message), "{error}");
}

// ------------------------------------------------------ Windows only (COM)
#[cfg(windows)]
mod windows_only {
    use super::*;
    use windows::Win32::System::Com::SAFEARRAY;
    use windows::Win32::System::Ole::{
        SafeArrayAccessData, SafeArrayCreateVector, SafeArrayUnaccessData,
    };
    use windows::Win32::System::Variant::{VT_BSTR, VT_I4};

    /// Run `f` on a fresh thread (COM gets a multithreaded apartment there).
    fn on_new_thread<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::spawn(f).join().expect("the thread finished")
    }

    #[test]
    fn test_smoke_creates_the_automation_object_without_asking_any_window() {
        let timeouts = on_new_thread(|| {
            let client = UiAutomation::with_timeouts(210, 480);
            // Reading the timeouts back proves the IUIAutomation2 binding.
            let timeouts = client.timeouts();
            client.close();
            timeouts
        });
        assert_eq!(timeouts, Ok(Some((210, 480))));
    }

    #[test]
    fn test_smoke_reads_an_integer_safearray() {
        // SAFETY: the array is created here, filled through its data pointer and handed to
        // `read_int_safearray`, which destroys it.
        unsafe {
            let psa = SafeArrayCreateVector(VT_I4, 0, 3);
            let mut data = std::ptr::null_mut();
            SafeArrayAccessData(psa, &mut data).expect("access");
            data.cast::<i32>().copy_from([42, 0x40A2C, -3].as_ptr(), 3);
            SafeArrayUnaccessData(psa).expect("unaccess");
            assert_eq!(
                com::read_int_safearray(psa),
                Ok(Some(vec![42, 0x40A2C, -3]))
            ); // and destroyed
            let other: *mut SAFEARRAY = SafeArrayCreateVector(VT_BSTR, 0, 1); // not a runtime id
            assert_eq!(com::read_int_safearray(other), Ok(None));
            assert_eq!(com::read_int_safearray(std::ptr::null_mut()), Ok(None));
        }
    }

    #[test]
    fn test_smoke_reads_a_bstr_cut_to_its_limit() {
        use windows::core::BSTR;
        assert_eq!(
            com::bstr_text(&BSTR::from("tiptap ProseMirror")),
            "tiptap ProseMirror"
        );
        let long = "x".repeat(5000);
        assert_eq!(com::bstr_text(&BSTR::from(long.as_str())).len(), 4096);
    }
}
