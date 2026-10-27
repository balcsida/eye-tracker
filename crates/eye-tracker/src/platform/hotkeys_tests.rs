//! Tests for global hotkey parsing, formatting and the native managers (tests/test_hotkeys.py).
//!
//! Nothing here synthesises key presses. On Windows the live tests register an unusual
//! combination and deliver `WM_HOTKEY` to the manager's own thread with `PostThreadMessageW`;
//! macOS and X11 code paths run against fakes.
//!
//! Not ported:
//! * `test_parse_rejects_non_string`, `test_invalid_register_arguments` (TypeError) and the
//!   `ValueError` case of `test_fourcc`: the Rust types rule these inputs out at compile time.
//! * `test_x11_parse_xkb_options` cases `str`, `array` and `int`: the property value is bytes.
//! * `test_factory_x11_without_xlib`: x11rb is compiled in, there is no import that can fail.
//! * `test_x11_error_catcher_protocol`'s return value: python-xlib's handler protocol; the
//!   "first error wins" part is ported.
//! * `test_mac_ctypes_binding_with_fake_library`, `test_mac_ctypes_layout_binding_with_fake_library`
//!   and `test_mac_layout_api_missing_falls_back_to_us_positions`: they replace the Carbon dylib
//!   behind ctypes; the Rust binding links the frameworks directly, so they cannot be faked.
//!   The manager logic above the binding is covered with [`FakeCarbon`].
//! * `test_mac_live_register_smoke`: Carbon must be called on the process's main thread, which
//!   `cargo test` never gives a test.
//! * `test_x11_live_register`: needs a real X server.
//! * `test_windows_punctuation_without_a_key_is_refused`: it monkeypatched `_win_virtual_key`;
//!   [`test_windows_virtual_key_prefers_active_layout_for_punctuation`] covers the logic.
//! * `test_x11_loop_waits_without_timeout` checks the constant the loop passes to `poll`
//!   instead of patching `select`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use parking_lot::Mutex;

use super::*;
use crate::config::{LEGACY_HOTKEYS, Platform, default_hotkey};
use crate::testing::capture_logs;

// Unusual combinations that no desktop or application binds by default.
const COMBO: &str = "ctrl+alt+shift+f24";
const COMBO_2: &str = "ctrl+alt+shift+f21";
const MAC_COMBO: &str = "ctrl+alt+shift+f20"; // macOS keyboards stop at F20

fn hk(text: &str) -> Hotkey {
    parse_hotkey(text).unwrap()
}

fn set(items: &[&'static str]) -> BTreeSet<&'static str> {
    items.iter().copied().collect()
}

fn noop() -> HotkeyCallback {
    Arc::new(|| {})
}

/// A callback that appends `value` to `log`.
fn push<T: Clone + Send + Sync + 'static>(log: &Arc<Mutex<Vec<T>>>, value: T) -> HotkeyCallback {
    let log = log.clone();
    Arc::new(move || log.lock().push(value.clone()))
}

fn defaults(platform: Platform) -> [String; 3] {
    ["t", "p", "c"].map(|key| default_hotkey(platform, key))
}

// =========================================================================== parsing
#[test]
fn test_parse_valid() {
    let cases = [
        ("Ctrl+Alt+P", "ctrl+alt+p"),
        ("ctrl+alt+p", "ctrl+alt+p"),
        ("CTRL+ALT+P", "ctrl+alt+p"),
        ("  ctrl +  alt+ p  ", "ctrl+alt+p"),
        ("alt+ctrl+p", "ctrl+alt+p"),
        ("Ctrl+Shift+F5", "ctrl+shift+f5"),
        ("ctrl+alt+shift+meta+f24", "ctrl+alt+shift+meta+f24"),
        ("Ctrl+Alt+Space", "ctrl+alt+space"),
        ("ctrl+alt+esc", "ctrl+alt+escape"),
        ("ctrl+alt+Return", "ctrl+alt+enter"),
        ("ctrl+alt+Del", "ctrl+alt+delete"),
        ("ctrl+alt+Ins", "ctrl+alt+insert"),
        ("ctrl+alt+PgUp", "ctrl+alt+pageup"),
        ("ctrl+alt+PgDown", "ctrl+alt+pagedown"),
        ("ctrl+alt+Page Up", "ctrl+alt+pageup"),
        ("ctrl+alt+page_down", "ctrl+alt+pagedown"),
        ("ctrl+alt+Left", "ctrl+alt+left"),
        ("ctrl+alt+Backspace", "ctrl+alt+backspace"),
        ("ctrl+alt+Tab", "ctrl+alt+tab"),
        ("ctrl+alt+Home", "ctrl+alt+home"),
        ("ctrl+alt+End", "ctrl+alt+end"),
        ("ctrl+alt+7", "ctrl+alt+7"),
        ("ctrl+alt+-", "ctrl+alt+minus"),
        ("ctrl+alt+=", "ctrl+alt+equal"),
        ("ctrl+alt+[", "ctrl+alt+bracketleft"),
        ("ctrl+alt+]", "ctrl+alt+bracketright"),
        ("ctrl+alt+\\", "ctrl+alt+backslash"),
        ("ctrl+alt+;", "ctrl+alt+semicolon"),
        ("ctrl+alt+'", "ctrl+alt+quote"),
        ("ctrl+alt+`", "ctrl+alt+grave"),
        ("ctrl+alt+,", "ctrl+alt+comma"),
        ("ctrl+alt+.", "ctrl+alt+period"),
        ("ctrl+alt+/", "ctrl+alt+slash"),
        ("ctrl+alt+comma", "ctrl+alt+comma"),
        ("shift+f5", "shift+f5"), // Shift alone is fine for non-typing keys
        ("shift+escape", "shift+escape"),
        ("⌃⌥P", "ctrl+alt+p"),
        ("⌃⌥⇧⌘F24", "ctrl+alt+shift+meta+f24"),
        ("⌘⇧G", "shift+meta+g"),
        ("⌃ + ⌥ + P", "ctrl+alt+p"),
        ("ctrl+⌥P", "ctrl+alt+p"),
        ("⌘Space", "meta+space"),
        ("⌃⌥⎋", "ctrl+alt+escape"),
        ("⌃⌥←", "ctrl+alt+left"),
    ];
    for (text, expected) in cases {
        assert_eq!(hk(text).to_string(), expected, "{text}");
    }
}

#[test]
fn test_modifier_aliases() {
    let cases = [
        ("cmd", "meta"),
        ("Command", "meta"),
        ("win", "meta"),
        ("Windows", "meta"),
        ("super", "meta"),
        ("META", "meta"),
        ("option", "alt"),
        ("Opt", "alt"),
        ("control", "ctrl"),
        ("Ctl", "ctrl"),
        ("Shift", "shift"),
    ];
    for (alias, canonical) in cases {
        let other = if canonical != "shift" {
            "shift"
        } else {
            "ctrl"
        };
        let parsed = hk(&format!("{alias}+{other}+g"));
        let expected: BTreeSet<&str> = [canonical, other].into_iter().collect();
        assert_eq!(parsed.modifiers(), &expected, "{alias}");
        assert_eq!(parsed.key(), "g");
    }
}

#[test]
fn test_parse_is_case_insensitive_and_whitespace_tolerant() {
    let variants = [
        "Ctrl+Alt+P",
        "ctrl+alt+p",
        "cTrL+aLt+P",
        " Ctrl + Alt + P ",
        "\tctrl+alt+p\n",
    ];
    let parsed: HashSet<Hotkey> = variants.iter().map(|v| hk(v)).collect();
    let expected: HashSet<Hotkey> = [Hotkey::new(["ctrl", "alt"], "p").unwrap()].into();
    assert_eq!(parsed, expected);
}

#[test]
fn test_duplicate_modifiers_are_merged() {
    assert_eq!(hk("ctrl+control+p"), hk("ctrl+p"));
}

#[test]
fn test_parse_invalid_raises() {
    let cases = [
        "",
        "   ",
        "p", // key only
        "F5",
        "space",
        "ctrl+alt", // modifiers only
        "ctrl",
        "⌃⌥",
        "ctrl+", // missing key
        "+p",    // empty modifier
        "ctrl++p",
        "ctrl++", // the '+' key
        "+",
        "hyper+p",      // unknown modifier
        "ctrl+alt+p+q", // 'p' is not a modifier
        "ctrl+f25",
        "ctrl+f0",
        "ctrl+alt+pp", // unknown key
        "ctrl+alt+é",
        "shift+a", // would block typing capitals
        "shift+1",
        "shift+space",
        "shift+tab",
        "shift+enter",
        "shift+/",
        // Shift+navigation extends selections (and scrolls terminals) everywhere.
        "shift+left",
        "shift+right",
        "shift+up",
        "shift+down",
        "shift+home",
        "shift+end",
        "shift+pageup",
        "shift+pagedown",
    ];
    for text in cases {
        let err = parse_hotkey(text).expect_err(text);
        assert!(err.0.chars().any(|c| !c.is_whitespace()), "{text}");
    }
}

#[test]
fn test_hotkey_validation_and_normalisation() {
    let built = Hotkey::new(["alt", "ctrl"], "p").unwrap();
    assert_eq!(built.modifiers(), &set(&["ctrl", "alt"]));
    assert_eq!(built, hk("ctrl+alt+p"));
    let hash = |h: &Hotkey| {
        use std::hash::{BuildHasher, RandomState};
        static STATE: std::sync::OnceLock<RandomState> = std::sync::OnceLock::new();
        STATE.get_or_init(RandomState::new).hash_one(h)
    };
    assert_eq!(hash(&built), hash(&hk("alt+ctrl+p")));
    assert_eq!(built.ordered_modifiers(), vec!["ctrl", "alt"]);
    assert!(
        Hotkey::new(["hyper"], "p")
            .unwrap_err()
            .0
            .contains("modifier")
    );
    // keys are canonical lower case
    assert!(Hotkey::new(["ctrl"], "P").unwrap_err().0.contains("key"));
    assert!(
        Hotkey::new(Vec::<&str>::new(), "p")
            .unwrap_err()
            .0
            .contains("at least one modifier")
    );
    assert!(
        Hotkey::new(["shift"], "a")
            .unwrap_err()
            .0
            .contains("Shift alone")
    );
}

#[test]
fn test_str_is_canonical_and_round_trips() {
    let parsed = hk("Win+Shift+Alt+Ctrl+K");
    assert_eq!(parsed.to_string(), "ctrl+alt+shift+meta+k");
    assert_eq!(hk(&parsed.to_string()), parsed);
}

#[test]
fn test_default_settings_hotkeys_per_platform() {
    // Windows: Ctrl+Alt is AltGr, so the Win key keeps the defaults from typing. Linux: Ctrl+Alt+T
    // opens a terminal, and Ctrl+Alt+Shift cannot be pressed with an Alt+Shift / Ctrl+Shift
    // layout switch. macOS: ⌃⌥T and ⌃⌥C are Rectangle's and Magnet's defaults.
    for platform in [Platform::Windows, Platform::Other, Platform::MacOs] {
        let parsed: Vec<Hotkey> = defaults(platform).iter().map(|t| hk(t)).collect();
        let keys: Vec<&str> = parsed.iter().map(Hotkey::key).collect();
        assert_eq!(keys, ["t", "p", "c"]);
        assert!(
            parsed
                .iter()
                .all(|p| p.modifiers() == &set(&["ctrl", "alt", "meta"]))
        );
    }
}

#[test]
fn test_windows_defaults_never_collide_with_altgr() {
    let probe = FakeLayoutProbe::everything("x"); // a layout on which every AltGr key types
    for text in defaults(Platform::Windows) {
        let hotkey = hk(&text);
        let vk = win_virtual_key(hotkey.key(), None, None).expect("virtual key");
        assert_eq!(win_altgr_conflict(&hotkey, vk, &probe), None, "{text}");
    }
    assert!(probe.queries.lock().is_empty()); // never even asked: Win+AltGr is not a typing chord
}

#[test]
fn test_linux_defaults_avoid_desktop_shortcuts() {
    let mut taken: HashSet<Hotkey> = (1..=12).map(|n| hk(&format!("ctrl+alt+f{n}"))).collect();
    taken.insert(hk("ctrl+alt+t"));
    for text in defaults(Platform::Other) {
        assert!(!taken.contains(&hk(&text)), "{text}");
    }
}

#[test]
fn test_key_table_is_complete() {
    assert_eq!(MODIFIERS, ["ctrl", "alt", "shift", "meta"]);
    assert_eq!(KEYS.iter().collect::<HashSet<_>>().len(), KEYS.len());
    for key in [
        "a", "z", "0", "9", "f1", "f24", "space", "escape", "tab", "enter",
    ] {
        assert!(KEYS.contains(&key));
    }
    for key in [
        "left", "right", "up", "down", "home", "end", "pageup", "pagedown",
    ] {
        assert!(KEYS.contains(&key));
    }
    for key in [
        "insert", "delete", "minus", "equal", "comma", "period", "slash",
    ] {
        assert!(KEYS.contains(&key));
    }
}

// ========================================================================= formatting
#[test]
fn test_format_default_style() {
    let cases = [
        ("ctrl+alt+p", "Ctrl+Alt+P"),
        ("shift+ctrl+f5", "Ctrl+Shift+F5"),
        ("ctrl+alt+space", "Ctrl+Alt+Space"),
        ("ctrl+alt+esc", "Ctrl+Alt+Esc"),
        ("ctrl+alt+pageup", "Ctrl+Alt+PgUp"),
        ("ctrl+alt+-", "Ctrl+Alt+-"),
    ];
    for (text, expected) in cases {
        assert_eq!(format_hotkey(&hk(text), Some(false)), expected);
    }
}

#[test]
fn test_format_meta_label_follows_platform() {
    let hotkey = hk("win+alt+g");
    assert_eq!(
        format_hotkey_on(&hotkey, Some(false), Platform::Windows),
        "Alt+Win+G"
    );
    assert_eq!(
        format_hotkey_on(&hotkey, Some(false), Platform::Other),
        "Alt+Super+G"
    );
    // macos=None follows the platform
    assert_eq!(
        format_hotkey_on(&hotkey, None, Platform::Other),
        "Alt+Super+G"
    );
    assert_eq!(format_hotkey_on(&hotkey, None, Platform::MacOs), "⌥⌘G");
}

#[test]
fn test_format_macos_style() {
    let cases = [
        ("ctrl+alt+p", "⌃⌥P"),
        ("cmd+shift+g", "⇧⌘G"),
        ("ctrl+alt+shift+cmd+f12", "⌃⌥⇧⌘F12"),
        ("cmd+space", "⌘Space"),
        ("ctrl+alt+esc", "⌃⌥⎋"),
        ("ctrl+alt+enter", "⌃⌥↩"),
        ("ctrl+alt+left", "⌃⌥←"),
    ];
    for (text, expected) in cases {
        assert_eq!(format_hotkey(&hk(text), Some(true)), expected);
    }
}

#[test]
fn test_format_parse_round_trip_for_every_key() {
    let modifier_sets: [&[&str]; 4] = [
        &["ctrl", "alt"],
        &["meta"],
        &["ctrl", "alt", "shift", "meta"],
        &["alt", "shift"],
    ];
    for macos in [false, true] {
        for modifiers in modifier_sets {
            for key in KEYS {
                let hotkey = Hotkey::new(modifiers.iter().copied(), key).unwrap();
                assert_eq!(hk(&format_hotkey(&hotkey, Some(macos))), hotkey, "{key}");
            }
        }
    }
}

// =============================================================== unsupported manager
#[test]
fn test_base_manager_is_a_safe_no_op() {
    let manager = UnsupportedHotkeyManager::new(Some("nope".to_string()));
    assert!(!manager.supported());
    assert_eq!(manager.note().as_deref(), Some("nope"));
    manager.start();
    assert!(!manager.register("t", hk(COMBO).into(), noop()));
    assert!(!manager.unregister("t"));
    assert!(manager.registered().is_empty());
    manager.unregister_all();
    manager.stop();
    manager.stop();
}

// ============================================================================ factory
fn env_of(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
    let vars: HashMap<String, String> = vars
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |name| vars.get(name).cloned()
}

fn eye_tracker_command(args: &[&str]) -> String {
    let mut parts = vec!["eye-tracker".to_string()];
    parts.extend(args.iter().map(|a| a.to_string()));
    crate::cli::format_command(&parts)
}

#[cfg(windows)]
#[test]
fn test_factory_windows() {
    let manager = create_for("windows", &env_of(&[]), &eye_tracker_command); // starts nothing
    assert_eq!(manager.name(), "win32");
    assert!(manager.supported());
}

#[test]
fn test_factory_macos() {
    let manager = create_for("macos", &env_of(&[]), &eye_tracker_command); // Carbon loads lazily
    assert_eq!(manager.name(), "carbon");
}

#[test]
fn test_factory_wayland_without_display_is_unsupported() {
    let env = env_of(&[("WAYLAND_DISPLAY", "wayland-0")]);
    let manager = create_for("linux", &env, &eye_tracker_command);
    assert_eq!(manager.name(), "none");
    assert!(!manager.supported());
    assert!(manager.note().expect("note").contains("ctl"));
}

/// Regression (r2-docs-01): the release packages put no "eye-tracker" command on the PATH, and
/// a desktop shortcut bound to a missing command fails silently.
#[cfg(unix)]
#[test]
fn test_wayland_notes_name_the_commands_of_this_copy() {
    let appimage = "/home/me/Apps/Eye Tracker.AppImage";
    let command_text = |args: &[&str]| {
        let mut parts = vec![appimage.to_string()];
        parts.extend(args.iter().map(|a| a.to_string()));
        crate::cli::format_command(&parts)
    };
    let wayland = create_for(
        "linux",
        &env_of(&[("WAYLAND_DISPLAY", "wayland-0")]),
        &command_text,
    )
    .note();
    let xwayland = create_for(
        "linux",
        &env_of(&[("WAYLAND_DISPLAY", "wayland-0"), ("DISPLAY", ":0")]), // XWayland
        &command_text,
    )
    .note();
    for note in [&wayland, &xwayland] {
        let note = note.as_deref().expect("note");
        assert!(!note.contains("eye-tracker ctl"));
        for action in ["toggle", "privacy-toggle", "calibrate"] {
            assert!(note.contains(&format!("'{}'", command_text(&["ctl", action]))));
        }
    }
    assert!(
        wayland
            .unwrap()
            .starts_with("Wayland does not allow applications to register global hotkeys.")
    );
    assert!(
        xwayland
            .unwrap()
            .contains("may only fire while an X11 window has focus")
    );
}

#[test]
fn test_factory_headless_linux() {
    let manager = create_for("linux", &env_of(&[]), &eye_tracker_command);
    assert_eq!(manager.name(), "none");
}

#[cfg(unix)]
#[test]
fn test_factory_x11() {
    for wayland in [false, true] {
        let session = if wayland { "wayland" } else { "x11" };
        let env = env_of(&[("DISPLAY", ":0"), ("XDG_SESSION_TYPE", session)]);
        let manager = create_for("linux", &env, &eye_tracker_command); // does not connect yet
        assert_eq!(manager.name(), "x11");
        assert_eq!(manager.note().is_some(), wayland);
    }
}

// ============================================================ Windows virtual keys
#[test]
fn test_windows_virtual_key_table() {
    let vk = |key: &str| win_virtual_key(key, None, None);
    for key in KEYS {
        assert!(vk(key).is_some(), "{key}");
    }
    assert_eq!(vk("a"), Some(0x41));
    assert_eq!(vk("z"), Some(0x5A));
    assert_eq!(vk("0"), Some(0x30));
    assert_eq!(vk("f1"), Some(0x70));
    assert_eq!(vk("f24"), Some(0x87));
    assert_eq!(vk("escape"), Some(0x1B));
    assert_eq!(vk("slash"), Some(0xBF));
}

#[test]
fn test_windows_virtual_key_prefers_active_layout_for_punctuation() {
    fn scan(result: i32) -> impl Fn(char) -> Result<i32, String> {
        move |_| Ok(result)
    }
    fn to_scan(result: u32) -> impl Fn(u8) -> Result<u32, String> {
        move |_| Ok(result)
    }
    let failing = |_: char| -> Result<i32, String> { Err("layout query failed".to_string()) };
    // Unshifted mapping from the layout wins …
    assert_eq!(
        win_virtual_key("period", Some(&scan(0xBF)), None),
        Some(0xBF)
    );
    // … a character the layout only types with Shift/AltGr has no key of its own: the US
    // virtual key would be another key (Turkish Q: 0xDB is 'ğ', not '[').
    assert_eq!(win_virtual_key("period", Some(&scan(0x0137)), None), None);
    assert_eq!(
        win_virtual_key("bracketleft", Some(&scan(0x0638)), None),
        None
    ); // AltGr+8
    // A character missing from the layout (Cyrillic) falls back to the US position …
    assert_eq!(win_virtual_key("period", Some(&scan(-1)), None), Some(0xBE));
    assert_eq!(
        win_virtual_key("period", Some(&scan(-1)), Some(&to_scan(0x34))),
        Some(0xBE)
    );
    // … unless no physical key produces that virtual key on this layout.
    assert_eq!(
        win_virtual_key("equal", Some(&scan(-1)), Some(&to_scan(0))),
        None
    );
    // A failing layout query is treated like "not on the layout".
    assert_eq!(
        win_virtual_key("period", Some(&failing), Some(&to_scan(0x34))),
        Some(0xBE)
    );
    // Letters never consult the layout.
    assert_eq!(
        win_virtual_key("q", Some(&scan(0x99)), Some(&to_scan(0))),
        Some(b'Q')
    );
}

#[test]
fn test_windows_modifier_mask() {
    assert_eq!(win_modifiers(["ctrl", "alt"]), 0x0003);
    assert_eq!(win_modifiers(["shift", "meta"]), 0x000C);
}

// ================================================== Windows AltGr (Ctrl+Alt) collisions
const TURKISH_Q: isize = 0x041F041F;
const POLISH_PROGRAMMERS: isize = 0x04150415;
const US: isize = 0x04090409;
const VK_T: u8 = 0x54;
const VK_C: u8 = 0x43;
const VK_A: u8 = 0x41;
const VK_8: u8 = 0x38;
const VK_OEM_1: u8 = 0xBA;

type LayoutTables = BTreeMap<isize, HashMap<(u8, bool), (&'static str, bool)>>;

/// What AltGr(+Shift) types, as read from the layout DLL tables (kbdtuq.dll, kbdpl1.dll) and
/// confirmed with ToUnicodeEx on an installed Turkish Q layout.
fn layout_tables() -> LayoutTables {
    BTreeMap::from([
        (
            TURKISH_Q,
            HashMap::from([
                ((VK_T, false), ("₺", false)),
                ((VK_8, false), ("[", false)),
                ((VK_A, true), ("Æ", false)),
                ((VK_OEM_1, false), ("´", true)), // dead key
            ]),
        ),
        (
            POLISH_PROGRAMMERS,
            HashMap::from([((VK_C, false), ("ć", false)), ((VK_C, true), ("Ć", false))]),
        ),
        (US, HashMap::new()),
    ])
}

/// Installed keyboard layouts as data (the real probe asks user32).
struct FakeLayoutProbe {
    tables: LayoutTables,
    everything: Option<&'static str>,
    queries: Mutex<Vec<(u8, bool, isize)>>,
    broken: bool,
}

impl FakeLayoutProbe {
    fn new(tables: LayoutTables) -> FakeLayoutProbe {
        FakeLayoutProbe {
            tables,
            everything: None,
            queries: Mutex::new(Vec::new()),
            broken: false,
        }
    }

    fn everything(text: &'static str) -> FakeLayoutProbe {
        FakeLayoutProbe {
            everything: Some(text),
            ..FakeLayoutProbe::new(BTreeMap::from([(US, HashMap::new())]))
        }
    }
}

impl WinLayoutProbe for FakeLayoutProbe {
    fn layouts(&self) -> Vec<isize> {
        if self.broken {
            panic!("user32 is gone");
        }
        self.tables.keys().copied().collect()
    }

    fn altgr_text(&self, vk: u8, shift: bool, hkl: isize) -> Option<(String, bool)> {
        self.queries.lock().push((vk, shift, hkl));
        if let Some(text) = self.everything {
            return Some((text.to_string(), false));
        }
        self.tables[&hkl]
            .get(&(vk, shift))
            .map(|(text, dead)| (text.to_string(), *dead))
    }

    fn layout_name(&self, hkl: isize) -> String {
        match hkl {
            TURKISH_Q => "Turkish Q".to_string(),
            POLISH_PROGRAMMERS => "Polish (Programmers)".to_string(),
            US => "US".to_string(),
            _ => format!("{hkl:#x}"),
        }
    }
}

fn altgr_conflict(text: &str, probe: &FakeLayoutProbe) -> Option<String> {
    let hotkey = hk(text);
    let vk = win_virtual_key(hotkey.key(), None, None).expect("virtual key");
    win_altgr_conflict(&hotkey, vk, probe)
}

#[test]
fn test_altgr_conflict_catches_the_old_defaults() {
    let probe = FakeLayoutProbe::new(layout_tables());
    assert_eq!(
        altgr_conflict("ctrl+alt+t", &probe).as_deref(),
        Some("Ctrl+Alt+T is AltGr+T and types '₺' on the Turkish Q keyboard layout")
    );
    assert_eq!(
        altgr_conflict("ctrl+alt+c", &probe).as_deref(),
        Some("Ctrl+Alt+C is AltGr+C and types 'ć' on the Polish (Programmers) keyboard layout")
    );
    assert_eq!(altgr_conflict("ctrl+alt+p", &probe), None); // nothing on these three layouts
    // Every installed layout is asked, because RegisterHotKey fires on any of them.
    let asked: BTreeSet<isize> = probe.queries.lock().iter().map(|q| q.2).collect();
    assert_eq!(asked, layout_tables().keys().copied().collect());
}

#[test]
fn test_altgr_conflict_with_shift_and_dead_keys() {
    let probe = FakeLayoutProbe::new(layout_tables());
    let message = altgr_conflict("ctrl+alt+shift+c", &probe).expect("conflict");
    assert!(message.contains("AltGr+Shift+C"));
    assert!(message.contains("'Ć'"));
    assert!(probe.queries.lock().iter().all(|q| q.1));
    let message = altgr_conflict("ctrl+alt+8", &probe).expect("conflict");
    assert!(message.contains("'['"));
    let dead = win_altgr_conflict(&hk("ctrl+alt+semicolon"), VK_OEM_1, &probe).expect("dead");
    assert!(dead.contains("the dead key '´'"));
}

#[test]
fn test_typed_characters_are_described_readably() {
    let cases = [
        ("ć", "'ć'"),
        ("ab", "'ab'"),
        ("'", "\"'\""),
        ("e\u{301}", "'é'"), // composed first
        // Invisible characters are named, never shown as escapes like '\xa0'.
        ("\u{a0}", "no-break space (U+00A0)"),
        ("\u{202f}", "narrow no-break space (U+202F)"),
        (" ", "space (U+0020)"),
        ("\u{200b}", "zero width space (U+200B)"),
        ("\u{301}", "combining acute accent (U+0301)"), // would sit on the quote sign
        ("\u{e000}", "U+E000"),                         // private use: no name
        (
            "x\u{a0}",
            "latin small letter x (U+0078) + no-break space (U+00A0)",
        ),
        ("", "nothing"),
    ];
    for (text, expected) in cases {
        assert_eq!(describe_typed(text), expected, "{text:?}");
    }
}

#[test]
fn test_altgr_conflict_messages_read_well() {
    assert_eq!(
        altgr_conflict("ctrl+alt+space", &FakeLayoutProbe::everything("\u{a0}")).as_deref(),
        Some(
            "Ctrl+Alt+Space is AltGr+Space and types no-break space (U+00A0) on the US keyboard \
             layout"
        )
    );
    // A punctuation key does not run into the sentence ("is AltGr+,, which types").
    assert_eq!(
        altgr_conflict("ctrl+alt+comma", &FakeLayoutProbe::everything("ç")).as_deref(),
        Some("Ctrl+Alt+, is AltGr+, and types 'ç' on the US keyboard layout")
    );
}

#[test]
fn test_altgr_conflict_only_checks_ctrl_alt_without_win() {
    for text in [
        "alt+t",
        "ctrl+t",
        "ctrl+shift+t",
        "ctrl+alt+meta+t",
        "alt+shift+meta+c",
    ] {
        let probe = FakeLayoutProbe::everything("x");
        assert_eq!(altgr_conflict(text, &probe), None, "{text}");
        assert!(probe.queries.lock().is_empty());
    }
}

#[test]
fn test_altgr_conflict_function_keys_never_type() {
    let probe = FakeLayoutProbe::new(layout_tables());
    assert_eq!(altgr_conflict("ctrl+alt+f9", &probe), None);
    assert_eq!(altgr_conflict("ctrl+alt+shift+f24", &probe), None);
}

/// `WindowsHotkeyManager::layout_conflict` minus the manager (it is Windows-only here).
#[test]
fn test_windows_manager_layout_conflict_uses_probe() {
    let probe = FakeLayoutProbe::new(layout_tables());
    let conflict = win_layout_conflict(&hk("ctrl+alt+t"), &probe).expect("conflict");
    assert!(conflict.contains("Turkish Q"));
    assert_eq!(win_layout_conflict(&hk("ctrl+alt+meta+t"), &probe), None);
    // Advisory only: a failing query never breaks the caller.
    let broken = FakeLayoutProbe {
        broken: true,
        ..FakeLayoutProbe::new(layout_tables())
    };
    assert_eq!(win_layout_conflict(&hk("ctrl+alt+t"), &broken), None);
}

#[cfg(windows)]
#[test]
fn test_windows_manager_layout_conflict_uses_probe_live_manager() {
    let manager = WindowsHotkeyManager::with_layout_probe(Some(Arc::new(FakeLayoutProbe::new(
        layout_tables(),
    ))));
    let conflict = manager.layout_conflict("ctrl+alt+t".into()).unwrap();
    assert!(conflict.expect("conflict").contains("Turkish Q"));
    assert_eq!(
        manager.layout_conflict(hk("ctrl+alt+meta+t").into()),
        Ok(None)
    );
    let err = manager
        .layout_conflict("ctrl+alt+nosuchkey".into())
        .unwrap_err();
    assert!(err.0.contains("unsupported key"));
    let broken = FakeLayoutProbe {
        broken: true,
        ..FakeLayoutProbe::new(layout_tables())
    };
    let manager = WindowsHotkeyManager::with_layout_probe(Some(Arc::new(broken)));
    assert_eq!(manager.layout_conflict("ctrl+alt+t".into()), Ok(None));
}

#[test]
fn test_base_manager_layout_conflict() {
    let manager = UnsupportedHotkeyManager::new(None);
    assert_eq!(manager.layout_conflict("ctrl+alt+t".into()), Ok(None));
    let err = manager.layout_conflict("ctrl+alt".into()).unwrap_err();
    assert!(err.0.contains("needs a key"));
}

/// Just enough of user32 for [`Win32LayoutProbe`].
struct FakeUser32 {
    /// (hkl, vk, shift) → (ToUnicodeEx result, buffer text)
    results: HashMap<(isize, u32, bool), (i32, &'static str)>,
    flags: Mutex<Vec<u32>>,
    hkls: Vec<isize>,
    layout_text: Mutex<Option<&'static str>>,
}

impl Win32LayoutApi for FakeUser32 {
    fn get_keyboard_layout_list(&self, handles: Option<&mut [isize]>) -> i32 {
        let Some(handles) = handles else {
            return self.hkls.len() as i32;
        };
        for (out, hkl) in handles.iter_mut().zip(&self.hkls) {
            *out = *hkl;
        }
        handles.len().min(self.hkls.len()) as i32
    }

    fn map_virtual_key_ex(&self, code: u32, map_type: u32, _hkl: isize) -> u32 {
        assert_eq!(map_type, 0); // MAPVK_VK_TO_VSC
        if code == 0xFF { 0 } else { 0x14 }
    }

    fn to_unicode_ex(
        &self,
        vk: u32,
        scan: u32,
        state: &[u8; 256],
        buf: &mut [u16],
        flags: u32,
        hkl: isize,
    ) -> i32 {
        self.flags.lock().push(flags);
        assert_eq!(scan, 0x14);
        // VK_CONTROL, VK_LCONTROL, VK_MENU, VK_RMENU
        for held in [0x11, 0xA2, 0x12, 0xA5] {
            assert_eq!(state[held], 0x80);
        }
        let shift = state[0x10] == 0x80;
        assert_eq!(state[0xA0] == 0x80, shift);
        let (count, text) = self
            .results
            .get(&(hkl, vk, shift))
            .copied()
            .unwrap_or((0, ""));
        for (slot, unit) in buf.iter_mut().zip(text.encode_utf16()) {
            *slot = unit;
        }
        count
    }

    fn get_locale_info(&self, lcid: u32, lctype: u32, buf: &mut [u16]) -> i32 {
        assert_eq!(lctype, 0x72); // LOCALE_SENGLISHDISPLAYNAME
        if lcid != 0x041F {
            return 0;
        }
        let text: Vec<u16> = "Turkish (Türkiye)".encode_utf16().collect();
        buf[..text.len()].copy_from_slice(&text);
        buf[text.len()] = 0;
        text.len() as i32 + 1
    }

    fn layout_text(&self, _hkl: isize) -> Option<String> {
        self.layout_text.lock().map(String::from)
    }
}

#[test]
fn test_win32_layout_probe_queries() {
    let probe = Win32LayoutProbe {
        api: FakeUser32 {
            results: HashMap::from([
                ((TURKISH_Q, u32::from(VK_T), false), (1, "₺")),
                ((TURKISH_Q, u32::from(VK_OEM_1), false), (-1, "´")),
                ((TURKISH_Q, 0x0D, false), (1, "\r")), // control characters are not typing
                ((TURKISH_Q, u32::from(VK_A), true), (1, "Æ")),
            ]),
            flags: Mutex::new(Vec::new()),
            hkls: vec![TURKISH_Q, US],
            layout_text: Mutex::new(None),
        },
    };
    let text = |s: &str, dead| Some((s.to_string(), dead));
    assert_eq!(probe.layouts(), vec![TURKISH_Q, US]);
    assert_eq!(probe.altgr_text(VK_T, false, TURKISH_Q), text("₺", false));
    assert_eq!(probe.altgr_text(VK_T, false, US), None);
    assert_eq!(
        probe.altgr_text(VK_OEM_1, false, TURKISH_Q),
        text("´", true)
    );
    assert_eq!(probe.altgr_text(0x0D, false, TURKISH_Q), None);
    assert_eq!(probe.altgr_text(VK_A, true, TURKISH_Q), text("Æ", false));
    assert_eq!(probe.altgr_text(0xFF, false, TURKISH_Q), None); // no key makes this VK
    // Flag 0x4: the query never changes the keyboard state (no dead key is armed).
    let flags: BTreeSet<u32> = probe.api.flags.lock().iter().copied().collect();
    assert_eq!(flags, BTreeSet::from([0x4]));
    assert_eq!(probe.layout_name(TURKISH_Q), "Turkish (Türkiye)");
    assert_eq!(probe.layout_name(0x12345678), "0x12345678");
    *probe.api.layout_text.lock() = Some("Turkish Q");
    assert_eq!(probe.layout_name(TURKISH_Q), "Turkish Q");
}

// ======================================================== generic threaded machinery
/// A portable threaded manager whose "OS" is a map, to test the shared plumbing.
pub(super) struct Loopback {
    core: Core,
    threaded: Threaded,
    wake_event: Event,
    os_registered: Mutex<BTreeMap<u32, Hotkey>>,
    taken: Mutex<HashSet<Hotkey>>,
    reasons: Mutex<HashMap<Hotkey, String>>, // taken combos that explain themselves
    native_threads: Mutex<BTreeSet<String>>,
}

native_manager!(LoopbackManager(Loopback));

impl LoopbackManager {
    fn new(call_timeout: Duration) -> LoopbackManager {
        let mut threaded = Threaded::new();
        threaded.call_timeout = call_timeout;
        LoopbackManager(Arc::new(Loopback {
            core: Core::new("loopback", None),
            threaded,
            wake_event: Event::default(),
            os_registered: Mutex::new(BTreeMap::new()),
            taken: Mutex::new(HashSet::new()),
            reasons: Mutex::new(HashMap::new()),
            native_threads: Mutex::new(BTreeSet::new()),
        }))
    }

    fn binding_id(&self, name: &str) -> u32 {
        self.0.core.table.lock().bindings[name].id
    }

    fn fire(&self, name: &str) {
        let (this, id) = (self.0.clone(), self.binding_id(name));
        call_on_thread(&self.0, move || this.core.dispatch(id), ());
    }

    fn thread(&self) -> Option<Arc<BackendThread>> {
        self.0.threaded.current_thread()
    }
}

impl Loopback {
    fn reg(&self, binding: &Binding) -> bool {
        let name = thread::current().name().unwrap_or_default().to_string();
        self.native_threads.lock().insert(name);
        let reason = self.reasons.lock().get(&binding.hotkey).cloned();
        if let Some(reason) = reason {
            return self.core.reject(binding, reason);
        }
        if self.taken.lock().contains(&binding.hotkey) {
            return false;
        }
        self.os_registered
            .lock()
            .insert(binding.id, binding.hotkey.clone());
        true
    }
}

impl Native for Loopback {
    fn core(&self) -> &Core {
        &self.core
    }

    fn backend_start(self: &Arc<Self>) -> bool {
        threaded_backend_start(self)
    }

    fn backend_stop(self: &Arc<Self>) {
        threaded_backend_stop(self)
    }

    fn native_register(self: &Arc<Self>, binding: &Arc<Binding>) -> bool {
        let (this, binding) = (self.clone(), binding.clone());
        call_on_thread(self, move || this.reg(&binding), false)
    }

    fn native_unregister(self: &Arc<Self>, binding: &Arc<Binding>) {
        let (this, id) = (self.clone(), binding.id);
        call_on_thread(
            self,
            move || {
                this.os_registered.lock().remove(&id);
            },
            (),
        )
    }

    fn native_layout_conflict(self: &Arc<Self>, _hotkey: &Hotkey) -> Option<String> {
        None
    }
}

impl ThreadedNative for Loopback {
    fn threaded(&self) -> &Threaded {
        &self.threaded
    }

    fn thread_setup(self: &Arc<Self>) -> bool {
        true
    }

    fn thread_loop(self: &Arc<Self>) {
        while !self.threaded.is_stopping() {
            self.wake_event.wait(Duration::from_millis(200));
            self.wake_event.clear();
            self.threaded.run_pending_calls();
        }
    }

    fn thread_teardown(self: &Arc<Self>) {
        self.os_registered.lock().clear();
    }

    fn wake(&self) -> bool {
        self.wake_event.set();
        true
    }
}

fn loopback() -> LoopbackManager {
    LoopbackManager::new(CALL_TIMEOUT)
}

#[test]
fn test_threaded_registration_runs_on_backend_thread() {
    let loopback = loopback();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let log = calls.clone();
    let callback: HotkeyCallback = Arc::new(move || {
        log.lock()
            .push(thread::current().name().unwrap_or_default().to_string())
    });
    assert!(loopback.register("t", COMBO.into(), callback));
    assert_eq!(
        *loopback.0.native_threads.lock(),
        BTreeSet::from(["eye-tracker-hotkeys-loopback".to_string()])
    );
    assert_eq!(
        loopback.registered(),
        HashMap::from([("t".to_string(), hk(COMBO))])
    );
    loopback.fire("t");
    assert_eq!(*calls.lock(), ["eye-tracker-hotkeys-loopback"]);
}

#[test]
fn test_double_registration_semantics() {
    let loopback = loopback();
    let fired = Arc::new(Mutex::new(Vec::new()));
    assert!(loopback.register("a", COMBO.into(), push(&fired, "first")));
    // Same name, same combo: the callback is replaced without touching the OS.
    assert!(loopback.register("a", COMBO.into(), push(&fired, "second")));
    assert_eq!(loopback.0.os_registered.lock().len(), 1);
    loopback.fire("a");
    assert_eq!(*fired.lock(), ["second"]);
    // Another name for a combo this manager already owns is refused.
    assert!(!loopback.register("b", COMBO.into(), noop()));
    assert_eq!(loopback.registered().keys().collect::<Vec<_>>(), ["a"]);
    // Re-binding a name to a new combo releases the old one.
    assert!(loopback.register("a", COMBO_2.into(), noop()));
    let os: Vec<Hotkey> = loopback.0.os_registered.lock().values().cloned().collect();
    assert_eq!(os, [hk(COMBO_2)]);
    assert!(loopback.register("b", COMBO.into(), noop()));
}

#[test]
fn test_os_rejection_leaves_no_trace() {
    let loopback = loopback();
    loopback.0.taken.lock().insert(hk(COMBO));
    assert!(!loopback.register("t", COMBO.into(), noop()));
    assert!(loopback.registered().is_empty());
    assert!(loopback.0.core.table.lock().by_id.is_empty());
}

#[test]
fn test_last_error_explains_failures() {
    let loopback = loopback();
    assert_eq!(loopback.last_error("t"), None); // never registered
    assert!(!loopback.register("t", "ctrl+alt+nosuchkey".into(), noop()));
    let error = loopback.last_error("t").expect("error");
    assert!(error.contains("unsupported key"));
    // A native hook that explains itself (e.g. an AltGr collision) wins …
    let reason = "Ctrl+Alt+T is AltGr+T, which types '₺' on the Turkish Q keyboard layout";
    loopback
        .0
        .reasons
        .lock()
        .insert(hk("ctrl+alt+t"), reason.to_string());
    // Run on this thread so that the warning is captured here.
    let logs = capture_logs();
    loopback.0.threaded.inline.store(true, Ordering::SeqCst);
    assert!(!loopback.register("t", "ctrl+alt+t".into(), noop()));
    loopback.0.threaded.inline.store(false, Ordering::SeqCst);
    assert_eq!(loopback.last_error("t").as_deref(), Some(reason));
    let warnings = logs.messages_at_least(log::Level::Warn);
    assert!(warnings.iter().any(|m| m.contains(reason)), "{warnings:?}");
    // … otherwise a generic message is recorded. Messages name a combination the way this
    // system shows it (macOS: modifier glyphs).
    let mac = cfg!(target_os = "macos");
    loopback.0.taken.lock().insert(hk(COMBO));
    assert!(!loopback.register("t", COMBO.into(), noop()));
    let combo = if mac {
        "⌃⌥⇧F24"
    } else {
        "Ctrl+Alt+Shift+F24"
    };
    assert_eq!(
        loopback.last_error("t"),
        Some(format!("{combo} could not be registered"))
    );
    // A clash inside this manager names the other action.
    assert!(loopback.register("a", COMBO_2.into(), noop()));
    assert!(!loopback.register("b", COMBO_2.into(), noop()));
    let combo_2 = if mac {
        "⌃⌥⇧F21"
    } else {
        "Ctrl+Alt+Shift+F21"
    };
    assert_eq!(
        loopback.last_error("b"),
        Some(format!("{combo_2} is already used for a"))
    );
    // Success clears the error.
    assert!(loopback.register("t", "ctrl+alt+shift+f20".into(), noop()));
    assert_eq!(loopback.last_error("t"), None);
    assert_eq!(loopback.last_error("a"), None);
}

#[test]
fn test_unsupported_manager_last_error() {
    let manager =
        UnsupportedHotkeyManager::new(Some("Wayland does not allow global hotkeys.".to_string()));
    assert!(!manager.register("t", COMBO.into(), noop()));
    assert_eq!(
        manager.last_error("t").as_deref(),
        Some("Wayland does not allow global hotkeys.")
    );
    assert!(!UnsupportedHotkeyManager::new(None).register("t", COMBO.into(), noop()));
}

#[test]
fn test_invalid_register_arguments() {
    let loopback = loopback();
    assert!(!loopback.register("t", "not a hotkey".into(), noop()));
}

#[test]
fn test_unregister_and_unregister_all() {
    let loopback = loopback();
    assert!(loopback.register("a", COMBO.into(), noop()));
    assert!(loopback.register("b", COMBO_2.into(), noop()));
    assert!(loopback.unregister("a"));
    assert!(!loopback.unregister("a"));
    assert_eq!(loopback.registered().keys().collect::<Vec<_>>(), ["b"]);
    loopback.unregister_all();
    assert!(loopback.registered().is_empty());
    assert!(loopback.0.os_registered.lock().is_empty());
    let thread = loopback.thread().expect("thread");
    assert!(thread.is_alive()); // unregister_all keeps the backend running
}

#[test]
fn test_stop_is_idempotent_and_restartable() {
    let loopback = loopback();
    loopback.stop(); // before anything started
    loopback.start();
    let thread = loopback.thread().expect("thread");
    assert!(thread.is_alive());
    assert!(loopback.register("t", COMBO.into(), noop()));
    loopback.stop();
    assert!(!thread.is_alive());
    assert!(loopback.registered().is_empty());
    loopback.stop();
    assert!(loopback.register("t", COMBO.into(), noop())); // lazily restarts
    let new_thread = loopback.thread().expect("thread");
    assert!(!Arc::ptr_eq(&new_thread, &thread));
    assert!(new_thread.is_alive());
}

#[test]
fn test_callback_exception_is_logged() {
    let loopback = loopback();
    let boom: HotkeyCallback = Arc::new(|| panic!("callback failure"));
    let fired = Arc::new(Mutex::new(Vec::new()));
    assert!(loopback.register("bad", COMBO.into(), boom));
    assert!(loopback.register("good", COMBO_2.into(), push(&fired, 1)));
    loopback.fire("bad"); // on the hotkey thread, which survives it
    // Dispatched here as well so that the log record is captured on this thread.
    let logs = capture_logs();
    loopback.0.core.dispatch(loopback.binding_id("bad"));
    let errors = logs.messages_at_least(log::Level::Error);
    assert!(
        errors.iter().any(|m| m.contains("callback failure")),
        "{errors:?}"
    );
    loopback.fire("good");
    assert_eq!(*fired.lock(), [1]);
}

#[test]
fn test_call_timeout_returns_default() {
    let manager = LoopbackManager::new(Duration::from_millis(50));
    manager.start();
    let release = Arc::new(Event::default());
    // Occupy the backend thread so the next call cannot start in time.
    let blocker = release.clone();
    manager
        .0
        .threaded
        .calls
        .lock()
        .push_back(PendingCall::new(move || {
            blocker.wait(Duration::from_secs(1));
        }));
    manager.0.wake();
    assert_eq!(call_on_thread(&manager.0, || "late", "default"), "default");
    release.set();
    manager.stop();
}

#[test]
fn test_dispatch_ignores_unknown_ids() {
    loopback().0.core.dispatch(12345); // must not panic
}

// ============================================================================== macOS
fn mac_code(key: &str) -> u32 {
    mac_us_keycode(key).expect("Mac key code")
}

/// A Mac layout's base layer `{keycode: char}`: US plus `overrides` by US key name.
fn mac_layout(overrides: &[(&str, &str)]) -> BTreeMap<u32, String> {
    let mut layout: BTreeMap<u32, String> = mac_ansi_chars()
        .map(|(ch, code)| (code, ch.to_string()))
        .collect();
    for (key, ch) in overrides {
        layout.insert(mac_code(key), ch.to_string());
    }
    layout
}

fn mac_us() -> BTreeMap<u32, String> {
    mac_layout(&[])
}

/// French AZERTY: A/Q, Z/W and M swapped around; the number row types &é"'(§è!çà.
#[rustfmt::skip]
fn mac_azerty() -> BTreeMap<u32, String> {
    mac_layout(&[
        ("a", "q"), ("q", "a"), ("w", "z"), ("z", "w"), ("semicolon", "m"), ("m", ","),
        ("comma", ";"), ("period", ":"), ("slash", "="),
        ("1", "&"), ("2", "é"), ("3", "\""), ("4", "'"), ("5", "("), ("6", "§"), ("7", "è"),
        ("8", "!"), ("9", "ç"), ("0", "à"),
        ("minus", ")"), ("equal", "-"), ("bracketleft", "^"), ("bracketright", "$"),
        ("quote", "ù"), ("backslash", "`"), ("grave", "<"),
    ])
}

/// Dvorak: the default ⌃⌥T/P/C sit on the keys at the US K, R and I positions.
#[rustfmt::skip]
fn mac_dvorak() -> BTreeMap<u32, String> {
    mac_layout(&[
        ("q", "'"), ("w", ","), ("e", "."), ("r", "p"), ("t", "y"), ("y", "f"), ("u", "g"),
        ("i", "c"), ("o", "r"), ("p", "l"), ("bracketleft", "/"), ("bracketright", "="),
        ("s", "o"), ("d", "e"), ("f", "u"), ("g", "i"), ("h", "d"), ("j", "h"), ("k", "t"),
        ("l", "n"), ("semicolon", "s"), ("quote", "-"), ("z", ";"), ("x", "q"), ("c", "j"),
        ("v", "k"), ("b", "x"), ("n", "b"), ("comma", "w"), ("period", "v"), ("slash", "z"),
        ("equal", "]"), ("minus", "["),
    ])
}

/// Russian: Cyrillic letters, so Latin shortcuts come from the ASCII-capable layout.
#[rustfmt::skip]
fn mac_russian() -> BTreeMap<u32, String> {
    mac_layout(&[
        ("a", "ф"), ("s", "ы"), ("d", "в"), ("f", "а"), ("t", "е"), ("p", "з"), ("c", "с"),
        ("k", "л"), ("comma", "б"), ("period", "ю"), ("slash", "."),
    ])
}

type Layout = Option<BTreeMap<u32, String>>;

#[derive(Default)]
struct FakeCarbonState {
    handler: Option<Arc<CarbonHandler>>,
    install_calls: usize,
    removed: Vec<usize>,
    /// ref → (keycode, modifiers, signature, id)
    registered: BTreeMap<usize, (u32, u32, u32, u32)>,
    reject: HashMap<(u32, u32), i32>,
    next: usize,
    // Keyboard layout: base layer of the current / ASCII-capable layout and the layers typed
    // with Option(+Shift), keyed by Carbon modifier mask.
    layout: Layout,
    ascii_layout: Layout,
    modified: HashMap<u32, BTreeMap<u32, String>>,
    layout_callback: Option<Arc<LayoutCallback>>,
    stopped_observing: usize,
    /// The fake "EventRef"s: index → (status, signature, id).
    events: Vec<(i32, u32, u32)>,
}

/// Stand-in for the Carbon binding.
#[derive(Clone)]
struct FakeCarbon(Arc<Mutex<FakeCarbonState>>);

const HANDLER_REF: usize = 0x4A4A;

impl FakeCarbon {
    fn new(layout: Layout) -> FakeCarbon {
        FakeCarbon::with_layers(layout.clone(), layout, HashMap::new())
    }

    fn with_layers(
        layout: Layout,
        ascii_layout: Layout,
        modified: HashMap<u32, BTreeMap<u32, String>>,
    ) -> FakeCarbon {
        FakeCarbon(Arc::new(Mutex::new(FakeCarbonState {
            layout,
            ascii_layout,
            modified,
            ..FakeCarbonState::default()
        })))
    }

    fn state(&self) -> parking_lot::MutexGuard<'_, FakeCarbonState> {
        self.0.lock()
    }

    fn switch_layout(&self, layout: BTreeMap<u32, String>) {
        let callback = {
            let mut st = self.state();
            st.layout = Some(layout.clone());
            st.ascii_layout = Some(layout);
            st.layout_callback.clone().expect("observing")
        };
        callback();
    }

    /// Key code at which Carbon currently holds the hot key `hotkey_id`.
    fn keycode_for(&self, hotkey_id: u32) -> Option<u32> {
        let codes: Vec<u32> = self
            .state()
            .registered
            .values()
            .filter(|v| v.3 == hotkey_id)
            .map(|v| v.0)
            .collect();
        assert!(codes.len() <= 1);
        codes.first().copied()
    }

    /// Run the installed handler on a fake event `(status, signature, id)`.
    fn handle(&self, event: (i32, u32, u32)) -> i32 {
        let (handler, index) = {
            let mut st = self.state();
            st.events.push(event);
            (st.handler.clone().expect("handler"), st.events.len() - 1)
        };
        handler(index)
    }

    fn press(&self, keycode: u32, modifiers: u32) -> i32 {
        let found = self
            .state()
            .registered
            .values()
            .find(|v| (v.0, v.1) == (keycode, modifiers))
            .copied();
        let (_, _, signature, hotkey_id) = found.expect("combination not registered");
        self.handle((0, signature, hotkey_id))
    }
}

impl CarbonApi for FakeCarbon {
    fn install_handler(&self, handler: CarbonHandler) -> (i32, usize) {
        let mut st = self.state();
        st.install_calls += 1;
        st.handler = Some(Arc::new(handler));
        (0, HANDLER_REF)
    }

    fn remove_handler(&self, handler_ref: usize) -> i32 {
        let mut st = self.state();
        st.removed.push(handler_ref);
        st.handler = None;
        0
    }

    fn register_hotkey(
        &self,
        keycode: u32,
        modifiers: u32,
        signature: u32,
        hotkey_id: u32,
    ) -> (i32, usize) {
        let mut st = self.state();
        let status = st.reject.get(&(keycode, modifiers)).copied().unwrap_or(0);
        if status != 0 {
            return (status, 0);
        }
        st.next += 1;
        let hotkey_ref = st.next;
        st.registered
            .insert(hotkey_ref, (keycode, modifiers, signature, hotkey_id));
        (0, hotkey_ref)
    }

    fn unregister_hotkey(&self, hotkey_ref: usize) -> i32 {
        self.state()
            .registered
            .remove(&hotkey_ref)
            .expect("registered");
        0
    }

    fn hotkey_id(&self, event: usize) -> (i32, u32, u32) {
        self.state().events[event]
    }

    fn layout_characters(
        &self,
        modifiers: u32,
        ascii_capable: bool,
    ) -> Option<BTreeMap<u32, String>> {
        let st = self.state();
        if modifiers != 0 {
            return Some(st.modified.get(&modifiers).cloned().unwrap_or_default());
        }
        if ascii_capable {
            st.ascii_layout.clone()
        } else {
            st.layout.clone()
        }
    }

    fn observe_layout_changes(&self, callback: LayoutCallback) -> bool {
        self.state().layout_callback = Some(Arc::new(callback));
        true
    }

    fn stop_observing_layout_changes(&self) {
        let mut st = self.state();
        st.layout_callback = None;
        st.stopped_observing += 1;
    }
}

fn mac_with(carbon: &FakeCarbon) -> MacHotkeyManager {
    MacHotkeyManager::with_carbon(Some(Arc::new(carbon.clone())))
}

fn mac_keycode_of(mac: &MacHotkeyManager, carbon: &FakeCarbon, name: &str) -> Option<u32> {
    let id = mac.0.core.table.lock().bindings[name].id;
    carbon.keycode_for(id)
}

#[test]
fn test_mac_register_uses_carbon_codes() {
    let carbon = FakeCarbon::new(None);
    let mac = mac_with(&carbon);
    let fired = Arc::new(Mutex::new(Vec::new()));
    assert!(mac.register("t", "ctrl+alt+p".into(), push(&fired, "t")));
    assert_eq!(carbon.state().install_calls, 1);
    let registered: Vec<_> = carbon.state().registered.values().copied().collect();
    let [(keycode, modifiers, signature, hotkey_id)] = registered[..] else {
        panic!("one registration expected: {registered:?}");
    };
    assert_eq!(keycode, 0x23); // kVK_ANSI_P
    assert_eq!(modifiers, 0x1000 | 0x0800); // controlKey | optionKey
    assert_eq!(signature, fourcc(b"EyTk"));
    assert_eq!(hotkey_id, mac.0.core.table.lock().bindings["t"].id);
    assert_eq!(carbon.press(0x23, 0x1800), 0); // noErr
    assert_eq!(*fired.lock(), ["t"]);
}

#[test]
fn test_mac_modifier_masks() {
    assert_eq!(mac_modifiers(["meta"]), 0x0100);
    assert_eq!(mac_modifiers(["shift"]), 0x0200);
    assert_eq!(mac_modifiers(["alt"]), 0x0800);
    assert_eq!(mac_modifiers(["ctrl"]), 0x1000);
}

#[test]
fn test_mac_foreign_events_are_passed_on() {
    let carbon = FakeCarbon::new(None);
    let mac = mac_with(&carbon);
    assert!(mac.register("t", "cmd+shift+g".into(), noop()));
    assert_eq!(carbon.handle((0, fourcc(b"XXXX"), 1)), -9874); // eventNotHandledErr
    assert_eq!(carbon.handle((-50, 0, 0)), -9874);
}

#[test]
fn test_mac_double_registration_and_rebinding() {
    let carbon = FakeCarbon::new(None);
    let mac = mac_with(&carbon);
    assert!(mac.register("a", MAC_COMBO.into(), noop()));
    assert!(mac.register("a", MAC_COMBO.into(), noop()));
    assert_eq!(carbon.state().registered.len(), 1);
    assert!(!mac.register("b", MAC_COMBO.into(), noop()));
    assert!(mac.register("a", "ctrl+alt+p".into(), noop()));
    let codes: Vec<u32> = carbon.state().registered.values().map(|v| v.0).collect();
    assert_eq!(codes, [0x23]);
}

#[test]
fn test_mac_rejections() {
    let carbon = FakeCarbon::new(None);
    let mac = mac_with(&carbon);
    carbon.state().reject.insert((0x23, 0x1800), -9878); // eventHotKeyExistsErr
    assert!(!mac.register("t", "ctrl+alt+p".into(), noop()));
    // Registered exclusively, so this means another application owns it.
    assert_eq!(
        mac.last_error("t").as_deref(),
        Some("⌃⌥P is already in use by another application")
    );
    assert!(!mac.register("t", "ctrl+alt+f24".into(), noop())); // no F24 on a Mac
    assert_eq!(
        mac.last_error("t").as_deref(),
        Some("⌃⌥F24: Mac keyboards have no F24")
    );
    carbon.state().reject.insert((0x23, 0x1800), -50);
    assert!(!mac.register("t", "ctrl+alt+p".into(), noop()));
    assert_eq!(
        mac.last_error("t").as_deref(),
        Some("RegisterEventHotKey(⌃⌥P) failed (OSStatus -50)")
    );
    assert!(mac.registered().is_empty());
}

#[test]
fn test_mac_letters_follow_the_keyboard_layout() {
    let carbon = FakeCarbon::new(Some(mac_azerty()));
    {
        let mac = mac_with(&carbon);
        for (name, text) in [
            ("a", "ctrl+alt+a"),
            ("m", "ctrl+alt+m"),
            ("comma", "ctrl+alt+,"),
        ] {
            assert!(mac.register(name, text.into(), noop()));
        }
        assert!(mac.register("one", "ctrl+alt+1".into(), noop()));
        assert!(mac.register("t", "ctrl+alt+t".into(), noop()));
        let code = |name| mac_keycode_of(&mac, &carbon, name);
        // The key labelled A on AZERTY is at the US Q position, and so on.
        assert_eq!(code("a"), Some(mac_code("q")));
        assert_eq!(code("m"), Some(mac_code("semicolon")));
        assert_eq!(code("comma"), Some(mac_code("m")));
        // Digits need Shift on AZERTY: the number row keeps its (labelled) US position.
        assert_eq!(code("one"), Some(mac_code("1")));
        assert_eq!(code("t"), Some(mac_code("t")));
        // '/' needs modifiers on AZERTY: no key is "the / key", so nothing is guessed.
        assert!(!mac.register("slash", "ctrl+alt+/".into(), noop()));
        assert_eq!(
            mac.last_error("slash").as_deref(),
            Some("⌃⌥/: no key types '/' without modifiers on the current keyboard layout")
        );
        mac.stop();
    }
    assert_eq!(carbon.state().stopped_observing, 1);
}

#[test]
fn test_mac_default_hotkeys_on_dvorak_use_the_labelled_keys() {
    let carbon = FakeCarbon::new(Some(mac_dvorak()));
    let mac = mac_with(&carbon);
    for (name, key) in [
        ("toggle_tracking", "t"),
        ("toggle_privacy", "p"),
        ("recalibrate", "c"),
    ] {
        assert!(mac.register(name, format!("ctrl+alt+{key}").as_str().into(), noop()));
    }
    assert_eq!(
        mac_keycode_of(&mac, &carbon, "toggle_tracking"),
        Some(mac_code("k"))
    );
    assert_eq!(
        mac_keycode_of(&mac, &carbon, "toggle_privacy"),
        Some(mac_code("r"))
    );
    assert_eq!(
        mac_keycode_of(&mac, &carbon, "recalibrate"),
        Some(mac_code("i"))
    );
}

#[test]
fn test_mac_non_latin_layout_uses_the_ascii_capable_layout() {
    let carbon = FakeCarbon::with_layers(Some(mac_russian()), Some(mac_dvorak()), HashMap::new());
    let mac = mac_with(&carbon);
    assert!(mac.register("t", "ctrl+alt+t".into(), noop()));
    assert_eq!(mac_keycode_of(&mac, &carbon, "t"), Some(mac_code("k"))); // Dvorak 't'
    assert!(mac.register("period", "ctrl+alt+.".into(), noop()));
    // The current layout wins where it has the character (Russian '.' at US '/').
    assert_eq!(
        mac_keycode_of(&mac, &carbon, "period"),
        Some(mac_code("slash"))
    );
}

#[test]
fn test_mac_without_layout_data_uses_us_positions() {
    let carbon = FakeCarbon::new(None);
    let mac = mac_with(&carbon);
    assert!(mac.register("a", "ctrl+alt+a".into(), noop()));
    assert!(mac.register("slash", "ctrl+alt+/".into(), noop()));
    assert_eq!(mac_keycode_of(&mac, &carbon, "a"), Some(0x00));
    assert_eq!(mac_keycode_of(&mac, &carbon, "slash"), Some(0x2C));
}

#[test]
fn test_mac_registrations_follow_layout_switches() {
    let carbon = FakeCarbon::new(Some(mac_us()));
    let fired = Arc::new(Mutex::new(Vec::new()));
    {
        let mac = mac_with(&carbon);
        assert!(mac.register("a", "ctrl+alt+a".into(), push(&fired, "a")));
        assert!(mac.register("t", "ctrl+alt+t".into(), noop()));
        assert!(mac.register("slash", "ctrl+alt+/".into(), noop()));
        let t_id = mac.0.core.table.lock().bindings["t"].id;
        let t_ref = *carbon
            .state()
            .registered
            .iter()
            .find(|(_, v)| v.3 == t_id)
            .unwrap()
            .0;
        carbon.switch_layout(mac_azerty());
        assert_eq!(mac_keycode_of(&mac, &carbon, "a"), Some(mac_code("q")));
        assert!(carbon.state().registered.contains_key(&t_ref)); // unchanged key: left alone
        // '/' has no key of its own on AZERTY: the hotkey is inactive, not forgotten.
        let names: BTreeSet<String> = mac.registered().into_keys().collect();
        assert_eq!(names, BTreeSet::from(["a".to_string(), "t".to_string()]));
        assert_eq!(mac_keycode_of(&mac, &carbon, "slash"), None);
        let error = mac.last_error("slash").expect("error");
        assert!(error.contains("no key types '/'"));
        assert_eq!(carbon.press(mac_code("q"), 0x1800), 0);
        assert_eq!(*fired.lock(), ["a"]);
        carbon.switch_layout(mac_us()); // back again: everything is restored
        assert_eq!(mac.registered().len(), 3);
        assert_eq!(mac_keycode_of(&mac, &carbon, "slash"), Some(0x2C));
        assert_eq!(mac_keycode_of(&mac, &carbon, "a"), Some(0x00));
        assert_eq!(mac.last_error("slash"), None);
        // Re-registering the inactive-then-restored name is a no-op like any other.
        assert!(mac.register("slash", "ctrl+alt+/".into(), noop()));
        assert_eq!(carbon.state().registered.len(), 3);
    }
    assert!(carbon.state().registered.is_empty());
}

#[test]
fn test_mac_register_retries_an_inactive_binding() {
    let carbon = FakeCarbon::new(Some(mac_us()));
    let mac = mac_with(&carbon);
    assert!(mac.register("slash", "ctrl+alt+/".into(), noop()));
    carbon.switch_layout(mac_azerty());
    assert!(mac.registered().is_empty());
    {
        // changed without a notification
        let mut st = carbon.state();
        st.layout = Some(mac_us());
        st.ascii_layout = Some(mac_us());
    }
    let keymap = mac.0.read_keymap(&carbon);
    mac.0.state.lock().keymap = keymap;
    assert!(mac.register("slash", "ctrl+alt+/".into(), noop())); // not the no-op path
    assert_eq!(
        mac.registered(),
        HashMap::from([("slash".to_string(), hk("ctrl+alt+/"))])
    );
}

#[test]
fn test_mac_option_combinations_that_type_are_refused() {
    let (option, shift) = (0x0800, 0x0200);
    let e_code = mac_code("e");
    let layer = BTreeMap::from([(e_code, "´".to_string())]);
    let carbon = FakeCarbon::with_layers(
        Some(mac_us()),
        Some(mac_us()),
        HashMap::from([(option, layer.clone()), (option | shift, layer)]),
    );
    let mac = mac_with(&carbon);
    assert!(!mac.register("e", "alt+e".into(), noop()));
    let typed = "⌥E types '´' on the current keyboard layout";
    assert_eq!(mac.last_error("e").as_deref(), Some(typed));
    assert!(!mac.register("e", "alt+shift+e".into(), noop()));
    assert_eq!(
        mac.layout_conflict("alt+e".into()),
        Ok(Some(typed.to_string()))
    );
    // Control or Command switch the typing layer off.
    assert_eq!(mac.layout_conflict("ctrl+alt+e".into()), Ok(None));
    assert!(mac.register("e", "ctrl+alt+e".into(), noop()));
    assert!(mac.register("f5", "alt+f5".into(), noop())); // types nothing
    assert_eq!(mac.layout_conflict("alt+r".into()), Ok(None)); // nothing on this fake's ⌥R
}

/// ⌥Space (Alfred's default) types U+00A0 on US/ABC: say so without escapes.
#[test]
fn test_mac_option_space_names_the_no_break_space() {
    let space = mac_code("space");
    let carbon = FakeCarbon::with_layers(
        Some(mac_us()),
        Some(mac_us()),
        HashMap::from([(0x0800, BTreeMap::from([(space, "\u{a0}".to_string())]))]),
    );
    let mac = mac_with(&carbon);
    assert!(!mac.register("s", "alt+space".into(), noop()));
    assert_eq!(
        mac.last_error("s").as_deref(),
        Some("⌥Space types no-break space (U+00A0) on the current keyboard layout")
    );
}

#[test]
fn test_mac_defaults_avoid_window_manager_shortcuts() {
    let carbon = FakeCarbon::new(None);
    let mac = mac_with(&carbon);
    let (control_option, command) = (0x1000 | 0x0800, 0x0100);
    // Rectangle and Magnet hold ⌃⌥T (Last Two Thirds) and ⌃⌥C (Center) by default.
    for key in ["t", "c"] {
        carbon
            .state()
            .reject
            .insert((mac_code(key), control_option), -9878);
    }
    let defaults = defaults(Platform::MacOs);
    for (name, text) in ["toggle_tracking", "toggle_privacy", "recalibrate"]
        .iter()
        .zip(&defaults)
    {
        assert!(mac.register(name, text.into(), noop()), "{name}");
    }
    let masks: BTreeSet<u32> = carbon.state().registered.values().map(|v| v.1).collect();
    assert_eq!(masks, BTreeSet::from([control_option | command]));
    assert_eq!(format_hotkey(&hk(&defaults[0]), Some(true)), "⌃⌥⌘T");
    // The old defaults: registered exclusively, so the collision is reported clearly instead
    // of both apps reacting to one press.
    assert!(!mac.register("old", LEGACY_HOTKEYS[0].into(), noop()));
    assert_eq!(
        mac.last_error("old").as_deref(),
        Some("⌃⌥T is already in use by another application")
    );
}

#[test]
fn test_mac_layout_conflict_before_start_and_off_main_thread() {
    let e_code = mac_code("e");
    let carbon = FakeCarbon::with_layers(
        Some(mac_us()),
        Some(mac_us()),
        HashMap::from([(0x0800, BTreeMap::from([(e_code, "´".to_string())]))]),
    );
    let mac = mac_with(&carbon);
    assert!(mac.layout_conflict("alt+e".into()).unwrap().is_some()); // reads the layout on demand
    assert_eq!(carbon.state().install_calls, 0);
    let result = thread::scope(|s| s.spawn(|| mac.layout_conflict("alt+e".into())).join());
    // TIS calls are main-thread only: no answer, no crash
    assert_eq!(result.unwrap(), Ok(None));
}

#[test]
fn test_mac_char_keymap_prefers_us_position_for_duplicates() {
    let chars: BTreeMap<u32, String> = [
        (0x0A, "<"),
        (0x32, "<"),
        (0x2B, ","),
        (0x31, " "),
        (0x24, "ab"),
        (0x00, "Q"),
    ]
    .into_iter()
    .map(|(code, text)| (code, text.to_string()))
    .collect();
    let keymap = mac_char_keymap(&chars);
    assert_eq!(
        keymap,
        HashMap::from([('<', 0x0A), (',', 0x2B), ('q', 0x00)])
    );
    let xs = BTreeMap::from([(0x05, "x".to_string()), (0x07, "x".to_string())]);
    assert_eq!(mac_char_keymap(&xs), HashMap::from([('x', 0x07)])); // US 'x' is 0x07
    // non-character keys: fixed codes
    assert_eq!(
        mac_keycode("f5", Some(&HashMap::from([('f', 0x03)]))),
        Some(0x60)
    );
    assert_eq!(mac_keycode("minus", Some(&HashMap::new())), None);
    assert_eq!(mac_keycode("minus", None), Some(0x1B));
}

#[test]
fn test_mac_stop_is_idempotent_and_restartable() {
    let carbon = FakeCarbon::new(None);
    let mac = mac_with(&carbon);
    mac.stop(); // never started: nothing to remove
    assert!(carbon.state().removed.is_empty());
    assert!(mac.register("t", "ctrl+alt+p".into(), noop()));
    mac.stop();
    mac.stop();
    assert_eq!(carbon.state().removed, [HANDLER_REF]);
    assert!(carbon.state().registered.is_empty());
    assert!(mac.register("t", "ctrl+alt+p".into(), noop()));
    assert_eq!(carbon.state().install_calls, 2);
}

#[test]
fn test_mac_refuses_other_threads() {
    let carbon = FakeCarbon::new(None);
    let mac = mac_with(&carbon);
    let result = thread::scope(|s| {
        s.spawn(|| mac.register("t", MAC_COMBO.into(), noop()))
            .join()
    });
    assert!(!result.unwrap());
    assert_eq!(carbon.state().install_calls, 0);
}

#[test]
fn test_mac_keycode_table() {
    let keys: BTreeSet<&str> = MAC_KEYCODES.iter().map(|(k, _)| *k).collect();
    let expected: BTreeSet<&str> = KEYS
        .iter()
        .copied()
        .filter(|k| !["f21", "f22", "f23", "f24"].contains(k))
        .collect();
    assert_eq!(keys, expected);
    let codes: BTreeSet<u32> = MAC_KEYCODES.iter().map(|(_, c)| *c).collect();
    assert_eq!(codes.len(), MAC_KEYCODES.len());
    assert_eq!(mac_code("a"), 0x00);
    assert_eq!(mac_code("space"), 0x31);
    assert_eq!(mac_code("escape"), 0x35);
    assert_eq!(mac_code("f1"), 0x7A);
    assert_eq!(mac_code("f20"), 0x5A);
}

#[test]
fn test_fourcc() {
    assert_eq!(fourcc(b"keyb"), 0x6B657962);
    assert_eq!(fourcc(b"hkid"), 0x686B6964);
    assert_eq!(fourcc(b"----"), 0x2D2D2D2D);
}

#[cfg(not(target_os = "macos"))]
#[test]
fn test_mac_carbon_unavailable() {
    // The real loader, with this test thread standing in for the main thread (as under pytest).
    let manager = MacHotkeyManager::build(None, Some(thread::current().id()));
    assert!(!manager.register("t", "cmd+alt+k".into(), noop()));
    assert!(manager.note().is_some());
    manager.stop();
}

// ================================================================================ X11
#[cfg(unix)]
mod x11_tests {
    use std::collections::VecDeque;
    use std::os::fd::RawFd;

    use super::*;
    use crate::config::LEGACY_HOTKEYS_X11;
    use crate::platform::hotkeys::x11::*;

    /// Fake keysyms: every name the module uses, numbered in sorted order.
    fn keysyms() -> BTreeMap<String, u32> {
        let mut names: BTreeSet<String> = KEYS
            .iter()
            .map(|k| x11_keysym_name(k).to_string())
            .collect();
        names.extend(["Num_Lock", "Super_L", "Super_R"].map(String::from));
        names
            .into_iter()
            .enumerate()
            .map(|(index, name)| (name, index as u32 + 1))
            .collect()
    }

    fn keysym(name: &str) -> u32 {
        keysyms()[name]
    }

    fn numlock_keycode() -> u8 {
        (keysym("Num_Lock") + 8) as u8
    }

    fn keycode_of(key: &'static str) -> u8 {
        (keysym(x11_keysym_name(canonical_key(key).unwrap())) + 8) as u8
    }

    #[derive(Default)]
    struct FakeRoot {
        grabs: BTreeSet<(u8, u16)>,
        foreign: BTreeSet<(u8, u16)>, // held by "another client"
        grab_calls: Vec<(u8, u16, bool, u8, u8)>,
        properties: HashMap<u32, Vec<u8>>, // atom → property value
        property_reads: usize,
        broken_reads: bool,
        event_mask: Option<u32>, // what this client selected on the root
    }

    struct FakeDisplayState {
        root: FakeRoot,
        keymap: HashMap<u32, u8>,
        modmap: Vec<Vec<u8>>,
        events: VecDeque<XEvent>,
        refreshed: Vec<String>,
        closed: bool,
        atoms: BTreeMap<String, u32>,
        broken_atoms: bool,
        r: RawFd,
        w: RawFd,
    }

    /// The fake X connection; clones share one state (the test keeps one, the manager another).
    #[derive(Clone)]
    struct FakeDisplay(Arc<Mutex<FakeDisplayState>>);

    impl FakeDisplay {
        fn new(selectable: bool) -> FakeDisplay {
            let (mut r, mut w) = (-1, -1);
            if selectable {
                let mut fds = [-1; 2];
                // SAFETY: fds has room for two descriptors.
                assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
                // SAFETY: fds[0] was just created.
                unsafe {
                    let flags = libc::fcntl(fds[0], libc::F_GETFL);
                    libc::fcntl(fds[0], libc::F_SETFL, flags | libc::O_NONBLOCK);
                }
                (r, w) = (fds[0], fds[1]);
            }
            let nl = numlock_keycode();
            FakeDisplay(Arc::new(Mutex::new(FakeDisplayState {
                root: FakeRoot::default(),
                keymap: keysyms()
                    .values()
                    .map(|sym| (*sym, (*sym + 8) as u8))
                    .collect(),
                // Modifier index 4 (Mod2) holds Num_Lock, as on virtually every keymap.
                modmap: vec![
                    vec![50],
                    vec![66],
                    vec![37],
                    vec![64],
                    vec![nl],
                    vec![],
                    vec![133],
                    vec![],
                ],
                events: VecDeque::new(),
                refreshed: Vec::new(),
                closed: false,
                atoms: BTreeMap::new(),
                broken_atoms: false,
                r,
                w,
            })))
        }

        fn st(&self) -> parking_lot::MutexGuard<'_, FakeDisplayState> {
            self.0.lock()
        }

        fn push(&self, event: XEvent) {
            let mut st = self.st();
            st.events.push_back(event);
            if st.w >= 0 {
                // SAFETY: writing one byte from a valid buffer.
                unsafe { libc::write(st.w, b"x".as_ptr().cast(), 1) };
            }
        }

        /// Record `options` in `_XKB_RULES_NAMES` as setxkbmap does; returns the atom.
        fn set_xkb_options(&self, options: &[&str]) -> u32 {
            let atom = self.intern(XKB_RULES_NAMES, false);
            let names = ["evdev", "pc105", "us,ru", ",", &options.join(",")];
            let mut value = names.join("\0").into_bytes();
            value.push(0);
            self.st().root.properties.insert(atom, value);
            atom
        }

        fn intern(&self, name: &str, only_if_exists: bool) -> u32 {
            let mut st = self.st();
            if let Some(atom) = st.atoms.get(name) {
                return *atom;
            }
            if only_if_exists {
                return 0; // X.NONE
            }
            let atom = 100 + st.atoms.len() as u32;
            st.atoms.insert(name.to_string(), atom);
            atom
        }
    }

    impl XDisplay for FakeDisplay {
        fn string_to_keysym(&self, name: &str) -> u32 {
            keysyms().get(name).copied().unwrap_or(0)
        }

        fn keysym_to_keycode(&mut self, keysym: u32) -> u8 {
            self.st().keymap.get(&keysym).copied().unwrap_or(0)
        }

        fn get_modifier_mapping(&mut self) -> Result<Vec<Vec<u8>>, String> {
            Ok(self.st().modmap.clone())
        }

        fn grab_key(
            &mut self,
            keycode: u8,
            modifiers: u16,
            owner_events: bool,
            pointer_mode: u8,
            keyboard_mode: u8,
            onerror: &mut XErrorCatcher,
        ) {
            let mut st = self.st();
            st.root.grab_calls.push((
                keycode,
                modifiers,
                owner_events,
                pointer_mode,
                keyboard_mode,
            ));
            if st.root.foreign.contains(&(keycode, modifiers)) {
                onerror.catch("BadAccess");
                return;
            }
            st.root.grabs.insert((keycode, modifiers));
        }

        fn ungrab_key(&mut self, keycode: u8, modifiers: u16, _onerror: &mut XErrorCatcher) {
            self.st().root.grabs.remove(&(keycode, modifiers));
        }

        fn sync(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn flush(&mut self) {}

        fn refresh_keyboard_mapping(&mut self, event: &XEvent) {
            self.st().refreshed.push(format!("{event:?}"));
        }

        fn intern_atom(&mut self, name: &str, only_if_exists: bool) -> Result<u32, String> {
            if self.st().broken_atoms {
                return Err("lost the X server".to_string());
            }
            Ok(self.intern(name, only_if_exists))
        }

        fn get_root_property(&mut self, atom: u32) -> Result<Option<Vec<u8>>, String> {
            let mut st = self.st();
            st.root.property_reads += 1;
            if st.root.broken_reads {
                return Err("lost the X server".to_string());
            }
            Ok(st.root.properties.get(&atom).cloned())
        }

        fn select_root_events(&mut self, event_mask: u32) -> Result<(), String> {
            self.st().root.event_mask = Some(event_mask);
            Ok(())
        }

        fn poll_event(&mut self) -> Result<Option<XEvent>, String> {
            let mut st = self.st();
            if st.r >= 0 {
                let mut buf = [0u8; 4096];
                // SAFETY: non-blocking read into a valid buffer.
                while unsafe { libc::read(st.r, buf.as_mut_ptr().cast(), buf.len()) } > 0 {}
            }
            Ok(st.events.pop_front())
        }

        fn fileno(&self) -> RawFd {
            self.st().r
        }

        fn close(&mut self) {
            let mut st = self.st();
            st.closed = true;
            for fd in [st.r, st.w] {
                if fd >= 0 {
                    // SAFETY: closing the fake's own pipe.
                    unsafe { libc::close(fd) };
                }
            }
            st.r = -1;
            st.w = -1;
        }
    }

    fn factory_of(display: &FakeDisplay) -> DisplayFactory {
        let display = display.clone();
        Box::new(move |_| Ok(Box::new(display.clone()) as Box<dyn XDisplay>))
    }

    fn property_event(atom: u32) -> XEvent {
        XEvent::PropertyNotify { atom }
    }

    fn press(detail: u8, state: u16, time: u32) -> XEvent {
        XEvent::KeyPress {
            detail,
            state,
            time,
        }
    }

    /// `_state` is what X reports; releases are matched by keycode alone.
    fn release(detail: u8, _state: u16, time: u32) -> XEvent {
        XEvent::KeyRelease { detail, time }
    }

    fn mapping(request: u8) -> XEvent {
        XEvent::MappingNotify { request }
    }

    /// An X11 manager wired to a fake display, driven synchronously (no thread).
    fn x11() -> (X11HotkeyManager, FakeDisplay) {
        let display = FakeDisplay::new(false);
        let manager = X11HotkeyManager::with_display_factory(None, None, factory_of(&display));
        attach(&manager, &display);
        let mut st = manager.0.state.lock();
        st.numlock_mask = X11Backend::find_numlock_mask(&mut st);
        drop(st);
        (manager, display)
    }

    fn attach(manager: &X11HotkeyManager, display: &FakeDisplay) {
        manager.0.state.lock().display = Some(Box::new(display.clone()));
    }

    fn bind_x11(
        manager: &X11HotkeyManager,
        name: &str,
        text: &str,
        callback: HotkeyCallback,
    ) -> Arc<Binding> {
        let core = &manager.0.core;
        let binding = Arc::new(Binding::new(
            core.allocate_id().unwrap(),
            name,
            hk(text),
            callback,
        ));
        let mut table = core.table.lock();
        table.by_id.insert(binding.id, binding.clone());
        table.bindings.insert(name.to_string(), binding.clone());
        binding
    }

    fn follow(manager: &X11HotkeyManager) {
        let mut st = manager.0.state.lock();
        manager.0.follow_xkb_options(&mut st);
    }

    fn check_super(manager: &X11HotkeyManager) {
        let mut st = manager.0.state.lock();
        manager.0.check_super_alone(&mut st);
    }

    fn xkb_options(manager: &X11HotkeyManager) -> Option<Vec<String>> {
        manager.0.xkb_options.lock().clone()
    }

    fn registered_names(manager: &X11HotkeyManager) -> BTreeSet<String> {
        manager.registered().into_keys().collect()
    }

    #[test]
    fn test_x11_lock_variants() {
        assert_eq!(x11_lock_variants(16), [0, 2, 16, 18]);
        assert_eq!(x11_lock_variants(32), [0, 2, 32, 34]);
        assert_eq!(x11_lock_variants(2), [0, 2]); // degenerate keymap: no duplicates
    }

    #[test]
    fn test_x11_modifier_masks() {
        assert_eq!(x11_modifiers(["ctrl", "alt"]), 4 | 8);
        assert_eq!(x11_modifiers(["shift", "meta"]), 1 | 64);
    }

    #[test]
    fn test_x11_keysym_names_cover_every_key() {
        for key in KEYS {
            assert_ne!(x11_string_to_keysym(x11_keysym_name(key)), 0, "{key}");
        }
    }

    #[test]
    fn test_x11_numlock_detection() {
        let (manager, display) = x11();
        assert_eq!(manager.0.state.lock().numlock_mask, 16);
        let nl = numlock_keycode();
        display.st().modmap = vec![
            vec![50],
            vec![66],
            vec![37],
            vec![64],
            vec![],
            vec![nl],
            vec![133],
            vec![],
        ];
        assert_eq!(
            X11Backend::find_numlock_mask(&mut manager.0.state.lock()),
            32
        );
        display.st().modmap = vec![vec![]; 8];
        // fallback: Mod2
        assert_eq!(
            X11Backend::find_numlock_mask(&mut manager.0.state.lock()),
            16
        );
    }

    #[test]
    fn test_x11_grab_includes_lock_variants() {
        let (manager, display) = x11();
        let binding = bind_x11(&manager, "t", "ctrl+alt+p", noop());
        assert!(manager.0.grab_binding(&binding));
        let code = keycode_of("p");
        assert_eq!(
            display.st().root.grabs,
            BTreeSet::from([(code, 12), (code, 14), (code, 28), (code, 30)])
        );
        assert!(
            display
                .st()
                .root
                .grab_calls
                .iter()
                .all(|c| (c.2, c.3, c.4) == (true, 1, 1))
        );
        manager.0.ungrab_binding(binding.id);
        assert!(display.st().root.grabs.is_empty());
    }

    #[test]
    fn test_x11_bad_access_rolls_back() {
        let (manager, display) = x11();
        let code = keycode_of("p");
        display.st().root.foreign.insert((code, 12 | 16)); // another client owns the NumLock variant
        let binding = bind_x11(&manager, "t", "ctrl+alt+p", noop());
        assert!(!manager.0.grab_binding(&binding));
        assert!(display.st().root.grabs.is_empty()); // partial grabs released
        assert!(manager.0.state.lock().lookup.is_empty());
    }

    #[test]
    fn test_x11_missing_key_on_layout() {
        let (manager, display) = x11();
        display.st().keymap.remove(&keysym("F24"));
        let binding = bind_x11(&manager, "t", COMBO, noop());
        assert!(!manager.0.grab_binding(&binding));
    }

    #[test]
    fn test_x11_event_dispatch_and_autorepeat() {
        let (manager, _display) = x11();
        let fired = Arc::new(Mutex::new(Vec::new()));
        let binding = bind_x11(&manager, "t", "ctrl+alt+p", push(&fired, 1));
        assert!(manager.0.grab_binding(&binding));
        let code = keycode_of("p");
        let handle = |event| manager.0.handle_event(event);
        // Caps Lock + Num Lock active: still matches.
        handle(press(code, 12 | 2 | 16, 1000));
        assert_eq!(*fired.lock(), [1]);
        // Auto-repeat: release and press share a timestamp → ignored.
        handle(release(code, 12, 1500));
        handle(press(code, 12, 1500));
        handle(release(code, 12, 1530));
        handle(press(code, 12, 1530));
        assert_eq!(*fired.lock(), [1]);
        // Real release, then a new press later → fires again.
        handle(release(code, 12, 1600));
        handle(press(code, 12, 1800));
        assert_eq!(*fired.lock(), [1, 1]);
        // Different modifiers or keys are not ours.
        handle(release(code, 12, 1900));
        handle(press(code, 4, 2000));
        handle(press(code + 1, 12, 2100));
        handle(XEvent::Other); // unrelated event type
        assert_eq!(*fired.lock(), [1, 1]);
    }

    #[test]
    fn test_x11_timestamp_wraparound() {
        let (manager, _display) = x11();
        let fired = Arc::new(Mutex::new(Vec::new()));
        let binding = bind_x11(&manager, "t", "ctrl+alt+p", push(&fired, 1));
        assert!(manager.0.grab_binding(&binding));
        let code = keycode_of("p");
        manager.0.handle_event(press(code, 12, 0xFFFFFF00));
        manager.0.handle_event(release(code, 12, 0xFFFFFFFF));
        manager.0.handle_event(press(code, 12, 0x00000100));
        assert_eq!(*fired.lock(), [1, 1]);
    }

    #[test]
    fn test_x11_releasing_modifiers_first_rearms_the_hotkey() {
        // A KeyRelease reports the modifiers held just before it: lifting Ctrl/Alt first gives
        // state 0 (or Mod1/Control only). The hotkey must fire on every press.
        for release_state in [0, 8, 4, 12 | 2 | 16] {
            let (manager, _display) = x11();
            let fired = Arc::new(Mutex::new(Vec::new()));
            let binding = bind_x11(&manager, "t", "ctrl+alt+p", push(&fired, 1));
            assert!(manager.0.grab_binding(&binding));
            let code = keycode_of("p");
            for n in 1..6 {
                manager.0.handle_event(press(code, 12, 1000 * n));
                manager
                    .0
                    .handle_event(release(code, release_state, 1000 * n + 90));
            }
            assert_eq!(*fired.lock(), [1; 5], "{release_state}");
        }
    }

    #[test]
    fn test_x11_release_of_another_key_keeps_the_hotkey_held() {
        let (manager, _display) = x11();
        let fired = Arc::new(Mutex::new(Vec::new()));
        let p = bind_x11(&manager, "p", "ctrl+alt+p", push(&fired, "p"));
        let k = bind_x11(&manager, "k", "ctrl+alt+shift+k", push(&fired, "k"));
        assert!(manager.0.grab_binding(&p));
        assert!(manager.0.grab_binding(&k));
        let handle = |event| manager.0.handle_event(event);
        handle(press(keycode_of("p"), 12, 100));
        handle(release(keycode_of("k"), 0, 200)); // not P's release
        handle(press(keycode_of("p"), 12, 300)); // P still held
        assert_eq!(*fired.lock(), ["p"]);
        handle(release(keycode_of("p"), 0, 400));
        handle(press(keycode_of("k"), 13, 500));
        handle(press(keycode_of("p"), 12, 600));
        assert_eq!(*fired.lock(), ["p", "k", "p"]);
    }

    #[test]
    fn test_x11_hotkey_lost_by_a_layout_change_comes_back() {
        let (manager, display) = x11();
        let fired = Arc::new(Mutex::new(Vec::new()));
        let binding = bind_x11(&manager, "t", "ctrl+alt+p", push(&fired, 1));
        assert!(manager.0.grab_binding(&binding));
        // e.g. `setxkbmap ru`: no Latin keysyms
        let code = display.st().keymap.remove(&keysym("p")).unwrap();
        manager.0.handle_event(mapping(1));
        assert!(display.st().root.grabs.is_empty());
        assert!(manager.registered().is_empty()); // not shown as working any more …
        let error = manager.last_error("t").expect("error");
        assert!(error.contains("not on the current keyboard layout"));
        display.st().keymap.insert(keysym("p"), code); // `setxkbmap us`
        manager.0.handle_event(mapping(1)); // … but retried on the next change
        assert!(display.st().root.grabs.contains(&(code, 12)));
        assert_eq!(
            manager.registered(),
            HashMap::from([("t".to_string(), hk("ctrl+alt+p"))])
        );
        assert_eq!(manager.last_error("t"), None);
        manager.0.handle_event(press(code, 12, 10));
        assert_eq!(*fired.lock(), [1]);
    }

    #[test]
    fn test_x11_regrab_keeps_in_flight_and_drops_removed_bindings() {
        let (manager, display) = x11();
        let core = &manager.0.core;
        // Registration in flight: published in by_id and grabbed on the hotkey thread, but
        // register() has not added it to bindings yet.
        let in_flight = Arc::new(Binding::new(
            core.allocate_id().unwrap(),
            "new",
            hk("ctrl+alt+k"),
            noop(),
        ));
        core.table
            .lock()
            .by_id
            .insert(in_flight.id, in_flight.clone());
        assert!(manager.0.grab_binding(&in_flight));
        // Removal in flight: remove() forgot the binding, its ungrab is still queued.
        let removed = Binding::new(core.allocate_id().unwrap(), "old", hk("ctrl+alt+j"), noop());
        assert!(manager.0.grab_binding(&removed));
        manager.0.handle_event(mapping(0));
        let codes: BTreeSet<u8> = display.st().root.grabs.iter().map(|g| g.0).collect();
        assert_eq!(codes, BTreeSet::from([keycode_of("k")]));
        let grabbed: Vec<u32> = manager.0.state.lock().grabs.keys().copied().collect();
        assert_eq!(grabbed, [in_flight.id]);
    }

    #[test]
    fn test_x11_loop_waits_without_timeout() {
        // No periodic wake-ups while idle: only events and the wake pipe end the wait.
        assert_eq!(SELECT_TIMEOUT, None);
    }

    #[test]
    fn test_x11_mapping_notify_regrabs() {
        let (manager, display) = x11();
        let fired = Arc::new(Mutex::new(Vec::new()));
        let binding = bind_x11(&manager, "t", "ctrl+alt+p", push(&fired, 1));
        assert!(manager.0.grab_binding(&binding));
        let new = keycode_of("p") + 100;
        display.st().keymap.insert(keysym("p"), new); // layout change moved the key
        manager.0.handle_event(mapping(1));
        assert_eq!(display.st().refreshed, [format!("{:?}", mapping(1))]);
        let codes: BTreeSet<u8> = display.st().root.grabs.iter().map(|g| g.0).collect();
        assert_eq!(codes, BTreeSet::from([new]));
        manager.0.handle_event(press(new, 12, 10));
        assert_eq!(*fired.lock(), [1]);
        // Pointer mapping changes are irrelevant.
        manager.0.handle_event(mapping(2));
        assert_eq!(display.st().refreshed.len(), 1);
    }

    #[test]
    fn test_x11_error_catcher_protocol() {
        let mut catcher = XErrorCatcher::default();
        catcher.catch("first");
        catcher.catch("second");
        assert_eq!(catcher.error.as_deref(), Some("first"));
    }

    // ---------------------------------------------- X11 keyboard-layout switches (XKB)
    const ALT_SHIFT: &str = "grp:alt_shift_toggle";
    const SWITCHES: &str = "which switches the keyboard layout";

    #[test]
    fn test_x11_parse_xkb_options() {
        let cases: [(Option<&[u8]>, &[&str]); 5] = [
            (
                Some(b"evdev\0pc105\0us,ru\0,\0grp:alt_shift_toggle,terminate:ctrl_alt_bksp\0"),
                &[ALT_SHIFT, "terminate:ctrl_alt_bksp"],
            ),
            (
                Some(b"evdev\0pc105\0us\0\0grp:ctrl_shift_toggle"),
                &["grp:ctrl_shift_toggle"],
            ),
            (
                Some(b"evdev\0pc105\0us\0\0 grp:lwin_toggle , \0"),
                &["grp:lwin_toggle"],
            ),
            (Some(b"base\0pc105\0us\0\0\0"), &[]),
            (Some(b"evdev\0pc105\0us"), &[]), // truncated
        ];
        for (value, expected) in cases {
            assert_eq!(x11_parse_xkb_options(value), expected, "{value:?}");
        }
        assert!(x11_parse_xkb_options(Some(b"")).is_empty());
        assert!(x11_parse_xkb_options(None).is_empty()); // no such property
    }

    #[test]
    fn test_x11_layout_switch_conflict() {
        let s = SWITCHES;
        let cases: Vec<(&str, Vec<&str>, Option<String>)> =
            vec![
            // The finding: Ctrl+Alt+Shift (the old Linux default) with an Alt+Shift switch.
            (
                "ctrl+alt+shift+t",
                vec![ALT_SHIFT],
                Some(format!(
                    "Ctrl+Alt+Shift+T includes Alt+Shift, {s} (XKB option {ALT_SHIFT})"
                )),
            ),
            // The first option that takes keys of the hotkey is named.
            (
                "ctrl+alt+shift+p",
                vec!["terminate:ctrl_alt_bksp", "grp:ctrl_shift_toggle", ALT_SHIFT],
                Some(format!(
                    "Ctrl+Alt+Shift+P includes Ctrl+Shift, {s} (XKB option grp:ctrl_shift_toggle)"
                )),
            ),
            // Today's default with the rare switches that do take its keys.
            (
                "ctrl+alt+meta+c",
                vec!["grp:ctrl_alt_toggle"],
                Some(format!(
                    "Ctrl+Alt+Super+C includes Ctrl+Alt, {s} (XKB option grp:ctrl_alt_toggle)"
                )),
            ),
            (
                "ctrl+alt+meta+t",
                vec!["grp:lwin_toggle"],
                Some(format!(
                    "Ctrl+Alt+Super+T includes Super, {s} (XKB option grp:lwin_toggle)"
                )),
            ),
            (
                "ctrl+alt+meta+t",
                vec!["lv3:lwin_switch"],
                Some(
                    "Ctrl+Alt+Super+T includes Super, which acts as AltGr (XKB option \
                     lv3:lwin_switch)"
                        .to_string(),
                ),
            ),
            // r3-platform-hotkeys-03: options that leave no Super (or no Ctrl) at all.
            (
                "ctrl+alt+meta+t",
                vec!["grp:win_menu_switch"],
                Some(format!(
                    "Ctrl+Alt+Super+T includes Super, {s} (XKB option grp:win_menu_switch)"
                )),
            ),
            (
                "ctrl+alt+meta+t",
                vec!["grp:win_menu_select"],
                Some(format!(
                    "Ctrl+Alt+Super+T includes Super, {s} (XKB option grp:win_menu_select)"
                )),
            ),
            (
                "ctrl+alt+meta+p",
                vec!["grp:ctrl_select"],
                Some(format!(
                    "Ctrl+Alt+Super+P includes Ctrl, {s} (XKB option grp:ctrl_select)"
                )),
            ),
            (
                "ctrl+alt+meta+p",
                vec!["grp:lctrl_rctrl_switch"],
                Some(format!(
                    "Ctrl+Alt+Super+P includes Ctrl, {s} (XKB option grp:lctrl_rctrl_switch)"
                )),
            ),
            (
                "ctrl+alt+meta+c",
                vec!["altwin:ctrl_win"],
                Some(
                    "Ctrl+Alt+Super+C includes Super, which turns the Win keys into Ctrl (XKB \
                     option altwin:ctrl_win)"
                        .to_string(),
                ),
            ),
            (
                "ctrl+alt+meta+c",
                vec!["altwin:alt_win"],
                Some(
                    "Ctrl+Alt+Super+C includes Super, which turns the Win keys into Alt (XKB \
                     option altwin:alt_win)"
                        .to_string(),
                ),
            ),
            (
                "ctrl+alt+meta+t",
                vec!["lv5:lwin_switch_lock"],
                Some(
                    "Ctrl+Alt+Super+T includes Super, which acts as a level-5 key (XKB option \
                     lv5:lwin_switch_lock)"
                        .to_string(),
                ),
            ),
            (
                "ctrl+alt+meta+t",
                vec!["lv5:lwin_switch_lock_cancel"],
                Some(
                    "Ctrl+Alt+Super+T includes Super, which acts as a level-5 key (XKB option \
                     lv5:lwin_switch_lock_cancel)"
                        .to_string(),
                ),
            ),
            // Swapping keys keeps every modifier available.
            (
                "ctrl+alt+meta+t",
                vec!["altwin:swap_alt_win", "altwin:ctrl_alt_win"],
                None,
            ),
            // Key toggles only concern their key.
            (
                "ctrl+alt+space",
                vec!["grp:alt_space_toggle"],
                Some(format!(
                    "Ctrl+Alt+Space includes Alt+Space, {s} (XKB option grp:alt_space_toggle)"
                )),
            ),
            (
                "meta+space",
                vec!["grp:win_space_toggle"],
                Some(
                    "Super+Space switches the keyboard layout (XKB option grp:win_space_toggle)"
                        .to_string(),
                ),
            ),
            (
                "ctrl+alt+t",
                vec!["grp:alt_space_toggle", "grp:win_space_toggle"],
                None,
            ),
            ("ctrl+shift+t", vec![ALT_SHIFT, "grp:nonexistent_option"], None),
            ("ctrl+alt+shift+t", vec![], None),
        ];
        for (text, options, expected) in cases {
            assert_eq!(
                x11_layout_switch_conflict(&hk(text), &options),
                expected,
                "{text} {options:?}"
            );
        }
    }

    /// Layout switches and other XKB options people commonly set (GNOME Tweaks, KDE and Xfce
    /// keyboard settings, setxkbmap guides), none of which takes keys of a default.
    const COMMON_XKB_OPTIONS: [&str; 26] = [
        ALT_SHIFT,
        "grp:alt_shift_toggle_bidir",
        "grp:lalt_lshift_toggle",
        "grp:ctrl_shift_toggle",
        "grp:lctrl_lshift_toggle",
        "grp:win_space_toggle",
        "grp:caps_toggle",
        "grp:shift_caps_toggle",
        "grp:alt_caps_toggle",
        "grp:shifts_toggle",
        "grp:alts_toggle",
        "grp:ctrls_toggle",
        "grp:toggle",
        "grp:menu_toggle",
        "grp:rwin_toggle",
        "grp:ralt_rshift_toggle",
        "grp:rctrl_ralt_toggle",
        "grp:sclk_toggle",
        "lv3:ralt_switch",
        "lv3:caps_switch",
        "compose:ralt",
        "compose:menu",
        "terminate:ctrl_alt_bksp",
        "grp_led:scroll",
        "ctrl:nocaps",
        "caps:escape",
    ];

    #[test]
    fn test_x11_default_hotkeys_survive_the_usual_layout_switches() {
        for text in defaults(Platform::Other) {
            assert_eq!(
                x11_layout_switch_conflict(&hk(&text), &COMMON_XKB_OPTIONS),
                None,
                "{text}"
            );
        }
        // The previous defaults could not be pressed with the two most common switches.
        for option in [ALT_SHIFT, "grp:ctrl_shift_toggle"] {
            for text in LEGACY_HOTKEYS_X11 {
                assert!(x11_layout_switch_conflict(&hk(text), &[option]).is_some());
            }
        }
    }

    #[test]
    fn test_x11_layout_switch_table_is_consistent() {
        for (option, taken) in &X_KEY_OPTIONS {
            assert!(
                ["grp:", "lv3:", "lv5:", "altwin:"]
                    .iter()
                    .any(|p| option.starts_with(p)),
                "{option}"
            );
            assert!(!taken.modifiers.is_empty(), "{option}");
            assert!(
                taken.modifiers.iter().all(|m| MODIFIERS.contains(m)),
                "{option}"
            );
            assert!(taken.key.is_none_or(|k| KEYS.contains(&k)), "{option}");
        }
        let names: HashSet<&str> = X_KEY_OPTIONS.iter().map(|(o, _)| *o).collect();
        assert_eq!(names.len(), X_KEY_OPTIONS.len());
        // Right-hand-only switches leave the keys that chords are pressed with alone.
        for option in [
            "grp:ralt_rshift_toggle",
            "grp:rctrl_rshift_toggle",
            "grp:rwin_toggle",
        ] {
            assert!(!names.contains(option));
        }
    }

    #[test]
    fn test_x11_refuses_a_hotkey_the_layout_switch_makes_unpressable() {
        let (manager, display) = x11();
        display.set_xkb_options(&[ALT_SHIFT, "terminate:ctrl_alt_bksp"]);
        follow(&manager);
        assert_eq!(
            xkb_options(&manager),
            Some(vec![
                ALT_SHIFT.to_string(),
                "terminate:ctrl_alt_bksp".to_string()
            ])
        );
        assert_eq!(display.st().root.event_mask, Some(X_PROPERTY_CHANGE_MASK));
        let blocked = bind_x11(&manager, "t", "ctrl+alt+shift+t", noop());
        // XGrabKey would succeed and the hotkey would silently never fire: refuse it.
        assert!(!manager.0.grab_binding(&blocked));
        assert!(display.st().root.grab_calls.is_empty());
        let expected =
            format!("Ctrl+Alt+Shift+T includes Alt+Shift, {SWITCHES} (XKB option {ALT_SHIFT})");
        assert_eq!(manager.last_error("t"), Some(expected.clone()));
        assert_eq!(
            manager.layout_conflict("ctrl+alt+shift+t".into()),
            Ok(Some(expected))
        );
        assert_eq!(
            manager.layout_conflict(hk("ctrl+alt+meta+t").into()),
            Ok(None)
        );
        assert!(!display.st().closed); // answered from what the hotkey thread knows
        let fine = bind_x11(&manager, "p", "ctrl+alt+meta+p", noop());
        assert!(manager.0.grab_binding(&fine));
        assert!(
            display
                .st()
                .root
                .grabs
                .contains(&(keycode_of("p"), 4 | 8 | 64))
        );
    }

    #[test]
    fn test_x11_follows_layout_switch_changes() {
        let (manager, display) = x11();
        follow(&manager); // no _XKB_RULES_NAMES yet: nothing to avoid
        assert_eq!(xkb_options(&manager), Some(vec![]));
        let fired = Arc::new(Mutex::new(Vec::new()));
        let binding = bind_x11(&manager, "t", "ctrl+alt+shift+t", push(&fired, 1));
        assert!(manager.0.grab_binding(&binding));
        let (code, mods) = (keycode_of("t"), 1 | 4 | 8);
        // The desktop applies the user's Alt+Shift switch after the app started (login).
        let atom = display.set_xkb_options(&[ALT_SHIFT]);
        let reads = display.st().root.property_reads;
        manager.0.handle_event(property_event(atom + 1)); // another root property: ignored
        assert_eq!(display.st().root.property_reads, reads);
        assert!(display.st().root.grabs.contains(&(code, mods)));
        manager.0.handle_event(property_event(atom));
        assert!(display.st().root.grabs.is_empty());
        assert!(manager.registered().is_empty()); // reported as not working …
        assert!(manager.last_error("t").expect("error").contains(ALT_SHIFT));
        // … an unchanged value (desktops rewrite it) does not churn the grabs …
        let grab_calls = display.st().root.grab_calls.len();
        manager.0.handle_event(property_event(atom));
        assert_eq!(display.st().root.grab_calls.len(), grab_calls);
        // … and the hotkey comes back when the switch moves to Super+Space.
        display.set_xkb_options(&["grp:win_space_toggle"]);
        manager.0.handle_event(property_event(atom));
        assert!(display.st().root.grabs.contains(&(code, mods)));
        assert_eq!(
            manager.registered(),
            HashMap::from([("t".to_string(), hk("ctrl+alt+shift+t"))])
        );
        assert_eq!(manager.last_error("t"), None);
        manager.0.handle_event(press(code, mods, 10));
        assert_eq!(*fired.lock(), [1]);
    }

    /// r3-platform-hotkeys-04: a hotkey refused when it was registered was never tried again,
    /// and its error kept naming an option that was gone.
    #[test]
    fn test_x11_hotkey_refused_at_registration_comes_back_with_the_option() {
        let (manager, display) = x11();
        let atom = display.set_xkb_options(&["grp:ctrl_alt_toggle"]);
        follow(&manager);
        // register() as the app calls it, run synchronously on this thread.
        manager.0.threaded.inline.store(true, Ordering::SeqCst);
        let fired = Arc::new(Mutex::new(Vec::new()));
        assert!(!manager.register("t", "ctrl+alt+meta+t".into(), push(&fired, 1)));
        assert!(manager.registered().is_empty());
        let error = manager.last_error("t").expect("error");
        assert!(error.contains("grp:ctrl_alt_toggle"));
        // The user moves the layout switch to Super+Space, as the message suggests.
        display.set_xkb_options(&["grp:win_space_toggle"]);
        manager.0.handle_event(property_event(atom));
        let (code, mods) = (keycode_of("t"), 4 | 8 | 64);
        assert!(display.st().root.grabs.contains(&(code, mods)));
        assert_eq!(
            manager.registered(),
            HashMap::from([("t".to_string(), hk("ctrl+alt+meta+t"))])
        );
        assert_eq!(manager.last_error("t"), None);
        manager.0.handle_event(press(code, mods, 10));
        assert_eq!(*fired.lock(), [1]);
        // Unregistering forgets it; nothing comes back with the next change.
        assert!(manager.unregister("t"));
        assert!(display.st().root.grabs.is_empty());
        display.set_xkb_options(&["grp:alt_space_toggle"]);
        manager.0.handle_event(property_event(atom));
        assert!(display.st().root.grabs.is_empty());
        // Other refusals (another program holds the chord) are not kept.
        display.st().root.foreign.insert((code, mods));
        assert!(!manager.register("t", "ctrl+alt+meta+t".into(), noop()));
        assert!(!manager.0.core.table.lock().bindings.contains_key("t"));
    }

    /// r3-platform-hotkeys-02: Xubuntu's menu grabs Super alone, which hides a chord pressed
    /// Super first from our grab although XGrabKey succeeded.
    #[test]
    fn test_x11_super_key_taken_by_another_program_is_explained() {
        let (manager, display) = x11();
        check_super(&manager);
        assert_eq!(manager.note(), None);
        assert!(display.st().root.grabs.is_empty()); // the probe released what it got
        let super_l = (keysym("Super_L") + 8) as u8;
        let numlock = manager.0.state.lock().numlock_mask;
        for variant in x11_lock_variants(numlock) {
            display.st().root.foreign.insert((super_l, variant)); // xfsettingsd's grab
        }
        check_super(&manager);
        let note = manager.note().expect("note");
        assert_eq!(note, SUPER_TAKEN_NOTE);
        assert!(note.contains("Ctrl or Alt is pressed before Super"));
        assert!(display.st().root.grabs.is_empty());
        // Re-checked after a keyboard mapping change: the menu was unbound.
        display.st().root.foreign.clear();
        manager.0.handle_event(mapping(1));
        assert_eq!(manager.note(), None);
        assert!(display.st().root.grabs.is_empty());
    }

    #[test]
    fn test_x11_super_key_note_keeps_the_managers_own_note() {
        let display = FakeDisplay::new(false);
        let manager = X11HotkeyManager::with_display_factory(
            None,
            Some("Running under Wayland.".to_string()),
            factory_of(&display),
        );
        attach(&manager, &display);
        let super_r = (keysym("Super_R") + 8) as u8;
        display.st().root.foreign.insert((super_r, 0));
        check_super(&manager);
        assert_eq!(
            manager.note(),
            Some(format!("Running under Wayland. {SUPER_TAKEN_NOTE}"))
        );
        manager.0.state.lock().super_taken = true;
        manager.0.thread_teardown();
        assert_eq!(manager.note().as_deref(), Some("Running under Wayland."));
    }

    #[test]
    fn test_x11_mapping_change_rereads_the_layout_switches() {
        let (manager, display) = x11();
        follow(&manager);
        let binding = bind_x11(&manager, "p", "ctrl+shift+p", noop());
        assert!(manager.0.grab_binding(&binding));
        display.set_xkb_options(&["grp:ctrl_shift_toggle"]);
        manager.0.handle_event(mapping(1));
        assert!(manager.registered().is_empty());
        let error = manager.last_error("p").expect("error");
        assert!(error.contains("grp:ctrl_shift_toggle"));
    }

    #[test]
    fn test_x11_failed_option_reads_keep_what_was_known() {
        let (manager, display) = x11();
        let atom = display.set_xkb_options(&[ALT_SHIFT]);
        follow(&manager);
        display.st().root.broken_reads = true;
        manager.0.handle_event(property_event(atom));
        // no flip-flop on a failed read
        assert_eq!(xkb_options(&manager), Some(vec![ALT_SHIFT.to_string()]));
    }

    #[test]
    fn test_x11_hotkeys_work_when_the_options_cannot_be_followed() {
        let (manager, display) = x11();
        display.st().broken_atoms = true;
        follow(&manager);
        assert_eq!(manager.0.state.lock().rules_atom, 0);
        assert_eq!(xkb_options(&manager), Some(vec![]));
        let binding = bind_x11(&manager, "t", "ctrl+alt+shift+t", noop());
        assert!(manager.0.grab_binding(&binding));
        manager.0.handle_event(property_event(0)); // nothing followed: ignored, no crash
        assert_eq!(
            manager.registered(),
            HashMap::from([("t".to_string(), hk("ctrl+alt+shift+t"))])
        );
    }

    #[test]
    fn test_x11_layout_conflict_before_the_hotkey_thread_runs() {
        let display = FakeDisplay::new(false);
        display.set_xkb_options(&[ALT_SHIFT]);
        let opened = Arc::new(Mutex::new(Vec::new()));
        let (log, shared) = (opened.clone(), display.clone());
        let factory: DisplayFactory = Box::new(move |name| {
            log.lock().push(name.map(String::from));
            Ok(Box::new(shared.clone()) as Box<dyn XDisplay>)
        });
        let manager = X11HotkeyManager::with_display_factory(Some(":1".to_string()), None, factory);
        assert_eq!(
            manager.layout_conflict("ctrl+alt+shift+c".into()),
            Ok(Some(format!(
                "Ctrl+Alt+Shift+C includes Alt+Shift, {SWITCHES} (XKB option {ALT_SHIFT})"
            )))
        );
        assert_eq!(manager.layout_conflict("ctrl+alt+meta+c".into()), Ok(None));
        // A short-lived, read-only connection per question: nothing grabbed or selected.
        let one = Some(":1".to_string());
        assert_eq!(*opened.lock(), [one.clone(), one]);
        assert!(display.st().closed);
        assert!(display.st().root.grab_calls.is_empty());
        assert_eq!(display.st().root.event_mask, None);
        // A server without the property: nothing to check, no atom created.
        let bare = FakeDisplay::new(false);
        let manager_bare = X11HotkeyManager::with_display_factory(None, None, factory_of(&bare));
        assert_eq!(
            manager_bare.layout_conflict("ctrl+alt+shift+c".into()),
            Ok(None)
        );
        assert!(bare.st().atoms.is_empty());
        // Advisory only: an unreachable server never breaks the caller.
        let unreachable = X11HotkeyManager::with_display_factory(
            None,
            None,
            Box::new(|_| Err("no X server".to_string())),
        );
        assert_eq!(
            unreachable.layout_conflict("ctrl+alt+shift+c".into()),
            Ok(None)
        );
        let err = manager.layout_conflict("ctrl+alt".into()).unwrap_err();
        assert!(err.0.contains("needs a key"));
    }

    #[test]
    fn test_x11_thread_with_fake_display() {
        let display = FakeDisplay::new(true);
        let manager = X11HotkeyManager::with_display_factory(None, None, factory_of(&display));
        let fired = Arc::new(Event::default());
        let threads = Arc::new(Mutex::new(Vec::new()));
        let (event, log) = (fired.clone(), threads.clone());
        let callback: HotkeyCallback = Arc::new(move || {
            log.lock()
                .push(thread::current().name().unwrap_or_default().to_string());
            event.set();
        });
        assert!(manager.register("t", "ctrl+alt+p".into(), callback.clone()));
        let code = keycode_of("p");
        assert!(display.st().root.grabs.contains(&(code, 12)));
        display.push(press(code, 12 | 16, 5));
        assert!(fired.wait(Duration::from_secs(2)));
        assert_eq!(*threads.lock(), ["eye-tracker-hotkeys-x11"]);
        assert!(!manager.register("dup", "ctrl+alt+p".into(), noop()));
        assert!(manager.unregister("t"));
        assert!(display.st().root.grabs.is_empty());
        assert!(manager.register("t", "ctrl+alt+p".into(), callback));
        manager.stop();
        manager.stop();
        assert!(display.st().closed);
        assert!(display.st().root.grabs.is_empty());
        assert!(manager.0.threaded.current_thread().is_none());
    }

    #[test]
    fn test_x11_thread_refuses_hotkeys_a_layout_switch_takes() {
        let display = FakeDisplay::new(true);
        let atom = display.set_xkb_options(&[ALT_SHIFT]);
        let manager = X11HotkeyManager::with_display_factory(None, None, factory_of(&display));
        assert!(!manager.register("t", "ctrl+alt+shift+t".into(), noop()));
        assert!(manager.last_error("t").expect("error").contains(ALT_SHIFT));
        assert!(display.st().root.grabs.is_empty());
        assert!(manager.register("t", "ctrl+alt+meta+t".into(), noop()));
        // cached options
        assert!(
            manager
                .layout_conflict("ctrl+alt+shift+t".into())
                .unwrap()
                .is_some()
        );
        assert_eq!(display.st().root.event_mask, Some(X_PROPERTY_CHANGE_MASK));
        // The switch changes while the app runs: the thread follows it.
        assert!(manager.register("p", "ctrl+shift+p".into(), noop()));
        display.set_xkb_options(&["grp:ctrl_shift_toggle"]);
        display.push(property_event(atom));
        // Each loop pass runs queued calls, then drains events, and it also runs calls queued
        // while it runs them, so round trips alone can all finish before the drain. Wait until
        // the thread took the event; the next call then runs after the pass that handled it.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !display.st().events.is_empty() && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        assert!(call_on_thread(&manager.0, || true, false));
        assert_eq!(
            registered_names(&manager),
            BTreeSet::from(["t".to_string()])
        );
        manager.stop();
        assert!(display.st().closed);
    }

    #[test]
    fn test_x11_unreachable_display() {
        let manager = X11HotkeyManager::with_display_factory(
            None,
            None,
            Box::new(|_| Err("no X server".to_string())),
        );
        assert!(!manager.register("t", "ctrl+alt+p".into(), noop()));
        assert!(manager.note().expect("note").contains("X server"));
        manager.stop();
    }
}

// ====================================================================== Windows live
#[cfg(windows)]
mod windows_live {
    use super::*;

    /// The tests register the same system-wide combinations, which one thread at a time can hold,
    /// so they must not overlap.
    static LIVE_LOCK: Mutex<()> = Mutex::new(());

    fn post_hotkey(manager: &WindowsHotkeyManager, name: &str) {
        // Deliver WM_HOTKEY to the manager's own thread (no input is synthesised).
        let id = manager.0.core.table.lock().bindings[name].id;
        let thread_id = manager.0.thread_id.load(Ordering::SeqCst);
        assert!(win32::post_thread_message(
            thread_id,
            WM_HOTKEY,
            id as usize
        ));
    }

    fn thread_of(manager: &WindowsHotkeyManager) -> Arc<BackendThread> {
        manager.0.threaded.current_thread().expect("thread")
    }

    #[test]
    fn test_windows_factory_returns_native_manager() {
        let manager = create_hotkey_manager();
        assert_eq!(manager.name(), "win32");
        manager.stop();
    }

    #[test]
    fn test_windows_live_register_dispatch_and_stop() {
        let _live = LIVE_LOCK.lock();
        let win = WindowsHotkeyManager::new();
        let fired = Arc::new(Event::default());
        let threads = Arc::new(Mutex::new(Vec::new()));
        let (event, log) = (fired.clone(), threads.clone());
        let callback: HotkeyCallback = Arc::new(move || {
            log.lock()
                .push(thread::current().name().unwrap_or_default().to_string());
            event.set();
        });
        assert!(win.register("t", hk(COMBO).into(), callback));
        assert_eq!(
            win.registered(),
            HashMap::from([("t".to_string(), hk(COMBO))])
        );
        let thread = thread_of(&win);
        assert!(thread.is_alive());
        post_hotkey(&win, "t");
        assert!(fired.wait(Duration::from_secs(2)));
        assert_eq!(*threads.lock(), ["eye-tracker-hotkeys-win32"]);
        win.stop();
        assert!(!thread.is_alive());
        assert!(win.registered().is_empty());
        win.stop(); // idempotent
    }

    #[test]
    fn test_windows_double_registration() {
        let _live = LIVE_LOCK.lock();
        let win = WindowsHotkeyManager::new();
        let fired = Arc::new(Mutex::new(Vec::new()));
        let done = Arc::new(Event::default());
        let (log, event) = (fired.clone(), done.clone());
        let second: HotkeyCallback = Arc::new(move || {
            log.lock().push("second");
            event.set();
        });
        assert!(win.register("a", COMBO.into(), push(&fired, "first")));
        assert!(win.register("a", COMBO.into(), second)); // same name + combo: callback replaced
        assert!(!win.register("b", COMBO.into(), noop())); // combo owned by "a"
        assert_eq!(win.registered().len(), 1);
        post_hotkey(&win, "a");
        assert!(done.wait(Duration::from_secs(2)));
        assert_eq!(*fired.lock(), ["second"]);
    }

    #[test]
    fn test_windows_combo_taken_by_another_owner() {
        let _live = LIVE_LOCK.lock();
        let (first, second) = (WindowsHotkeyManager::new(), WindowsHotkeyManager::new());
        assert!(first.register("t", COMBO.into(), noop()));
        // The OS refuses a combination that another thread/app already registered.
        assert!(!second.register("t", COMBO.into(), noop()));
        assert!(second.registered().is_empty());
        assert_eq!(
            second.last_error("t").as_deref(),
            Some("Ctrl+Alt+Shift+F24 is already in use by another application")
        );
        first.stop();
        assert!(second.register("t", COMBO.into(), noop())); // released by stop()
    }

    #[test]
    fn test_windows_unregister_releases_os_registration() {
        let _live = LIVE_LOCK.lock();
        let win = WindowsHotkeyManager::new();
        let other = WindowsHotkeyManager::new();
        assert!(win.register("t", COMBO.into(), noop()));
        assert!(win.unregister("t"));
        assert!(other.register("t", COMBO.into(), noop()));
        other.unregister_all();
        assert!(win.register("t", COMBO.into(), noop()));
    }

    #[test]
    fn test_windows_callback_may_register_and_stop() {
        let _live = LIVE_LOCK.lock();
        let win = Arc::new(WindowsHotkeyManager::new());
        let results = Arc::new(Mutex::new(Vec::new()));
        let done = Arc::new(Event::default());
        let (w, log, event) = (Arc::downgrade(&win), results.clone(), done.clone());
        let register_more: HotkeyCallback = Arc::new(move || {
            let win = w.upgrade().unwrap();
            log.lock()
                .push(win.register("second", COMBO_2.into(), noop()));
            event.set();
        });
        assert!(win.register("first", COMBO.into(), register_more));
        post_hotkey(&win, "first");
        assert!(done.wait(Duration::from_secs(2)));
        assert_eq!(*results.lock(), [true]);
        assert_eq!(win.registered().len(), 2);

        let stopped = Arc::new(Event::default());
        let (w, event) = (Arc::downgrade(&win), stopped.clone());
        let stop_from_callback: HotkeyCallback = Arc::new(move || {
            w.upgrade().unwrap().stop();
            event.set();
        });
        assert!(win.register("first", COMBO.into(), stop_from_callback));
        let thread = thread_of(&win);
        post_hotkey(&win, "first");
        assert!(stopped.wait(Duration::from_secs(2)));
        thread.join(Duration::from_secs(2));
        assert!(!thread.is_alive());
        assert!(win.registered().is_empty());
        assert!(win.register("again", COMBO.into(), noop())); // restarts cleanly
    }

    #[test]
    fn test_windows_callback_exception_keeps_thread_alive() {
        let _live = LIVE_LOCK.lock();
        let win = WindowsHotkeyManager::new();
        let done = Arc::new(Event::default());
        let event = done.clone();
        assert!(win.register("bad", COMBO.into(), Arc::new(|| panic!("callback failure"))));
        assert!(win.register("good", COMBO_2.into(), Arc::new(move || event.set())));
        post_hotkey(&win, "bad");
        post_hotkey(&win, "good");
        assert!(done.wait(Duration::from_secs(2)));
        assert!(thread_of(&win).is_alive());
    }

    #[test]
    fn test_windows_start_prewarms_thread() {
        let win = WindowsHotkeyManager::new();
        win.start();
        let thread = thread_of(&win);
        assert!(thread.is_alive());
        win.stop();
        assert!(!thread.is_alive());
    }

    #[test]
    fn test_windows_altgr_collision_is_refused_before_registering() {
        // A layout on which even AltGr+Shift+F24 would type: the manager must refuse it without
        // calling RegisterHotKey, so another owner can still take the combo.
        let _live = LIVE_LOCK.lock();
        let refusing = WindowsHotkeyManager::with_layout_probe(Some(Arc::new(
            FakeLayoutProbe::everything("x"),
        )));
        let other = WindowsHotkeyManager::new();
        assert!(!refusing.register("t", COMBO.into(), noop()));
        assert!(refusing.registered().is_empty());
        assert_eq!(
            refusing.last_error("t").as_deref(),
            Some("Ctrl+Alt+Shift+F24 is AltGr+Shift+F24 and types 'x' on the US keyboard layout")
        );
        assert!(other.register("t", COMBO.into(), noop())); // nothing was registered
        // With Win held, AltGr is not involved: no check, registration proceeds.
        assert!(refusing.register("w", "ctrl+alt+meta+f22".into(), noop()));
    }

    #[test]
    fn test_windows_live_layout_queries_are_read_only_and_sane() {
        let probe = Win32LayoutProbe { api: win32::User32 };
        let installed = probe.layouts();
        assert!(!installed.is_empty()); // every session has at least one keyboard layout
        for hkl in &installed {
            assert_eq!(probe.altgr_text(0x87, false, *hkl), None); // VK_F24 never types
            assert!(!probe.layout_name(*hkl).is_empty());
        }
        assert_eq!(win32::layout_text(US).as_deref(), Some("US"));
        // "Layout Id" 0002
        assert_eq!(
            win32::layout_text(0xF0020409_isize).as_deref(),
            Some("United States-Dvorak")
        );
        let manager = WindowsHotkeyManager::new();
        for text in defaults(Platform::Windows) {
            assert_eq!(manager.layout_conflict(text.as_str().into()), Ok(None));
        }
        if installed.contains(&TURKISH_Q) {
            // the maintainer's layout: Ctrl+Alt+T is AltGr+T = '₺'
            let conflict = manager.layout_conflict("ctrl+alt+t".into()).unwrap();
            assert!(conflict.expect("conflict").contains("'₺'"));
        }
    }
}
