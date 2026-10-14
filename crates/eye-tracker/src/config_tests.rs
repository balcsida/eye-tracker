// Ported from tests/test_config.py (pytest parametrization became loops; Python's `linux` and
// `freebsd14` platforms are both `Platform::Other`). The new tests at the end cover the
// forgiving loading rules (`_coerce`, `_apply`), `load`/`save` and `atomic_write_text`, which
// the Python suite only exercised through other test files.

use serde_json::json;

use super::*;
use crate::testing;

const PLATFORMS: [Platform; 3] = [Platform::Windows, Platform::MacOs, Platform::Other];

fn legacy_trio() -> Value {
    json!({
        "toggle_tracking": LEGACY_HOTKEYS[0],
        "toggle_privacy": LEGACY_HOTKEYS[1],
        "recalibrate": LEGACY_HOTKEYS[2],
    })
}

fn legacy_x11_trio() -> Value {
    json!({
        "toggle_tracking": LEGACY_HOTKEYS_X11[0],
        "toggle_privacy": LEGACY_HOTKEYS_X11[1],
        "recalibrate": LEGACY_HOTKEYS_X11[2],
    })
}

/// What a missing, malformed or old "version" key looks like in a settings file.
fn old_versions() -> Vec<Value> {
    vec![
        json!({}),
        json!({"version": 1}),
        json!({"version": 0}),
        json!({"version": "2"}),
        json!({"version": true}),
        json!({"version": null}),
    ]
}

/// `{**a, **b}`
fn merged(a: &Value, b: &Value) -> Value {
    let mut out = a.as_object().cloned().unwrap_or_default();
    out.extend(b.as_object().cloned().unwrap_or_default());
    Value::Object(out)
}

fn trio(settings: &Settings) -> [String; 3] {
    let h = &settings.hotkeys;
    [
        h.toggle_tracking.clone(),
        h.toggle_privacy.clone(),
        h.recalibrate.clone(),
    ]
}

fn default_trio(platform: Platform) -> [String; 3] {
    let d = HotkeySettings::defaults(platform);
    [d.toggle_tracking, d.toggle_privacy, d.recalibrate]
}

fn strs(items: [&str; 3]) -> [String; 3] {
    items.map(String::from)
}

fn row(key: &str) -> SettingRow {
    describe_settings()
        .into_iter()
        .find(|r| r.key == key)
        .expect("setting exists")
}

#[test]
fn test_legacy_backend_names_are_renamed() {
    for (stored, expected) in [("mediapipe", "facemesh"), ("opencv", "lite")] {
        let logs = testing::capture_logs();
        let data = json!({"general": {"backend": stored, "notifications": false}});
        let before = data.clone();
        let settings = Settings::from_dict(&data);
        assert_eq!(settings.general.backend, expected);
        assert!(!settings.general.notifications); // the rest of the section is kept
        assert_eq!(data, before); // the caller's dict is not modified
        assert!(logs.records().iter().any(|(level, msg)| {
            *level == log::Level::Info && msg.contains(stored) && msg.contains(expected)
        }));
        // Nothing is logged as a warning: the old name is valid, just renamed.
        assert!(logs.messages_at_least(log::Level::Warn).is_empty());
    }
}

#[test]
fn test_legacy_backend_names_are_renamed_in_files_of_every_version() {
    for version in [1, CONFIG_VERSION, CONFIG_VERSION + 1] {
        let data = json!({"version": version, "general": {"backend": "opencv"}});
        assert_eq!(Settings::from_dict(&data).general.backend, "lite");
    }
}

#[test]
fn test_current_and_unknown_backend_names_are_not_migrated() {
    let backend = |data: Value| Settings::from_dict(&data).general.backend;
    assert_eq!(backend(json!({"general": {"backend": "lite"}})), "lite");
    assert_eq!(backend(json!({"general": {"backend": "tflite"}})), "auto");
    assert_eq!(backend(json!({"general": {"backend": 3}})), "auto");
    assert_eq!(backend(json!({"general": "broken"})), "auto");
}

#[test]
fn test_the_file_format_version() {
    assert_eq!(CONFIG_VERSION, 2);
    assert_eq!(
        Settings::default().to_dict()["version"],
        json!(CONFIG_VERSION)
    );
    // Loading an old file stamps today's version, so the next save writes it.
    assert_eq!(
        Settings::from_dict(&json!({"version": 1})).version,
        CONFIG_VERSION
    );
    assert_eq!(Settings::from_dict(&json!({})).version, CONFIG_VERSION);
}

#[test]
fn test_every_platform_defaults_to_ctrl_alt_meta() {
    for platform in PLATFORMS {
        assert_eq!(
            default_trio(platform),
            strs(["ctrl+alt+meta+t", "ctrl+alt+meta+p", "ctrl+alt+meta+c"])
        );
    }
}

#[test]
fn test_the_old_default_hotkeys_become_todays_defaults() {
    for version in old_versions() {
        for platform in PLATFORMS {
            let data = merged(
                &version,
                &json!({"hotkeys": merged(&json!({"enabled": false}), &legacy_trio())}),
            );
            let before = data.clone();
            let settings = Settings::from_dict_for(&data, platform);
            assert_eq!(trio(&settings), default_trio(platform));
            assert_eq!(settings.hotkeys.toggle_tracking, "ctrl+alt+meta+t");
            assert!(!settings.hotkeys.enabled); // other hotkey settings are kept
            assert_eq!(data, before);
        }
    }
}

#[test]
fn test_old_defaults_are_recognised_regardless_of_case_and_spaces() {
    let stored: Map<String, Value> = HOTKEY_FIELDS
        .iter()
        .zip(LEGACY_HOTKEYS)
        .map(|(name, combo)| {
            (
                (*name).to_string(),
                json!(format!(" {} ", combo.to_uppercase())),
            )
        })
        .collect();
    let settings = Settings::from_dict_for(&json!({"hotkeys": stored}), Platform::Other);
    assert_eq!(settings.hotkeys.toggle_tracking, "ctrl+alt+meta+t");
}

#[test]
fn test_the_first_x11_defaults_are_migrated() {
    // Ctrl+Alt+Shift+T/P/C was the Linux default of later version 1 builds.
    let logs = testing::capture_logs();
    let data = json!({"version": 1, "hotkeys": legacy_x11_trio()});
    let settings = Settings::from_dict_for(&data, Platform::Other);
    assert_eq!(trio(&settings), default_trio(Platform::Other));
    assert!(
        logs.records()
            .iter()
            .any(|(_, m)| m.contains("ctrl+alt+meta+t"))
    );
    assert!(logs.messages_at_least(log::Level::Warn).is_empty());
}

#[test]
fn test_the_x11_defaults_are_a_choice_elsewhere() {
    // Ctrl+Alt+Shift+T/P/C never was a Windows or macOS default: the user picked it.
    for platform in [Platform::Windows, Platform::MacOs] {
        let data = json!({"version": 1, "hotkeys": legacy_x11_trio()});
        let settings = Settings::from_dict_for(&data, platform);
        assert_eq!(trio(&settings), strs(LEGACY_HOTKEYS_X11));
    }
}

#[test]
fn test_old_defaults_chosen_in_a_current_file_are_kept() {
    // A version 2 file holds what the user chose, even if it equals an old default.
    for platform in PLATFORMS {
        for (stored, expected) in [
            (legacy_trio(), LEGACY_HOTKEYS),
            (legacy_x11_trio(), LEGACY_HOTKEYS_X11),
        ] {
            let logs = testing::capture_logs();
            let data = json!({"version": CONFIG_VERSION, "hotkeys": stored});
            let settings = Settings::from_dict_for(&data, platform);
            assert_eq!(trio(&settings), strs(expected));
            assert!(!logs.records().iter().any(|(_, m)| m.contains("Hotkeys")));
            // A file from a newer version is not second-guessed either.
            let data = json!({"version": CONFIG_VERSION + 1, "hotkeys": stored});
            assert_eq!(
                trio(&Settings::from_dict_for(&data, platform)),
                strs(expected)
            );
        }
    }
}

#[test]
fn test_a_chosen_old_default_survives_restarts() {
    // The user re-records Ctrl+Alt+T/P/C in Settings: it must not be reset at login.
    for platform in PLATFORMS {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("settings.json");
        let mut settings = Settings::defaults(platform);
        settings.hotkeys.toggle_tracking = LEGACY_HOTKEYS[0].into();
        settings.hotkeys.toggle_privacy = LEGACY_HOTKEYS[1].into();
        settings.hotkeys.recalibrate = LEGACY_HOTKEYS[2].into();
        settings.save(&path).unwrap();
        let written: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["version"], json!(CONFIG_VERSION));
        for _restart in 0..3 {
            let reloaded = Settings::load_for(&path, platform);
            assert_eq!(trio(&reloaded), strs(LEGACY_HOTKEYS));
            reloaded.save(&path).unwrap();
        }
    }
}

#[test]
fn test_hotkeys_the_user_changed_are_kept() {
    // Only the complete, untouched legacy trio is migrated.
    for changed in HOTKEY_FIELDS {
        let stored = merged(&legacy_trio(), &json!({ changed: "ctrl+shift+f9" }));
        let data = json!({"version": 1, "hotkeys": stored});
        let hotkeys = Settings::from_dict_for(&data, Platform::Windows).hotkeys;
        assert_eq!(
            hotkeys.toggle_tracking,
            stored["toggle_tracking"].as_str().unwrap()
        );
        assert_eq!(
            hotkeys.toggle_privacy,
            stored["toggle_privacy"].as_str().unwrap()
        );
        assert_eq!(hotkeys.recalibrate, stored["recalibrate"].as_str().unwrap());
    }
}

#[test]
fn test_a_mixed_old_trio_is_kept() {
    // Two defaults of one build and one of the other: somebody chose that.
    let stored = merged(
        &legacy_trio(),
        &json!({"recalibrate": LEGACY_HOTKEYS_X11[2]}),
    );
    let settings =
        Settings::from_dict_for(&json!({"version": 1, "hotkeys": stored}), Platform::Other);
    assert_eq!(
        trio(&settings),
        strs([LEGACY_HOTKEYS[0], LEGACY_HOTKEYS[1], LEGACY_HOTKEYS_X11[2]])
    );
}

#[test]
fn test_a_partial_legacy_trio_is_kept() {
    let stored = json!({"toggle_tracking": "ctrl+alt+t", "toggle_privacy": "ctrl+alt+p"});
    let data = json!({"version": 1, "hotkeys": stored});
    let hotkeys = Settings::from_dict_for(&data, Platform::Windows).hotkeys;
    assert_eq!(hotkeys.toggle_tracking, "ctrl+alt+t");
    assert_eq!(hotkeys.toggle_privacy, "ctrl+alt+p");
    assert_eq!(hotkeys.recalibrate, HotkeySettings::default().recalibrate); // missing: the default
}

#[test]
fn test_malformed_hotkeys_in_an_old_file_are_not_migrated() {
    let data = json!({"version": 1, "hotkeys": "ctrl+alt+t"});
    assert_eq!(
        Settings::from_dict_for(&data, Platform::Other).hotkeys,
        HotkeySettings::default()
    );
    let stored = merged(&legacy_trio(), &json!({"recalibrate": 3}));
    let data = json!({"version": 1, "hotkeys": stored});
    let hotkeys = Settings::from_dict_for(&data, Platform::Other).hotkeys;
    assert_eq!(hotkeys.toggle_tracking, "ctrl+alt+t"); // not a complete trio: left alone
    assert_eq!(hotkeys.recalibrate, HotkeySettings::default().recalibrate); // wrong type: the default
}

#[test]
fn test_a_migrated_file_round_trips() {
    for platform in PLATFORMS {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("settings.json");
        let data =
            json!({"version": 1, "general": {"backend": "opencv"}, "hotkeys": legacy_trio()});
        fs::write(&path, data.to_string()).unwrap();
        let migrated = Settings::load_for(&path, platform);
        assert_eq!(migrated.general.backend, "lite");
        assert_eq!(migrated.hotkeys, HotkeySettings::defaults(platform));
        migrated.save(&path).unwrap();
        let written: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["version"], json!(CONFIG_VERSION));
        let reloaded = Settings::load_for(&path, platform);
        assert_eq!(reloaded.general.backend, "lite");
        assert_eq!(reloaded.hotkeys, HotkeySettings::defaults(platform));
    }
}

#[test]
fn test_an_old_file_that_is_never_saved_migrates_on_every_load() {
    // Until something is saved the file still says version 1 and still holds the old defaults,
    // so every load reaches the same result: the migration is stable.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("settings.json");
    fs::write(
        &path,
        json!({"version": 1, "hotkeys": legacy_trio()}).to_string(),
    )
    .unwrap();
    let first = Settings::load_for(&path, Platform::MacOs);
    let second = Settings::load_for(&path, Platform::MacOs);
    assert_eq!(first.hotkeys, HotkeySettings::defaults(Platform::MacOs));
    assert_eq!(first.hotkeys, second.hotkeys);
    assert_eq!(
        trio(&first),
        strs(["ctrl+alt+meta+t", "ctrl+alt+meta+p", "ctrl+alt+meta+c"])
    );
}

#[test]
fn test_camera_device_documentation_mentions_stable_links_and_offline() {
    let doc = row("camera.device").doc;
    assert!(doc.contains("/dev/v4l/by-id"));
    assert!(doc.contains("URLs"));
    assert!(doc.contains("offline"));
}

#[test]
fn test_split_panes_are_experimental_and_off_by_default() {
    let panes = Settings::default().panes;
    assert!(!panes.enabled);
    assert_eq!(
        (
            panes.dwell_ms,
            panes.precision,
            panes.hysteresis,
            panes.min_pane_px
        ),
        (400, 2.5, 0.5, 240)
    );
    assert!(panes.tmux && panes.wezterm && panes.windows_terminal);
    assert!(row("panes.enabled").doc.starts_with("Experimental"));
}

#[test]
fn test_split_pane_values_are_validated_like_the_others() {
    let data = json!({"panes": {"enabled": "yes", "precision": 99, "dwell_ms": 10, "tmux": false}});
    let settings = Settings::from_dict(&data);
    assert!(!settings.panes.enabled); // wrong type: the default
    assert_eq!(settings.panes.precision, 6.0); // clamped
    assert_eq!(settings.panes.dwell_ms, 100);
    assert!(!settings.panes.tmux);
    // Files written before the section existed get the defaults.
    let older = Settings::from_dict(&json!({"switching": {"enabled": false}}));
    assert_eq!(older.panes, Settings::default().panes);
}

// ------------------------------------------------------------ new tests (no Python original)

#[test]
fn test_values_are_coerced_like_the_python_loader() {
    let data = json!({
        "camera": {"width": 99999, "height": 479.5, "device": 3, "api": "nope"},
        "switching": {"dwell_ms": true, "smoothing": -1, "hysteresis": "x", "cursor_target": "gaze"},
        "privacy": {"pause_for_apps": ["a.exe", "  ", "b"]},
        "unknown": 1,
        "ui": [],
    });
    let s = Settings::from_dict(&data);
    assert_eq!(s.camera.width, 1920); // clamped
    assert_eq!(s.camera.height, 480); // rounded (half to even)
    assert_eq!(s.camera.device, "0"); // not a string
    assert_eq!(s.camera.api, "auto"); // not a choice
    assert_eq!(s.switching.dwell_ms, 300); // a bool is no number
    assert_eq!(s.switching.smoothing, 0.0);
    assert_eq!(s.switching.hysteresis, 0.06);
    assert_eq!(s.switching.cursor_target, "gaze");
    assert_eq!(s.privacy.pause_for_apps, ["a.exe", "b"]); // blanks dropped
    assert_eq!(s.ui, UISettings::default());
    let bad_list = Settings::from_dict(&json!({"privacy": {"pause_for_apps": ["a", 1]}}));
    assert!(bad_list.privacy.pause_for_apps.is_empty());
    assert_eq!(Settings::from_dict(&json!([1])), Settings::default());
}

#[test]
fn test_load_and_save() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sub").join("settings.json");
    assert_eq!(Settings::load(&path), Settings::default()); // missing file
    let mut settings = Settings::default();
    settings.switching.dwell_ms = 123;
    settings.save(&path).unwrap();
    let text = fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("{\n  \"version\": 2,\n  \"general\": {\n    \"backend\": \"auto\""));
    assert!(text.ends_with("}\n"));
    assert_eq!(Settings::load(&path).switching.dwell_ms, 123);

    // Invalid JSON is moved aside and the defaults are used.
    fs::write(&path, "{ nope").unwrap();
    assert_eq!(Settings::load(&path), Settings::default());
    assert!(!path.exists());
    assert_eq!(
        fs::read_to_string(tmp.path().join("sub").join("settings.json.corrupt")).unwrap(),
        "{ nope"
    );
}

#[test]
fn test_atomic_write_text_replaces_the_file_and_leaves_no_temp_file() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("a").join("f.txt");
    atomic_write_text(&path, "one").unwrap();
    atomic_write_text(&path, "two").unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap(), "two");
    assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
}

#[test]
fn test_describe_settings_lists_every_setting_in_file_order() {
    let rows = describe_settings();
    assert_eq!(rows[0].key, "general.backend");
    assert_eq!(rows[0].kind, "str");
    assert_eq!(rows[0].choices, ["auto", "facemesh", "lite", "apple"]);
    let width = row("camera.width");
    assert_eq!(
        (width.kind, width.lo, width.hi, width.default.clone()),
        ("int", Some(160.0), Some(1920.0), json!(640))
    );
    assert_eq!(row("privacy.pause_for_apps").kind, "list[str]");
    assert_eq!(row("switching.hysteresis").default, json!(0.06));
    assert_eq!(
        rows.last().map(|r| r.key.as_str()),
        Some("ui.show_gaze_overlay")
    );
    assert_eq!(rows.len(), 59);
    // Every row's key is a key of the serialized settings.
    let dict = Settings::default().to_dict();
    for r in &rows {
        assert_eq!(dict[r.section][r.field], r.default, "{}", r.key);
    }
}
