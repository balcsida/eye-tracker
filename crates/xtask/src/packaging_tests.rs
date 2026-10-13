//! Tests of the packaging: the Windows installer script, the Linux and macOS scripts, the
//! release notes script and the workflows. Port of the applicable tests of
//! `tests/test_build_scripts.py`.
//!
//! Not ported, because what they tested no longer exists:
//!
//! * the PyInstaller spec (its names, models, licences, `MACOS_MINIMUM` from the wheels, the
//!   pruning of Qt plugins, OpenSSL and unused macOS libraries, the Mach-O binding reader, the
//!   dangling-symlink pass) and `packaging/pyinstaller/entry.py` (its entry-point choice is
//!   `cli_command` / `app_command` in `crates/eye-tracker/src/cli.rs`, tested there);
//! * the Python bundle: `scripts/smoke_test_bundle.py` against a fake CLI (see
//!   `smoke_test_bundle_tests.rs`), `pyproject.toml`, the sdist, `uv.lock`, the Python
//!   model-pin and icon scripts (tested in their own xtask modules);
//! * the checks that the Linux build bundles libxcb-cursor and the X11 libraries: the Rust
//!   executable links the system's libraries at run time.
//!
//! Tests of the text of the installer script and of the workflows pin the contract between
//! them (names, task ordering, pins); the behaviour itself runs on the CI machines.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use regex::Regex;

use crate::repo_root;

fn root() -> PathBuf {
    repo_root()
}

fn read(relative: &str) -> String {
    let path = root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The version in `[workspace.package]` of `Cargo.toml`.
fn cargo_version() -> String {
    let re = Regex::new(r#"(?m)^version = "([^"]+)"$"#).unwrap();
    re.captures(&read("Cargo.toml")).unwrap()[1].to_string()
}

// ======================================================================== the installer
fn iss() -> String {
    let bytes = fs::read(root().join("packaging/windows/installer.iss")).unwrap();
    assert!(
        bytes.starts_with(b"\xef\xbb\xbf"),
        "installer.iss must keep its UTF-8 BOM"
    );
    String::from_utf8(bytes[3..].to_vec())
        .unwrap()
        .replace("\r\n", "\n")
}

fn iss_define(name: &str) -> String {
    let re = Regex::new(&format!(r#"(?m)^#define {name} "([^"]*)""#)).unwrap();
    re.captures(&iss())
        .unwrap_or_else(|| panic!("#define {name}"))[1]
        .to_string()
}

fn iss_code() -> String {
    let text = iss();
    text[text.find("[Code]").unwrap()..].to_string()
}

/// The non-comment lines of one section of installer.iss.
fn iss_section(name: &str) -> Vec<String> {
    let text = iss();
    let start = text.find(&format!("\n[{name}]\n")).unwrap() + name.len() + 4;
    let body = &text[start..];
    let body = &body[..body.find("\n[").unwrap()];
    body.lines()
        .filter(|line| !line.is_empty() && !line.starts_with(';'))
        .map(String::from)
        .collect()
}

/// `text[from..to]` for two markers.
fn between<'a>(text: &'a str, from: &str, to: &str) -> &'a str {
    let start = text
        .find(from)
        .unwrap_or_else(|| panic!("missing {from:?}"));
    let end = start
        + text[start..]
            .find(to)
            .unwrap_or_else(|| panic!("missing {to:?}"));
    &text[start..end]
}

#[test]
fn test_installer_matches_the_app() {
    use eye_tracker::platform::autostart;
    assert_eq!(iss_define("AppName"), eye_tracker::APP_NAME);
    assert_eq!(iss_define("AppExeName"), "EyeTracker.exe");
    assert_eq!(iss_define("CliExeName"), "eye-tracker-cli.exe");
    assert_eq!(iss_define("AppUserModelID"), eye_tracker::APP_ID);
    assert_eq!(iss_define("RunValueName"), autostart::RUN_VALUE_NAME);
    assert_eq!(iss_define("RunKey"), autostart::RUN_KEY);
    assert_eq!(
        iss_define("StartupApprovedKey"),
        autostart::STARTUP_APPROVED_KEY
    );
    let text = iss();
    assert!(text.contains("PrivilegesRequired=lowest"));
    assert!(text.contains(r"DefaultDirName={userpf}\{#AppName}"));
    assert!(text.contains("OutputBaseFilename=EyeTracker-{#AppVersion}-windows-x64-setup"));
    // The Run value is exactly what autostart writes for a background start.
    assert!(text.contains(r#"ValueData: """{app}\{#AppExeName}"" --background""#));
    assert_eq!(autostart::BACKGROUND_FLAG, "--background");
}

#[test]
fn test_installer_ships_the_models_notice_and_the_icon() {
    let text = iss();
    assert_eq!(iss_define("IconFileName"), "eye-tracker.ico");
    assert!(text.contains(r#"UninstallDisplayIcon={app}\{#IconFileName}"#));
    assert!(text.contains(r#"#if !FileExists(AddBackslash(BundleDir) + "NOTICE.md")"#));
    assert!(iss_section("Files")[0].starts_with(r#"Source: "{#BundleDir}\*"; DestDir: "{app}""#));
    assert!(root().join("packaging/windows/eye-tracker.ico").is_file());
    assert!(root().join("crates/eye-tracker/models/NOTICE.md").is_file());
    // Python-era builds kept their libraries in _internal; an upgrade removes it.
    assert!(
        iss_section("InstallDelete")
            .contains(&r#"Type: filesandordirs; Name: "{app}\_internal""#.to_string())
    );
    // Nothing of the Python builds is left in the script.
    for word in ["Python", "PyInstaller", "pyinstaller"] {
        assert!(!text.contains(word), "{word}");
    }
}

#[test]
fn test_installer_app_id_never_changes() {
    // Upgrades and the upgrade detection in [Code] both depend on it.
    assert_eq!(
        iss_define("AppGuid"),
        "F2F98F7C-0EEF-44D5-ADDE-173D9D8BA82B"
    );
    let text = iss();
    assert!(text.contains("AppId={{{#AppGuid}}"));
    assert!(text.contains(
        r#"#define UninstallKey "Software\Microsoft\Windows\CurrentVersion\Uninstall\{""#
    ));
    assert!(text.contains(r#"+ AppGuid + "}_is1""#));
}

#[test]
fn test_installer_takes_the_version_from_the_build() {
    // The script has no version of its own: the build passes the one read from Cargo.toml.
    assert!(iss().contains("#error Pass the version"));
    let build = read(".github/workflows/build.yml");
    assert!(build.contains(
        r#"& $iscc "/DAppVersion=$env:VERSION" "/DAppVersionNumeric=$env:VERSION_NUMERIC" packaging\windows\installer.iss"#
    ));
    assert!(
        build.contains(
            r#"version="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)""#
        )
    );
}

#[test]
fn test_installer_upgrades_keep_the_users_autostart_choice() {
    let text = iss();
    let registry: Vec<_> = text
        .lines()
        .filter(|line| line.starts_with("Root: HKCU") && line.contains("startup"))
        .collect();
    assert_eq!(registry.len(), 2);
    for line in registry {
        assert!(
            line.ends_with("Tasks: startup; Check: ShouldWriteAutostart"),
            "{line}"
        );
    }
    let code = iss_code();
    for fragment in [
        "function ShouldWriteAutostart: Boolean;",
        "else if AutostartWasOn then\n    Result := False\n  else\n    Result := StartupChoiceIsExplicit;",
        "procedure CurPageChanged(CurPageID: Integer);",
        "WizardSelectTasks('!startup')",
        "IsUpgrade := RegKeyExists(HKCU, '{#UninstallKey}');",
        "AutostartWasOn := IsUpgrade and AutostartEnabledFor(ExpandConstant('{app}'))",
        // Task Manager's "disabled" flag: an odd first byte in StartupApproved.
        "((Ord(Approved[1]) and 1) = 1)",
    ] {
        assert!(code.contains(fragment), "{fragment}");
    }
    // The state is captured before the [Registry] section runs (ssInstall).
    assert!(code.find("if CurStep = ssInstall then") < code.find("CurStep = ssPostInstall"));
}

#[test]
fn test_installer_honours_an_explicit_startup_choice_on_upgrades() {
    let code = iss_code();
    let page = between(
        &code,
        "procedure CurPageChanged",
        "function StartupChoiceIs",
    );
    assert!(page.contains(
        "if (CurPageID = wpSelectTasks) and IsUpgrade and not StartupPageSynced and\n     (StartupTaskParam = -1) then"
    ));
    assert!(code.contains(
        "function StartupChoiceIsExplicit: Boolean;\nbegin\n  Result := (StartupTaskParam <> -1) or (StartupPageSynced and not WizardSilent);\nend;"
    ));
    assert!(code.contains(
        "if AutostartWasOn and not WizardIsTaskSelected('startup') and StartupChoiceIsExplicit then"
    ));
    // The command line is read the way Setup reads it: /TASKS deselects what it does not
    // list, /MERGETASKS changes only what it names.
    let task_param = between(&code, "function StartupTaskParam", "procedure CurPage");
    for fragment in [
        "if Pos('/tasks=', Param) = 1 then",
        "else if Pos('/mergetasks=', Param) = 1 then",
        "if (Task = 'startup') or (Task = '*startup') then\n        Result := 1",
        "else if Task = '!startup' then\n        Result := 0;",
    ] {
        assert!(task_param.contains(fragment), "{fragment}");
    }
}

#[test]
fn test_installer_puts_the_command_on_path() {
    assert_eq!(iss_define("CliCommandExeName"), "eye-tracker.exe");
    let tasks = iss_section("Tasks");
    let task = tasks
        .iter()
        .find(|l| l.starts_with(r#"Name: "addtopath""#))
        .unwrap();
    assert!(!task.contains("unchecked"), "on by default");
    assert!(task.contains(r#"Description: "{cm:AddToPath}""#));
    assert!(iss_section("Files").contains(
        &r#"Source: "{#BundleDir}\{#CliExeName}"; DestDir: "{app}"; DestName: "{#CliCommandExeName}"; Flags: ignoreversion"#
            .to_string()
    ));
    assert!(iss_section("Setup").contains(&"ChangesEnvironment=yes".to_string()));
    let text = iss();
    for language in ["english", "turkish"] {
        for message in ["AddToPath", "CommandLineGroup"] {
            assert!(
                text.contains(&format!("{language}.{message}=")),
                "{language}.{message}"
            );
        }
    }
    let code = iss_code();
    for fragment in [
        // Added (or, when unticked in an upgrade, removed) after the files are in place.
        "if WizardIsTaskSelected('addtopath') then\n      AddAppToPath\n    else\n      RemoveAppFromPath;",
        // Removed again on uninstall.
        "QuitRunningApp;\n    RemoveAutostartIfOurs;\n    RemoveAppFromPath;",
        // Unexpanded in and out, so %VARIABLES% in the user's PATH survive.
        "RegQueryStringValue(HKCU, EnvironmentKey, 'Path', PathList)",
        "RegWriteExpandStringValue(HKCU, EnvironmentKey, 'Path', PathList + Folder)",
        "RegWriteExpandStringValue(HKCU, EnvironmentKey, 'Path', Kept)",
        "EnvironmentKey = 'Environment';",
    ] {
        assert!(code.contains(fragment), "{fragment}");
    }
    let add = between(
        &code,
        "procedure AddAppToPath;",
        "procedure RemoveAppFromPath;",
    );
    assert!(
        add.contains("if Found then\n    Exit;"),
        "never added twice"
    );
    // The Run value still names the windowed executable.
    let registry = iss_section("Registry");
    let run_value = registry
        .iter()
        .find(|l| l.contains("ValueType: string"))
        .unwrap();
    assert!(run_value.contains(r#"ValueData: """{app}\{#AppExeName}"" --background""#));
    assert!(!run_value.contains("CliCommandExeName"));
}

#[test]
fn test_installer_restarts_the_app_after_a_silent_upgrade() {
    assert_eq!(
        iss_section("Run"),
        [
            r#"Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent"#,
            r#"Filename: "{app}\{#AppExeName}"; Parameters: "--background"; Flags: nowait runasoriginaluser; Check: RelaunchAfterSilentUpgrade"#,
        ]
    );
    let code = iss_code();
    assert!(code.contains("function RelaunchAfterSilentUpgrade: Boolean;"));
    assert!(code.contains("Result := WizardSilent and AppWasRunning;"));
    // Only an instance that acknowledged "ctl quit" counts as running.
    let quit_app = between(
        &code,
        "procedure QuitRunningApp;",
        "function PrepareToInstall",
    );
    assert!(quit_app.contains("(ResultCode <> 0) then\n    Exit;\n  AppWasRunning := True;"));
}

#[test]
fn test_installer_never_replaces_a_path_it_cannot_read() {
    let code = iss_code();
    let add = between(
        &code,
        "procedure AddAppToPath;",
        "{ Remove the installation",
    );
    let read = add
        .find("if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', PathList) then")
        .unwrap();
    let guard = add
        .find("if RegValueExists(HKCU, EnvironmentKey, 'Path') then")
        .unwrap();
    let exit = guard + add[guard..].find("Exit;").unwrap();
    let empty = add.find("PathList := '';").unwrap();
    assert!(read < guard && guard < exit && exit < empty);
    assert!(empty < add.find("RegWriteExpandStringValue").unwrap());
    let remove = between(
        &code,
        "procedure RemoveAppFromPath;",
        "{ Remove the \"start",
    );
    let read_or_exit =
        "if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', PathList) then\n    Exit;";
    assert!(remove.find(read_or_exit).unwrap() < remove.find("RegDeleteValue").unwrap());
}

#[test]
fn test_update_flags_are_accepted_by_the_installer_smoke_test() {
    // crates/eye-tracker/src/update/installer.rs runs the setup program with these; the
    // installer smoke test in build.yml runs the very same ones for its upgrades.
    let build = read(".github/workflows/build.yml");
    let flags = eye_tracker::update::installer::SETUP_ARGUMENTS.join(r#"", ""#);
    assert!(
        build.contains(&format!(r#"@("{flags}", "/LOG=$log")"#)),
        "{flags}"
    );
}

// ======================================================================== Linux
#[test]
fn test_linux_desktop_entry() {
    let text = read("packaging/linux/eye-tracker.desktop");
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("[Desktop Entry]"));
    let entries: std::collections::HashMap<_, _> =
        lines.filter_map(|line| line.split_once('=')).collect();
    assert_eq!(entries["Type"], "Application");
    assert_eq!(entries["Name"], eye_tracker::APP_NAME);
    assert_eq!(entries["Exec"], "eye-tracker");
    assert_eq!(entries["Icon"], "eye-tracker");
    assert!(entries["Categories"].ends_with(';'));
}

#[test]
fn test_apprun_starts_the_one_executable_and_keeps_the_library_path() {
    let text = read("packaging/linux/AppRun");
    assert!(text.starts_with("#!/bin/sh\n"));
    // The variable platform/linux.rs restores (or clears) for the tools it runs.
    assert!(text.contains(r#"export LD_LIBRARY_PATH_ORIG="${LD_LIBRARY_PATH:-}""#));
    assert!(read("crates/eye-tracker/src/platform/linux.rs").contains("\"LD_LIBRARY_PATH_ORIG\""));
    assert!(text.contains(r#"exec "$HERE/usr/bin/eye-tracker" "$@""#));
}

#[test]
fn test_appimage_script_packs_one_executable_with_the_models_notice() {
    let text = read("packaging/linux/build_appimage.sh");
    for fragment in [
        r#"install -m 755 "$BINARY" "$APPDIR/usr/bin/eye-tracker""#,
        r#"install -m 755 "$BINARY" "$WORK/tar/eye-tracker/eye-tracker""#,
        r#"install -m 644 "$MODELS/NOTICE.md" "$WORK/tar/eye-tracker/NOTICE.md""#,
        r#"cp -R "$MODELS/licenses" "$WORK/tar/eye-tracker/licenses""#,
        r#"BASENAME="EyeTracker-$VERSION-linux-$ARCH""#,
    ] {
        assert!(text.contains(fragment), "{fragment}");
    }
    assert!(!text.contains("PyInstaller") && !text.contains("_internal"));
}

#[test]
fn test_appimage_tools_are_pinned() {
    let text = read("packaging/linux/build_appimage.sh");
    assert!(
        !text
            .replace("\"continuous\" channel", "")
            .contains("continuous")
    );
    assert!(
        Regex::new(r#"(?m)^APPIMAGETOOL_VERSION="\d+\.\d+\.\d+""#)
            .unwrap()
            .is_match(&text)
    );
    assert!(
        Regex::new(r#"(?m)^APPIMAGE_RUNTIME_VERSION="\d{8}""#)
            .unwrap()
            .is_match(&text)
    );
    for tool in ["appimagetool", "runtime"] {
        for arch in ["x86_64", "aarch64"] {
            let re = Regex::new(&format!(
                r#"(?m)^\s+{tool}:{arch}\) echo "([0-9a-f]{{64}})" ;;$"#
            ))
            .unwrap();
            assert!(re.is_match(&text), "{tool} {arch}");
        }
    }
    assert!(text.contains(r#"TOOL_ARGS=(--no-appstream --runtime-file "$RUNTIME_FILE")"#));
    assert!(text.contains("--allow-unpinned"));
    assert!(text.contains("refusing to download"));
}

fn bash(script: &Path, args: &[&str], dir: &Path, envs: &[(&str, &str)]) -> std::process::Output {
    Command::new("bash")
        .arg(script)
        .args(args)
        .current_dir(dir)
        .envs(envs.iter().copied())
        .output()
        .unwrap()
}

#[cfg(unix)]
fn executable(path: &Path, content: &str) {
    use std::os::unix::fs::PermissionsExt;
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

/// A copy of the repository's packaging inputs: scripts, desktop file, icon, models' notice.
#[cfg(unix)]
fn fake_repository(dir: &Path) {
    for name in [
        "build_appimage.sh",
        "AppRun",
        "eye-tracker.desktop",
        "eye-tracker.png",
    ] {
        let to = dir.join("packaging/linux").join(name);
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::copy(root().join("packaging/linux").join(name), to).unwrap();
    }
    fs::write(
        dir.join("Cargo.toml"),
        "[workspace.package]\nversion = \"9.9.9\"\n",
    )
    .unwrap();
    fs::write(dir.join("LICENSE"), "MIT\n").unwrap();
    let models = dir.join("crates/eye-tracker/models");
    fs::create_dir_all(models.join("licenses")).unwrap();
    fs::write(models.join("NOTICE.md"), "notice\n").unwrap();
    fs::write(models.join("licenses/LICENSE-X.txt"), "x\n").unwrap();
    executable(&dir.join("target/release/eye-tracker"), "#!/bin/sh\n");
}

const FAKE_CURL: &str = r#"#!/bin/sh
out=""; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    --output) out="$2"; shift 2 ;;
    --retry) shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
echo "$url" >> "$FAKE_CURL_LOG"
case "$url" in
  *appimagetool*) cat "$FAKE_TOOL" > "$out" ;;
  *) printf 'runtime' > "$out" ;;
esac
"#;

const FAKE_TOOL: &str = r#"#!/bin/sh
echo "$@" > "$FAKE_TOOL_LOG"
for last; do :; done
: > "$last"
"#;

#[cfg(unix)]
#[test]
fn test_build_appimage_verifies_its_downloads() {
    use sha2::{Digest, Sha256};
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    fake_repository(&repo);
    let bin = dir.path().join("bin");
    executable(&bin.join("curl"), FAKE_CURL);
    // The script refuses other systems; the downloads are checked before it needs GNU tar.
    executable(
        &bin.join("uname"),
        "#!/bin/sh\n[ \"$1\" = -s ] && echo Linux || echo x86_64\n",
    );
    if !cfg!(target_os = "linux") {
        // macOS has no GNU sha256sum, which the script verifies its downloads with.
        executable(
            &bin.join("sha256sum"),
            "#!/bin/sh\nexec shasum -a 256 \"$@\"\n",
        );
    }
    let tool = dir.path().join("tool");
    executable(&tool, FAKE_TOOL);
    let tool_log = dir.path().join("tool.log");
    let curl_log = dir.path().join("curl.log");
    let path = format!("{}:/usr/bin:/bin:/usr/sbin:/sbin", bin.display());
    let script = repo.join("packaging/linux/build_appimage.sh");
    let run = |extra: &[(&str, &str)]| {
        let mut envs = vec![
            ("PATH", path.as_str()),
            ("ARCH", "x86_64"),
            ("FAKE_TOOL", tool.to_str().unwrap()),
            ("FAKE_TOOL_LOG", tool_log.to_str().unwrap()),
            ("FAKE_CURL_LOG", curl_log.to_str().unwrap()),
        ];
        envs.extend_from_slice(extra);
        bash(&script, &["--version", "9.9.9"], &repo, &envs)
    };

    // The real pinned checksums do not match the fake downloads: refused, nothing kept.
    let failed = run(&[]);
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("checksum mismatch"));
    let kept = fs::read_dir(repo.join("build/tools")).map_or(0, |d| d.count());
    assert_eq!(kept, 0);

    let tool_sha = hex::encode(Sha256::digest(fs::read(&tool).unwrap()));
    let runtime_sha = hex::encode(Sha256::digest(b"runtime"));
    let done = run(&[
        ("APPIMAGETOOL_SHA256", &tool_sha),
        ("APPIMAGE_RUNTIME_SHA256", &runtime_sha),
    ]);
    let downloads = fs::read_to_string(&curl_log).unwrap();
    let urls: Vec<_> = downloads.split_whitespace().collect();
    assert_eq!(
        urls[urls.len() - 2..],
        [
            "https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage",
            "https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64",
        ]
    );
    if cfg!(target_os = "linux") {
        assert!(
            done.status.success(),
            "{}",
            String::from_utf8_lossy(&done.stderr)
        );
        let arguments = fs::read_to_string(&tool_log).unwrap();
        let arguments: Vec<_> = arguments.split_whitespace().collect();
        let runtime = repo.join("build/tools/runtime-20251108-x86_64");
        let at = arguments
            .iter()
            .position(|a| *a == "--runtime-file")
            .unwrap();
        assert_eq!(arguments[at + 1], runtime.to_str().unwrap());
        assert!(
            repo.join("dist/EyeTracker-9.9.9-linux-x86_64.tar.gz")
                .is_file()
        );
    }

    // A custom download without a checksum is refused.
    let unpinned = run(&[("APPIMAGETOOL_URL", "https://example.invalid/tool.AppImage")]);
    assert!(!unpinned.status.success());
    assert!(String::from_utf8_lossy(&unpinned.stderr).contains("refusing to download"));
}

// ======================================================================== macOS
#[test]
fn test_make_dmg_can_sign_with_a_stable_identity() {
    let text = read("packaging/macos/make_dmg.sh");
    assert!(text.contains(r#"IDENTITY="${MACOS_SIGN_IDENTITY:--}""#));
    assert!(text.contains(r#"IDENTITY="${2:?--identity needs a name"#));
    assert!(text.contains(r#"SIGN_ARGS=(--force --deep --sign "$IDENTITY" --timestamp=none)"#));
    // A certificate-based signature must not leave a build-specific requirement.
    assert!(text.contains(r#""$REQUIREMENT" == *cdhash*"#));
}

#[test]
fn test_make_dmg_refuses_dangling_symlinks_before_signing() {
    let text = read("packaging/macos/make_dmg.sh");
    let check = text
        .find(r#"if ! BROKEN="$(find -L "$APP" -type l)"; then"#)
        .unwrap();
    let xattr = text.find(r#"xattr -cr "$APP""#).unwrap();
    let sign = text.find(r#"codesign "${SIGN_ARGS[@]}" "$APP""#).unwrap();
    assert!(check < xattr && xattr < sign);
    assert!(
        text.contains("the app bundle is incomplete; rebuild it with packaging/macos/make_app.sh")
    );
}

#[test]
fn test_make_dmg_builds_the_app_when_it_is_missing() {
    let text = read("packaging/macos/make_dmg.sh");
    assert!(text.contains(r#"bash "$ROOT/packaging/macos/make_app.sh""#));
    assert!(!text.to_lowercase().contains("pyinstaller"));
}

#[cfg(target_os = "macos")]
#[test]
fn test_make_dmg_names_the_dangling_symlinks() {
    let dir = tempfile::tempdir().unwrap();
    let app = dir.path().join("Eye Tracker.app");
    let frameworks = app.join("Contents/Frameworks");
    fs::create_dir_all(frameworks.join("cv2")).unwrap();
    fs::write(app.join("Contents/Info.plist"), "<plist/>\n").unwrap();
    fs::write(frameworks.join("cv2/libavformat.61.dylib"), "x").unwrap();
    std::os::unix::fs::symlink(
        "cv2/libavformat.61.dylib",
        frameworks.join("libavformat.61.dylib"),
    )
    .unwrap();
    std::os::unix::fs::symlink("cv2/libX11.6.dylib", frameworks.join("libX11.6.dylib")).unwrap();
    let out = dir.path().join("out.dmg");
    let output = bash(
        &root().join("packaging/macos/make_dmg.sh"),
        &[
            "--app",
            app.to_str().unwrap(),
            "--version",
            "9.9.9",
            "--out",
            out.to_str().unwrap(),
        ],
        dir.path(),
        &[],
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("Contents/Frameworks/libX11.6.dylib -> cv2/libX11.6.dylib"));
    assert!(!stderr.contains("libavformat"));
    assert!(stderr.contains("the app bundle is incomplete"));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("Signing"));
    assert!(!out.exists());
}

#[cfg(target_os = "macos")]
fn plist_value(plist: &Path, key: &str) -> String {
    let output = Command::new("plutil")
        .args(["-extract", key, "raw"])
        .arg(plist)
        .output()
        .unwrap();
    assert!(output.status.success(), "{key}");
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}

#[cfg(target_os = "macos")]
#[test]
fn test_make_app_assembles_the_bundle() {
    let dir = tempfile::tempdir().unwrap();
    let bin = dir.path().join("release");
    executable(&bin.join("eye-tracker-gui"), "#!/bin/sh\n");
    executable(&bin.join("eye-tracker"), "#!/bin/sh\n");
    let app = dir.path().join("out/Eye Tracker.app");
    let output = bash(
        &root().join("packaging/macos/make_app.sh"),
        &[
            "--bin-dir",
            bin.to_str().unwrap(),
            "--out",
            app.to_str().unwrap(),
        ],
        dir.path(),
        &[],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // The names cli_command and app_command look for.
    assert!(app.join("Contents/MacOS/Eye Tracker").is_file());
    assert!(app.join("Contents/MacOS/eye-tracker-cli").is_file());
    assert!(app.join("Contents/Resources/eye-tracker.icns").is_file());
    assert!(app.join("Contents/Resources/NOTICE.md").is_file());
    assert!(app.join("Contents/Resources/licenses").is_dir());
    assert!(app.join("Contents/Resources/LICENSE").is_file());
    let plist = app.join("Contents/Info.plist");
    assert_eq!(
        plist_value(&plist, "CFBundleIdentifier"),
        eye_tracker::APP_ID
    );
    assert_eq!(
        plist_value(&plist, "CFBundleShortVersionString"),
        cargo_version()
    );
    assert_eq!(plist_value(&plist, "CFBundleExecutable"), "Eye Tracker");
    assert_eq!(plist_value(&plist, "LSUIElement"), "true");
    assert_eq!(plist_value(&plist, "LSMinimumSystemVersion"), "14.0");
    assert!(plist_value(&plist, "NSCameraUsageDescription").contains("never saved or sent"));
    assert_eq!(plist_value(&plist, "NSHighResolutionCapable"), "true");
}

#[test]
fn test_make_app_refuses_what_is_missing() {
    let dir = tempfile::tempdir().unwrap();
    let output = bash(
        &root().join("packaging/macos/make_app.sh"),
        &[
            "--bin-dir",
            dir.path().to_str().unwrap(),
            "--out",
            dir.path().join("a.app").to_str().unwrap(),
        ],
        dir.path(),
        &[],
    );
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing"));
}

// ======================================================================== shell scripts
const SCRIPTS: [&str; 5] = [
    "packaging/linux/AppRun",
    "packaging/linux/build_appimage.sh",
    "packaging/macos/make_app.sh",
    "packaging/macos/make_dmg.sh",
    "packaging/release_notes.sh",
];

#[test]
fn test_shell_scripts_use_lf_line_endings() {
    for path in SCRIPTS {
        let data = fs::read(root().join(path)).unwrap();
        assert!(data.starts_with(b"#!"), "{path}");
        assert!(
            !data.windows(2).any(|w| w == b"\r\n"),
            "{path} must use LF line endings"
        );
    }
}

#[test]
fn test_script_help_prints_the_whole_header() {
    for (path, range) in [
        ("packaging/linux/build_appimage.sh", "2,28"),
        ("packaging/macos/make_app.sh", "2,20"),
        ("packaging/macos/make_dmg.sh", "2,25"),
    ] {
        let text = read(path);
        assert!(text.contains(&format!("sed -n '{range}p'")), "{path}");
        let (first, last) = range.split_once(',').unwrap();
        let (first, last): (usize, usize) = (first.parse().unwrap(), last.parse().unwrap());
        let lines: Vec<_> = text.lines().collect();
        assert!(
            lines[first - 1..last].iter().all(|l| l.starts_with('#')),
            "{path}"
        );
        assert!(
            !lines[last].starts_with('#'),
            "{path}: the line after the header"
        );
    }
}

// ======================================================================== release notes
fn release_notes(
    dir: &Path,
    changelog: &str,
    version: &str,
    ref_type: &str,
) -> std::process::Output {
    fs::write(dir.join("CHANGELOG.md"), changelog).unwrap();
    let _ = fs::remove_file(dir.join("release-notes.md"));
    bash(
        &root().join("packaging/release_notes.sh"),
        &[],
        dir,
        &[
            ("VERSION", version),
            ("GITHUB_REF_TYPE", ref_type),
            ("MACOS_MINIMUM", "14.0"),
            ("MACOS_STABLE_SIGNATURE", "false"),
        ],
    )
}

fn changelog(unreleased: &str) -> String {
    format!(
        "# Changelog\n\n## [Unreleased]\n{unreleased}\n## [1.2.0] - 2026-10-01\n\n### Fixed\n\n- The fix that 1.2.0 ships.\n\n[Unreleased]: https://github.com/bugraskl/eye-tracker/compare/v1.2.0...HEAD\n[1.2.0]: https://github.com/bugraskl/eye-tracker/releases/tag/v1.2.0\n"
    )
}

#[test]
fn test_release_notes_are_the_versions_section() {
    for unreleased in ["", "\n### Added\n\n### Fixed\n"] {
        let dir = tempfile::tempdir().unwrap();
        let output = release_notes(dir.path(), &changelog(unreleased), "1.2.0", "tag");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        let notes = fs::read_to_string(dir.path().join("release-notes.md")).unwrap();
        assert!(notes.starts_with("### Fixed\n\n- The fix that 1.2.0 ships.\n\n### Downloads\n"));
        assert!(notes.contains("`EyeTracker-1.2.0-macos-arm64.dmg`"));
        assert!(notes.contains("| macOS 14 or later (Apple silicon) |"));
        assert!(!String::from_utf8_lossy(&output.stdout).contains("::"));
    }
}

#[test]
fn test_release_fails_while_unreleased_has_notes() {
    let dir = tempfile::tempdir().unwrap();
    let text = changelog("\n### Fixed\n\n- Not moved yet.\n");
    let output = release_notes(dir.path(), &text, "1.2.0", "tag");
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stdout).contains(
        "::error file=CHANGELOG.md::CHANGELOG.md still has notes under '## [Unreleased]': move them under '## [1.2.0] - YYYY-MM-DD' before tagging v1.2.0"
    ));
    assert!(!dir.path().join("release-notes.md").exists());
    // A build that is not for a tag (the Bundle check, a manual run) only says so.
    let output = release_notes(dir.path(), &text, "1.2.0", "branch");
    assert!(output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("::notice file=CHANGELOG.md::CHANGELOG.md still has notes under")
    );
    let notes = fs::read_to_string(dir.path().join("release-notes.md")).unwrap();
    assert!(!notes.contains("Not moved yet"));
}

#[test]
fn test_release_fails_without_a_section_for_the_version() {
    let dir = tempfile::tempdir().unwrap();
    let text = changelog("");
    let output = release_notes(dir.path(), &text, "1.3.0", "tag");
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("has no non-empty '## [1.3.0] - YYYY-MM-DD' section")
    );
    let output = release_notes(dir.path(), &text, "1.3.0", "branch");
    assert!(output.status.success());
    let notes = fs::read_to_string(dir.path().join("release-notes.md")).unwrap();
    assert!(notes.starts_with("Development build.\n"));
}

#[test]
fn test_changelog_has_a_section_for_the_current_version() {
    let version = cargo_version();
    let re = Regex::new(&format!(
        r"(?m)^## \[{}\] - \d{{4}}-\d{{2}}-\d{{2}}$",
        regex::escape(&version)
    ))
    .unwrap();
    assert!(
        re.is_match(&read("CHANGELOG.md")),
        "CHANGELOG.md needs a '## [{version}] - YYYY-MM-DD' section for the release job"
    );
}

#[test]
fn test_the_packaging_scripts_read_the_cargo_version() {
    // Both default to the workspace version, so no second copy can drift.
    for path in [
        "packaging/linux/build_appimage.sh",
        "packaging/macos/make_app.sh",
    ] {
        assert!(
            read(path).contains(r#"sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/Cargo.toml""#),
            "{path}"
        );
    }
    assert!(read("Cargo.toml").contains(&format!("version = \"{}\"", cargo_version())));
}

// ======================================================================== workflows
fn workflow(name: &str) -> String {
    read(&format!(".github/workflows/{name}"))
}

const WORKFLOW_FILES: [&str; 5] = [
    "ci.yml",
    "build.yml",
    "release.yml",
    "bundle.yml",
    "pages.yml",
];
/// How release.yml and bundle.yml run the one build definition (same commit, no tag).
const BUILD_CALL: &str = "./.github/workflows/build.yml";

/// The text of each job of a workflow, by job id (two-space indented keys).
fn jobs(text: &str) -> Vec<(String, String)> {
    let body = &text[text.find("\njobs:\n").unwrap()..];
    let re = Regex::new(r"(?m)^  ([A-Za-z0-9_-]+):\n").unwrap();
    let marks: Vec<_> = re
        .captures_iter(body)
        .map(|c| (c.get(0).unwrap(), c[1].to_string()))
        .collect();
    marks
        .iter()
        .enumerate()
        .map(|(i, (m, id))| {
            let end = marks
                .get(i + 1)
                .map_or(body.len(), |(next, _)| next.start());
            (id.clone(), body[m.end()..end].to_string())
        })
        .collect()
}

fn job(text: &str, id: &str) -> String {
    jobs(text)
        .into_iter()
        .find(|(name, _)| name == id)
        .unwrap_or_else(|| panic!("job {id}"))
        .1
}

/// `text` with backslash-continued shell lines joined.
fn joined(text: &str) -> String {
    Regex::new(r" *\\\n *")
        .unwrap()
        .replace_all(text, " ")
        .into_owned()
}

#[test]
fn test_workflow_actions_are_pinned_to_commits() {
    let pin = Regex::new(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40} # v\d+(\.\d+)*$").unwrap();
    let uses = Regex::new(r"(?m)^\s*(?:-\s+)?uses:\s*(.+)$").unwrap();
    for name in WORKFLOW_FILES {
        let text = workflow(name);
        let all: Vec<_> = uses
            .captures_iter(&text)
            .map(|c| c[1].to_string())
            .collect();
        assert!(!all.is_empty(), "{name}");
        for reference in all {
            assert!(
                reference == BUILD_CALL || pin.is_match(&reference),
                "{name}: {reference}"
            );
        }
    }
}

#[test]
fn test_workflows_pin_one_version_of_each_action() {
    let re = Regex::new(r"uses:\s*([\w.-]+/[\w./-]+)@([0-9a-f]{40} # v\S+)").unwrap();
    let mut pins: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
        Default::default();
    for name in WORKFLOW_FILES {
        for c in re.captures_iter(&workflow(name)) {
            pins.entry(c[1].to_string())
                .or_default()
                .insert(c[2].to_string());
        }
    }
    for action in [
        "actions/checkout",
        "actions/upload-artifact",
        "dtolnay/rust-toolchain",
        "Swatinem/rust-cache",
    ] {
        assert!(pins.contains_key(action), "{action}");
    }
    let lagging: Vec<_> = pins.iter().filter(|(_, pin)| pin.len() > 1).collect();
    assert!(lagging.is_empty(), "{lagging:?}");
}

#[test]
fn test_dependabot_keeps_actions_and_cargo_updated() {
    let text = read(".github/dependabot.yml");
    assert!(text.contains("package-ecosystem: github-actions"));
    assert!(text.contains("package-ecosystem: cargo"));
}

#[test]
fn test_nothing_in_the_workflows_or_scripts_is_left_of_the_python_build() {
    let banned =
        Regex::new(r"(?i)python|pyinstaller|\buv\b|pytest|\bpip\b|pyproject|__version__").unwrap();
    let mut paths: Vec<String> = WORKFLOW_FILES
        .iter()
        .map(|n| format!(".github/workflows/{n}"))
        .collect();
    paths.push(".github/dependabot.yml".into());
    paths.extend(SCRIPTS.iter().map(|s| s.to_string()));
    paths.push("packaging/windows/installer.iss".into());
    for path in paths {
        let text = read(&path);
        assert!(
            !banned.is_match(&text),
            "{path}: {:?}",
            banned.find(&text).unwrap().as_str()
        );
    }
    assert!(!root().join("packaging/pyinstaller").exists());
}

const PACKAGES: [&str; 5] = [
    "windows-x64-setup.exe",
    "windows-x64-portable.zip",
    "macos-arm64.dmg",
    "linux-x86_64.AppImage",
    "linux-x86_64.tar.gz",
];

#[test]
fn test_release_workflow_artifact_names() {
    for name in ["build.yml", "release.yml"] {
        let text = workflow(name);
        for suffix in PACKAGES {
            assert!(
                text.contains(&format!("EyeTracker-${{{{ env.VERSION }}}}-{suffix}")),
                "{name} {suffix}"
            );
        }
    }
    let text = workflow("release.yml");
    assert!(text.contains("softprops/action-gh-release@"));
    assert!(text.contains("SHA256SUMS.txt"));
    assert!(text.contains("actions/attest-build-provenance@"));
}

#[test]
fn test_packages_are_built_and_tested_in_one_place() {
    let release = jobs(&workflow("release.yml"));
    let bundle = jobs(&workflow("bundle.yml"));
    let ids = |jobs: &[(String, String)]| jobs.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
    assert_eq!(ids(&release), ["build", "release"]);
    assert_eq!(ids(&bundle), ["build"]);
    for (_, text) in [&release[0], &bundle[0]] {
        assert!(text.contains(&format!("    uses: {BUILD_CALL}\n")));
    }
    assert!(release[1].1.contains("    needs: build\n"));
    for name in ["release.yml", "bundle.yml"] {
        let text = workflow(name);
        for step in [
            "cargo build",
            "cargo xtask",
            "packaging\\windows\\installer.iss",
            "bash packaging/macos/make_dmg.sh",
            "bash packaging/linux/build_appimage.sh",
            "runs-on: windows",
            "runs-on: macos",
        ] {
            assert!(!text.contains(step), "{name}: {step}");
        }
    }
    let build = jobs(&workflow("build.yml"));
    assert_eq!(
        ids(&build),
        ["prepare", "build-windows", "build-macos", "build-linux"]
    );
    for (id, runner) in [
        ("build-windows", "windows-latest"),
        ("build-macos", "macos-14"),
        ("build-linux", "ubuntu-22.04"),
    ] {
        let text = job(&workflow("build.yml"), id);
        assert!(text.contains(&format!("    runs-on: {runner}\n")), "{id}");
        assert!(text.contains("    needs: prepare\n"), "{id}");
        // Packaging and its smoke tests run whether or not the packages are kept.
        for step in text.split("\n      - name: ") {
            assert_eq!(
                step.contains("if: inputs.upload"),
                step.contains("upload-artifact@"),
                "{id}: {}",
                step.lines().next().unwrap()
            );
        }
    }
}

#[test]
fn test_the_bundle_check_keeps_and_publishes_nothing() {
    let write = Regex::new(r"(?m)^\s+[\w-]+: write$").unwrap();
    for name in ["build.yml", "bundle.yml", "release.yml"] {
        assert!(
            workflow(name).contains("\npermissions:\n  contents: read\n"),
            "{name}"
        );
    }
    assert!(!write.is_match(&workflow("build.yml")));
    let bundle = workflow("bundle.yml");
    assert!(!write.is_match(&bundle));
    assert!(!bundle.contains("secrets"));
    assert!(bundle.contains("    with:\n      upload: false\n"));
    let release = jobs(&workflow("release.yml"));
    assert!(!write.is_match(&release[0].1));
    assert!(release[0].1.contains("    with:\n      upload: true\n"));
    assert!(!workflow("release.yml").contains("secrets: inherit"));
    for secret in ["CERT_P12", "CERT_PASSWORD", "IDENTITY"] {
        let line =
            format!("      MACOS_SIGNING_{secret}: ${{{{ secrets.MACOS_SIGNING_{secret} }}}}\n");
        assert!(release[0].1.contains(&line), "{secret}");
    }
    let publish = &release[1].1;
    assert!(publish.contains("    if: github.ref_type == 'tag'\n"));
    for fragment in [
        "      contents: write\n",
        "      id-token: write\n",
        "      attestations: write\n",
        "uses: actions/attest-build-provenance@",
        "uses: softprops/action-gh-release@",
    ] {
        assert!(publish.contains(fragment), "{fragment}");
    }
    assert!(!workflow("build.yml").contains("attest-build-provenance"));
    assert!(!workflow("build.yml").contains("action-gh-release"));
}

#[test]
fn test_release_rebuilds_the_website() {
    let release = workflow("release.yml");
    assert!(release.contains("gh workflow run pages.yml"));
    assert!(release.contains("actions: write"));
    let pages = workflow("pages.yml");
    assert!(pages.contains("workflow_dispatch:"));
    assert!(pages.contains("cargo xtask build-site --release release.json --out _site"));
    assert!(pages.contains("crates/xtask/src/build_site.rs"));
}

#[test]
fn test_workflow_runs_are_grouped() {
    assert!(!workflow("build.yml").contains("concurrency"));
    for (name, cancel) in [
        ("ci.yml", "true"),
        ("bundle.yml", "true"),
        ("release.yml", "false"),
    ] {
        let text = workflow(name);
        assert!(
            text.contains("\nconcurrency:\n  group: ${{ github.workflow }}-"),
            "{name}"
        );
        assert!(
            text.contains(&format!("\n  cancel-in-progress: {cancel}\n")),
            "{name}"
        );
    }
}

#[test]
fn test_ci_runs_the_rust_checks_on_every_os() {
    let ci = workflow("ci.yml");
    for fragment in [
        "- ubuntu-24.04\n",
        "- windows-latest\n",
        "- macos-latest\n",
        "run: cargo fmt --check\n",
        "run: cargo clippy --all-targets -- -D warnings\n",
        "run: cargo test\n",
        "run: cargo xtask check-privacy\n",
        "run: cargo xtask fetch-models --check\n",
        "clang libclang-dev libxcb1-dev libxcb-randr0-dev pkg-config",
        "desktop-file-validate packaging/linux/eye-tracker.desktop",
    ] {
        assert!(ci.contains(fragment), "{fragment}");
    }
    let shellcheck = joined(&ci);
    for script in SCRIPTS {
        assert!(shellcheck.contains(script), "shellcheck {script}");
    }
}

#[test]
fn test_release_builds_pass_the_bundle_privacy_gate() {
    let build = workflow("build.yml");
    for (id, bundle, packaging_step) in [
        (
            "build-windows",
            "dist/EyeTracker",
            "Create the portable ZIP",
        ),
        (
            "build-macos",
            "\"dist/Eye Tracker.app\"",
            "Sign and create the disk image",
        ),
        (
            "build-linux",
            "target/release/eye-tracker",
            "Create the AppImage and the tarball",
        ),
    ] {
        let text = job(&build, id);
        let built = text.find("cargo build --release").unwrap();
        let gate = text
            .find(&format!(
                "run: cargo xtask check-privacy --bundle {bundle}\n"
            ))
            .unwrap();
        assert!(built < gate, "{id}");
        assert!(gate < text.find(packaging_step).unwrap(), "{id}");
        assert!(
            gate < text.find("cargo xtask smoke-test-bundle").unwrap(),
            "{id}"
        );
        assert!(
            text.find("cargo xtask fetch-models --check").unwrap() < built,
            "{id}"
        );
    }
}

#[test]
fn test_linux_builds_check_that_gtk_and_gio_stay_out() {
    let linux = job(&workflow("build.yml"), "build-linux");
    let check = "ldd target/release/eye-tracker | grep -E 'libgtk|libgio'";
    assert!(linux.find(check).unwrap() < linux.find("check-privacy --bundle").unwrap());
}

#[test]
fn test_windows_bundle_has_the_names_the_app_looks_for() {
    let windows = job(&workflow("build.yml"), "build-windows");
    for fragment in [
        r#"cp target/release/eye-tracker-gui.exe "$bundle/EyeTracker.exe""#,
        r#"cp target/release/eye-tracker.exe "$bundle/eye-tracker-cli.exe""#,
        r#"cp crates/eye-tracker/models/NOTICE.md LICENSE packaging/windows/eye-tracker.ico "$bundle/""#,
        r#"cp -R crates/eye-tracker/models/licenses "$bundle/licenses""#,
    ] {
        assert!(windows.contains(fragment), "{fragment}");
    }
    assert!(eye_tracker::cli::FROZEN_CLI_NAME == "eye-tracker-cli");
    let cargo = read("crates/eye-tracker/Cargo.toml");
    assert!(
        cargo.contains("name = \"eye-tracker\"") && cargo.contains("name = \"eye-tracker-gui\"")
    );
}

#[test]
fn test_release_checks_the_documented_macos_minimum() {
    let text = workflow("build.yml");
    let re = Regex::new(r#"(?m)^  MACOS_MINIMUM: "(\d+\.\d+)"$"#).unwrap();
    assert_eq!(re.captures_iter(&text).count(), 1);
    assert_eq!(&re.captures(&text).unwrap()[1], "14.0");
    for name in ["release.yml", "bundle.yml"] {
        assert!(
            !workflow(name).contains("MACOS_MINIMUM"),
            "{name}: defined once"
        );
    }
    let macos = job(&text, "build-macos");
    assert!(macos.contains(r#"export MACOSX_DEPLOYMENT_TARGET="$MACOS_MINIMUM""#));
    assert!(macos.contains(r#"minimum="$(plutil -extract LSMinimumSystemVersion raw "$plist")""#));
    assert!(macos.contains(r#"if [[ "$minimum" != "$MACOS_MINIMUM" ]]; then"#));
    assert!(macos.contains("otool -l"));
    assert!(
        macos.find("LSMinimumSystemVersion").unwrap()
            < macos.find("cargo xtask smoke-test-bundle").unwrap()
    );
    assert!(
        read("packaging/macos/make_app.sh").contains(r#"MACOS_MINIMUM="${MACOS_MINIMUM:-14.0}""#)
    );
}

#[test]
fn test_installer_smoke_test_covers_start_at_sign_in_on_upgrades() {
    let windows = job(&workflow("build.yml"), "build-windows");
    let start = windows.find("- name: Smoke-test the installer").unwrap();
    let smoke = &windows[start..windows.find("- name: Upload the packages").unwrap()];
    let order = [
        r#"Invoke-Setup "install" @("/MERGETASKS=startup,!desktopicon")"#,
        r#"& "$app\eye-tracker.exe" --version"#,
        r#"(Get-Autostart $runKey) -notlike "*\EyeTracker.exe*""#,
        r#"if ((Get-PathCount) -ne 1) { throw "the installer did not add $app to the user PATH" }"#,
        r#"cargo xtask smoke-test-bundle --version $env:VERSION "$app\eye-tracker.exe""#,
        "Remove-ItemProperty -Path $runKey -Name EyeTracker",
        r#"Invoke-Setup "upgrade-silent" @()"#,
        r#"if (Get-Autostart $runKey) { throw "a silent upgrade re-enabled start at sign-in" }"#,
        r#"Invoke-Setup "upgrade-enable" @("/MERGETASKS=startup")"#,
        r#"if (-not (Get-Autostart $runKey)) { throw "/MERGETASKS=startup did not enable it" }"#,
        r#"Invoke-Setup "upgrade-disable" @("/MERGETASKS=!startup")"#,
        r#"Invoke-Setup "upgrade-disabled" @()"#,
        "throw \"an upgrade cleared Task Manager's 'disabled' flag\"",
        r#"Invoke-Setup "upgrade-enable-disabled" @("/MERGETASKS=startup")"#,
        "throw \"/MERGETASKS=startup left Task Manager's 'disabled' flag\"",
        r#"Invoke-Setup "upgrade-no-path" @("/MERGETASKS=!addtopath")"#,
        r#"Invoke-Setup "upgrade-unreadable-path" @("/MERGETASKS=addtopath")"#,
        r#"throw "setup replaced a user PATH that it cannot read""#,
        "finally {",
        r#"$app\unins000.exe"#,
        r#"throw "uninstall left the StartupApproved value behind""#,
        r#"if ((Get-PathCount) -ne 0) { throw "uninstall left $app on the user PATH" }"#,
    ];
    let positions: Vec<_> = order
        .iter()
        .map(|f| smoke.find(f).unwrap_or_else(|| panic!("{f}")))
        .collect();
    assert!(positions.windows(2).all(|w| w[0] < w[1]), "{positions:?}");
    // The setup logs (which say why start at sign-in was or was not written) on failure.
    assert!(smoke.contains("catch {"));
    assert!(smoke.contains("Get-Content $log"));
}

#[test]
fn test_every_package_is_smoke_tested_as_users_get_it() {
    let build = workflow("build.yml");
    let windows = joined(&job(&build, "build-windows"));
    assert!(windows.contains(
        r#"cargo xtask smoke-test-bundle --version "$VERSION" dist/EyeTracker/eye-tracker-cli.exe"#
    ));
    assert!(
        windows.find("dist/EyeTracker/eye-tracker-cli.exe").unwrap()
            < windows.find("Create the portable ZIP").unwrap()
    );
    let macos = joined(&job(&build, "build-macos"));
    for fragment in [
        r#"hdiutil attach -nobrowse -readonly -mountpoint "$mnt""#,
        r#"broken="$(find -L "$app" -type l)""#,
        r#"codesign --verify --deep --strict --verbose=2 "$app""#,
        r#"cli="$app/Contents/MacOS/eye-tracker-cli""#,
        r#"cargo xtask smoke-test-bundle --version "$VERSION" "$cli""#,
    ] {
        assert!(macos.contains(fragment), "{fragment}");
    }
    let linux = joined(&job(&build, "build-linux"));
    for fragment in [
        r#"cargo xtask smoke-test-bundle --version "$VERSION" "dist/EyeTracker-$VERSION-linux-x86_64.AppImage""#,
        r#"broken="$(find -L "$RUNNER_TEMP/eye-tracker" -type l)""#,
        r#"test -f "$RUNNER_TEMP/eye-tracker/NOTICE.md""#,
    ] {
        assert!(linux.contains(fragment), "{fragment}");
    }
}

#[test]
fn test_release_notes_explain_macos_permissions_after_updates() {
    let build = workflow("build.yml");
    let prepare = job(&build, "prepare");
    assert!(
        prepare.contains("MACOS_STABLE_SIGNATURE: ${{ secrets.MACOS_SIGNING_CERT_P12 != '' }}")
    );
    assert!(prepare.contains("bash packaging/release_notes.sh"));
    assert!(
        read("packaging/release_notes.sh")
            .contains("remove Eye Tracker with \"−\" and add it again.")
    );
    // Signing with a stable certificate is optional; without the secret the import step
    // exits early and make_dmg.sh falls back to an ad-hoc signature.
    let macos = job(&build, "build-macos");
    assert!(macos.contains(r#"if [[ -z "$CERT_P12" ]]; then"#));
    assert!(macos.contains("MACOS_SIGN_IDENTITY: ${{ steps.signing.outputs.identity }}"));
    assert!(macos.contains("security delete-keychain"));
}

#[test]
fn test_bundle_workflow_runs_for_what_decides_the_packages() {
    let text = workflow("bundle.yml");
    let triggers = &text[..text.find("\npermissions:").unwrap()];
    for path in [
        "crates/**",
        "vendor/**",
        "packaging/**",
        "Cargo.toml",
        "Cargo.lock",
        "CHANGELOG.md",
        ".github/workflows/build.yml",
        ".github/workflows/bundle.yml",
        ".github/workflows/release.yml",
    ] {
        // pull_request and push
        assert_eq!(
            triggers.matches(&format!("      - \"{path}\"\n")).count(),
            2,
            "{path}"
        );
    }
}

#[test]
fn test_documentation_links_point_to_real_pages() {
    let pages = [
        "docs/troubleshooting.md",
        "docs/privacy.md",
        "docs/platform-support.md",
    ];
    let blob = Regex::new(r"eye-tracker/blob/main/([\w./-]+)").unwrap();
    let docs = Regex::new(r"\$docs/([\w.-]+\.md)").unwrap();
    for path in [
        ".github/workflows/build.yml",
        ".github/workflows/release.yml",
        "packaging/release_notes.sh",
    ] {
        let text = read(path);
        assert!(!text.contains("tree/main/docs"), "{path}");
        for c in blob.captures_iter(&text) {
            assert!(
                pages.contains(&&c[1]) || ["CHANGELOG.md", "docs"].contains(&&c[1]),
                "{path}: {}",
                &c[1]
            );
        }
        for c in docs.captures_iter(&text) {
            assert!(
                pages.contains(&format!("docs/{}", &c[1]).as_str()),
                "{path}: {}",
                &c[1]
            );
        }
    }
}

#[test]
fn test_model_files_are_binary_in_git() {
    let attributes = read(".gitattributes");
    let models = root().join("crates/eye-tracker/models");
    let mut suffixes = std::collections::BTreeSet::new();
    for entry in fs::read_dir(&models).unwrap() {
        let path = entry.unwrap().path();
        if let Some(ext) = path
            .extension()
            .filter(|ext| path.is_file() && *ext != "md")
        {
            suffixes.insert(ext.to_string_lossy().into_owned());
        }
    }
    assert!(!suffixes.is_empty());
    for suffix in suffixes {
        assert!(
            attributes
                .lines()
                .any(|l| l == format!("*.{suffix} binary")),
            "{suffix}"
        );
    }
}
