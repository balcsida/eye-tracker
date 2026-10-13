// Ported from tests/test_panes_command.py (the provider command runner and process helpers).
//
// Changed: the Python tests replaced `subprocess.run` and psutil so no process was started. Here
// `run_command` is exercised against the standard Unix tools (`sh`, `sleep`), so those three
// tests are `cfg(unix)`; "passes an argument vector without a shell" and the stdin/CREATE_NO_WINDOW
// checks of test_run_command_passes_an_argument_vector_without_a_shell are covered by the argument
// reaching the program as one word. The process-tree helpers are tested on a hand-made process
// table (`ProcessTable`) instead of a fake `psutil.Process`.

use super::*;

#[cfg(unix)]
fn argv(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[cfg(unix)]
#[test]
fn test_run_command_passes_an_argument_vector_without_a_shell() {
    // `$1` is one argument even though it holds shell syntax: nothing is interpreted.
    let result = run_command(
        &argv(&[
            "sh",
            "-c",
            "printf '%s' \"$1\"",
            "sh",
            "#{pane_id}; echo hi",
        ]),
        5.0,
    )
    .unwrap();
    assert_eq!(result, CommandResult::new(0, "#{pane_id}; echo hi", ""));
    assert!(result.ok());
}

#[cfg(unix)]
#[test]
fn test_run_command_failures_are_pane_errors() {
    let error = run_command(&argv(&["sleep", "5"]), 0.2).unwrap_err();
    assert!(error.0.contains("did not answer"), "{error}");

    let error = run_command(&argv(&["/nonexistent/eye-tracker-test-program"]), 1.0).unwrap_err();
    assert!(error.0.contains("could not be started"), "{error}");

    let result = run_command(
        &argv(&["sh", "-c", "echo 'no server running' >&2; exit 1"]),
        5.0,
    );
    assert_eq!(
        result.unwrap(),
        CommandResult::new(1, "", "no server running\n")
    );
}

fn table(entries: &[(u32, Option<u32>, &str)]) -> ProcessTable {
    ProcessTable(
        entries
            .iter()
            .map(|(pid, parent, name)| {
                (
                    *pid,
                    ProcessEntry {
                        parent: *parent,
                        name: name.to_string(),
                    },
                )
            })
            .collect(),
    )
}

#[test]
fn test_process_tree_helpers() {
    let processes = table(&[
        (1, None, "init"),
        (49, Some(1), "explorer"),
        (50, Some(49), "WindowsTerminal.exe"),
        (51, Some(50), "OpenConsole.exe"),
        (52, Some(50), "wsl.exe"),
        (53, Some(52), "bash"),
    ]);
    assert_eq!(processes.parents(50), vec![49, 1]);
    assert_eq!(processes.parents(404), Vec::<u32>::new());
    let expected: HashSet<String> = ["openconsole", "wsl", "bash"].map(String::from).into();
    assert_eq!(processes.descendant_names(50), expected);
    assert_eq!(processes.descendant_names(404), HashSet::new());
}

#[test]
fn test_window_key() {
    let window = |handle, pid| WindowRef {
        handle: WindowHandle::Hwnd(handle),
        pid: Some(pid),
        rect: None,
    };
    assert_eq!(window_key(&window(12, 3)), window_key(&window(12, 3)));
    // A handle reused by another process's window is another window.
    assert_ne!(window_key(&window(12, 4)), window_key(&window(12, 3)));
    // The Python test also covered an unhashable handle; handles here are always hashable.
}

#[test]
fn test_process_basename() {
    assert_eq!(process_basename("WindowsTerminal.exe"), "windowsterminal");
    assert_eq!(process_basename("/usr/bin/kitty"), "kitty");
    assert_eq!(
        process_basename("C:\\Apps\\WezTerm\\wezterm-gui.exe"),
        "wezterm-gui"
    );
    assert_eq!(process_basename("bash"), "bash");
}
