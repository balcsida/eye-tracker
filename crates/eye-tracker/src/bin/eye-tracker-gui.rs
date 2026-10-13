#![cfg_attr(windows, windows_subsystem = "windows")]

fn main() {
    std::process::exit(eye_tracker::cli::gui_main());
}
