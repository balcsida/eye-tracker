//! Embeds the application icon into the Windows executables (the PyInstaller spec did the same).

fn main() {
    // The target, not the host: the executables are also cross-compiled.
    if std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows") {
        winresource::WindowsResource::new()
            .set_icon("../../packaging/windows/eye-tracker.ico")
            .compile()
            .expect("could not embed the Windows icon");
    }
    println!("cargo:rerun-if-changed=../../packaging/windows/eye-tracker.ico");
}
