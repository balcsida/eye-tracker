# Rust port: conventions

This file is the contract for the Python → Rust rewrite. Every porting task follows it. It
describes decisions, not progress. The Python sources under `src/eye_tracker/` and the tests
under `tests/` are the specification; port them faithfully, module by module, and keep the
behaviour, file formats, names and defaults. Do not redesign.

## Layout

```
Cargo.toml                      workspace: crates/eye-tracker, crates/xtask
crates/eye-tracker/             the app (library + two binaries)
  Cargo.toml
  src/lib.rs                    pub mod … for every module below; APP_NAME, APP_SLUG, APP_ID,
                                APP_AUTHOR, REPO_URL, VERSION (from CARGO_PKG_VERSION)
  src/bin/eye-tracker.rs        console entry: std::process::exit(eye_tracker::cli::main())
  src/bin/eye-tracker-gui.rs    same, with #![cfg_attr(windows, windows_subsystem = "windows")]
                                and eye_tracker::cli::gui_main()
  src/types.rs                  ← types.py
  src/config.rs                 ← config.py
  src/paths.rs                  ← paths.py
  src/trace.rs                  ← trace.py
  src/logging_setup.rs          ← logging_setup.py
  src/ipc.rs                    ← ipc.py
  src/diagnostics.rs            ← diagnostics.py
  src/cli.rs                    ← cli.py
  src/app.rs                    ← app.py
  src/vision/{mod,camera,motion,threads,worker}.rs
  src/vision/backends/{mod,base,face_geometry,facemesh_backend,lite_backend,overlay}.rs
  src/gaze/{mod,model,filters,calibration,learning,store}.rs
  src/engine/{mod,decision,reading,presence,guard,scheduler,input_state,window_memory,
              camera_yield,controller}.rs
  src/panes/{mod,types,decider,registry,worker}.rs
  src/panes/providers/{mod,command,tmux,wezterm,windows_terminal,chromium_apps}.rs
  src/platform/{mod,base,windows,macos,linux,autostart,hotkeys,win_uia}.rs
  src/ui/{mod,tray,settings_dialog,calibration_window,wizard,overlay,curtain,countdown,about,
          preview,icons,util,update_dialog}.rs
  src/update/{mod,release,fetch,installer,service,winhttp}.rs
  src/testing/                  #[cfg(test)] shared test helpers ← tests/gaze_synth.py,
                                tests/face_drawing.py, tests/panes_fakes.py, tests/conftest.py
  models/                       the three model files, NOTICE.md and licenses/ (moved from
                                src/eye_tracker/vision/models; embedded with include_bytes!)
crates/xtask/                   `cargo xtask <command>` ← scripts/*.py (check-privacy,
                                fetch-models, gen-config-docs, make-icons, build-site,
                                smoke-test-bundle)
```

One Python module → one Rust module with the same name and the same public functions, types and
constants (snake_case / CamelCase as Rust requires). `mod.rs` files re-export what the Python
`__init__.py` exported. Platform-only modules are `#[cfg(target_os = "…")]`.

## Tests

The Python tests are ported with the module they test. Tests for `src/foo.rs` live next to it in
`src/foo_tests.rs`, declared at the bottom of `foo.rs` as

```rust
#[cfg(test)]
#[path = "foo_tests.rs"]
mod tests;
```

so they can use private items like the Python tests did. Keep the Python test names
(`test_…`) and their assertions and tolerances. Shared fixtures go to `src/testing/`. A test that
cannot be ported (it drove a Qt widget, or needs a real camera or face photo) is listed in the
module's `*_tests.rs` header comment with the reason, so nothing is lost silently. Tests that
needed `EYE_TRACKER_TEST_FACE` keep that environment variable and skip (return early) without it.

`cargo test`, `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` must pass
before a task is done.

## Crates (fixed; do not add alternatives)

| Need | Crate |
|---|---|
| CLI | `clap` (derive) |
| Config, calibration store, IPC messages | `serde`, `serde_json` (same JSON schema and file names as the Python app) |
| User directories (platformdirs) | `directories` |
| Logging | `log` facade + `flexi_logger` (size-rotated file, 1 MB × 3, same format) |
| Errors | `thiserror` for typed errors, `anyhow` only at the CLI/app edge |
| Threads | `std::thread`, `crossbeam-channel`, `parking_lot`; no async runtime |
| Numerics | `nalgebra` (regression, PnP), `ndarray` where an array API reads best |
| Images | `image`, `fast_image_resize`, `imageproc` |
| Inference | `tract-tflite` (face landmarks), `tract-onnx` (YuNet); fallback only if the spike fails: the `opencv` crate |
| Camera | `nokhwa` without default features (MSMF / AVFoundation / V4L2); MJPEG decoded with `image`; image files and frame directories replace video-file input |
| UI | `egui` + `eframe` (hidden root viewport; dialogs, calibration, overlays and curtain are viewports) |
| Tray | `tray-icon` on Windows/macOS, `ksni` on Linux |
| Notifications | `notify-rust` |
| File picker | `egui-file-dialog` (pure egui, every OS) |
| Fatal error box | `rfd::MessageDialog` on Windows/macOS only; stderr plus a notification on Linux |
| Global hotkeys | `global-hotkey` for registration where it fits; the layout and availability checks are ported directly |
| IPC | `interprocess` local sockets (named pipes on Windows, Unix sockets elsewhere) |
| Processes | `sysinfo`, `std::process` |
| Hashing | `sha2`, `hex` |
| Windows | `windows` (Win32, UI Automation, WinHTTP, registry) |
| macOS | `objc2`, `objc2-foundation`, `objc2-app-kit`, `objc2-core-graphics`, `objc2-application-services` / `core-foundation` as needed |
| Linux | `x11rb` (with xtest and xkb), `zbus` (D-Bus), `libc` |
| Time | `std::time`; pure logic takes `now: f64` seconds as the Python did |
| xtask network | `ureq` (dev tooling only, never in the app) |

Add a crate only through `[workspace.dependencies]` in the root `Cargo.toml` and
`workspace = true` in the crate. Every crate above is pre-declared by the scaffold.

## Rules carried over from the Python project

- **Privacy is a hard constraint.** No network code in `crates/eye-tracker` except
  `update/winhttp.rs`. No frame is ever written to disk. `cargo xtask check-privacy` enforces it.
- **Degrade, never crash.** Platform calls return `bool` / `Option`; a failure logs and continues.
  No `unwrap()`/`expect()` outside tests except on invariants the code itself guarantees.
- **Pure logic stays pure.** `gaze/*`, `engine/*` (except `controller.rs`) and `panes/decider.rs`
  do not touch the camera, the UI or the OS, and take time as an argument.
- **Coordinates** are native pixels on Windows and Linux/X11 and Cocoa points on macOS, exactly
  as before. No scaling anywhere.
- **Config and data files** keep their names, locations, JSON keys, versions and migrations, so a
  user upgrading from the Python app keeps settings and calibrations.
- Comments explain the why, as the Python docstrings did; keep the module docstrings as `//!`.

## Qt idioms

- **Signals.** A component collects what it would have emitted in a `Vec<Event>` (an enum per
  component) that the method producing them returns, or that `drain_events()` hands over, and the
  owner (controller, app) dispatches. Nothing stores callbacks. Cross-thread signals are a
  `crossbeam_channel::Sender<Event>` given to the thread at start.
- **Timers.** `QTimer` becomes a `tick(now: f64)` method the owner calls; the component keeps its
  own due time. The app loop ticks the controller at a fixed cadence.
- **QObject workers on QThread** become a `std::thread` with a request channel in and an event
  channel out, joined on drop or `stop()`.
- **QLocalServer/QLocalSocket** become `interprocess` local sockets served on their own thread.

## Vision stack: what the spike established

The spike project at `/tmp/eye-tracker-port/spike` (report:
`/tmp/eye-tracker-port/notes/spike-vision.md`) is the reference for these points.

- `tract-tflite` 0.23.8 needs two extra TFLite ops (`DEQUANTIZE` for float16 weights, `PRELU`,
  about 30 lines in `src/ops/nn.rs`). First check whether tract's git `main` already has them; if
  so depend on that commit, otherwise vendor the patched crate under `vendor/tract-tflite/` with
  the `.patch` file next to it and `[patch.crates-io]` in the root `Cargo.toml`.
- Landmark net: input `input_12` 1×256×256×3 NHWC RGB 0..1; outputs `Identity` (1434 = 478×3) and
  `Identity_1` (presence logit). `into_optimized()` gives 13 ms per frame on an M3. The load-time
  self-test with `self_test_blob` values must stay.
- YuNet: `tract_onnx::onnx().with_ignore_value_info(true).with_ignore_output_shapes(true)`, one
  optimised plan per input size (pad to multiples of 32), post-processing as OpenCV's
  `FaceDetectorYN` (spike `src/yunet.rs`). Lives in `vision/backends/yunet.rs`.
- PnP: Rodrigues, projectPoints and an LM PnP with DLT start (spike `src/pnp.rs`) live in
  `vision/backends/pnp.rs` and replace `cv2.solvePnP` for both the SQPNP and ITERATIVE paths.
- Camera on macOS: do not request MJPEG from AVFoundation; take the device's native format and
  check that NV12 frames decode correctly (nokhwa maps `420v`/`420f` to YUYV, which may be wrong).
  Pick the device by the configured name, never blindly index 0 (Continuity Camera comes first).

## Threads

| Thread | Owns |
|---|---|
| UI thread (eframe) | controller, UI, gaze model, decisions, platform calls |
| Vision worker | camera, motion gate, face backend |
| Hotkey thread (Windows, X11) | hotkey message loop |
| Pane worker | pane providers |

Workers send messages over `crossbeam-channel`; the UI thread drains them in `update()` and calls
`ctx.request_repaint()` / `request_repaint_after()` instead of Qt timers.
