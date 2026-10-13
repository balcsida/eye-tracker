# Contributing

Thanks for helping make Eye Tracker better. Reproducible bug reports, accuracy data from real
setups, and small focused pull requests are the most valuable contributions.

## Ground rules

- **Privacy is a hard constraint.** No network code under `crates/eye-tracker/src` (except
  `update/winhttp.rs`), no frames written to disk, no telemetry. `cargo xtask check-privacy`
  enforces this in CI; please don't work around it.
- **Degrade, never crash.** Platform features are best effort. An unsupported call returns
  `false`/`None` and the app keeps running.
- **Pure logic stays pure.** `gaze/*` and `engine/*` (except `controller.rs`) touch neither the
  camera, the UI nor the operating system and take time as an argument, so they can be tested
  deterministically.
- **Measure performance claims.** If a change affects CPU or latency, include `eye-tracker bench`
  numbers before and after.

## Development setup

You need a current stable [Rust toolchain](https://rustup.rs/). On Linux (Debian/Ubuntu names) also
`clang`, `libclang-dev`, `libxcb1-dev`, `libxcb-randr0-dev` and `pkg-config`.

```bash
git clone https://github.com/bugraskl/eye-tracker.git
cd eye-tracker
cargo run --release           # start the tray app from source
cargo run --release -- doctor # environment and permission report
```

No webcam? Point the app at a photo, or at a folder of image frames played in file-name order and
looped (video files are not supported):

```bash
cargo run --release -- --camera path/to/face.jpg
```

`--camera`, `--backend` and `--trace` only apply when Eye Tracker starts: quit a running instance
first (tray menu → **Quit Eye Tracker**, or `eye-tracker ctl quit`), otherwise the command only
hands over to it and says that the options were not applied.

## Checks

Run everything CI runs before opening a pull request:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo xtask check-privacy
cargo xtask fetch-models --check
cargo xtask gen-config-docs --check
```

After changing a setting in `crates/eye-tracker/src/config.rs`, regenerate the configuration
reference with `cargo xtask gen-config-docs` (the check above fails while it is out of date). CI
also runs `shellcheck` on the packaging shell scripts and `desktop-file-validate` on the Linux
desktop entry.

After a local release build ([docs/building.md](docs/building.md)), also run the bundle privacy
gate, which scans every native binary of the bundle:

```bash
cargo xtask check-privacy --bundle target/release
```

Pull requests that change `packaging/`, the privacy gate or the dependencies also build the
bundles on all three platforms in CI (the *Bundle* workflow).

Tests need no display or camera. Tests that need a real face photo return early unless
`EYE_TRACKER_TEST_FACE` points to one (face photos are never committed).

### Testing Linux from a Mac

Run the Linux build and tests in a container (the first run compiles everything and takes about ten
minutes; the named volumes keep the registry and the Linux `target` directory between runs):

```bash
docker run --rm -v "$PWD":/src -v et-cargo-registry:/usr/local/cargo/registry \
  -v et-target-linux:/target -w /src -e CARGO_TARGET_DIR=/target rust:1.99-bookworm \
  bash -c 'apt-get update -qq >/dev/null && apt-get install -y -qq clang libclang-dev >/dev/null && cargo test'
```

[docs/building.md](docs/building.md#cross-checking-windows-from-another-system) describes the
MinGW check for Windows.

## Project layout

```text
crates/eye-tracker/
  src/
    vision/     camera capture, motion gate, face backends (facemesh: MediaPipe's landmark
                model run with tract; lite: YuNet), worker thread
    gaze/       gaze regression model, smoothing, calibration, implicit learning
    engine/     switching decision, presence, shoulder guard, rate scheduler, controller
    panes/      split-pane focus (experimental)
    platform/   OS integration (Windows, macOS, Linux), autostart, global hotkeys
    ui/         tray, settings, calibration, overlays, first-run wizard (egui)
    update/     the opt-in update check; winhttp.rs is the only network file
    app.rs      application wiring
    cli.rs      command-line interface
  models/       the face models (embedded in the program), NOTICE.md and licence texts
crates/xtask/   cargo xtask: privacy gate, model fetch/verification, icons, docs, website
packaging/      Windows installer, macOS app and DMG, Linux AppImage
docs/           user and developer documentation
```

Tests live next to the module they test, as `foo_tests.rs` beside `foo.rs`.
[docs/architecture.md](docs/architecture.md) explains how the pieces fit together and lists the
conventions.

GitHub Actions are pinned to full commit SHAs; Dependabot proposes updates.

## Pull requests

1. Open an issue first for anything larger than a bug fix, so we can agree on the approach.
2. Keep pull requests focused; one behaviour change per PR.
3. Add or update tests. Logic changes in `engine/` or `gaze/` need unit tests (in the module's `*_tests.rs`).
4. Update `CHANGELOG.md` under **Unreleased** (start that section if there is none). Until a
   version has been released, its own section describes it as it is, so describe changes to it
   there.
5. Platform-specific changes: say which OS and desktop environment you tested on.

## Reporting bugs

Please use the bug report template and paste the output of `eye-tracker doctor` (the template says
how to run it with each package). It contains your OS, camera and monitor layout and feature
support, and **no images or personal data**.

## Code of conduct

Be kind and constructive. Harassment or personal attacks are not tolerated in issues, pull requests
or discussions; maintainers may remove such content and block repeat offenders.
