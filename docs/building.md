# Building from source

Releases are built on GitHub's runners whenever a `v*` tag is pushed, by the steps in
[`.github/workflows/build.yml`](../.github/workflows/build.yml) (see
[Continuous integration](#continuous-integration)). You can run the same steps locally. Build each
package on the operating system it targets; the one cross-check that works from anywhere is
[the MinGW check](#cross-checking-windows-from-another-system).

## Prerequisites

- A current stable [Rust toolchain](https://rustup.rs/)
- **Windows:** [Inno Setup 6](https://jrsoftware.org/isinfo.php) for the installer (optional)
- **macOS:** Xcode command line tools (`xcode-select --install`)
- **Linux** (Debian/Ubuntu package names): `clang`, `libclang-dev`, `libxcb1-dev`,
  `libxcb-randr0-dev` and `pkg-config`

## Common steps

```bash
cargo build --release                     # target/release/eye-tracker and eye-tracker-gui
cargo xtask fetch-models --check          # verify the committed models and licence texts
cargo xtask make-icons                    # .ico / .icns / .png from the app's own icon code
```

The result is two binaries and nothing else to ship: `eye-tracker` (the command line) and
`eye-tracker-gui` (the tray app; on Windows it has no console window). The face models are embedded
in the program, and there is no Qt, OpenCV or Python runtime to bundle. The packaging scripts below
turn the binaries into the packages.

The release profile (`[profile.release]` in the root `Cargo.toml`) uses fat LTO, one codegen unit
and stripped symbols, which makes a release build take a few minutes longer (about 4 minutes on
an M-series Mac) and the binary over a third smaller. The optimisation level stays at 3, because the
face models' inference is the hot path, and panics still unwind, because the app catches them.

Then run the privacy gate on the result. It reads every native binary and fails on telemetry
markers, on networking libraries and on imported networking functions that are not on its reviewed
allow-list ([how it works](privacy.md#the-privacy-gate)). The success line says how many native
binaries it checked:

```bash
cargo xtask check-privacy --bundle target/release           # the plain build
cargo xtask check-privacy --bundle "dist/Eye Tracker.app"   # a macOS app
```

The allow-list (`BUNDLE_ALLOWLIST` in
[`crates/xtask/src/check_privacy.rs`](../crates/xtask/src/check_privacy.rs)) names each file that may
contain networking code, exactly what it may link, and why; the gate prints those as notices.

Finally, `cargo xtask smoke-test-bundle --version X.Y.Z <command-line tool or AppImage>` starts a
built bundle in a temporary configuration folder, without opening a camera, and checks `--version`,
`ctl status` and `ctl quit`; the release workflow does this for each package the way users get it.
The tool inside a macOS app is `Eye Tracker.app/Contents/MacOS/eye-tracker-cli`, a symbolic link to
the app's own executable.

## Packages

The scripts live under `packaging/`; their headers describe their arguments. Run them from the
repository root after the common steps above. Every package is written to `dist/`.

**Windows installer and portable ZIP**

The installer script packs a bundle folder, `dist\EyeTracker`, which the build workflow assembles:
the windowed app as `EyeTracker.exe`, the console tool as `eye-tracker-cli.exe`, and `NOTICE.md`,
`LICENSE`, `eye-tracker.ico` and the `licenses` folder. The portable ZIP is that folder.

```bash
mkdir -p dist/EyeTracker
cp target/release/eye-tracker-gui.exe dist/EyeTracker/EyeTracker.exe
cp target/release/eye-tracker.exe dist/EyeTracker/eye-tracker-cli.exe
cp crates/eye-tracker/models/NOTICE.md LICENSE packaging/windows/eye-tracker.ico dist/EyeTracker/
cp -R crates/eye-tracker/models/licenses dist/EyeTracker/licenses
(cd dist && 7z a -tzip -mx=9 EyeTracker-0.3.0-windows-x64-portable.zip EyeTracker)
```

```powershell
iscc /DAppVersion=0.3.0 packaging\windows\installer.iss
```

This writes `dist\EyeTracker-0.3.0-windows-x64-setup.exe`. A version such as `0.3.0rc1` also needs
`/DAppVersionNumeric=0.3.0` (the Windows file version must be numeric); `/DBundleDir` and
`/DOutputDir` change the two folders. The installer is per user (no administrator rights), creates a Start menu shortcut, and optionally
registers start at sign-in. It also installs the command-line tool and, with the task *Add the
"eye-tracker" command to PATH* (on by default; `/MERGETASKS=!addtopath` turns it off in a silent
install), adds the installation folder to the user's PATH; uninstalling removes that entry. A user
PATH that is not stored as a string value (some tools write it as `REG_MULTI_SZ`) cannot be read,
and is left as it is. Upgrades keep your choices, and a silent upgrade (winget, `/SILENT`,
`/VERYSILENT`) restarts Eye Tracker in the background if it had to close it.

Start at sign-in is never turned back on by an upgrade: it stays as you left it in the app or in
Task Manager (which can disable it without removing it), whatever the first installation chose. An
explicit choice wins, also in a silent upgrade: `/MERGETASKS=startup` turns it on (and clears Task
Manager's "disabled" flag), `/MERGETASKS=!startup` turns it off, and `/TASKS=...` turns it on only
when it lists `startup`.

**macOS app and disk image**

```bash
bash packaging/macos/make_app.sh
bash packaging/macos/make_dmg.sh
```

`make_app.sh` assembles `dist/Eye Tracker.app` from the release build (`--bin-dir`, `--out` and
`--version` override the defaults; it needs the icon from `cargo xtask make-icons`). The app holds
one executable, the console binary `eye-tracker` as `Contents/MacOS/Eye Tracker`, and
`Contents/MacOS/eye-tracker-cli` is a relative symbolic link to it; `eye-tracker-gui` only differs
on Windows, so it is not copied. `make_dmg.sh`
signs it (`--identity`, `--keychain`) and builds `dist/EyeTracker-<version>-macos-<arch>.dmg`, running
`make_app.sh` first when the app does not exist yet. The release workflow builds with
`MACOSX_DEPLOYMENT_TARGET` set to its `MACOS_MINIMUM` (14.0), which `make_app.sh` also writes into
the Info.plist; do the same when you build for other Macs. The app is signed ad hoc unless you provide a certificate. With a stable identity (a
Developer ID, or a self-signed certificate as described at the top of `make_dmg.sh`), the Camera and
Accessibility permissions survive updates. In CI, set the repository secrets
`MACOS_SIGNING_CERT_P12`, `MACOS_SIGNING_CERT_PASSWORD` and optionally `MACOS_SIGNING_IDENTITY`.

**Linux AppImage and tarball**

```bash
bash packaging/linux/build_appimage.sh
```

This writes `dist/EyeTracker-<version>-linux-<arch>.AppImage` and
`dist/EyeTracker-<version>-linux-<arch>.tar.gz` (`--version`, `--dist` and `--binary` override the
defaults). `appimagetool` and the AppImage runtime are downloaded (with `curl`) into `build/tools/` at
pinned versions and verified against their SHA-256. Where FUSE is unavailable (containers, CI), export `APPIMAGE_EXTRACT_AND_RUN=1`.

## Cross-checking Windows from another system

A Windows build cannot be tested away from Windows, but it can be type-checked, which catches most
`cfg(windows)` mistakes:

```bash
rustup target add x86_64-pc-windows-gnu
cargo check --target x86_64-pc-windows-gnu --all-targets
```

This needs MinGW-w64 (`brew install mingw-w64`, or `gcc-mingw-w64-x86-64` on Debian/Ubuntu). The
Linux build can be checked from a Mac the way
[CONTRIBUTING.md](../CONTRIBUTING.md#testing-linux-from-a-mac) describes.

## Face models

The models live in [`crates/eye-tracker/models/`](../crates/eye-tracker/models/) together with
`NOTICE.md` (licences, upstream URLs and checksums) and, in its `licenses/` folder, the full licence
texts that must ship with them (Apache-2.0 and the YuNet MIT notice). To re-download and re-extract
the models from their upstream sources:

```bash
cargo xtask fetch-models
```

The licence texts are never downloaded: `cargo xtask fetch-models --check` fails when one is
missing or altered (restore it from git). The program embeds the model files with `include_bytes!`,
so a build fails when one is missing.

### The Core ML package (macOS)

`models/FaceLandmarks.mlpackage` is `face_landmarks_detector.tflite` converted for Core ML, which
the facemesh backend runs on macOS (see [architecture](architecture.md)). It is committed, so a
build needs neither Python nor Xcode. Rerun the conversion only when `face_landmarks_detector.tflite`
or the conversion itself changes; the app compiles the new package on its next start. The tool
lives in [`tools/coreml/`](../tools/coreml/) with pinned versions (Python 3.12, macOS):

```bash
python3.12 -m venv /tmp/coreml-venv
/tmp/coreml-venv/bin/pip install -r tools/coreml/requirements.txt
/tmp/coreml-venv/bin/python tools/coreml/convert.py   # rewrites crates/eye-tracker/models/FaceLandmarks.mlpackage
/tmp/coreml-venv/bin/python tools/coreml/check.py     # parity with TFLite, compute plan, timing
```

`check.py` also needs `xcrun coremlcompiler` (the Xcode command line tools) for the compute plan.
Then run `cargo test`: the facemesh tests check the package against tract.

## Continuous integration

- [`ci.yml`](../.github/workflows/ci.yml) runs `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, the privacy
  gate, the model check and the configuration reference check on Linux, Windows and macOS.
- [`build.yml`](../.github/workflows/build.yml) is the one definition of how the packages are
  built and tested. It is a reusable workflow, never run on its own: the Bundle check and the
  release both call it, so a pull request checks exactly what a release ships. It reads the version
  and writes the release notes (`packaging/release_notes.sh`), then builds each platform on its own runner (`windows-latest`,
  `macos-14`, and `ubuntu-22.04` for the oldest supported glibc):
  - **every platform:** `cargo xtask fetch-models --check`, `cargo build --release`,
    `cargo xtask make-icons`, the bundle privacy gate, the packages, and
    `cargo xtask smoke-test-bundle` on each package the way users get it;
  - **Windows:** the bundle (the portable ZIP's contents), then the installer, which a disposable
    runner installs, upgrades and uninstalls: start at sign-in after silent upgrades, with and
    without `/MERGETASKS=startup` or `!startup` and with Task Manager's "disabled" flag, the PATH
    entry (also with an unreadable PATH value), and the installed copy smoke-tested through the
    `eye-tracker` command;
  - **macOS:** the minimum macOS version, signing (with the optional certificate) and the disk
    image; the app is then checked inside the mounted image: no symbolic link without a target,
    `codesign --verify --deep --strict`, the Info.plist, and the smoke test of its
    `eye-tracker-cli`;
  - **Linux:** a check that the executable links no GTK or GIO library, the privacy gate on the
    executable, the AppImage (smoke test) and the tarball (its version, `.desktop` file, `NOTICE.md`,
    `licenses/` and links).
- [`bundle.yml`](../.github/workflows/bundle.yml) runs `build.yml` whenever a pull request or a push
  to `main` changes the app (`crates/`), `packaging/`, `Cargo.toml`, `Cargo.lock` or the release workflows. It passes no secret (the macOS app
  is signed ad hoc), keeps no artifact and publishes nothing. Start it by hand on any branch with
  `gh workflow run bundle.yml --ref <branch>`.
- [`release.yml`](../.github/workflows/release.yml) runs `build.yml` for a `v*` tag, or by hand to
  try a release without publishing it; it keeps the packages and the release notes as workflow
  artifacts. Only for a tag does its publishing job (the only one with write access) compute
  `SHA256SUMS.txt`, attach build-provenance attestations and publish the GitHub release.

Every workflow has read-only permissions unless a job needs more, uses actions pinned to commit
SHAs (kept current by Dependabot), and runs in a concurrency group: a newer push cancels an older
check, while release runs never cancel each other.

## Website

The project website, https://bugraskl.github.io/eye-tracker/, is built from [`site/`](../site/)
(an English and a Turkish page sharing one stylesheet and script). It loads nothing from other
sites: no web fonts, no analytics, no cookies. The download buttons are filled in from the latest
GitHub release at build time, so [`pages.yml`](../.github/workflows/pages.yml) rebuilds and
deploys it on every published release and on changes to the site.

To build and preview it locally:

```bash
gh api repos/bugraskl/eye-tracker/releases/latest > release.json   # optional
cargo xtask build-site --release release.json --out _site
python3 -m http.server 8000 --directory _site
```

## Releasing

1. Update `version` under `[workspace.package]` in the root `Cargo.toml` (`cargo build` then updates
   `Cargo.lock`) and move the **Unreleased** notes in
   `CHANGELOG.md` under a new `## [x.y.z] - YYYY-MM-DD` heading, leaving `## [Unreleased]` empty.
2. Commit, then tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
   The installed apps look for `EyeTracker-<version>-windows-x64-setup.exe` and `SHA256SUMS.txt` in
   the latest release to update themselves ([`update/release.rs`](../crates/eye-tracker/src/update/release.rs),
   checked by its tests): keep those two names when you change `release.yml`.
3. The release workflow builds and tests every package as described above and publishes the GitHub
   release with the notes from `CHANGELOG.md`. It fails before building anything when the tag does
   not match the version in `Cargo.toml`, when `CHANGELOG.md` has no section for the version, or when notes are
   still under **Unreleased** (they would be missing from the release notes). A run that is not
   for a tag only reports the two `CHANGELOG.md` problems.
