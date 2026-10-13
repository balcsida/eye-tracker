#!/usr/bin/env bash
# Assemble "Eye Tracker.app" from the release build.
#
# Usage (from the repository root, after `cargo build --release` and `cargo xtask make-icons`):
#   bash packaging/macos/make_app.sh [--bin-dir target/release] [--out "dist/Eye Tracker.app"]
#        [--version X.Y.Z]
#
# Defaults: --bin-dir $CARGO_TARGET_DIR/release (or target/release), --out dist/Eye Tracker.app,
# --version from the workspace Cargo.toml.
#
# The app holds both executables of the build:
#   Contents/MacOS/Eye Tracker       the windowed menu-bar app (eye-tracker-gui)
#   Contents/MacOS/eye-tracker-cli   the command-line tool (eye-tracker)
# The names are the ones cli_command and app_command in crates/eye-tracker/src/cli.rs look for.
# The face models are compiled into the executables; their NOTICE.md and licence texts go to
# Contents/Resources, next to the project's LICENSE.
#
# The oldest macOS the app runs on is $MACOS_MINIMUM (default 14.0); the same value should be
# exported as MACOSX_DEPLOYMENT_TARGET for `cargo build`. The result is not signed:
# make_dmg.sh does that.
set -euo pipefail

die() { echo "make_app: $*" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_DIR="${CARGO_TARGET_DIR:-$ROOT/target}/release"
APP="$ROOT/dist/Eye Tracker.app"
VERSION=""
MACOS_MINIMUM="${MACOS_MINIMUM:-14.0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin-dir) BIN_DIR="${2:?--bin-dir needs a path}"; shift 2 ;;
    --out) APP="${2:?--out needs a path}"; shift 2 ;;
    --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/Cargo.toml" | head -n 1)"
  [[ -n "$VERSION" ]] || die "cannot read the version from Cargo.toml"
fi
# CFBundleVersion must be purely numeric.
NUMERIC="$(grep -oE '^[0-9]+(\.[0-9]+){0,2}' <<<"$VERSION" || true)"
[[ -n "$NUMERIC" ]] || die "not a version number: $VERSION"
APP_ID="$(sed -n 's/^pub const APP_ID: &str = "\(.*\)";$/\1/p' "$ROOT/crates/eye-tracker/src/lib.rs")"
[[ -n "$APP_ID" ]] || die "cannot read APP_ID from crates/eye-tracker/src/lib.rs"

MODELS="$ROOT/crates/eye-tracker/models"
ICON="$ROOT/packaging/macos/eye-tracker.icns"
[[ -x "$BIN_DIR/eye-tracker-gui" ]] || die "missing $BIN_DIR/eye-tracker-gui (run: cargo build --release)"
[[ -x "$BIN_DIR/eye-tracker" ]] || die "missing $BIN_DIR/eye-tracker (run: cargo build --release)"
[[ -f "$ICON" ]] || die "missing $ICON; run: cargo xtask make-icons"
[[ -f "$MODELS/NOTICE.md" && -d "$MODELS/licenses" ]] || die "missing the models' NOTICE.md or licenses/ in $MODELS"

echo "==> Assembling $APP ($VERSION)"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
install -m 755 "$BIN_DIR/eye-tracker-gui" "$APP/Contents/MacOS/Eye Tracker"
install -m 755 "$BIN_DIR/eye-tracker" "$APP/Contents/MacOS/eye-tracker-cli"
install -m 644 "$ICON" "$APP/Contents/Resources/eye-tracker.icns"
install -m 644 "$ROOT/LICENSE" "$APP/Contents/Resources/LICENSE"
install -m 644 "$MODELS/NOTICE.md" "$APP/Contents/Resources/NOTICE.md"
cp -R "$MODELS/licenses" "$APP/Contents/Resources/licenses"

# LSUIElement: a menu-bar (tray) app has no Dock icon, but it does show UI.
cat >"$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>Eye Tracker</string>
  <key>CFBundleIconFile</key>
  <string>eye-tracker</string>
  <key>CFBundleIdentifier</key>
  <string>$APP_ID</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>Eye Tracker</string>
  <key>CFBundleDisplayName</key>
  <string>Eye Tracker</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>$VERSION</string>
  <key>CFBundleVersion</key>
  <string>$NUMERIC</string>
  <key>LSApplicationCategoryType</key>
  <string>public.app-category.productivity</string>
  <key>LSBackgroundOnly</key>
  <false/>
  <key>LSMinimumSystemVersion</key>
  <string>$MACOS_MINIMUM</string>
  <key>LSUIElement</key>
  <true/>
  <key>NSCameraUsageDescription</key>
  <string>Eye Tracker uses the camera to see which monitor you are looking at and whether you are at your desk. Frames are analysed on this Mac and are never saved or sent anywhere.</string>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSHumanReadableCopyright</key>
  <string>Copyright © 2026 Buğra Şıkel. MIT License.</string>
  <key>NSSupportsAutomaticGraphicsSwitching</key>
  <true/>
</dict>
</plist>
PLIST
plutil -lint "$APP/Contents/Info.plist" >/dev/null
echo "==> Done: $APP"
