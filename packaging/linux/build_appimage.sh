#!/usr/bin/env bash
# Package the Linux release build as an AppImage and a tarball.
#
# Usage (from the repository root, after `cargo build --release` and `cargo xtask make-icons`):
#   bash packaging/linux/build_appimage.sh [--version X.Y.Z] [--dist DIR] [--binary PATH]
#        [--appimagetool PATH] [--runtime-file PATH] [--allow-unpinned]
#
# Writes to DIST (default: dist/):
#   EyeTracker-<version>-linux-<arch>.AppImage
#   EyeTracker-<version>-linux-<arch>.tar.gz   (eye-tracker/ with the executable, desktop file,
#                                              icon, licence, models' NOTICE.md and licenses/)
#
# appimagetool is taken from --appimagetool, $APPIMAGETOOL or PATH; if none is
# found, the pinned release below is downloaded into build/tools/ and verified
# against its SHA-256. The AppImage runtime (the ELF header users execute first)
# is pinned the same way and passed with --runtime-file, so appimagetool never
# embeds whatever its "continuous" channel serves at build time. Override with
# --runtime-file / $APPIMAGE_RUNTIME (a local file, checked if
# $APPIMAGE_RUNTIME_SHA256 is set) or $APPIMAGETOOL_URL + $APPIMAGETOOL_SHA256.
# A download without a checksum is refused unless --allow-unpinned is given.
# Where FUSE is unavailable (containers, CI) export APPIMAGE_EXTRACT_AND_RUN=1.
# Set UPDATE_INFORMATION to embed AppImage update information; a .zsync file is
# written next to the AppImage when appimagetool can produce one.
#
# --binary defaults to $CARGO_TARGET_DIR/release/eye-tracker (or target/release/eye-tracker). The
# version defaults to the workspace Cargo.toml's. The face models are compiled into the
# executable, so the packages hold no other library; the system's X11 or Wayland and
# OpenGL libraries are loaded at run time.
set -euo pipefail

die() { echo "build_appimage: $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DIST="$ROOT/dist"
VERSION=""
APPIMAGETOOL="${APPIMAGETOOL:-}"
RUNTIME_FILE="${APPIMAGE_RUNTIME:-}"
ALLOW_UNPINNED=0
BINARY="${CARGO_TARGET_DIR:-$ROOT/target}/release/eye-tracker"

# Pinned third-party build tools. To update: pick a tagged release, and copy the
# SHA-256 of each asset from the release page (or `sha256sum` a downloaded copy).
APPIMAGETOOL_VERSION="1.9.1"        # https://github.com/AppImage/appimagetool/releases
APPIMAGE_RUNTIME_VERSION="20251108" # https://github.com/AppImage/type2-runtime/releases

pinned_sha256() {
  case "$1:$2" in
    appimagetool:x86_64) echo "ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0" ;;
    appimagetool:aarch64) echo "f0837e7448a0c1e4e650a93bb3e85802546e60654ef287576f46c71c126a9158" ;;
    runtime:x86_64) echo "2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d" ;;
    runtime:aarch64) echo "00cbdfcf917cc6c0ff6d3347d59e0ca1f7f45a6df1a428a0d6d8a78664d87444" ;;
    *) return 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
    --dist) DIST="${2:?--dist needs a path}"; shift 2 ;;
    --binary) BINARY="${2:?--binary needs a path}"; shift 2 ;;
    --appimagetool) APPIMAGETOOL="${2:?--appimagetool needs a path}"; shift 2 ;;
    --runtime-file) RUNTIME_FILE="${2:?--runtime-file needs a path}"; shift 2 ;;
    --allow-unpinned) ALLOW_UNPINNED=1; shift ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || die "this script builds Linux packages; run it on Linux"

if [[ -z "$VERSION" ]]; then
  VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/Cargo.toml" | head -n 1)"
  [[ -n "$VERSION" ]] || die "cannot read the version from Cargo.toml"
fi
ARCH="${ARCH:-$(uname -m)}"
DIST="$(mkdir -p "$DIST" && cd "$DIST" && pwd)"

MODELS="$ROOT/crates/eye-tracker/models"
DESKTOP="$HERE/eye-tracker.desktop"
ICON="$HERE/eye-tracker.png"
[[ -x "$BINARY" ]] || die "executable not found: $BINARY (run: cargo build --release)"
[[ -f "$ICON" ]] || die "missing $ICON; run: cargo xtask make-icons"
[[ -f "$MODELS/NOTICE.md" && -d "$MODELS/licenses" ]] || die "missing the models' NOTICE.md or licenses/ in $MODELS"

BASENAME="EyeTracker-$VERSION-linux-$ARCH"
TARBALL="$DIST/$BASENAME.tar.gz"
APPIMAGE="$DIST/$BASENAME.AppImage"

# ------------------------------------------------------------------ build tools
is_verified() {
  local file="$1" sha256="$2"
  [[ -f "$file" ]] && echo "$sha256  $file" | sha256sum --check --status
}

# Download URL to DEST (atomically) unless a copy with the expected checksum is
# already there. An empty checksum is refused unless --allow-unpinned was given.
fetch_verified() {
  local url="$1" dest="$2" sha256="$3"
  if [[ -n "$sha256" ]] && is_verified "$dest" "$sha256"; then
    return 0
  fi
  if [[ -z "$sha256" ]]; then
    [[ "$ALLOW_UNPINNED" == 1 ]]       || die "refusing to download $url without a SHA-256 (set one, or pass --allow-unpinned)"
    echo "build_appimage: warning: $url is not verified (--allow-unpinned)" >&2
  fi
  command -v curl >/dev/null 2>&1 || die "curl is needed to download $url"
  echo "==> Downloading $url"
  mkdir -p "$(dirname "$dest")"
  curl --fail --location --silent --show-error --retry 3 --output "$dest.part" "$url"
  if [[ -n "$sha256" ]] && ! is_verified "$dest.part" "$sha256"; then
    rm -f "$dest.part"
    die "checksum mismatch for $url (expected $sha256)"
  fi
  mv "$dest.part" "$dest"
}

find_appimagetool() {
  if [[ -n "$APPIMAGETOOL" ]]; then
    if [[ -x "$APPIMAGETOOL" ]]; then return 0; fi
    if command -v "$APPIMAGETOOL" >/dev/null 2>&1; then
      APPIMAGETOOL="$(command -v "$APPIMAGETOOL")"
      return 0
    fi
    die "appimagetool not found: $APPIMAGETOOL"
  fi
  if command -v appimagetool >/dev/null 2>&1; then
    APPIMAGETOOL="$(command -v appimagetool)"
    return 0
  fi

  local url sha256 name
  if [[ -n "${APPIMAGETOOL_URL:-}" ]]; then
    url="$APPIMAGETOOL_URL"
    sha256="${APPIMAGETOOL_SHA256:-}"
    name="appimagetool-custom-$ARCH.AppImage"
  else
    url="https://github.com/AppImage/appimagetool/releases/download/$APPIMAGETOOL_VERSION/appimagetool-$ARCH.AppImage"
    sha256="${APPIMAGETOOL_SHA256:-$(pinned_sha256 appimagetool "$ARCH" || true)}"
    name="appimagetool-$APPIMAGETOOL_VERSION-$ARCH.AppImage"
  fi
  APPIMAGETOOL="$ROOT/build/tools/$name"
  fetch_verified "$url" "$APPIMAGETOOL" "$sha256"
  chmod 755 "$APPIMAGETOOL"
}

find_runtime() {
  if [[ -n "$RUNTIME_FILE" ]]; then
    [[ -f "$RUNTIME_FILE" ]] || die "AppImage runtime not found: $RUNTIME_FILE"
    if [[ -n "${APPIMAGE_RUNTIME_SHA256:-}" ]]; then
      is_verified "$RUNTIME_FILE" "$APPIMAGE_RUNTIME_SHA256"         || die "checksum mismatch for $RUNTIME_FILE (expected $APPIMAGE_RUNTIME_SHA256)"
    fi
    return 0
  fi
  local sha256
  sha256="${APPIMAGE_RUNTIME_SHA256:-$(pinned_sha256 runtime "$ARCH" || true)}"
  RUNTIME_FILE="$ROOT/build/tools/runtime-$APPIMAGE_RUNTIME_VERSION-$ARCH"
  fetch_verified     "https://github.com/AppImage/type2-runtime/releases/download/$APPIMAGE_RUNTIME_VERSION/runtime-$ARCH"     "$RUNTIME_FILE" "$sha256"
}

find_appimagetool
find_runtime

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eye-tracker-appimage.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# ------------------------------------------------------------------------ tarball
echo "==> Creating $TARBALL"
mkdir -p "$WORK/tar/eye-tracker"
install -m 755 "$BINARY" "$WORK/tar/eye-tracker/eye-tracker"
install -m 644 "$MODELS/NOTICE.md" "$WORK/tar/eye-tracker/NOTICE.md"
cp -R "$MODELS/licenses" "$WORK/tar/eye-tracker/licenses"
install -m 644 "$DESKTOP" "$WORK/tar/eye-tracker/eye-tracker.desktop"
install -m 644 "$ICON" "$WORK/tar/eye-tracker/eye-tracker.png"
install -m 644 "$ROOT/LICENSE" "$WORK/tar/eye-tracker/LICENSE"
rm -f "$TARBALL"
tar -C "$WORK/tar" --sort=name --owner=0 --group=0 --numeric-owner -czf "$TARBALL" eye-tracker

# ------------------------------------------------------------------------ AppImage
echo "==> Assembling AppDir"
APPDIR="$WORK/EyeTracker.AppDir"
mkdir -p \
  "$APPDIR/usr/bin" \
  "$APPDIR/usr/share/eye-tracker" \
  "$APPDIR/usr/share/applications" \
  "$APPDIR/usr/share/icons/hicolor/512x512/apps"
install -m 755 "$BINARY" "$APPDIR/usr/bin/eye-tracker"
install -m 755 "$HERE/AppRun" "$APPDIR/AppRun"
install -m 644 "$DESKTOP" "$APPDIR/eye-tracker.desktop"
install -m 644 "$DESKTOP" "$APPDIR/usr/share/applications/eye-tracker.desktop"
install -m 644 "$ICON" "$APPDIR/eye-tracker.png"
install -m 644 "$ICON" "$APPDIR/usr/share/icons/hicolor/512x512/apps/eye-tracker.png"
install -m 644 "$ROOT/LICENSE" "$APPDIR/usr/share/eye-tracker/LICENSE"
install -m 644 "$MODELS/NOTICE.md" "$APPDIR/usr/share/eye-tracker/NOTICE.md"
cp -R "$MODELS/licenses" "$APPDIR/usr/share/eye-tracker/licenses"
ln -s eye-tracker.png "$APPDIR/.DirIcon"

echo "==> Running appimagetool ($APPIMAGETOOL)"
# The pinned runtime becomes the AppImage's entry point.
TOOL_ARGS=(--no-appstream --runtime-file "$RUNTIME_FILE")
if [[ -n "${UPDATE_INFORMATION:-}" ]]; then
  TOOL_ARGS+=(--updateinformation "$UPDATE_INFORMATION")
fi
rm -f "$APPIMAGE" "$APPIMAGE.zsync"
# appimagetool writes the .zsync file into the working directory.
(cd "$DIST" && ARCH="$ARCH" VERSION="$VERSION" "$APPIMAGETOOL" "${TOOL_ARGS[@]}" "$APPDIR" "$APPIMAGE")
chmod 755 "$APPIMAGE"

echo "==> Done"
ls -lh "$TARBALL" "$APPIMAGE"
if [[ -f "$APPIMAGE.zsync" ]]; then ls -lh "$APPIMAGE.zsync"; fi
