#!/usr/bin/env bash
# Write release-notes.md (in the current folder) from the version's section of CHANGELOG.md
# plus the download table and the notes every release carries.
#
# Usage (from the repository root; .github/workflows/build.yml runs it):
#   VERSION=X.Y.Z MACOS_MINIMUM=14.0 [GITHUB_REF_TYPE=tag] [MACOS_STABLE_SIGNATURE=true] \
#     bash packaging/release_notes.sh
#
# For a tag it fails when CHANGELOG.md has no non-empty "## [X.Y.Z] - YYYY-MM-DD" section or
# still has notes under "## [Unreleased]" (a tag ships every change made before it, so they
# would be missing here and announced again with the next version). Any other run only
# reports both and writes "Development build." when the section is missing.
set -euo pipefail

: "${VERSION:?VERSION is required}"
: "${MACOS_MINIMUM:?MACOS_MINIMUM is required}"
is_tag=false
if [[ "${GITHUB_REF_TYPE:-}" == "tag" ]]; then is_tag=true; fi
docs="https://github.com/bugraskl/eye-tracker/blob/main/docs"

# Keep a Changelog: a section starts at "## [name]" and ends at the next "## " heading or at
# the link reference definitions ("[X.Y.Z]: url").
section() {
  awk -v head="## [$1]" '
    index($0, head) == 1 { on = 1; next }
    on && (/^## / || /^\[[^]]+\]:/) { exit }
    on { print }' CHANGELOG.md | sed -e '/./,$!d'
}

report() { # level message
  echo "::$1 file=CHANGELOG.md::$2"
}

# Notes under [Unreleased]; Keep a Changelog's empty "### Added" headings are no notes.
pending="$(section Unreleased)"
if grep -Eqv '^[[:space:]]*(#|$)' <<<"$pending"; then
  message="CHANGELOG.md still has notes under '## [Unreleased]': move them under '## [$VERSION] - YYYY-MM-DD' before tagging v$VERSION"
  if $is_tag; then
    report error "$message"
    exit 1
  fi
  report notice "$message"
fi

body="$(section "$VERSION")"
if [[ -z "${body//[[:space:]]/}" ]]; then
  message="CHANGELOG.md has no non-empty '## [$VERSION] - YYYY-MM-DD' section"
  if $is_tag; then
    report error "$message"
    exit 1
  fi
  report warning "$message"
  body="Development build."
fi

{
  printf '%s\n\n' "$body"
  cat <<FOOTER
### Downloads

| Platform | File |
|---|---|
| Windows 10/11 (x64), installer | \`EyeTracker-$VERSION-windows-x64-setup.exe\` |
| Windows 10/11 (x64), portable | \`EyeTracker-$VERSION-windows-x64-portable.zip\` |
| macOS ${MACOS_MINIMUM%.0} or later (Apple silicon) | \`EyeTracker-$VERSION-macos-arm64.dmg\` |
| Linux (x86_64), AppImage | \`EyeTracker-$VERSION-linux-x86_64.AppImage\` |
| Linux (x86_64), tarball | \`EyeTracker-$VERSION-linux-x86_64.tar.gz\` |

Verify a download with \`SHA256SUMS.txt\`, e.g. \`sha256sum -c SHA256SUMS.txt --ignore-missing\`,
or its build provenance with \`gh attestation verify <file> --repo bugraskl/eye-tracker\`.

Command line (\`doctor\`, \`ctl\`, \`bench\`…): \`eye-tracker\` in a new terminal after
installing on Windows (the installer adds it to PATH unless you untick that option;
the portable ZIP has \`eye-tracker-cli.exe\`),
\`"/Applications/Eye Tracker.app/Contents/MacOS/eye-tracker-cli"\` on macOS, and the
\`.AppImage\` file itself (or \`eye-tracker/eye-tracker\` from the tarball) on Linux.

The builds are not signed with a paid certificate: Windows SmartScreen may ask you to
confirm ("More info" → "Run anyway"), and on macOS the first launch needs
"Open Anyway" in System Settings → Privacy & Security.

FOOTER
  if [[ "${MACOS_STABLE_SIGNATURE:-false}" == "true" ]]; then
    cat <<'SIGNED'
**Updating on macOS:** this build is signed with the project's stable certificate,
so the Camera and Accessibility permissions carry over to later updates. Coming
from an ad-hoc-signed version, re-grant them once: in System Settings → Privacy &
Security → Accessibility, remove Eye Tracker with "−" and add it again.
SIGNED
  else
    cat <<'ADHOC'
**Updating on macOS:** this build is signed ad hoc, so macOS treats every version
as a new app. It asks for camera access again, and the Accessibility switch for
Eye Tracker still looks on but no longer applies: if keyboard focus stops following
your gaze, open System Settings → Privacy & Security → Accessibility, remove
Eye Tracker with "−" and add it again.
ADHOC
  fi
  cat <<LINKS

Help: [troubleshooting]($docs/troubleshooting.md) ·
[platform support]($docs/platform-support.md) · [privacy]($docs/privacy.md)
LINKS
} > release-notes.md
