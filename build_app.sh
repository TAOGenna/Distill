#!/bin/bash
# Build Distill.app — a lightweight macOS wrapper that auto-updates on launch.
# The app points back to this cloned repo, so it always runs the latest code.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$REPO_DIR/dist/Distill.app"
ICON="$REPO_DIR/AppIcon.icns"

echo "==> Building Distill.app..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/"{MacOS,Resources}

# ── Launcher script ──────────────────────────────────────────────────────────
cat > "$APP_DIR/Contents/MacOS/Distill" << 'LAUNCHER'
#!/bin/bash
# Distill launcher — pulls latest code and starts the web UI.

REPO_DIR="__REPO_DIR__"

if [ ! -d "$REPO_DIR/.git" ]; then
    osascript -e 'display alert "Distill" message "Repo not found at '"$REPO_DIR"'.\nDid you move or delete it?" as critical'
    exit 1
fi

cd "$REPO_DIR"

# Pull latest (skip if offline or repo is dirty)
if git diff --quiet 2>/dev/null; then
    git pull --ff-only 2>/dev/null || true
fi

# Sync deps (no-op if lock file unchanged)
uv sync --quiet 2>/dev/null || true

exec .venv/bin/python -m distill "$@"
LAUNCHER

# Inject the actual repo path
sed -i '' "s|__REPO_DIR__|$REPO_DIR|g" "$APP_DIR/Contents/MacOS/Distill"
chmod +x "$APP_DIR/Contents/MacOS/Distill"

# ── Info.plist ───────────────────────────────────────────────────────────────
cat > "$APP_DIR/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Distill</string>
    <key>CFBundleDisplayName</key>
    <string>Distill</string>
    <key>CFBundleIdentifier</key>
    <string>com.kenyi.distill</string>
    <key>CFBundleVersion</key>
    <string>0.3.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.3.0</string>
    <key>CFBundleExecutable</key>
    <string>Distill</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

# ── Icon ─────────────────────────────────────────────────────────────────────
if [ -f "$ICON" ]; then
    cp "$ICON" "$APP_DIR/Contents/Resources/AppIcon.icns"
fi

echo "==> Done! App is at: dist/Distill.app"
echo "    To install: cp -R dist/Distill.app /Applications/"
echo "    On launch it will git pull + uv sync automatically."
