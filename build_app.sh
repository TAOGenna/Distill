#!/bin/bash
# Build standalone Distill.app using PyInstaller
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Cleaning previous build..."
rm -rf build/ dist/

echo "==> Building Distill.app..."
.venv/bin/python -m PyInstaller distill.spec --noconfirm

echo ""
echo "==> Done! App is at: dist/Distill.app"
echo "    To install: cp -R dist/Distill.app /Applications/"
echo "    Size: $(du -sh dist/Distill.app | cut -f1)"
