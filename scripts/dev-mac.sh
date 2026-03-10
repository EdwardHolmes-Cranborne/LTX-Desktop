#!/bin/bash
# Build and launch LTX Desktop on macOS (Apple Silicon)
# Handles code signing issues that cause crashes on macOS 26+

set -e

cd "$(dirname "$0")/.."

MODE="${1:-release}"

if [ "$MODE" = "release" ]; then
    APP_PATH="release/mac-arm64/LTX Desktop.app"

    echo "Building release..."
    pnpm build:mac:skip-python

    echo "Re-signing all binaries (fixes Team ID mismatch crash on macOS 26+)..."
    find "$APP_PATH" -type f \( -name "*.dylib" -o -name "*.so" -o -name "python3*" \) -exec codesign --force --sign - {} \;
    codesign --force --deep --sign - "$APP_PATH"

    echo "Launching..."
    open "$APP_PATH"
elif [ "$MODE" = "dev" ]; then
    echo "Starting dev server..."
    exec pnpm dev
else
    echo "Usage: $0 [release|dev]"
    exit 1
fi
