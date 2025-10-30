#!/usr/bin/env bash
set -e

OS="$(uname -s)"

if [[ "$OS" == "Linux"* ]]; then
    echo "Detected Linux system. Installing LibreOffice (if not installed) and openpyxl..."
    if ! command -v libreoffice >/dev/null 2>&1; then
        sudo apt update -qq
        sudo apt install -y libreoffice >/dev/null
    fi
    uv pip install openpyxl
elif [[ "$OS" == "Darwin"* || "$OS" == MINGW* || "$OS" == MSYS* || "$OS" == CYGWIN* ]]; then
    echo "Detected macOS/Windows system. Installing xlwings and openpyxl..."
    uv pip install openpyxl xlwings
else
    echo "Unsupported OS: $OS" >&2
    exit 1
fi
