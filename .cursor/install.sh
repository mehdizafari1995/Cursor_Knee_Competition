#!/usr/bin/env bash
# Idempotent dependency bootstrap for the Cursor Knee Competition project.
# Safe to run repeatedly: apt install is a no-op when packages are present,
# venv creation skips an existing environment, and pip reconciles requirements.
set -euo pipefail

cd "$(dirname "$0")/.."

# python3-venv is not part of the base image and is required to create the
# project virtual environment.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-pip
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "Install complete. Activate with: source .venv/bin/activate"
