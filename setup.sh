#!/usr/bin/env bash
# Evolutioner MK1 setup — Arch/CachyOS fish syntax from the spec, portable bash here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "[1/4] Checking system packages (bubblewrap optional but recommended)..."
if ! command -v bwrap >/dev/null 2>&1; then
  echo "  ! bubblewrap not found. Install it for the deterministic verifier sandbox:"
  echo "      sudo pacman -S bubblewrap     (Arch/CachyOS)"
  echo "      sudo apt install bubblewrap   (Debian/Ubuntu)"
  echo "  (continuing with resource-limited subprocess fallback)"
fi

echo "[2/4] Creating venv..."
python3 -m venv env
source env/bin/activate

echo "[3/4] Installing dependencies (llama-cpp-python may compile a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] Running selftest with mock backend (no model download)..."
python -m evolutioner.cli selftest

cat <<'EOF'

Setup complete.
  Activate:        source env/bin/activate
  Run (mock):      python -m evolutioner.cli selftest
  Real model:      python -m evolutioner.cli query "What is 12*12?" --config config/default.json
  Download check:  python -m evolutioner.cli models
  Usage stats:     python -m evolutioner.cli usage
EOF
