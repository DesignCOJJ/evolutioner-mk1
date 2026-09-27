#!/usr/bin/env bash
# ============================================================================
#  Migrate the Ollama store from ~/.ollama/models to /var/lib/ollama
#  (matches the systemd unit: User=ollama, OLLAMA_MODELS=/var/lib/ollama)
#
#  Run AFTER all model downloads/imports are finished. Safe: copies first,
#  verifies, and only then offers to remove the old store.
#  Usage:  bash scripts/migrate_store.sh
# ============================================================================
set -euo pipefail

SRC="$HOME/.ollama/models"
DST="/var/lib/ollama/models"

G=$'\e[1;32m'; Y=$'\e[1;33m'; R=$'\e[1;31m'; N=$'\e[0m'

[ -d "$SRC" ] || { echo "no source store at $SRC — nothing to migrate"; exit 0; }
[ "$(id -u)" = "0" ] || { echo "run with sudo:  sudo bash scripts/migrate_store.sh"; exit 1; }

echo "1/5 stopping any running ollama (systemd + manual)..."
systemctl stop ollama 2>/dev/null || true
pkill -x ollama 2>/dev/null || true
sleep 2

echo "2/5 copying store  $SRC -> $DST ..."
mkdir -p "$DST/blobs" "$DST/manifests"
cp -a "$SRC/blobs/."   "$DST/blobs/"
cp -a "$SRC/manifests/." "$DST/manifests/"
chown -R ollama:ollama /var/lib/ollama

echo "3/5 starting systemd service (OLLAMA_MODELS=/var/lib/ollama)..."
systemctl start ollama
for _ in $(seq 1 20); do
  curl -s --max-time 2 http://localhost:11434/api/version >/dev/null 2>&1 && break
  sleep 0.5
done

echo "4/5 verifying..."
COUNT=$(ollama list | tail -n +2 | wc -l)
if [ "$COUNT" -eq 0 ]; then
  echo -e "${R}  ✘ no models visible — DO NOT delete the old store. Investigate first.${N}"
  exit 1
fi
ollama list

echo "5/5 old store at $SRC is now redundant."
echo -e "${Y}  verify the list above, then remove it manually:${N}"
echo  "    rm -rf $SRC"
echo -e "${G}  ✔ migration complete — future pulls/creates land in /var/lib/ollama${N}"
