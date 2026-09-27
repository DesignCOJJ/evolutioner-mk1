#!/usr/bin/env bash
# ============================================================================
#  Evolutioner MK1 — dedicated downloader / bootstrap
#
#  One-liner:
#    curl -fsSL https://raw.githubusercontent.com/wallpiller-lm/EvolutionerMK1/main/get.sh | bash
#
#  Idempotent — safe to re-run to update an existing install.
#  Env overrides:
#    EVOLUTIONER_DIR=~/evolutioner-mk1   install location
#    EVOLUTIONER_REPO=<git url>          alternate repo
#    SKIP_DEPS=1                         skip pip install
# ============================================================================
set -euo pipefail

REPO_URL="${EVOLUTIONER_REPO:-https://github.com/wallpiller-lm/EvolutionerMK1.git}"
DIR="${EVOLUTIONER_DIR:-$HOME/evolutioner-mk1}"
BRANCH="main"

C=$'\e[1;36m'; G=$'\e[1;32m'; Y=$'\e[1;33m'; R=$'\e[1;31m'; D=$'\e[2m'; N=$'\e[0m'
step() { printf "\n${C}==>${N} %s\n" "$1"; }
ok()   { printf "  ${G}✔${N} %s\n" "$1"; }
warn() { printf "  ${Y}!${N} %s\n" "$1"; }
die()  { printf "  ${R}✘ %s${N}\n" "$1"; exit 1; }

printf "${C}"
cat <<'BANNER'
  ▗▄▄▖ ▗▄▄▖  ▄▄▄  ▄▄▄▖ ▗▄▖ ▗▄▄▖  ▗▄▄▖  ▄▄▄  ▄▄▄▖ ▗▄▖
 ▐▛   ▘▐▛ ▜▌█   █ █▄▄▘█▛▜▌▐▛ ▜▌▐▛   ▘█   █ █▄▄▘█▛▜▌
 ▐▙▄▄▖▐▛ ▜▌▐▌  █ █▄▄▖█▌ ▐▌▐▙▄▟▌▐▙▄▄▖▐▌  ▐▌█▄▄▖█▌ ▐▌
  ▝▀▀▘▝▘  ▘ ▀▀▀  ▝▀▀ ▝▘ ▝▘ ▝▀▀▘ ▝▀▀▘ ▀▀▀  ▝▀▀ ▝▘ ▝▘
BANNER
printf "${N}${D}   downloader / bootstrap · wallpiller-lm/EvolutionerMK1${N}\n"

# ── 1. repository ────────────────────────────────────────────────────────────
step "1/5 repository → ${DIR}"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch origin "$BRANCH" --quiet
  git -C "$DIR" reset --hard "origin/$BRANCH" --quiet
  ok "updated to $(git -C "$DIR" rev-parse --short HEAD)"
else
  git clone --depth 1 -b "$BRANCH" "$REPO_URL" "$DIR"
  ok "cloned $(git -C "$DIR" rev-parse --short HEAD)"
fi
cd "$DIR"

# ── 2. python env ────────────────────────────────────────────────────────────
step "2/5 python environment"
command -v python3 >/dev/null || die "python3 not found (need 3.10+)"
[ -x env/bin/python ] || python3 -m venv env
if [ "${SKIP_DEPS:-0}" != "1" ]; then
  env/bin/pip install --upgrade pip --quiet
  env/bin/pip install -r requirements.txt --quiet
  ok "dependencies installed (llama-cpp-python may compile on first run)"
else
  warn "SKIP_DEPS=1 — reusing existing env"
fi
export PYTHONPATH="$DIR/src" PYTHONUNBUFFERED=1

# ── 3. ollama models ─────────────────────────────────────────────────────────
step "3/5 ollama models (primary source: your local ollama store)"
if ! command -v ollama >/dev/null; then
  warn "ollama CLI not found — install it (https://ollama.com), then re-run this script"
else
  if ! curl -s --max-time 2 http://localhost:11434/api/version >/dev/null 2>&1; then
    printf "  ${D}starting ollama serve...${N}\n"
    (setsid ollama serve >/tmp/evolutioner_ollama.log 2>&1 < /dev/null &)
    for _ in $(seq 1 20); do
      curl -s --max-time 2 http://localhost:11434/api/version >/dev/null 2>&1 && break
      sleep 0.5
    done
  fi
  TAGS="$(ollama list | awk 'NR>1{print $1}')"
  has() { printf "%s\n" "$TAGS" | grep -Eiq "$1" && return 0 || return 1; }

  if has "(evolutionermk1|evolu)" && has "(evolutionermk1|evolu).*3b"; then
    ok "portfolio detected in ollama store (tag resolver maps names automatically):"
    printf "%s\n" "$TAGS" | grep -Ei "(evolutionermk1|evolu)" | sed 's/^/      /'
  else
    warn "portfolio models not found locally — trying registry pull"
    if ollama pull wallpillar-lm/evolutionermk1-3B:latest \
       && ollama pull wallpillar-lm/evolutionermk1-1.5B:latest; then
      ok "pulled 3B + 1.5B from registry"
    else
      printf "  ${Y}registry pull unavailable.${N} Import from GGUF files instead:\n"
      printf "      ${D}printf 'FROM ./your-model.gguf\\n' > /tmp/mf && ollama create wallpillar-lm/evolu-general-3B -f /tmp/mf${N}\n"
      printf "      ${D}(or use the launcher: bash launch.sh → menu 8 → 2)${N}\n"
    fi
  fi

  # ── 4. integrated model (methodology baked in via Modelfile) ──────────────
  step "4/5 integrated model (evolutionermk1-integrated-3B)"
  if printf "%s" "$TAGS" | grep -Eqi "integrated-3b"; then
    ok "already present"
  elif has "(evolutionermk1|evolu).*3b"; then
    ollama create evolutionermk1-integrated-3B -f Modelfile >/dev/null \
      && ok "created from Modelfile (MCTS-style reasoning + python verification baked in)"
  else
    warn "skipped — needs a base 3B model first"
  fi
fi

# ── 5. verify ────────────────────────────────────────────────────────────────
step "5/5 verify"
SELFTEST=0
if env/bin/python -m evolutioner.cli selftest; then SELFTEST=1; else warn "selftest reported failures — see above"; fi
printf "\n"
env/bin/python -m evolutioner.cli status || true

printf "\n"
if [ "$SELFTEST" = "1" ]; then
  printf "${G}  ✔ Evolutioner MK1 is ready.${N}\n"
else
  printf "${R}  ✘ install incomplete${N} ${D}(most common cause: missing deps — re-run without SKIP_DEPS)${N}\n"
  exit 1
fi
printf "    Run it:         bash launch.sh          (menu 1–9)\n"
printf "    Chat REPL:      menu 9   ·   /solve <q> = full MCTS + verified harness\n"
printf "    Headless:       PYTHONPATH=src env/bin/python -m evolutioner.cli query \"7*6? Verify.\"\n"
