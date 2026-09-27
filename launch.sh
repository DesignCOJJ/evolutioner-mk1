#!/usr/bin/env bash
# Evolutioner MK1 — unified app launcher (harness + Ollama control)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="env/bin/python"
[ -x "$PY" ] || { echo "run bash setup.sh first"; exit 1; }
export PYTHONPATH="$ROOT/src"

C=$'\e[1;36m'; M=$'\e[1;35m'; G=$'\e[1;32m'; Y=$'\e[1;33m'; R=$'\e[1;31m'; D=$'\e[2m'; N=$'\e[0m'

ollama_up() { curl -s --max-time 2 http://localhost:11434/api/version >/dev/null 2>&1; }
ensure_ollama() {
  if ollama_up; then return 0; fi
  printf "${Y}starting ollama serve...${N}\n"
  (setsid ollama serve >/tmp/evolutioner_ollama.log 2>&1 < /dev/null &)
  for _ in $(seq 1 20); do ollama_up && return 0; sleep 0.5; done
  printf "${R}ollama failed to start — see /tmp/evolutioner_ollama.log${N}\n"; return 1
}

menu() {
  clear
  printf "${C}"
  cat <<'EOF'
  ▗▄▄▖ ▗▄▄▖  ▄▄▄  ▄▄▄▖ ▗▄▖ ▗▄▄▖  ▗▄▄▖  ▄▄▄  ▄▄▄▖ ▗▄▖
 ▐▛   ▘▐▛ ▜▌█   █ █▄▄▘█▛▜▌▐▛ ▜▌▐▛   ▘█   █ █▄▄▘█▛▜▌
 ▐▙▄▄▖▐▛ ▜▌▐▌  █ █▄▄▖█▌ ▐▌▐▙▄▟▌▐▙▄▄▖▐▌  ▐▌█▄▄▖█▌ ▐▌
  ▝▀▀▘▝▘  ▘ ▀▀▀  ▝▀▀ ▝▘ ▝▘ ▝▀▀▘ ▝▀▀▘ ▀▀▀  ▝▀▀ ▝▘ ▝▘
     MCTS // RLVR SANDBOX // VIRTUAL CONTEXT // TIERED PORTFOLIO
EOF
  printf "${N}\n"
  if ollama_up; then
    printf "  ${G}● OLLAMA${N} ${D}online${N}   ${M}● MK-1${N} ${D}ready${N}\n\n"
  else
    printf "  ${R}● OLLAMA${N} ${D}offline (auto-starts on demand)${N}   ${M}● MK-1${N} ${D}ready${N}\n\n"
  fi
  printf "  ${C}1${N}  TUI dashboard ${D}(cyberpunk telemetry console)${N}\n"
  printf "  ${C}2${N}  Query the harness ${D}(tiered MCTS + verified answer)${N}\n"
  printf "  ${C}3${N}  Agent task ${D}(tools + sub-agent spawning)${N}\n"
  printf "  ${C}4${N}  Portfolio status ${D}(which variants are installed)${N}\n"
  printf "  ${C}5${N}  Usage stats ${D}(token/call tracker)${N}\n"
  printf "  ${C}6${N}  Selftest ${D}(mock backend, no model needed)${N}\n"
  printf "  ${C}7${N}  Run tests ${D}(unittest suite)${N}\n"
  printf "  ${C}8${N}  Ollama controls ${D}(serve / pull / models / chat / stop)${N}\n"
  printf "  ${C}q${N}  Quit\n\n"
}

pull_menu() {
  printf "${C}variants:${N} 1)1.5B 2)2B 3)3B 4)5B  5)all  a)any tag\n"
  read -rp "pull which? " v
  case "$v" in
    1) ollama pull wallpillar-lm/evolutionermk1-1.5B:latest ;;
    2) ollama pull wallpillar-lm/evolutionermk1-2B:latest ;;
    3) ollama pull wallpillar-lm/evolutionermk1-3B:latest ;;
    4) ollama pull wallpillar-lm/evolutionermk1-5B:latest ;;
    5) for t in 1.5B 2B 3B 5B; do ollama pull "wallpillar-lm/evolutionermk1-$t:latest"; done ;;
    a) read -rp "model tag: " t; ollama pull "$t" ;;
  esac
}

ollama_menu() {
  printf "${C}ollama:${N} 1)serve 2)models 3)pull 4)chat 5)ps 6)stop\n"
  read -rp "select: " o
  case "$o" in
    1) ensure_ollama && printf "${G}ollama online${N}\n" ;;
    2) ollama list ;;
    3) pull_menu ;;
    4) read -rp "model [wallpillar-lm/evolutionermk1-3B:latest]: " m
       ollama chat "${m:-wallpillar-lm/evolutionermk1-3B:latest}" ;;
    5) curl -s http://localhost:11434/api/ps | head -c 2000; echo ;;
    6) pkill -x ollama 2>/dev/null && printf "${Y}stopped${N}\n" || printf "${D}not running${N}\n" ;;
  esac
}

while true; do
  menu
  read -rp "evolutioner :: ❯ " choice || { echo; exit 0; }
  case "$choice" in
  1) "$PY" -m evolutioner.cli tui ;;
  2) read -rp "query: " Q
     ensure_ollama
     "$PY" -m evolutioner.cli query ${Q:-"What is 7 * 6? Verify with python."} ;;
  3) read -rp "agent task: " A
     ensure_ollama
     "$PY" -m evolutioner.cli agent ${A:-"List the files in the workspace and summarize what you see."} --tier 3B ;;
  4) ensure_ollama && "$PY" -m evolutioner.cli status ;;
  5) "$PY" -m evolutioner.cli usage ;;
  6) "$PY" -m evolutioner.cli selftest ;;
  7) "$PY" -m unittest discover -s tests -v ;;
  8) ollama_menu ;;
    q|Q) exit 0 ;;
    *) printf "${Y}unknown option: '%s'${N}  ${D}(enter 1-8 or q)${N}\n" "$choice"; sleep 1 ;;
  esac
  printf "${D}\n(press enter to return to menu)${N}\n"
done
