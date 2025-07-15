#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────────
# setup.sh  •  One-command bootstrap for the Resume-to-Job-Rec project
#
#  ▸ Creates (or recreates) a virtualenv in .venv/
#  ▸ Installs Python dependencies from requirements.txt
#  ▸ Ensures spaCy’s small English model + key NLTK corpora are available
#  ▸ Optionally registers an IPyKernel for Jupyter / VS Code
#
# Usage:
#     ./setup.sh                 # full setup, auto-detect python
#     ./setup.sh -p /usr/bin/python3.12    # use specific interpreter
#     ./setup.sh -k no           # skip kernel registration
#     ./setup.sh -h              # show help
# ────────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ╭──────────────────────────  formatting utils  ──────────────────────────────╮
cyan()  { printf '\033[0;36m%s\033[0m\n' "$*"; }
green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
err()   { printf '\033[0;31m%s\033[0m\n' "$*" >&2; }
trap 'err "❌  Error on line $LINENO"; exit 1' ERR

# ╭──────────────────────────  parse CLI flags  ───────────────────────────────╮
PYTHON_BIN="python"
REGISTER_KERNEL="yes"
PROJECT_NAME="jobrec"

show_help() {
  cat <<EOF
Usage: ./setup.sh [options]

Options:
  -p, --python PATH   Use explicit python interpreter (default: first on \$PATH)
  -k, --kernel [yes|no]
                      Register Jupyter kernel (default: yes)
  -h, --help          Show this help and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--python)     PYTHON_BIN="$2"; shift 2 ;;
    -k|--kernel)     REGISTER_KERNEL="${2,,}"; shift 2 ;;
    -h|--help)       show_help; exit 0 ;;
    *) err "Unknown option: $1"; show_help; exit 1 ;;
  esac
done

# ╭──────────────────────────  ensure python  ─────────────────────────────────╮
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  err "Python not found at '$PYTHON_BIN'"
  exit 1
fi
cyan "▶ Using interpreter: $("$PYTHON_BIN" --version 2>&1)"

# ╭──────────────────────────  project paths  ─────────────────────────────────╮
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
VENV_DIR=".venv"

# ╭──────────────────────────  create venv  ───────────────────────────────────╮
cyan "▶ Creating virtual environment ➜ $VENV_DIR"
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --quiet --upgrade pip setuptools wheel

# ╭──────────────────────────  install deps  ──────────────────────────────────╮
cyan "▶ Installing Python requirements"
pip install -r requirements.txt
pip install -e .

# ╭──────────────────────────  spaCy model  ───────────────────────────────────╮
cyan "▶ Ensuring spaCy model (en_core_web_sm)"
python - <<'PY'
import subprocess, sys, importlib.util
import spacy
try:
    spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
PY

# ╭──────────────────────────  NLTK corpora  ──────────────────────────────────╮
cyan "▶ Ensuring NLTK corpora (stopwords, punkt, wordnet)"
python - <<'PY'
import nltk, sys, warnings
pkgs = ["stopwords", "punkt", "wordnet"]
for p in pkgs:
    try:
        nltk.data.find(f"tokenizers/{p}" if p == "punkt" else f"corpora/{p}")
    except LookupError:
        warnings.warn(f"Downloading NLTK {p} ...")
        nltk.download(p, quiet=True)
PY

# ╭──────────────────────────  editable install  ──────────────────────────────╮
cyan "▶ Installing src/ package in editable mode"
pip install --quiet -e .

# ╭──────────────────────────  Jupyter kernel  ────────────────────────────────╮
if [[ "$REGISTER_KERNEL" == "yes" ]]; then
  KERNEL_NAME=$PROJECT_NAME
  cyan "▶ Registering IPyKernel ➜ $KERNEL_NAME"
  python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)" >/dev/null
fi

# ╭──────────────────────────  done  ──────────────────────────────────────────╮
green "✅  Environment ready."
green "   Activate anytime with:  source $VENV_DIR/bin/activate"
[[ "$REGISTER_KERNEL" == "yes" ]] && green "   Jupyter kernel name:  Python ($PROJECT_NAME)"
