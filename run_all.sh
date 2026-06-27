#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/codes/pinn_fracenergy.py" ]]; then
  PROJECT_DIR="$SCRIPT_DIR"
  CODE_DIR="$SCRIPT_DIR/codes"
elif [[ -f "$SCRIPT_DIR/pinn_fracenergy.py" ]]; then
  PROJECT_DIR="$SCRIPT_DIR"
  CODE_DIR="$SCRIPT_DIR"
elif [[ -f "$SCRIPT_DIR/../codes/pinn_fracenergy.py" ]]; then
  PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
  CODE_DIR="$PROJECT_DIR/codes"
else
  echo "ERROR: could not find pinn_fracenergy.py relative to run_all.sh" >&2
  exit 1
fi
LOG_DIR="${RUN_LOG_DIR:-$PROJECT_DIR/run_logs}"
TMP_BASE="${TMPDIR:-/tmp}"
MPL_DIR="${MPLCONFIGDIR:-$TMP_BASE/fracenergy_mplconfig}"
PYCACHE_DIR="${PYTHONPYCACHEPREFIX:-$TMP_BASE/fracenergy_pycache}"

MODE="paper"
COMPILE_TEX=0
INSTALL_DEPS=0
PYTHON_OVERRIDE="${PYTHON_BIN:-}"
EXTRA_ARGS=()

PIP_DEPS=(numpy pandas scikit-learn matplotlib seaborn tensorflow)

usage() {
  cat <<'EOF'
Usage:
  bash run_all.sh [mode] [options]

Modes:
  --paper             Reproduce the main paper run and figures (default).
  --quick            Fast smoke test with small sample/epoch settings.
  --learning-curves  Run the longer learning-curve study and plot figures.
  --all              Run the paper reproduction and learning-curve study.

Options:
  --compile-tex      Compile pgnr.tex after the computational run, if LaTeX is installed.
  --install-deps     Install Python dependencies into the selected Python environment.
  --python PATH      Use a specific Python executable.
  -h, --help         Show this help message.

Examples:
  bash run_all.sh
  bash run_all.sh --quick
  bash run_all.sh --learning-curves
  bash run_all.sh --all --compile-tex
  PYTHON_BIN=/usr/bin/python3 bash run_all.sh
  bash run_all.sh --learning-curves -- --samples 500 1000 --seeds 42

Outputs:
  Root-level PNG figures are written where pgnr.tex expects them.
  Run logs are written to run_logs/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paper)
      MODE="paper"
      shift
      ;;
    --quick)
      MODE="quick"
      shift
      ;;
    --learning-curves)
      MODE="learning-curves"
      shift
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --compile-tex)
      COMPILE_TEX=1
      shift
      ;;
    --install-deps)
      INSTALL_DEPS=1
      shift
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --python requires a path." >&2
        exit 2
      fi
      PYTHON_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

python_has_deps() {
  local py="$1"
  "$py" - <<'PY' >/dev/null 2>&1
import numpy
import pandas
import sklearn
import matplotlib
import seaborn
import tensorflow
PY
}

python_missing_deps() {
  local py="$1"
  "$py" - <<'PY'
mods = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("tensorflow", "tensorflow"),
]
missing = []
for module, package in mods:
    try:
        __import__(module)
    except Exception:
        missing.append(package)
print(" ".join(missing))
PY
}

select_python() {
  local candidates=()
  local first_available=""
  local candidate

  if [[ -n "$PYTHON_OVERRIDE" ]]; then
    candidates+=("$PYTHON_OVERRIDE")
  else
    candidates+=("python3" "/usr/bin/python3" "python")
  fi

  for candidate in "${candidates[@]}"; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if [[ -z "$first_available" ]]; then
      first_available="$candidate"
    fi
    if python_has_deps "$candidate"; then
      PYTHON_BIN="$candidate"
      return 0
    fi
  done

  if [[ -z "$first_available" ]]; then
    echo "ERROR: no Python executable was found." >&2
    exit 1
  fi

  PYTHON_BIN="$first_available"
  if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    echo "Installing Python dependencies with: $PYTHON_BIN -m pip install ${PIP_DEPS[*]}"
    "$PYTHON_BIN" -m pip install "${PIP_DEPS[@]}"
    if python_has_deps "$PYTHON_BIN"; then
      return 0
    fi
  fi

  local missing
  missing="$(python_missing_deps "$PYTHON_BIN")"
  cat >&2 <<EOF
ERROR: required Python packages are missing for: $PYTHON_BIN
Missing packages: ${missing:-unknown}

Install them with:
  $PYTHON_BIN -m pip install ${PIP_DEPS[*]}

Or choose another environment:
  PYTHON_BIN=/path/to/python bash run_all.sh
EOF
  exit 1
}

run_paper() {
  local log="$LOG_DIR/paper_$(timestamp).log"
  echo "Running main paper workflow with $PYTHON_BIN"
  echo "Log: $log"
  (
    cd "$CODE_DIR"
    env \
      MPLCONFIGDIR="$MPL_DIR" \
      PYTHONPYCACHEPREFIX="$PYCACHE_DIR" \
      FIG_DIR="${FIG_DIR:-$PROJECT_DIR}" \
      NUM_SAMPLES="${NUM_SAMPLES:-2000}" \
      RUN_PLOTS="${RUN_PLOTS:-1}" \
      RUN_SENSITIVITY="${RUN_SENSITIVITY:-0}" \
      RUN_3D_SWEEP="${RUN_3D_SWEEP:-1}" \
      SWEEP_REPEATS="${SWEEP_REPEATS:-2}" \
      EPOCHS_1D="${EPOCHS_1D:-400}" \
      EPOCHS_2D="${EPOCHS_2D:-400}" \
      EPOCHS_3D="${EPOCHS_3D:-600}" \
      BATCH_SIZE="${BATCH_SIZE:-64}" \
      BATCH_SIZE_3D="${BATCH_SIZE_3D:-64}" \
      "$PYTHON_BIN" pinn_fracenergy.py
  ) 2>&1 | tee "$log"
}

run_quick() {
  local log="$LOG_DIR/quick_$(timestamp).log"
  echo "Running quick smoke test with $PYTHON_BIN"
  echo "Log: $log"
  (
    cd "$CODE_DIR"
    env \
      MPLCONFIGDIR="$MPL_DIR" \
      PYTHONPYCACHEPREFIX="$PYCACHE_DIR" \
      FIG_DIR="${FIG_DIR:-$PROJECT_DIR}" \
      NUM_SAMPLES="${NUM_SAMPLES:-300}" \
      RUN_PLOTS="${RUN_PLOTS:-0}" \
      RUN_SENSITIVITY="${RUN_SENSITIVITY:-0}" \
      RUN_3D_SWEEP="${RUN_3D_SWEEP:-0}" \
      USE_EARLY_STOP_1D="${USE_EARLY_STOP_1D:-1}" \
      USE_EARLY_STOP_2D="${USE_EARLY_STOP_2D:-1}" \
      USE_EARLY_STOP_3D="${USE_EARLY_STOP_3D:-1}" \
      EPOCHS_1D="${EPOCHS_1D:-5}" \
      EPOCHS_2D="${EPOCHS_2D:-5}" \
      EPOCHS_3D="${EPOCHS_3D:-5}" \
      BATCH_SIZE="${BATCH_SIZE:-64}" \
      BATCH_SIZE_3D="${BATCH_SIZE_3D:-64}" \
      "$PYTHON_BIN" pinn_fracenergy.py
  ) 2>&1 | tee "$log"
}

run_learning_curves() {
  local log="$LOG_DIR/learning_curves_$(timestamp).log"
  echo "Running learning-curve workflow with $PYTHON_BIN"
  echo "Log: $log"
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Extra learning-curve arguments: ${EXTRA_ARGS[*]}"
  fi
  (
    cd "$PROJECT_DIR"
    env \
      MPLCONFIGDIR="$MPL_DIR" \
      PYTHONPYCACHEPREFIX="$PYCACHE_DIR" \
      "$PYTHON_BIN" "$CODE_DIR/learning_curve_study.py" "${EXTRA_ARGS[@]}"
    env \
      MPLCONFIGDIR="$MPL_DIR" \
      PYTHONPYCACHEPREFIX="$PYCACHE_DIR" \
      "$PYTHON_BIN" "$CODE_DIR/plot_learning_curves.py" --out-dir "$PROJECT_DIR"
  ) 2>&1 | tee "$log"
}

compile_tex() {
  local log="$LOG_DIR/latex_$(timestamp).log"
  echo "Compiling pgnr.tex"
  echo "Log: $log"
  (
    cd "$PROJECT_DIR"
    if [[ ! -f pgnr.tex ]]; then
      echo "Skipping PDF compilation: pgnr.tex is not present in this repository."
    elif command -v latexmk >/dev/null 2>&1; then
      latexmk -pdf -interaction=nonstopmode pgnr.tex
    elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
      pdflatex -interaction=nonstopmode pgnr.tex
      bibtex pgnr
      pdflatex -interaction=nonstopmode pgnr.tex
      pdflatex -interaction=nonstopmode pgnr.tex
    else
      echo "Skipping PDF compilation: latexmk, or pdflatex+bibtex, is not installed."
    fi
  ) 2>&1 | tee "$log"
}

mkdir -p "$LOG_DIR" "$MPL_DIR" "$PYCACHE_DIR"
select_python

case "$MODE" in
  paper)
    run_paper
    ;;
  quick)
    run_quick
    ;;
  learning-curves)
    run_learning_curves
    ;;
  all)
    run_paper
    run_learning_curves
    ;;
  *)
    echo "ERROR: unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ "$COMPILE_TEX" -eq 1 ]]; then
  compile_tex
fi

echo "Done. Logs are in: $LOG_DIR"
