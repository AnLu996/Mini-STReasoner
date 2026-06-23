#!/usr/bin/env bash
set -Eeuo pipefail

# download_ecgqa_full.bash
# Descarga ECG-QA + señales ECG fuente y genera JSON mapeados con ecg_path.
# Fuentes soportadas:
#   ptbxl  -> ECG-QA original basado en PTB-XL, señales desde PhysioNet PTB-XL records500.
#   mimic  -> ECG-QA expandido basado en MIMIC-IV-ECG, señales desde PhysioNet MIMIC-IV-ECG.
#   both   -> ambos. Ojo: MIMIC-IV-ECG pesa ~90 GB descomprimido.
#
# Ejemplos:
#   bash download_ecgqa_full.bash --root ~/datasets/ecgqa_full --source ptbxl
#   bash download_ecgqa_full.bash --root ~/datasets/ecgqa_full --source both --yes
#   bash download_ecgqa_full.bash --root ~/datasets/ecgqa_full --source mimic --yes

ROOT="$HOME/datasets/ecgqa_full"
SOURCE="both"
YES="false"
SKIP_MAP="false"
ECGQA_REPO="https://github.com/Jwoo5/ecg-qa.git"
ECGQA_TAG="v1.0.2"
PTBXL_URL="https://physionet.org/files/ptb-xl/1.0.3/records500/"
MIMIC_URL="https://physionet.org/files/mimic-iv-ecg/1.0/"

usage() {
  cat <<USAGE
Uso:
  bash download_ecgqa_full.bash [opciones]

Opciones:
  --root DIR        Carpeta destino. Default: $ROOT
  --source X        ptbxl | mimic | both. Default: $SOURCE
  --yes             No pedir confirmación para MIMIC-IV-ECG.
  --skip-map        Solo descargar; no ejecutar scripts de mapeo.
  -h, --help        Mostrar ayuda.

Salida principal:
  ROOT/ecg-qa/                         Repo ECG-QA con JSON originales.
  ROOT/signals/ptbxl/records500/       Señales PTB-XL WFDB.
  ROOT/signals/mimic-iv-ecg/files/     Señales MIMIC-IV-ECG WFDB.
  ROOT/mapped/ptbxl/                   JSON ECG-QA con ecg_path para PTB-XL.
  ROOT/mapped/mimic-iv-ecg/            JSON ECG-QA con ecg_path para MIMIC-IV-ECG.
USAGE
}

log() { printf '\n\033[1;34m[ECG-QA]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[AVISO]\033[0m %s\n' "$*"; }
err() { printf '\n\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --yes|-y) YES="true"; shift ;;
    --skip-map) SKIP_MAP="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "Opción no reconocida: $1"; usage; exit 1 ;;
  esac
done

case "$SOURCE" in
  ptbxl|mimic|both) ;;
  *) err "--source debe ser: ptbxl, mimic o both"; exit 1 ;;
esac

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Falta instalar '$1'. En Ubuntu: sudo apt update && sudo apt install -y $2"
    exit 1
  fi
}

install_python_deps() {
  log "Preparando entorno Python local"
  need_cmd python3 "python3 python3-venv python3-pip"

  if [[ ! -d "$ROOT/.venv-ecgqa" ]]; then
    python3 -m venv "$ROOT/.venv-ecgqa"
  fi
  # shellcheck disable=SC1091
  source "$ROOT/.venv-ecgqa/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install pandas tqdm wfdb numpy scipy
}

clone_or_update_repo() {
  log "Descargando repo ECG-QA ($ECGQA_TAG)"
  mkdir -p "$ROOT"
  if [[ ! -d "$ROOT/ecg-qa/.git" ]]; then
    if ! git clone --depth 1 --branch "$ECGQA_TAG" "$ECGQA_REPO" "$ROOT/ecg-qa"; then
      warn "No se pudo clonar la etiqueta $ECGQA_TAG. Intentando con master."
      git clone --depth 1 "$ECGQA_REPO" "$ROOT/ecg-qa"
    fi
  else
    git -C "$ROOT/ecg-qa" fetch --tags --depth 1 || true
    if git -C "$ROOT/ecg-qa" rev-parse "$ECGQA_TAG" >/dev/null 2>&1; then
      git -C "$ROOT/ecg-qa" checkout "$ECGQA_TAG"
    else
      git -C "$ROOT/ecg-qa" pull --ff-only || true
    fi
  fi
}

# Descarga recursiva desde PhysioNet preservando estructura útil, con reanudación.
# Usa wget porque PhysioNet lo documenta oficialmente para estos datasets.
wget_recursive() {
  local url="$1"
  local dest="$2"
  local cut_dirs="$3"
  mkdir -p "$dest"
  wget \
    --recursive \
    --no-parent \
    --continue \
    --timestamping \
    --no-host-directories \
    --cut-dirs="$cut_dirs" \
    --reject 'index.html*' \
    --directory-prefix="$dest" \
    "$url"
}

download_ptbxl_signals() {
  local dest="$ROOT/signals/ptbxl"
  if [[ -d "$dest/records500" ]] && find "$dest/records500" -name '*.dat' -print -quit | grep -q .; then
    log "PTB-XL ya parece descargado en $dest"
    return 0
  fi
  log "Descargando señales PTB-XL records500 hacia $dest"
  wget_recursive "$PTBXL_URL" "$dest" 3
}

download_mimic_signals() {
  local dest="$ROOT/signals/mimic-iv-ecg"
  if [[ -d "$dest/files" ]] && [[ -f "$dest/record_list.csv" ]] && find "$dest/files" -name '*.dat' -print -quit | grep -q .; then
    log "MIMIC-IV-ECG ya parece descargado en $dest"
    return 0
  fi

  warn "MIMIC-IV-ECG completo ocupa aprox. 90.4 GB descomprimido; la descarga ZIP oficial pesa aprox. 33.8 GB."
  if [[ "$YES" != "true" ]]; then
    read -r -p "¿Continuar con MIMIC-IV-ECG? [y/N]: " ans
    case "$ans" in
      y|Y|yes|YES|s|S|si|SI|sí|SÍ) ;;
      *) warn "Saltando MIMIC-IV-ECG. Para evitar pregunta usa --yes."; return 0 ;;
    esac
  fi

  log "Descargando señales MIMIC-IV-ECG hacia $dest"
  wget_recursive "$MIMIC_URL" "$dest" 3
}

map_ptbxl() {
  local ecgqa_root="$ROOT/ecg-qa/ecgqa/ptbxl"
  local signal_root="$ROOT/signals/ptbxl"
  local out="$ROOT/mapped/ptbxl"
  log "Mapeando ECG-QA PTB-XL: ecg_id -> ecg_path"
  python "$ROOT/ecg-qa/mapping_ptbxl_samples.py" "$ecgqa_root" \
    --ptbxl-data-dir "$signal_root" \
    --dest "$out"
}

map_mimic() {
  local ecgqa_root="$ROOT/ecg-qa/ecgqa/mimic-iv-ecg"
  local signal_root="$ROOT/signals/mimic-iv-ecg"
  local out="$ROOT/mapped/mimic-iv-ecg"

  if [[ ! -f "$signal_root/record_list.csv" ]]; then
    warn "No encuentro $signal_root/record_list.csv; no puedo mapear MIMIC-IV-ECG."
    return 0
  fi

  log "Mapeando ECG-QA MIMIC-IV-ECG: study_id -> ecg_path"
  python "$ROOT/ecg-qa/mapping_mimic_iv_ecg_samples.py" "$ecgqa_root" \
    --mimic-iv-ecg-data-dir "$signal_root" \
    --dest "$out"
}

verify_outputs() {
  log "Verificando salida"
  echo "ROOT: $ROOT"

  if [[ "$SOURCE" == "ptbxl" || "$SOURCE" == "both" ]]; then
    echo "PTB-XL signals:" $(find "$ROOT/signals/ptbxl/records500" -name '*.dat' 2>/dev/null | wc -l) "archivos .dat"
    if [[ -d "$ROOT/mapped/ptbxl" ]]; then
      echo "PTB-XL mapped JSON:" $(find "$ROOT/mapped/ptbxl" -name '*.json' 2>/dev/null | wc -l) "archivos"
    fi
  fi

  if [[ "$SOURCE" == "mimic" || "$SOURCE" == "both" ]]; then
    if [[ -d "$ROOT/signals/mimic-iv-ecg/files" ]]; then
      echo "MIMIC signals:" $(find "$ROOT/signals/mimic-iv-ecg/files" -name '*.dat' 2>/dev/null | wc -l) "archivos .dat"
    fi
    if [[ -d "$ROOT/mapped/mimic-iv-ecg" ]]; then
      echo "MIMIC mapped JSON:" $(find "$ROOT/mapped/mimic-iv-ecg" -name '*.json' 2>/dev/null | wc -l) "archivos"
    fi
  fi

  cat <<NEXT

Listo. Para tu modelo, usa preferentemente los JSON mapeados en:
  $ROOT/mapped/ptbxl
  $ROOT/mapped/mimic-iv-ecg

Cada muestra conserva question/answer/ecg_id y agrega ecg_path, que apunta al archivo WFDB sin extensión.
Ejemplo para leer una señal con Python:
  import wfdb
  record = wfdb.rdrecord('/ruta/sin_extension')
  signal = record.p_signal
NEXT
}

main() {
  need_cmd git "git"
  need_cmd wget "wget"
  mkdir -p "$ROOT/signals" "$ROOT/mapped"
  install_python_deps
  clone_or_update_repo

  case "$SOURCE" in
    ptbxl)
      download_ptbxl_signals
      [[ "$SKIP_MAP" == "true" ]] || map_ptbxl
      ;;
    mimic)
      download_mimic_signals
      [[ "$SKIP_MAP" == "true" ]] || map_mimic
      ;;
    both)
      download_ptbxl_signals
      download_mimic_signals
      if [[ "$SKIP_MAP" != "true" ]]; then
        map_ptbxl
        map_mimic
      fi
      ;;
  esac

  verify_outputs
}

main "$@"
