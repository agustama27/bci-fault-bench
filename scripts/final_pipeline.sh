#!/usr/bin/env bash
# Cadena final del Entregable 2: descarga de la campaña → análisis → tablas y números →
# docx/pdf → datos y etiqueta v1.0 en el repositorio público.
# Uso: bash scripts/final_pipeline.sh <IP> <pem> <horas_campana>
set -euo pipefail
IP="$1"; PEM="$2"; HORAS="$3"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"
MAN="$ROOT/../30-TFG/Entregas/Modulo-2"
REPO="/c/Users/agustin.tamagusuku/Desktop/bci-fault-bench"
export PYTHONIOENCODING=utf-8

echo "== 1. descarga de la campaña"
rm -rf "$ROOT/results/vm/campana"
ssh -i "$PEM" ubuntu@"$IP" "cd ~/bcibench/results && tar czf - --exclude='*.log' campana" < /dev/null | tar xzf - -C "$ROOT/results/vm"
N=$(ls -d "$ROOT"/results/vm/campana/s0*/ | wc -l); echo "   ejecuciones: $N"
ssh -i "$PEM" ubuntu@"$IP" "tail -1 ~/bcibench/results/campana/runner.log" < /dev/null

echo "== 2. análisis (W=8, umbral = mínimo de la referencia) + alternativas"
"$PY" "$ROOT/scripts/analyze.py" --root "$ROOT/results/vm/campana" --out "$ROOT/results/analysis" --window 8 --thr-rule min 2>&1 | grep -E "ejecuciones|listo"
"$PY" "$ROOT/scripts/analyze.py" --root "$ROOT/results/vm/campana" --out "$ROOT/results/analysis-alt-p5" --window 8 --thr-rule percentile --q 5 2>&1 | grep -E "listo"
"$PY" "$ROOT/scripts/analyze.py" --root "$ROOT/results/vm/campana" --out "$ROOT/results/analysis-alt-w12" --window 12 --thr-rule min 2>&1 | grep -E "listo"

echo "== 3. tablas y números del manuscrito"
"$PY" "$ROOT/scripts/resultados_tablas.py" --analysis "$ROOT/results/analysis" --manuscrito "$MAN"
"$PY" "$ROOT/scripts/resultados_numeros.py" --analysis "$ROOT/results/analysis" --campana "$ROOT/results/vm/campana" \
  --piloto "$ROOT/results/vm/piloto" --solo "$ROOT/results/vm/piloto-solo" --offline "$ROOT/results/offline_bacc.csv" \
  --manuscrito "$MAN" --horas "$HORAS" | head -3

echo "== 4. docx y pdf"
cd "$MAN/src" && NODE_PATH="$(npm root -g)" node build.js .. | tail -1 && python topdf.py "../Tamagusuku_Agustin - Entregable 2 - Resultados.docx" | tail -1

echo "== 5. datos y análisis al repositorio público"
mkdir -p "$REPO/campaign-2026-09-27"
rm -rf "$REPO/campaign-2026-09-27/raw" "$REPO/campaign-2026-09-27/analysis" "$REPO/campaign-2026-09-27/pilot"
cp -r "$ROOT/results/vm/campana" "$REPO/campaign-2026-09-27/raw"
cp -r "$ROOT/results/analysis" "$REPO/campaign-2026-09-27/analysis"
mkdir -p "$REPO/campaign-2026-09-27/pilot" && cp -r "$ROOT/results/vm/piloto" "$ROOT/results/vm/piloto-solo" "$REPO/campaign-2026-09-27/pilot/"
cp "$ROOT/results/offline_bacc.csv" "$REPO/campaign-2026-09-27/"
find "$REPO/campaign-2026-09-27" -name "*.log" -delete
cp "$ROOT/src/bcibench/"*.py "$REPO/src/bcibench/" && cp "$ROOT/scripts/"*.py "$ROOT/scripts/"*.sh "$REPO/scripts/" && cp "$ROOT/requirements.txt" "$REPO/"
rm -f "$REPO/scripts/debug_"*.py
echo "   tamaño datos: $(du -sh "$REPO/campaign-2026-09-27" | cut -f1)"
echo "== listo; falta: git add/commit/tag/push en $REPO y apagar la VM"
