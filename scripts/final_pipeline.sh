#!/usr/bin/env bash
# Cadena final del Entregable 2 (versión 2): descarga de la campaña → re-decodificación de
# segmentos (EEGNet, CSP fuera de línea) → análisis por decodificador → tablas y números →
# docx/pdf → datos al repositorio público.
# Uso: bash scripts/final_pipeline.sh <IP> <pem> <horas_campana> [campana2]
set -euo pipefail
IP="$1"; PEM="$2"; HORAS="$3"; CAMP="${4:-campana2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"
MAN="$ROOT/../30-TFG/Entregas/Modulo-2"
REPO="/c/Users/agustin.tamagusuku/Desktop/bci-fault-bench"
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT/src"

echo "== 1. descarga de la campaña ($CAMP)"
rm -rf "$ROOT/results/vm/$CAMP"
ssh -i "$PEM" ubuntu@"$IP" "cd ~/bcibench/results && tar czf - --exclude='*.log' $CAMP" < /dev/null | tar xzf - -C "$ROOT/results/vm"
N=$(ls -d "$ROOT"/results/vm/$CAMP/s0*/ | wc -l); echo "   ejecuciones: $N"
ssh -i "$PEM" ubuntu@"$IP" "tail -1 ~/bcibench/results/$CAMP/runner.log" < /dev/null

echo "== 2. re-decodificación de segmentos (EEGNet y CSP fuera de línea)"
"$PY" "$ROOT/scripts/redecode_segments.py" --root "$ROOT/results/vm/$CAMP" --models "$ROOT/results/models" | tail -1

echo "== 3. análisis (W=8, umbral = mínimo de la referencia): CSP en línea, EEGNet, alternativas"
"$PY" "$ROOT/scripts/analyze.py" --root "$ROOT/results/vm/$CAMP" --out "$ROOT/results/analysis" --window 8 --thr-rule min 2>&1 | grep -E "ejecuciones|listo"
"$PY" "$ROOT/scripts/analyze.py" --root "$ROOT/results/vm/$CAMP" --out "$ROOT/results/analysis-eegnet" --window 8 --thr-rule min --trials-file trials_eegnet.csv 2>&1 | grep -E "ejecuciones|listo"
"$PY" "$ROOT/scripts/analyze.py" --root "$ROOT/results/vm/$CAMP" --out "$ROOT/results/analysis-csp-offline" --window 8 --thr-rule min --trials-file trials_csp_offline.csv 2>&1 | grep -E "listo"
"$PY" "$ROOT/scripts/analyze.py" --root "$ROOT/results/vm/$CAMP" --out "$ROOT/results/analysis-alt-p5" --window 8 --thr-rule percentile --q 5 2>&1 | grep -E "listo"
"$PY" "$ROOT/scripts/analyze.py" --root "$ROOT/results/vm/$CAMP" --out "$ROOT/results/analysis-alt-w12" --window 12 --thr-rule min 2>&1 | grep -E "listo"

echo "== 4. tablas y números del manuscrito"
"$PY" "$ROOT/scripts/resultados_tablas.py" --analysis "$ROOT/results/analysis" --analysis-eegnet "$ROOT/results/analysis-eegnet" --manuscrito "$MAN"
"$PY" "$ROOT/scripts/resultados_numeros.py" --analysis "$ROOT/results/analysis" --analysis-eegnet "$ROOT/results/analysis-eegnet" \
  --analysis-csp-offline "$ROOT/results/analysis-csp-offline" --campana "$ROOT/results/vm/$CAMP" \
  --piloto "$ROOT/results/vm/piloto" --solo "$ROOT/results/vm/piloto-solo" --offline "$ROOT/results/offline_bacc.csv" \
  --eegnet-offline "$ROOT/results/eegnet_offline.csv" --manuscrito "$MAN" --horas "$HORAS" | head -3

echo "== 5. docx y pdf"
cd "$MAN/src" && NODE_PATH="$(npm root -g)" node build.js .. | tail -1 && python topdf.py "../Tamagusuku_Agustin - Entregable 2 - Resultados.docx" | tail -1

echo "== 6. datos y análisis al repositorio público (sin segmentos: van a Zenodo)"
DST="$REPO/campaign-2026-09-27"
rm -rf "$DST/raw" "$DST/analysis" "$DST/analysis-eegnet" "$DST/pilot"
mkdir -p "$DST/raw" "$DST/pilot"
(cd "$ROOT/results/vm/$CAMP" && for d in s0*/; do mkdir -p "$DST/raw/$d"; cp "$d"/*.csv "$d"/*.json "$d"/*.jsonl "$DST/raw/$d" 2>/dev/null || true; done)
cp -r "$ROOT/results/analysis" "$DST/analysis"; cp -r "$ROOT/results/analysis-eegnet" "$DST/analysis-eegnet"
cp -r "$ROOT/results/vm/piloto" "$ROOT/results/vm/piloto-solo" "$DST/pilot/"
cp "$ROOT/results/offline_bacc.csv" "$ROOT/results/eegnet_offline.csv" "$DST/"
find "$DST" -name "*.log" -delete
cp "$ROOT/src/bcibench/"*.py "$REPO/src/bcibench/" && cp "$ROOT/scripts/"*.py "$ROOT/scripts/"*.sh "$REPO/scripts/" && cp "$ROOT/requirements.txt" "$REPO/"
rm -f "$REPO/scripts/debug_"*.py
echo "   tamaño datos: $(du -sh "$DST" | cut -f1)"
echo "== listo; falta: git add/commit/tag/push en $REPO y apagar la VM"
