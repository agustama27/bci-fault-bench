#!/usr/bin/env bash
# Despliegue del banco a la VM Ubuntu. Uso: bash scripts/deploy.sh <IP> <ruta/al/tfg-bench.pem>
set -euo pipefail
IP="$1"; PEM="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH="ssh -i $PEM -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 ubuntu@$IP"
echo "== conexión"
$SSH "uname -a && nproc && free -g | head -2 && df -h / | tail -1"
echo "== copia del código (sin .venv, data ni results/raw)"
$SSH "mkdir -p ~/bcibench/results"
tar -C "$ROOT" --exclude=.venv --exclude=data --exclude='results/smoke' --exclude='results/runner-test' \
    --exclude='results/mini' --exclude='results/dry' --exclude='results/analysis-test' --exclude=__pycache__ \
    -czf - src scripts requirements.txt results/models | $SSH "tar -C ~/bcibench -xzf -"
echo "== instalación"
$SSH "bash ~/bcibench/scripts/vm_setup.sh"
echo "== timer resolution check"
$SSH "cd ~/bcibench && PYTHONPATH=src .venv/bin/python -c \"
import time; from mne_lsl.lsl import local_clock
errs=[]
for _ in range(200):
    t=local_clock()+0.001;
    while local_clock()<t: pass
    errs.append((local_clock()-t)*1e6)
import statistics; print('busy-wait overshoot us: p50', round(statistics.median(errs),1), 'max', round(max(errs),1))
errs=[]
for _ in range(100):
    t=time.perf_counter(); time.sleep(0.001); errs.append((time.perf_counter()-t-0.001)*1e6)
print('sleep(1ms) overshoot us: p50', round(statistics.median(errs),1), 'max', round(max(errs),1))
\""
echo "== descarga del dataset (9 sujetos) en segundo plano en la VM → ~/bcibench/results/download.log"
$SSH "cd ~/bcibench && PYTHONPATH=src nohup .venv/bin/python -c \"
import warnings; warnings.filterwarnings('ignore')
from bcibench import data
for s in range(1,10): d=data.load_subject(s); print('sujeto', s, 'ok', flush=True)
print('DESCARGA COMPLETA', flush=True)
\" > results/download.log 2>&1 &
echo lanzado"
echo "== despliegue listo"
