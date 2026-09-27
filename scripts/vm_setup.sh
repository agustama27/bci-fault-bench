#!/usr/bin/env bash
# Instalación del banco en la VM Ubuntu 24.04 (se ejecuta en la VM, como usuario ubuntu).
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip libpugixml1v5 tmux htop > /dev/null
cd ~/bcibench
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import mne, mne_lsl, moabb, sklearn; print('deps ok', mne.__version__, mne_lsl.__version__, moabb.__version__)"
# liblsl: mne_lsl la descarga si no está en el sistema
.venv/bin/python -c "from mne_lsl.lsl import local_clock; print('lsl ok', local_clock())"
# permite más archivos abiertos y procesos (6-8 ejecuciones en paralelo)
ulimit -n 8192 || true
echo "setup listo"
