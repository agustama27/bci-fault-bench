"""Prueba de humo local: una ejecución truncada (consumidor + reproductor).

Uso: python scripts/smoke_run.py [--fault loss --severity 0.05] [--max-seconds 60]
     python scripts/smoke_run.py --consumer bcipy --consumer-python .venv-bcipy/Scripts/python.exe
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--subject", type=int, default=1)
ap.add_argument("--run", default="0")
ap.add_argument("--fault", default="none")
ap.add_argument("--severity", type=float, default=0.0)
ap.add_argument("--mode", default="random")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--max-seconds", type=float, default=60)
ap.add_argument("--consumer", default="own", choices=["own", "bcipy"],
                help="own = bcibench.consumer; bcipy = bcibench.consumer_bcipy")
ap.add_argument("--consumer-python", default=PY, help="intérprete del consumidor (p. ej. .venv-bcipy)")
a = ap.parse_args()
if os.path.exists(a.consumer_python):
    a.consumer_python = os.path.abspath(a.consumer_python)

exec_id = f"smoke-{int(time.time())}"
outdir = os.path.join(ROOT, "results", "smoke", exec_id)
os.makedirs(outdir, exist_ok=True)
model = os.path.join(ROOT, "results", "models", f"decoder_s{a.subject:02d}.pkl")

CONSUMER_MODULE = {"own": "bcibench.consumer", "bcipy": "bcibench.consumer_bcipy"}[a.consumer]
cons = subprocess.Popen([a.consumer_python, "-m", CONSUMER_MODULE, "--exec-id", exec_id, "--model", model,
                         "--outdir", outdir, "--duration", str(a.max_seconds)], env=ENV, cwd=ROOT)
time.sleep(2)
prod = subprocess.Popen([PY, "-m", "bcibench.replay", "--subject", str(a.subject), "--run", a.run,
                         "--exec-id", exec_id, "--fault", a.fault, "--severity", str(a.severity),
                         "--mode", a.mode, "--seed", str(a.seed), "--outdir", outdir,
                         "--max-seconds", str(a.max_seconds)], env=ENV, cwd=ROOT)
prod.wait(timeout=a.max_seconds + 120)
cons.wait(timeout=60)
print("consumer", a.consumer, "producer rc", prod.returncode, "consumer rc", cons.returncode)
for fn in ("producer.json", "consumer.json"):
    p = os.path.join(outdir, fn)
    if os.path.exists(p):
        d = json.load(open(p))
        print(fn, json.dumps({k: d[k] for k in d if k in ("label", "n_pushed", "n_dropped", "reconnections",
                                                         "timing_err_ms", "n_trials", "n_valid",
                                                         "balanced_accuracy", "exceptions", "consumer",
                                                         "get_data_assertions")}))
print("outdir:", outdir)
