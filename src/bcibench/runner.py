"""Ejecutor de campañas: planifica, corre en paralelo, persiste y se reanuda.

Plan = sujetos × corridas × condiciones. Cada ejecución = un consumidor + un
reproductor en procesos separados, con nombres de flujo LSL únicos. El orden es
POR BLOQUES: primero la corrida A de todos los sujetos y condiciones, después
la B. Así una campaña interrumpida deja bloques completos analizables.

Reanudación: una ejecución con `done.json` en su carpeta no se repite.

Uso:
  python -m bcibench.runner --plan piloto  --subjects 1 3 8 --runs 5 --workers 4 --out results/piloto
  python -m bcibench.runner --plan campana --subjects 1 2 3 4 5 6 7 8 9 --runs 0 1 --workers 6 --out results/campana
  python -m bcibench.runner --plan campana ... --dry-run     # solo lista
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .injector import FaultSpec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
RUN_DURATION_S = 420.0     # tope de respaldo por ejecución (una corrida dura ≈ 6,5 min); el fin real lo marca el reproductor

# Severidades exploratorias (Tabla 1). El piloto puede fijar otras: --severities JSON.
DEFAULT_CONDITIONS = [FaultSpec("none")] + [
    FaultSpec("loss", s, "random") for s in (0.01, 0.05, 0.10)] + [
    FaultSpec("jitter", s) for s in (0.010, 0.050, 0.100)] + [
    FaultSpec("delay", s) for s in (0.050, 0.100, 0.250)] + [
    FaultSpec("disconnect", s) for s in (0.5, 1.0, 3.0)]
# Familia 2: fallos estructurados, sincronizados con el ensayo (sin referencia: ya existe en la familia 1)
STRUCTURED_CONDITIONS = [
    FaultSpec("burst_trial", s) for s in (0.10, 0.25, 0.40)] + [
    FaultSpec("disconnect_trial", s) for s in (0.5, 1.0, 2.0)]

# Familia 3 (Módulo 3, exploratoria): barrido de exposición (cortes de 1 s, K por corrida),
# repetición de la última muestra en la ventana, y referencia repetida en la misma sesión de máquina.
M3_CONDITIONS = [FaultSpec("none")] + [
    FaultSpec("disconnect", 1.0, f"n{k}") for k in (2, 5, 10, 20)] + [
    FaultSpec("hold_trial", s) for s in (0.10, 0.25, 0.40)]
# Diagnóstico del piso de reconexión: cortes intermedios (el sujeto no influye: basta una corrida)
M3_FLOOR = [FaultSpec("disconnect", d, "n5") for d in (1.25, 1.75, 2.25)]
# Confirmación del mecanismo del piso: mismas condiciones bajo dos valores de tuning.MulticastMinRTT
# (lsl_api.cfg vía LSLAPICFG). Predicción: hueco = max(1,52; 0,52 + RTT * ceil(d / RTT)).
M3_RTT = [FaultSpec("disconnect", d, "n5") for d in (0.5, 1.1, 1.25, 1.75)]


def exec_id(subject: int, run: str, spec: FaultSpec) -> str:
    return f"s{subject:02d}-r{run}-{spec.label()}"


def seed_for(subject: int, run: str, spec: FaultSpec) -> int:
    # semilla determinista y distinta por (sujeto, corrida, condición)
    h = hash((subject, run, spec.kind, spec.mode, round(spec.severity, 6))) & 0x7FFFFFFF
    return h


def build_plan(subjects, runs, conditions, session="1test"):
    plan = []
    for run in runs:                      # bloque externo = corrida
        for s in subjects:
            for spec in conditions:
                sp = FaultSpec(spec.kind, spec.severity, spec.mode, seed_for(s, run, spec))
                plan.append(dict(subject=s, session=session, run=str(run), spec=sp, exec_id=exec_id(s, run, sp)))
    return plan


def run_one(item: dict, out_root: str, models_dir: str, duration: float, max_seconds=None) -> dict:
    eid = item["exec_id"]
    outdir = os.path.join(out_root, eid)
    done = os.path.join(outdir, "done.json")
    if os.path.exists(done):
        return dict(exec_id=eid, status="skipped")
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    model = os.path.join(models_dir, f"decoder_s{item['subject']:02d}.pkl")
    logc = open(os.path.join(outdir, "consumer.log"), "w")
    logp = open(os.path.join(outdir, "producer.log"), "w")
    t_start = time.time()
    cons = subprocess.Popen([PY, "-m", "bcibench.consumer", "--exec-id", eid, "--model", model,
                             "--outdir", outdir, "--duration", str(max_seconds or duration)],
                            env=env, cwd=ROOT, stdout=logc, stderr=subprocess.STDOUT)
    time.sleep(3)
    sp: FaultSpec = item["spec"]
    args = [PY, "-m", "bcibench.replay", "--subject", str(item["subject"]), "--session", item["session"],
            "--run", item["run"], "--exec-id", eid, "--fault", sp.kind, "--severity", str(sp.severity),
            "--mode", sp.mode, "--seed", str(sp.seed), "--outdir", outdir]
    if max_seconds:
        args += ["--max-seconds", str(max_seconds)]
    prod = subprocess.Popen(args, env=env, cwd=ROOT, stdout=logp, stderr=subprocess.STDOUT)
    try:
        prc = prod.wait(timeout=(max_seconds or duration) + 180)
        crc = cons.wait(timeout=120)
    except subprocess.TimeoutExpired:
        prod.kill(); cons.kill()
        prc = crc = -9
    logc.close(); logp.close()
    ok = prc == 0 and crc == 0 and os.path.exists(os.path.join(outdir, "consumer.json"))
    rec = dict(exec_id=eid, status="ok" if ok else "failed", producer_rc=prc, consumer_rc=crc,
               seconds=round(time.time() - t_start, 1), subject=item["subject"], run=item["run"],
               fault=sp.to_dict(), label=sp.label())
    if ok:
        with open(done, "w") as f:
            json.dump(rec, f, indent=2)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="campana")
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--session", default="1test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=os.path.join(ROOT, "results", "models"))
    ap.add_argument("--severities", default=None, help="JSON: {kind: [sev, ...]} para sobrescribir")
    ap.add_argument("--only", nargs="*", default=None, help="solo estas condiciones (labels)")
    ap.add_argument("--family", default="uniform", choices=["uniform", "structured", "all", "m3", "m3floor", "m3rtt"])
    ap.add_argument("--max-seconds", type=float, default=None, help="solo desarrollo")
    ap.add_argument("--duration", type=float, default=RUN_DURATION_S)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    conditions = {"uniform": list(DEFAULT_CONDITIONS), "structured": list(STRUCTURED_CONDITIONS),
                  "all": list(DEFAULT_CONDITIONS) + list(STRUCTURED_CONDITIONS),
                  "m3": list(M3_CONDITIONS), "m3floor": list(M3_FLOOR), "m3rtt": list(M3_RTT)}[a.family]
    if a.severities:
        sev = json.loads(a.severities)
        conditions = [FaultSpec("none")]
        for kind, vals in sev.items():
            for v in vals:
                conditions.append(FaultSpec(kind, float(v), "random"))
    if a.only:
        conditions = [c for c in conditions if c.label() in a.only]
    plan = build_plan(a.subjects, a.runs, conditions, a.session)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "plan.json"), "w") as f:
        json.dump([dict(p, spec=p["spec"].to_dict()) for p in plan], f, indent=1)
    est_h = len(plan) * (a.max_seconds or a.duration) / 3600 / a.workers
    print(f"plan: {len(plan)} ejecuciones, {a.workers} en paralelo, ~{est_h:.1f} h de reloj", flush=True)
    if a.dry_run:
        for p in plan:
            print("  ", p["exec_id"])
        return
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, p, a.out, a.models, a.duration, a.max_seconds): p for p in plan}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            el = (time.time() - t0) / 60
            print(f"[{i}/{len(plan)}] {r['exec_id']:<32} {r['status']:<8} {el:6.1f} min", flush=True)
            with open(os.path.join(a.out, "progress.jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")
    n_ok = sum(r["status"] in ("ok", "skipped") for r in results)
    print(f"fin: {n_ok}/{len(plan)} ok, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
