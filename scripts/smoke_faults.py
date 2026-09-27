"""Prueba corta de cada modelo de fallo (secuencial, para no contaminar tiempos)."""
import json, os, subprocess, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cases = [("none", 0.0, "random"), ("loss", 0.10, "random"), ("loss", 0.10, "burst"),
         ("jitter", 0.05, "random"), ("delay", 0.25, "random"), ("disconnect", 1.0, "random")]
for fault, sev, mode in cases:
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "smoke_run.py"), "--fault", fault,
                          "--severity", str(sev), "--mode", mode, "--max-seconds", "60", "--seed", "3"],
                         capture_output=True, text=True, cwd=ROOT)
    lines = [l for l in out.stdout.splitlines() if l.startswith("producer.json") or l.startswith("consumer.json") or l.startswith("outdir")]
    print(f"### {fault} {sev} {mode}")
    for l in lines:
        print("  ", l)
    outdir = [l.split("outdir: ")[1] for l in lines if l.startswith("outdir")]
    if outdir:
        import csv
        rows = list(csv.DictReader(open(os.path.join(outdir[0], "telemetry.csv"))))
        n = sum(int(r["n_samples"]) for r in rows); gaps = sum(int(r["n_gaps"]) for r in rows)
        lat = [float(r["lat_mean"]) for r in rows if r["lat_mean"]]
        ia = [float(r["ia_std"]) for r in rows if r["ia_std"]]
        print(f"   telemetry: samples={n} gaps={gaps} lat_mean_ms={1000*sum(lat)/max(len(lat),1):.1f} "
              f"ia_std_ms_mean={1000*sum(ia)/max(len(ia),1):.1f} secs_without_data={sum(1 for r in rows if int(r['n_samples'])==0)}")
