"""Decodificador híbrido con vigía (línea futura 5), prototipo fuera de línea.

Por ensayo, el vigía mira la telemetría de la ventana [2, 6] s (trials_sentinel.csv,
escrito por redecode_variants.py) y decide:
  ventana completa según la regla → decisión de EEGNet
  ventana con hueco               → decisión de CSP+LDA (fuera de línea, sobre las
                                    muestras presentes: trials_csp_offline.csv)
Reglas:
  complete : n_samples == 1000
  gap40    : máximo intervalo entre muestras presentes (con bordes) <= 40 ms
  gap200   : ídem <= 200 ms
Red: EEGNet original (trials_eegnet.csv) o endurecida (trials_eegnet_gapaug.csv, sufijo _gapaug).

Salidas: <exec>/trials_hybrid_<regla>[_gapaug].csv (columnas de trials_eegnet.csv + route)
         <out>/ruteo.csv y ruteo.md: fracción de ensayos válidos enviados a CSP por condición.
Uso: python scripts/hybrid_decoder.py --root results/campana2-eval --out results/robustez
"""
import argparse
import glob
import os

import pandas as pd

RULES = {
    "complete": lambda s: s["n_samples"] == 1000,
    "gap40": lambda s: s["max_gap_ms"] <= 40.0,
    "gap200": lambda s: s["max_gap_ms"] <= 200.0,
}
NETS = {"": "trials_eegnet.csv", "_gapaug": "trials_eegnet_gapaug.csv"}

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

route_rows = []
for d in sorted(glob.glob(os.path.join(a.root, "s*"))):
    sp, cp = os.path.join(d, "trials_sentinel.csv"), os.path.join(d, "trials_csp_offline.csv")
    if not (os.path.exists(sp) and os.path.exists(cp)):
        continue
    sen, csp = pd.read_csv(sp), pd.read_csv(cp)
    assert len(sen) == len(csp), d
    eid = os.path.basename(d)
    label = eid.split("-", 2)[2]
    for suf, fn in NETS.items():
        np_ = os.path.join(d, fn)
        if not os.path.exists(np_):
            continue
        net = pd.read_csv(np_)
        assert len(net) == len(sen), d
        for rule, f in RULES.items():
            use_net = f(sen).to_numpy()
            out = net.copy()
            out["route"] = ["net" if u else "csp" for u in use_net]
            for c in ["pred", "proba"]:
                out[c] = net[c].where(use_net, csp[c])
            valid = sen["valid"].to_numpy() == 1
            out["valid"] = valid.astype(int)
            out.loc[~valid, ["pred", "proba"]] = None
            out.loc[~valid, "route"] = ""
            out["n_samples"] = sen["n_samples"]
            out.to_csv(os.path.join(d, f"trials_hybrid_{rule}{suf}.csv"), index=False)
            if suf == "":
                route_rows.append(dict(exec_id=eid, subject=int(eid[1:3]), label=label, rule=rule,
                                       n_valid=int(valid.sum()), n_csp=int((valid & ~use_net).sum())))

R = pd.DataFrame(route_rows)
R.to_csv(os.path.join(a.out, "ruteo_por_ejecucion.csv"), index=False)
t = R.groupby(["label", "rule"])[["n_valid", "n_csp"]].sum().reset_index()
t["frac_csp"] = t["n_csp"] / t["n_valid"]
w = t.pivot(index="label", columns="rule", values="frac_csp").reset_index()
w = w[["label", "complete", "gap40", "gap200"]]
w.to_csv(os.path.join(a.out, "ruteo.csv"), index=False)
with open(os.path.join(a.out, "ruteo.md"), "w", encoding="utf-8") as fh:
    fh.write("Fracción de ensayos válidos enviados a CSP+LDA (el resto va a EEGNet), por condición y regla\n\n")
    fh.write(w.to_markdown(index=False, floatfmt=".3f"))
print(w.to_string())
