"""M3b · Tarea A: ensayos tocados vs. no tocados (¿importa DÓNDE cae la falla?). EXPLORATORIO, post hoc.

Cada ensayo de una ejecución con fallo se empareja con EL MISMO ensayo (sujeto, corrida, índice; onsets idénticos
a 1e-6 s y etiquetas iguales, verificado en m3b_common.load_paired) de la ejecución de referencia ('ref').
  tocado   : n_samples < 1000 (ventana [onset+2, onset+6] s incompleta) o ensayo inválido.
  cambio   : la decisión difiere de la de la referencia (decisión que desaparece = cambio).
  acierto  : decisión válida e igual a la etiqueta (inválido = error, como en la balanced accuracy móvil).
Unidad de inferencia = sujeto: se agrega por sujeto (ensayos de sus 5 corridas) antes de cualquier prueba.

Salidas (analysis-m3b/):
  m3b_A_trials.csv         ensayos emparejados (ambos decodificadores)
  m3b_A_subject.csv        por condición × sujeto × grupo (tocado / no tocado)
  m3b_A_cond.csv           por condición (y grupos 'loss-*', 'disconnect-*'): tasas y pruebas a nivel sujeto
  m3b_A_dosis.csv          dosis-respuesta por bin de n_samples (todas las condiciones con fallo)
  m3b_A_dosis_familia.csv  lo mismo desglosado por familia de fallo
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

from m3b_common import (DECODERS, FAULT_LABELS, LABEL_ORDER, OUT, load_paired, wilcoxon_paired, wilson)

TAGS = {"CSP+LDA": "csp", "EEGNet": "net"}
BINS = ["1000", "900-999", "750-899", "500-749", "125-499", "inválido"]


def bin_of(n, valid):
    if valid == 0:
        return "inválido"
    if n >= 1000:
        return "1000"
    if n >= 900:
        return "900-999"
    if n >= 750:
        return "750-899"
    if n >= 500:
        return "500-749"
    return "125-499"


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m); tmp[idx] = adj
    out[ok] = tmp
    return out


def main():
    D = load_paired()
    for tag in TAGS.values():       # cambio dirigido: perjudicial (acertaba, ya no) / favorable (fallaba, ahora acierta)
        D[f"harm_{tag}"] = ((D[f"ref_ok_{tag}"] == 1) & (D[f"ok_{tag}"] == 0)).astype(int)
        D[f"help_{tag}"] = ((D[f"ref_ok_{tag}"] == 0) & (D[f"ok_{tag}"] == 1)).astype(int)
    D.to_csv(os.path.join(OUT, "m3b_A_trials.csv"), index=False)
    F = D[D.label_cond != "ref"].copy()
    F["grupo"] = np.where(F.touched, "tocado", "no_tocado")
    print("ensayos con fallo:", len(F), "| ref:", (D.label_cond == "ref").sum())
    # chequeo de determinismo del propio emparejamiento: ref contra ref
    R = D[D.label_cond == "ref"]
    assert (R.chg_csp == 0).all() and (R.chg_net == 0).all() and not R.touched.any()
    assert (R.valid_csp == 1).all() and (R.valid_net == 1).all()

    # pseudo-condiciones agrupadas (solo para tener más potencia donde el contraste tocado/no tocado existe)
    F["cond"] = F.label_cond
    extra = []
    for name, mask in (("loss-* (3)", F.kind == "loss"),
                       ("disconnect-* (3)", F.kind == "disconnect"),
                       ("jitter-*+delay-* (6)", F.kind.isin(["jitter", "delay"])),
                       ("burst_trial-* (3)", F.kind == "burst_trial"),
                       ("disconnect_trial-* (3)", F.kind == "disconnect_trial")):
        g = F[mask].copy(); g["cond"] = name; extra.append(g)
    FA = pd.concat([F] + extra, ignore_index=True)

    # ------------------------------------------------------------ por condición × sujeto × grupo
    sub_rows = []
    for dec, tag in TAGS.items():
        g = (FA.groupby(["cond", "subject", "grupo"])
               .agg(n=("i", "size"), chg=(f"chg_{tag}", "sum"), ok=(f"ok_{tag}", "sum"), ref_ok=(f"ref_ok_{tag}", "sum"),
                    harm=(f"harm_{tag}", "sum"), help=(f"help_{tag}", "sum"),
                    inval=(f"valid_{tag}", lambda s: int((s == 0).sum())),
                    chg_valid_n=(f"chg_valid_{tag}", "count"), chg_valid=(f"chg_valid_{tag}", "sum"))
               .reset_index())
        g.insert(0, "decoder", dec)
        sub_rows.append(g)
    S = pd.concat(sub_rows, ignore_index=True)
    S["chg_rate"] = S.chg / S.n; S["acc"] = S.ok / S.n; S["acc_ref"] = S.ref_ok / S.n
    S.to_csv(os.path.join(OUT, "m3b_A_subject.csv"), index=False)

    # ------------------------------------------------------------ por condición
    conds = FAULT_LABELS + ["loss-* (3)", "disconnect-* (3)", "jitter-*+delay-* (6)", "burst_trial-* (3)",
                            "disconnect_trial-* (3)"]
    out = []
    for dec in TAGS:
        for c in conds:
            s = S[(S.decoder == dec) & (S.cond == c)]
            row = dict(decoder=dec, cond=c)
            piv = {}
            for gname in ("tocado", "no_tocado"):
                gg = s[s.grupo == gname]
                n, k, ok, rok = gg.n.sum(), gg.chg.sum(), gg.ok.sum(), gg.ref_ok.sum()
                row.update({f"n_{gname}": int(n), f"n_inval_{gname}": int(gg.inval.sum()),
                            f"chg_{gname}": k / n if n else np.nan,
                            f"chg_{gname}_lo": wilson(k, n)[0], f"chg_{gname}_hi": wilson(k, n)[1],
                            f"chg_valid_{gname}": gg.chg_valid.sum() / gg.chg_valid_n.sum() if gg.chg_valid_n.sum() else np.nan,
                            f"harm_{gname}": gg.harm.sum() / n if n else np.nan,
                            f"help_{gname}": gg.help.sum() / n if n else np.nan,
                            f"acc_{gname}": ok / n if n else np.nan,
                            f"acc_ref_mismos_{gname}": rok / n if n else np.nan,
                            f"d_acc_{gname}": (ok - rok) / n if n else np.nan,
                            f"n_subj_{gname}": int((gg.n > 0).sum())})
                piv[gname] = gg.set_index("subject")
                # pruebas de un grupo contra 0 / contra la referencia (nivel sujeto)
                if len(gg):
                    nz, _, p = wilcoxon_paired(gg.chg_rate.to_numpy(), np.zeros(len(gg)))
                    row[f"p_chg_vs0_{gname}"] = p
                    row[f"subj_chg_gt0_{gname}"] = int((gg.chg_rate > 0).sum())
                    nz, _, p = wilcoxon_paired(gg.acc.to_numpy(), gg.acc_ref.to_numpy())
                    row[f"p_dacc_vs_ref_{gname}"] = p
                    row[f"med_subj_dacc_{gname}"] = float((gg.acc - gg.acc_ref).median())
                    row[f"med_subj_chg_{gname}"] = float(gg.chg_rate.median())
            # contraste tocado vs no tocado, apareado por sujeto (solo sujetos con ambos tipos)
            both = piv["tocado"].index.intersection(piv["no_tocado"].index)
            row["n_subj_ambos"] = len(both)
            if len(both) >= 3:
                a, b = piv["tocado"].loc[both], piv["no_tocado"].loc[both]
                _, _, row["p_chg_t_vs_u"] = wilcoxon_paired(a.chg_rate.to_numpy(), b.chg_rate.to_numpy())
                row["med_subj_diff_chg"] = float((a.chg_rate - b.chg_rate).median())
                _, _, row["p_acc_t_vs_u"] = wilcoxon_paired(a.acc.to_numpy(), b.acc.to_numpy())
                row["med_subj_diff_acc"] = float((a.acc - b.acc).median())
                row["subj_chg_t_gt_u"] = int((a.chg_rate > b.chg_rate).sum())
            out.append(row)
    C = pd.DataFrame(out)
    C["_o"] = C.cond.map({c: i for i, c in enumerate(conds)})
    C = C.sort_values(["decoder", "_o"]).drop(columns="_o")
    C.to_csv(os.path.join(OUT, "m3b_A_cond.csv"), index=False)

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 200)
    cols = ["decoder", "cond", "n_tocado", "n_no_tocado", "n_inval_tocado", "chg_tocado", "chg_no_tocado",
            "acc_tocado", "acc_ref_mismos_tocado", "acc_no_tocado", "acc_ref_mismos_no_tocado"]
    print("\n== A1/A2: tasas por condición (agrupado en ensayos)")
    print(C[cols].round(3).to_string(index=False))
    cols2 = ["decoder", "cond", "n_subj_ambos", "n_subj_tocado", "med_subj_chg_tocado", "med_subj_chg_no_tocado",
             "p_chg_t_vs_u", "subj_chg_t_gt_u", "p_acc_t_vs_u", "p_chg_vs0_tocado", "p_chg_vs0_no_tocado",
             "med_subj_dacc_tocado", "p_dacc_vs_ref_tocado"]
    print("\n== pruebas a nivel sujeto")
    print(C[cols2].round(4).to_string(index=False))

    # ------------------------------------------------------------ dosis-respuesta
    F["bin"] = [bin_of(n, v) for n, v in zip(F.n_samples, F.valid_online)]
    F["bin"] = pd.Categorical(F["bin"], BINS, ordered=True)
    rows, fam_rows = [], []
    for dec, tag in TAGS.items():
        Sb = (F.groupby(["subject", "bin"], observed=True)
                .agg(n=("i", "size"), chg=(f"chg_{tag}", "sum"), ok=(f"ok_{tag}", "sum"), ref_ok=(f"ref_ok_{tag}", "sum"))
                .reset_index())
        Sb["chg_rate"] = Sb.chg / Sb.n; Sb["acc"] = Sb.ok / Sb.n; Sb["acc_ref"] = Sb.ref_ok / Sb.n
        base = Sb[Sb.bin == "1000"].set_index("subject")
        res = []
        for b in BINS:
            g = F[F["bin"] == b]
            sb = Sb[Sb.bin == b].set_index("subject")
            n = len(g); k = int(g[f"chg_{tag}"].sum()); ok = int(g[f"ok_{tag}"].sum()); rok = int(g[f"ref_ok_{tag}"].sum())
            nv = int((g[f"valid_{tag}"] == 1).sum())
            both = sb.index.intersection(base.index)
            p_chg = p_acc = np.nan
            if b != "1000" and len(both) >= 3:
                _, _, p_chg = wilcoxon_paired(sb.loc[both].chg_rate.to_numpy(), base.loc[both].chg_rate.to_numpy())
                _, _, p_acc = wilcoxon_paired(sb.loc[both].acc.to_numpy(), base.loc[both].acc.to_numpy())
            res.append(dict(decoder=dec, bin=b, n_trials=n, n_subj=len(sb), n_valid=nv,
                            chg=k / n if n else np.nan, chg_lo=wilson(k, n)[0], chg_hi=wilson(k, n)[1],
                            acc=ok / n if n else np.nan, acc_lo=wilson(ok, n)[0], acc_hi=wilson(ok, n)[1],
                            acc_valid=(g[f"ok_{tag}"][g[f"valid_{tag}"] == 1].sum() / nv) if nv else np.nan,
                            acc_ref_mismos=rok / n if n else np.nan,
                            harm=float(g[f"harm_{tag}"].mean()) if n else np.nan,
                            help=float(g[f"help_{tag}"].mean()) if n else np.nan,
                            med_subj_chg=float(sb.chg_rate.median()) if len(sb) else np.nan,
                            q1_subj_chg=float(sb.chg_rate.quantile(.25)) if len(sb) else np.nan,
                            q3_subj_chg=float(sb.chg_rate.quantile(.75)) if len(sb) else np.nan,
                            med_subj_acc=float(sb.acc.median()) if len(sb) else np.nan,
                            n_subj_vs1000=len(both), p_chg_vs1000=p_chg, p_acc_vs1000=p_acc))
        R_ = pd.DataFrame(res)
        R_["p_chg_vs1000_holm"] = holm(R_.p_chg_vs1000.to_numpy())
        R_["p_acc_vs1000_holm"] = holm(R_.p_acc_vs1000.to_numpy())
        rows.append(R_)
        # desglose por familia
        for fam, gf in F.groupby("family"):
            for b in BINS:
                g = gf[gf["bin"] == b]
                if len(g) == 0:
                    continue
                fam_rows.append(dict(decoder=dec, family=fam, bin=b, n_trials=len(g),
                                     chg=g[f"chg_{tag}"].mean(), acc=g[f"ok_{tag}"].mean(),
                                     acc_ref_mismos=g[f"ref_ok_{tag}"].mean()))
    Dz = pd.concat(rows, ignore_index=True)
    Dz.to_csv(os.path.join(OUT, "m3b_A_dosis.csv"), index=False)
    Fz = pd.DataFrame(fam_rows)
    Fz.to_csv(os.path.join(OUT, "m3b_A_dosis_familia.csv"), index=False)
    print("\n== A3: dosis-respuesta (18 condiciones con fallo, agrupado)")
    print(Dz.round(4).to_string(index=False))
    print("\n== dosis-respuesta por familia")
    print(Fz.round(3).to_string(index=False))

    # correlación continua (ensayo) entre fracción perdida y cambio, solo ensayos válidos, todas las condiciones con fallo
    print("\nSpearman fracción perdida vs cambio (ensayo, válidos, todas las condiciones con fallo):")
    for dec, tag in TAGS.items():
        v = F[(F[f"valid_{tag}"] == 1) & (F[f"ref_valid_{tag}"] == 1)]
        lost = 1 - v.n_samples / 1000
        r = stats.spearmanr(lost, v[f"chg_valid_{tag}"])  # p=0 significa < 1e-300
        rt = v[v.touched]
        r2 = stats.spearmanr(1 - rt.n_samples / 1000, rt[f"chg_valid_{tag}"])
        print(f"  {dec}: todos rho={r.statistic:.3f} p={r.pvalue:.2g} (n={len(v)}) | solo tocados rho={r2.statistic:.3f} p={r2.pvalue:.2g} (n={len(rt)})")


if __name__ == "__main__":
    main()
