#!/usr/bin/env python3
"""E_IS(T) from the local LAMMPS runs: width of the Gaussian spectrum, its fit, and the comparison with calorimetry.
Usage: analyse_spectrum.py root outjson. Rungs where the liquid no longer equilibrates (E_IS flat in time) are flagged."""
import glob, json, os, sys
import numpy as np

root, outp = sys.argv[1], sys.argv[2]
Rg = 8.314462; F = 96485.33
OUT = {}
for d in sorted(glob.glob(os.path.join(root, "x0.*"))):
    x = float(os.path.basename(d)[1:])
    a = np.loadtxt(os.path.join(d, "spectrum.dat"))
    if a.ndim < 2 or len(a) < 20:
        print(x, "too few rows"); continue
    Ts = np.unique(a[:, 0])[::-1]
    rows = []
    for T in Ts:
        s = a[a[:, 0] == T]
        e = s[:, 5] * F; h = len(e) // 2
        tr = (e[h:].mean() - e[:h].mean())                       # J/mol, drift within the sampling window
        rows.append(dict(T=float(T), n=len(e), EIS=float(e.mean()), se=float(e.std(ddof=1) / np.sqrt(len(e))), sd=float(e.std(ddof=1)), drift=float(tr),
                         PE=float(s[:, 2].mean() * F), V=float(s[:, 3].mean()), VIS=float(s[:, 6].mean())))
    T = np.array([r["T"] for r in rows]); E = np.array([r["EIS"] for r in rows]); se = np.array([max(r["se"], 1.0) for r in rows])
    res = {}
    for label, mask in (("all", T >= 0), ("T>=1000", T >= 1000), ("T>=1100", T >= 1100)):
        X = np.column_stack([np.ones(mask.sum()), -1 / (Rg * T[mask])])
        wts = 1 / se[mask]
        c, *_ = np.linalg.lstsq(X * wts[:, None], E[mask] * wts, rcond=None)
        cov = np.linalg.inv((X * wts[:, None]).T @ (X * wts[:, None]))
        w2 = c[1]
        res[label] = dict(w=float(np.sqrt(max(w2, 0))), w_se=float(np.sqrt(cov[1, 1]) / (2 * np.sqrt(max(w2, 1)))), Einf=float(c[0]), chi2=float((((E[mask] - X @ c) / se[mask]) ** 2).sum()), n=int(mask.sum()))
        # linear alternative, constant C
        cl = np.polyfit(T[mask], E[mask], 1, w=wts)
        res[label]["C_linear"] = float(cl[0])
        res[label]["chi2_linear"] = float((((E[mask] - np.polyval(cl, T[mask])) / se[mask]) ** 2).sum())
    OUT[f"{x:.2f}"] = dict(rows=rows, fits=res)
    print(f"\nx_Zr {x:.2f}, {len(a)} quenches")
    for r in rows:
        print(f"   {r['T']:5.0f} K  E_IS {r['EIS'] / 1e3:9.3f} +- {r['se'] / 1e3:5.3f} kJ/mol  drift {r['drift'] / 1e3:+6.3f}  V_IS {r['VIS']:.3f}")
    for k, v in res.items():
        print(f"   fit {k:8s} w {v['w'] / 1e3:5.2f} +- {v['w_se'] / 1e3:4.2f} kJ/mol   chi2/n {v['chi2'] / max(v['n'] - 2, 1):6.2f}   linear C {v['C_linear']:5.2f} J/mol/K, chi2/n {v['chi2_linear'] / max(v['n'] - 2, 1):6.2f}")
    for fit in ("T>=1000",):
        w = res[fit]["w"]
        print("   C_conf (J/mol/K) at 700, 1000, 1208 K: " + ", ".join(f"{w ** 2 / (Rg * t ** 2):.1f}" for t in (700, 1000, 1208)) + "   (calorimetry for CuZr 15.0, 8.4, 3.8)")
json.dump(OUT, open(outp, "w"), indent=1)
