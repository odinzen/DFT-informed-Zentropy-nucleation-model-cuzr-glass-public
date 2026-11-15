#!/usr/bin/env python3
"""Configurational heat capacity of each liquid, dE_IS/dT from E_IS = a + b/T + c/T^2 over every rung of the three cells,
beside the Mendelev values of spectrum_local and the CuZr calorimetry. Usage: python cconf_table.py <sol results folder>"""
import glob, json, os, sys
import numpy as np

RES = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sol_spectrum")); F = 96485.33
HERE = os.path.dirname(os.path.abspath(__file__))
TT = (800, 900, 1000, 1208)
out = {}
for comp in ("cu64zr36", "cu50zr50", "cu46zr54"):
    rs = [dict(np.load(p)) for p in glob.glob(os.path.join(RES, "runs", f"{comp}_*", "rung_*K.npz"))]
    T = np.array([float(r["T"]) for r in rs]); E = np.array([r["is_energy0_eV"].mean() * F for r in rs])
    X = np.column_stack([np.ones_like(T), 1 / T, 1 / T ** 2]); p, *_ = np.linalg.lstsq(X, E, rcond=None)
    # spread between the three cells, each fitted alone
    cells = []
    for c in "abc":
        q = [r for r, path in zip(rs, glob.glob(os.path.join(RES, "runs", f"{comp}_*", "rung_*K.npz"))) if f"{comp}_{c}" in path]
        Tc = np.array([float(r["T"]) for r in q]); Ec = np.array([r["is_energy0_eV"].mean() * F for r in q])
        pc, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(Tc), 1 / Tc, 1 / Tc ** 2]), Ec, rcond=None); cells.append(pc)
    cc = lambda q, t: -q[1] / t ** 2 - 2 * q[2] / t ** 3
    out[comp] = {t: dict(C=float(cc(p, t)), cell_sd=float(np.std([cc(q, t) for q in cells], ddof=1))) for t in TT}
    print(f"{comp:9s} rungs {sorted(set(T.astype(int)))}  " + "  ".join(f"{t} K {out[comp][t]['C']:.1f} (sd {out[comp][t]['cell_sd']:.1f})" for t in TT))
cal = lambda t: (55.979 - 0.01688 * t) - (25.5 + 0.00517 * t)
print("calorimetry CuZr " + "  ".join(f"{t} K {cal(t):.1f}" for t in TT))
json.dump(out, open(os.path.join(HERE, "cconf_table.json"), "w"), indent=1)
