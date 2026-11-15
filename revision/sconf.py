#!/usr/bin/env python3
"""Configurational entropy of the liquid at the liquidus and the temperature where it runs out.

S_conf(Tl) is the liquid entropy from thermodynamic integration (Appendix B) minus the harmonic vibrational entropy of
the inherent structures quenched at Tl on Sol. Below Tl it falls by the integral of C_conf/T, with C_conf = dE_IS/dT from
E_IS = a + b/T + c/T^2 over every rung. The anharmonic share is not separated in the harmonic S_vib, so it is removed in
steps of 0, 2 and 4 J/mol/K. Usage: python sconf.py [sol results folder]"""
import glob, json, os, sys
import numpy as np

RES = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sol_spectrum"))
WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration_free_liquidus", "ti_driver", "results_local")
F = 96485.33; kB = 8.617333e-5; R = 8.314462; H_OVER_K = 47.992   # h/kB in K per THz


def svib(freq, T):
    f = np.sort(freq); f = f[f > 0.05]; x = H_OVER_K * f / T
    return R * np.sum(x / np.expm1(x) - np.log1p(-np.exp(-x))) / (len(freq) / 3)


out = {}
for comp, Tl in (("cu50zr50", 1208), ("cu64zr36", 1230)):
    S = []
    for f in glob.glob(os.path.join(TI, f"Fliquid_{comp}_*_{Tl}K.json")):
        d = json.load(open(f)); S.append((d["epot_mean_eV"] + 1.5 * kB * Tl - d["F_liquid_eV"]) / Tl * F); V = d["volume_A3_per_atom"]
    rs = [dict(np.load(p)) for p in glob.glob(os.path.join(RES, "runs", f"{comp}_*", f"rung_{Tl}K.npz"))]
    sv = [svib(f, Tl) for r in rs for f in r["is_freq_THz"]]
    Vis = np.mean([r["is_volume0"].mean() for r in rs])
    allr = [dict(np.load(p)) for p in glob.glob(os.path.join(RES, "runs", f"{comp}_*", "rung_*K.npz"))]
    T = np.array([float(r["T"]) for r in allr]); E = np.array([r["is_energy0_eV"].mean() * F for r in allr])
    p = np.linalg.lstsq(np.column_stack([np.ones_like(T), 1 / T, 1 / T ** 2]), E, rcond=None)[0]
    C = lambda t: -p[1] / t ** 2 - 2 * p[2] / t ** 3
    base = np.mean(S) - np.mean(sv)
    print(f"{comp}: S_liquid {np.mean(S):.1f} (cells {np.round(S, 1).tolist()}), harmonic S_vib of inherent structures {np.mean(sv):.1f} "
          f"({len(sv)} structures); volume liquid {V:.2f}, inherent structures {Vis:.2f} A^3/atom")
    out[comp] = dict(S_liquid=float(np.mean(S)), S_vib=float(np.mean(sv)), V_liquid=float(V), V_IS=float(Vis), zero={})
    for anh in (0, 2, 4):
        s0 = base - anh; Tg = np.linspace(Tl, 300, 4000)
        s = s0 - np.concatenate([[0], np.cumsum((C(Tg[:-1]) / Tg[:-1] + C(Tg[1:]) / Tg[1:]) / 2 * -np.diff(Tg))])
        TK = float(Tg[np.argmax(s <= 0)]) if (s <= 0).any() else float("nan")
        out[comp]["zero"][anh] = dict(S_conf_Tl=float(s0), T_zero=TK)
        print(f"   anharmonic share {anh}: S_conf(Tl) {s0:.1f} J/mol/K, zero at {TK:.0f} K")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sconf.json"), "w"), indent=1)
