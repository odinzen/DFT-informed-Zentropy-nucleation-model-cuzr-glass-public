#!/usr/bin/env python3
"""The decisive test of the structure idea. Same viscosity, same kinetics, assessed Gibbs energies, three nuclei:
the lever mixture the paper assumes, the crystal observed to form first, and whichever real phase nucleates fastest.
The Turnbull coefficient is set on Cu50Zr50 = 2 mm inside each case (as Castellero did), then carried to the others.
An assessment is kept only if its liquid gives the assessed Cu8Zr3 liquidus within 40 K of 1245 K; the rest are
printed with the reason. Every value is written out, nothing is dropped from the JSON."""
import glob, json, os, sys, warnings
from dataclasses import replace
import numpy as np
from scipy.optimize import brentq
from pycalphad import Database, calculate
warnings.filterwarnings("ignore")

import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
kB = 1.380649e-23; Rg = 8.314462; NA = 6.02214076e23
ETA0 = 2.3e-4; LAM = 2.8e-10; X = 1e-6
V_CU, V_ZR = 7.11e-6, 14.02e-6
ROOT = os.environ.get("HALLSTEDT_TDB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdb"))
XZR = {"CU51ZR14": 14 / 65, "CU8ZR3": 3 / 11, "CU10ZR7": 7 / 17, "CUZR": 0.5, "CUZR2": 2 / 3}
# viscosity anchors, Gangopadhyay Table 1; measured diameters
ALLOYS = {"Cu50Zr50": dict(x=0.50, Tg=675, Ts=1064, dc=2.0, first=["CUZR"], first2=["CU10ZR7", "CUZR2"]),
          "Cu64Zr36": dict(x=0.36, Tg=733, Ts=1090, dc=2.0, first=["CU8ZR3", "CU10ZR7"], first2=["CU10ZR7", "CU51ZR14"]),
          "Cu46Zr54": dict(x=0.54, Tg=633, Ts=1030, dc=2.0, first=["CUZR", "CUZR2"], first2=["CU10ZR7", "CUZR2"])}
TG = np.arange(680.0, 1300.0, 10.0)


def vft(Tg, Ts):
    a, b = np.log(1e12 / ETA0), np.log(0.1 / ETA0); T0 = (a * Tg - b * Ts) / (a - b); return b * (Ts - T0), T0


def tables(db):
    phs = {}
    for p in db.phases:
        q = p.upper()
        for k in XZR:
            # names such as CUZR_B2 or CU10ZR7_S; CUZR2 must not match CUZR
            if q == k or (q.startswith(k) and not q[len(k)].isdigit()):
                phs.setdefault(k, p)
    if "CUZR" not in phs:
        for p in db.phases:
            if p.upper() in ("BCC_B2", "B2", "CUZR_B2", "B2_BCC"):
                phs["CUZR"] = p
    xs = np.linspace(0.18, 0.72, 109); GL = np.zeros((len(TG), len(xs))); Gc = {k: np.zeros(len(TG)) for k in phs}
    for i, T in enumerate(TG):
        for j, x in enumerate(xs):
            GL[i, j] = float(calculate(db, ["CU", "ZR"], "LIQUID", P=101325, T=T, N=1, output="GM", points=np.array([[1 - x, x]])).GM.values.squeeze())
        for k, p in phs.items():
            r = calculate(db, ["CU", "ZR"], p, P=101325, T=T, N=1, output="GM", pdens=50)
            xx = np.atleast_1d(r.X.sel(component="ZR").values.squeeze()); ee = np.atleast_1d(r.GM.values.squeeze())
            Gc[k][i] = float(ee[np.argmin(np.abs(xx - XZR[k]))])
    return xs, GL, Gc


def forces(xs, GL, Gc, x0):
    """Parallel tangent force per mole of atoms of nucleus at each T; mixture force at x0."""
    D = {k: np.zeros(len(TG)) for k in Gc}; mix = np.zeros(len(TG))
    for i in range(len(TG)):
        g0 = np.interp(x0, xs, GL[i]); s0 = np.interp(x0, xs, np.gradient(GL[i], xs))
        for k in Gc:
            D[k][i] = g0 + (XZR[k] - x0) * s0 - Gc[k][i]
        best = None
        ks = sorted(Gc, key=lambda k: XZR[k])
        for a in ks:
            for b in ks:
                if XZR[a] < x0 < XZR[b]:
                    f = (x0 - XZR[a]) / (XZR[b] - XZR[a]); e = (1 - f) * Gc[a][i] + f * Gc[b][i]
                    best = e if best is None else min(best, e)
            if abs(XZR[a] - x0) < 1e-9:
                best = Gc[a][i] if best is None else min(best, Gc[a][i])
        mix[i] = g0 - best if best is not None else np.nan
    return D, mix


def fusion_H(xs, GL, Gc, k):
    """Enthalpy of fusion of phase k at its own composition, near 1200 K, from G and dG/dT. For a nucleus of k."""
    i = np.argmin(np.abs(TG - 1200.0)); g = np.array([np.interp(XZR[k], xs, GL[t]) - Gc[k][t] for t in range(len(TG))])
    dS = -np.gradient(g, TG)[i]
    return float(g[i] + 1200.0 * dS)


def dc_of(forces_list, al, aT):
    """forces_list: (dG array, dHf, Vm). Rates add; growth is taken from the fastest channel at each T."""
    B, T0 = vft(al["Tg"], al["Ts"])
    Ts = np.linspace(al["Tg"] + 5, 1215, 700); out = np.empty_like(Ts)
    TlEff = None
    for n, T in enumerate(Ts):
        eta = ETA0 * np.exp(B / (T - T0)); I = 0.0; U = 1e-30
        for g_arr, dH, Vm in forces_list:
            g = np.interp(T, TG, g_arr)
            if g <= 1.0:
                continue
            sig = aT * dH / (NA ** (1 / 3) * Vm ** (2 / 3))
            bar = 16 * np.pi * sig ** 3 / (3 * (g / Vm) ** 2)
            I += (NA / Vm) * (kB * T / (3 * np.pi * LAM ** 3 * eta)) * np.exp(-bar / (kB * T))
            U = max(U, (kB * T / (3 * np.pi * LAM ** 2 * eta)) * (1 - np.exp(-g / (Rg * T))))
        out[n] = np.exp(0.25 * (np.log(3 * X / np.pi) - np.log(max(I, 1e-300)) - 3 * np.log(U)))
    # liquidus of this channel set
    tl = max([TG[np.where(g_arr > 0)[0].max()] if (g_arr > 0).any() else 0 for g_arr, _, _ in forces_list])
    j = int(np.argmin(out)); Rc = max(tl - Ts[j], 1.0) / out[j]
    return 10 * np.sqrt(10 / Rc), float(Ts[j]), float(Rc), float(tl)


def vm(x):
    return (1 - x) * V_CU + x * V_ZR


def phs_names(db):
    return {p: "" for p in db.phases if p.upper().startswith("CU") or "B2" in p.upper()}


OUT = {}
for f in sorted(glob.glob(os.path.join(ROOT, "CuZr-*.tdb"))):
    name = os.path.basename(f)[5:-4]
    db = Database(f)
    try:
        xs, GL, Gc = tables(db)
    except Exception as e:
        print(f"\n{name}: could not be evaluated, {str(e)[:90]}")
        OUT[name] = dict(failed=str(e)[:200], kept=False)
        continue
    missing = [k for k in XZR if k not in Gc]
    D36, mix36 = forces(xs, GL, Gc, 0.36)
    tl8 = float("nan")
    if "CU8ZR3" in Gc:
        d8 = D36["CU8ZR3"]
        for a, b, ta, tb in zip(d8[:-1], d8[1:], TG[:-1], TG[1:]):
            if a > 0 >= b:
                tl8 = ta + (tb - ta) * a / (a - b)
    H = {k: fusion_H(xs, GL, Gc, k) for k in Gc}
    keep = ("CU8ZR3" in Gc) and abs(tl8 - 1245) < 40 and all(k in Gc for k in ("CU10ZR7", "CUZR"))
    rec = dict(tl_Cu8Zr3=tl8, H={k: v for k, v in H.items()}, missing=missing, kept=bool(keep))
    print("  phase names:", {k: v for k, v in sorted(phs_names(db).items())})
    print(f"\n{name}: Cu8Zr3 liquidus at x=0.36 {tl8:.0f} K; missing {missing}; " + ("kept" if keep else "NOT KEPT (liquid or phase description does not give the assessed liquidus)"))
    print("  fusion enthalpy near 1200 K, kJ/mol: " + "  ".join(f"{k} {v / 1e3:.1f}" for k, v in H.items()))
    if not keep:
        OUT[name] = rec
        continue
    cases = {}
    for label in ("mixture", "equilibrium phases, each its own nucleus", "crystals observed first", "fastest real phase"):
        def channels(key):
            al = ALLOYS[key]; D, mix = forces(xs, GL, Gc, al["x"])
            if label == "mixture":
                dH = 1200.0 * (-np.gradient(mix, TG)[np.argmin(np.abs(TG - 1200.0))]) + mix[np.argmin(np.abs(TG - 1200.0))]
                return [(mix, dH, vm(al["x"]))]
            if label == "equilibrium phases, each its own nucleus":
                ks = [k for k in al["first"] if k in Gc]
            elif label == "crystals observed first":
                ks = [k for k in al["first2"] if k in Gc]
            else:
                ks = list(Gc)
            return [(D[k], H[k], vm(XZR[k])) for k in ks]
        ch50 = channels("Cu50Zr50")
        try:
            at = brentq(lambda a: np.log(dc_of(ch50, ALLOYS["Cu50Zr50"], a)[0] / 2.0), 0.2, 1.5, xtol=1e-3)
        except ValueError:
            print(f"  {label}: no aT reaches 2 mm for Cu50Zr50"); continue
        r = {k: dc_of(channels(k), ALLOYS[k], at) for k in ALLOYS}
        cases[label] = dict(aT=at, dc={k: r[k][0] for k in r}, nose={k: r[k][1] for k in r}, Tl={k: r[k][3] for k in r})
        print(f"  {label:26s} aT {at:.3f}  " + "  ".join(f"{k} {r[k][0]:8.2f} mm (nose {r[k][1]:.0f})" for k in r))
    rec["cases"] = cases
    # same coefficient for mixture and real nuclei at Cu64Zr36, how much the barrier moves at the nose
    i850 = np.argmin(np.abs(TG - 850.0))
    rec["forces850"] = {"mix": float(mix36[i850]), **{k: float(D36[k][i850]) for k in Gc}}
    print("  forces at 850 K, Cu64Zr36, kJ/mol: mix %.2f; " % (mix36[i850] / 1e3) + "  ".join(f"{k} {D36[k][i850] / 1e3:.2f}" for k in Gc))
    OUT[name] = rec
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_phase_kinetics.json"), "w"), default=float, indent=1)
print("\nmeasured: 2 mm for all three")
