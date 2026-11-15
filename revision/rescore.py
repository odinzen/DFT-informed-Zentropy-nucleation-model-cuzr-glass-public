#!/usr/bin/env python3
"""Rescore the crystal and liquid model with the MACE spectrum from Sol, beside the earlier liquids.

Heat capacity forms per composition (x = 0.36, 0.50, 0.537), smoothed quadratically in x like the Mendelev runs:
  MACE configurational: dE_IS/dT, E_IS = a + b/T + c/T^2 over all cells and rungs at zero pressure.
  MACE configurational + vibrational: adds the liquid's vibrational heat capacity (dU/dT minus dE_IS/dT, no PV term,
  which is pressure noise) minus that of the crystals at the same composition. Electronic heat capacity is in neither
  phase of a classical potential and is taken to cancel.
Outputs every value to rescore.json."""
import glob, json, os, sys, time
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import hyalos_reality as r

RES = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sol_spectrum"))
F = 96485.33; kB = 8.617333e-5; Rg = r.Rg
ROOT = os.environ.get("HALLSTEDT_TDB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdb"))
ASSESSMENTS = ["06He", "06Wan", "11Gie", "16Hsi", "17Liu"]
MEAS = {0.34: 1.0, 0.36: 2.0, 0.38: 1.0, 0.46: 1.0, 0.48: 1.0, 0.50: 2.0, 0.52: 1.0, 0.54: 2.0}
XA, TGA, TSA = [0.36, 0.50, 0.54], [733.0, 675.0, 633.0], [1090.0, 1064.0, 1030.0]
COMPS = {"cu64zr36": 0.36, "cu50zr50": 0.50, "cu46zr54": 58 / 108}
REF = {"cu50zr50": {"CuZr_B2": 1.0}, "cu64zr36": {"Cu8Zr3": 0.372, "Cu10Zr7": 0.628}, "cu46zr54": {"CuZr_B2": 0.778, "CuZr2": 0.222}}


def anchors(x):
    tg = np.polyval(np.polyfit(XA[:2], TGA[:2], 1), x) if x < 0.36 else np.interp(x, XA, TGA)
    ts = np.polyval(np.polyfit(XA[:2], TSA[:2], 1), x) if x < 0.36 else np.interp(x, XA, TSA)
    return float(tg), float(ts)


def load(tag):
    return [dict(np.load(p)) for p in sorted(glob.glob(os.path.join(RES, "runs", tag, "rung_*K.npz")))]


# ---- heat capacity forms from the Sol runs
crys = {}
for name in ("CuZr_B2", "Cu10Zr7", "CuZr2", "Cu8Zr3"):
    rs = load(name); T = np.array([float(q["T"]) for q in rs]); U = np.array([(q["epot_eV"].mean() + 1.5 * kB * float(q["T"])) * F for q in rs])
    crys[name] = np.polyfit(T, U, 2)
conf, vib = {}, {}
for comp, x in COMPS.items():
    rs = [q for c in "abc" for q in load(f"{comp}_{c}")]
    T = np.array([float(q["T"]) for q in rs])
    E = np.array([q["is_energy0_eV"].mean() * F for q in rs]); U = np.array([(q["epot_eV"].mean() + 1.5 * kB * float(q["T"])) * F for q in rs])
    X = np.column_stack([np.ones_like(T), 1 / T, 1 / T ** 2]); p, *_ = np.linalg.lstsq(X, E, rcond=None)
    pu = np.polyfit(T, U, 2); pe = np.polyfit(T, E, 2)
    conf[x] = p
    cry_d = lambda t, comp=comp: sum(f * (2 * crys[k][0] * t + crys[k][1]) for k, f in REF[comp].items())
    # liquid vibrational minus crystal, linear in T over the run range, held at its 850 K value below it
    tt = np.array([850.0, 1250.0])
    dv = np.array([(2 * pu[0] * t + pu[1]) - (2 * pe[0] * t + pe[1]) - cry_d(t) for t in tt])
    vib[x] = np.polyfit(tt, dv, 1)
    print(f"{comp}: C_conf at 800/900/1000/1208 K " + " ".join(f"{-p[1] / t ** 2 - 2 * p[2] / t ** 3:5.2f}" for t in (800, 900, 1000, 1208))
          + "   vibrational excess at 850/1000/1208 K " + " ".join(f"{np.polyval(vib[x], t):5.2f}" for t in (850, 1000, 1208)))
xs = np.array(sorted(conf)); P = np.array([conf[x] for x in xs]); V = np.array([vib[x] for x in xs])
bc = [np.polyfit(xs, P[:, k], 2) for k in (1, 2)]; vc = [np.polyfit(xs, V[:, k], 2) for k in (0, 1)]


def mace_conf(x, T):
    xx = np.clip(x, xs.min(), xs.max()); b, c = np.polyval(bc[0], xx), np.polyval(bc[1], xx)
    return np.maximum(-b / T ** 2 - 2 * c / T ** 3, 0.0)


def mace_total(x, T):
    xx = np.clip(x, xs.min(), xs.max())
    Tv = np.maximum(T, 850.0)
    return np.maximum(mace_conf(x, T) + np.polyval(vc[0], xx) * Tv + np.polyval(vc[1], xx), 0.0)


# the Mendelev form used before, for the side-by-side
S = json.load(open(os.path.join(HERE, "spectrum_widths.json")))
mx = np.array(sorted(float(k) for k in S)); abc = []
for k in mx:
    rows = S[f"{k:.2f}"]["rows"]; T = np.array([q["T"] for q in rows]); E = np.array([q["EIS"] for q in rows]); se = np.array([max(q["se"], 1.0) for q in rows])
    abc.append(np.linalg.lstsq(np.column_stack([np.ones(len(T)), 1 / T, 1 / T ** 2]) / se[:, None], E / se, rcond=None)[0])
abc = np.array(abc); mb, mc = np.polyfit(mx, abc[:, 1], 2), np.polyfit(mx, abc[:, 2], 2)


def mendelev(x, T):
    xx = np.clip(x, mx.min(), mx.max()); return np.maximum(-np.polyval(mb, xx) / T ** 2 - 2 * np.polyval(mc, xx) / T ** 3, 0.0)


def calorimetric(x, T):
    return np.maximum(55.979 - 25.5 + (-0.01688 - 0.00517) * T, 0.0)


LIQ = {"Mendelev configurational": mendelev, "MACE configurational": mace_conf, "MACE configurational + vibrational": mace_total, "calorimetric dCp of CuZr": calorimetric}
print("\nheat capacity used for Cu50Zr50, J/mol/K, at 800/900/1000/1208 K")
for k, f in LIQ.items():
    print(f"  {k:36s} " + " ".join(f"{float(f(0.5, np.array(t))):5.2f}" for t in (800.0, 900.0, 1000.0, 1208.0)))
print("  calorimetry (liquid minus Neumann-Kopp crystal)    " + " ".join(f"{calorimetric(0.5, np.array(t)):5.2f}" for t in (800.0, 900.0, 1000.0, 1208.0)))

RULE = ["CU51ZR14", "CU10ZR7", "CUZR2"]
OUT = dict(assessments={}, measured={str(k): v for k, v in MEAS.items()})
t0 = time.time()
for name in ASSESSMENTS:
    base = r.Tables.from_tdb(os.path.join(ROOT, f"CuZr-{name}.tdb"), name)
    tabs = {"assessed liquid": base}; tabs.update({k: r.with_spectrum(base, f) for k, f in LIQ.items()})
    rec = {"check": {k: {T: float(r.forces(tb, 0.5, ["CUZR"])["CUZR"][int(np.argmin(np.abs(r.TGRID - T)))] / 1e3) for T in (800, 900, 1000, 1100)} for k, tb in tabs.items()}}
    print(f"\n{name}: CuZr dG at 800/900/1000/1100 K, kJ/mol: " + ";  ".join(f"{k} " + "/".join(f"{v:.2f}" for v in rec['check'][k].values()) for k in tabs) + ";  calorimetry 2.52/2.05/1.46/0.79")
    for liq, tb in tabs.items():
        cases = {"mixture": lambda x, tb=tb: r.channel_mixture(tb, x),
                 "rule, attachment growth": lambda x, tb=tb: r.channels_real(tb, x, [k for k in RULE if k in tb.Gc], False),
                 "rule, diffusion growth": lambda x, tb=tb: r.channels_real(tb, x, [k for k in RULE if k in tb.Gc], True)}
        for lab, make in cases.items():
            key = f"{liq} | {lab}"
            tg, ts = anchors(0.50)
            try:
                at = r.fit_coefficient(make(0.50), tg, ts, 2.0)
            except ValueError:
                rec[key] = dict(failed="no coefficient reaches 2 mm for Cu50Zr50"); continue
            res = {}
            for x in MEAS:
                tg, ts = anchors(x); q = r.ttt(make(x), tg, ts, at); res[f"{x:.2f}"] = dict(dc=q["dc"], Tn=q["Tn"])
            m = np.array([MEAS[x] for x in MEAS if abs(x - .5) > 1e-6]); p = np.array([res[f"{x:.2f}"]["dc"] for x in MEAS if abs(x - .5) > 1e-6]); e = np.log10(p / m)
            rec[key] = dict(aT=at, res=res, rms=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()))
    OUT["assessments"][name] = rec
    print(f"  elapsed {time.time() - t0:.0f} s", flush=True)
print("\nSummary, across the five assessments (measured 1-2 mm; constant guess scores 0.146)")
keys = sorted({k for a in OUT["assessments"].values() for k in a if "|" in k})
for k in keys:
    A = [a[k] for a in OUT["assessments"].values() if k in a and "rms" in a[k]]
    med = {x: float(np.median([q["res"][x]["dc"] for q in A])) for x in A[0]["res"]}
    e = np.array([np.log10(q["res"][x]["dc"] / MEAS[float(x)]) for q in A for x in q["res"] if x != "0.50"])
    print(f"  {k:62s} rms {np.median([q['rms'] for q in A]):5.2f} ({min(q['rms'] for q in A):.2f}-{max(q['rms'] for q in A):.2f})  within x3 {np.mean(abs(e) < np.log10(3)) * 100:3.0f}%  "
          f"aT {min(q['aT'] for q in A):.2f}-{max(q['aT'] for q in A):.2f}  Cu64Zr36 {med['0.36']:.3g}  Cu46Zr54 {med['0.54']:.3g}  rank {spearmanr(list(med.values()), list(MEAS.values()))[0]:+.2f}")
    OUT.setdefault("summary", {})[k] = dict(median_rms=float(np.median([q['rms'] for q in A])), med=med)
json.dump(OUT, open(os.path.join(HERE, "rescore.json"), "w"), indent=1, default=float)
