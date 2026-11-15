#!/usr/bin/env python3
"""Every combination of crystal and liquid treatment, every usable assessment, eight binary compositions with reported
diameters. The coefficient is set on Cu50Zr50 = 2 mm inside each case. All values go to reality_run.json.

Usage: reality_run.py spectrum.json|placeholder outname
Measured diameters are Ge et al. 2008 Table 5. Viscosity anchors are the three of Gangopadhyay et al. 2017, interpolated
linearly in composition for the other alloys, which is an assumption of this test.
"""
import itertools, json, os, sys, time
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import hyalos_reality as r
ROOT = os.environ.get("HALLSTEDT_TDB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdb"))
ASSESSMENTS = ["06He", "06Wan", "11Gie", "16Hsi", "17Liu"]
MEAS = {0.34: 1.0, 0.36: 2.0, 0.38: 1.0, 0.46: 1.0, 0.48: 1.0, 0.50: 2.0, 0.52: 1.0, 0.54: 2.0}
XA, TGA, TSA = [0.36, 0.50, 0.54], [733.0, 675.0, 633.0], [1090.0, 1064.0, 1030.0]
TM_GE = {0.34: 1284, 0.36: 1250, 0.38: 1212, 0.46: 1196, 0.48: 1205, 0.50: 1209, 0.52: 1205, 0.54: 1199}


def anchors(x):
    tg = np.polyval(np.polyfit(XA[:2], TGA[:2], 1), x) if x < 0.36 else np.interp(x, XA, TGA)
    ts = np.polyval(np.polyfit(XA[:2], TSA[:2], 1), x) if x < 0.36 else np.interp(x, XA, TSA)
    return float(tg), float(ts)


# ---- spectrum from the inherent structure runs
S = json.load(open(sys.argv[1]))
xs = np.array(sorted(float(k) for k in S))
ws = np.array([S[f"{k:.2f}"]["fits"]["all"]["w"] for k in xs])
wcoef = np.polyfit(xs, ws, 2)
wfun = lambda x: float(np.polyval(wcoef, np.clip(x, xs.min(), xs.max())))
# measured form, E_IS = a + b/T + c/T^2, so C_conf = -b/T^2 - 2c/T^3; coefficients smoothed in composition
abc = []
for k in xs:
    rows = S[f"{k:.2f}"]["rows"]; T = np.array([q["T"] for q in rows]); E = np.array([q["EIS"] for q in rows]); se = np.array([max(q["se"], 1.0) for q in rows])
    X = np.column_stack([np.ones(len(T)), 1 / T, 1 / T ** 2]) / se[:, None]
    # scale for conditioning
    p, *_ = np.linalg.lstsq(X, E / se, rcond=None); abc.append(p)
    C = -p[1] / T ** 2 - 2 * p[2] / T ** 3
    print(f"x {k:.2f}: chi2/n {(((E - np.column_stack([np.ones(len(T)), 1 / T, 1 / T ** 2]) @ p) / se) ** 2).sum() / (len(T) - 3):5.2f}  C_conf at 900/1000/1100/1200 K " + " ".join(f"{np.interp(t, T[::-1], C[::-1]):5.2f}" for t in (900, 1000, 1100, 1200)) + f"  extrapolated 800 K {-p[1] / 800 ** 2 - 2 * p[2] / 800 ** 3:5.2f}  700 K {-p[1] / 700 ** 2 - 2 * p[2] / 700 ** 3:5.2f}")
abc = np.array(abc)
bc = np.polyfit(xs, abc[:, 1], 2); cc = np.polyfit(xs, abc[:, 2], 2)
def measured_form(x, T):
    xx = np.clip(x, xs.min(), xs.max())
    b, c = np.polyval(bc, xx), np.polyval(cc, xx)
    return np.maximum(-b / T ** 2 - 2 * c / T ** 3, 0.0)
def calorimetric(x, T):
    return np.maximum(55.979 - 25.5 + (-0.01688 - 0.00517) * T, 0.0)
WS = {f"{k:.2f}": float(v) for k, v in zip(xs, ws)}
print("Gaussian widths, all rungs", WS)
LIQUIDS = {"Gaussian spectrum": r.gaussian_spectrum(wfun), "measured spectrum": measured_form, "calorimetric dCp of CuZr": calorimetric}

PHASE_SETS = {"all real phases": ["CU51ZR14", "CU8ZR3", "CU10ZR7", "CUZR", "CUZR2"],
              "rule: without B2 and Cu8Zr3": ["CU51ZR14", "CU10ZR7", "CUZR2"]}
OUT = dict(spectrum=WS, assessments={}, measured={str(k): v for k, v in MEAS.items()})
t0 = time.time()
for name in ASSESSMENTS:
    base = r.Tables.from_tdb(os.path.join(ROOT, f"CuZr-{name}.tdb"), name)
    tables = {"assessed liquid": base}
    tables.update({k: r.with_spectrum(base, f) for k, f in LIQUIDS.items()})
    rec = {}
    # thermodynamic check against calorimetry of CuZr
    ii = {T: int(np.argmin(np.abs(r.TGRID - T))) for T in (800, 900, 1000, 1100)}
    dcal = {800: 2.52, 900: 2.05, 1000: 1.46, 1100: 0.79}
    chk = {}
    for label, tb in tables.items():
        fB2 = r.forces(tb, 0.5, ["CUZR"])["CUZR"]
        chk[label] = {T: float(fB2[i] / 1e3) for T, i in ii.items()}
        chk[label + " dCp"] = {T: float(r.nk_difference(tb, 0.5)[ii[T]]) for T in (800, 1000, 1100)}
    rec["check"] = chk
    print(f"\n{name}: CuZr dG kJ/mol at 800/900/1000/1100 K: " + ";  ".join(f"{lab} " + "/".join(f"{chk[lab][T]:.2f}" for T in ii) for lab in tables) + ";  calorimetry 2.52/2.05/1.46/0.79")
    for liq, tb in tables.items():
        # mixture, the paper's construction
        cases = {"mixture": lambda x, tb=tb: r.channel_mixture(tb, x)}
        for setname, ph in PHASE_SETS.items():
            for part in (False, True):
                lab = setname + (", diffusion growth" if part else ", attachment growth")
                cases[lab] = (lambda x, tb=tb, ph=ph, part=part: r.channels_real(tb, x, [k for k in ph if k in tb.Gc], part))
        for lab, make in cases.items():
            key = liq + " | " + lab
            tg, ts = anchors(0.50)
            try:
                at = r.fit_coefficient(make(0.50), tg, ts, 2.0)
            except ValueError:
                rec[key] = dict(failed="no coefficient reaches 2 mm for Cu50Zr50"); print("  ", key, "no coefficient"); continue
            res = {}
            for x in MEAS:
                tg, ts = anchors(x)
                q = r.ttt(make(x), tg, ts, at)
                res[f"{x:.2f}"] = dict(dc=q["dc"], Tn=q["Tn"], Rc=q["Rc"], Tl=q["Tl"])
            m = np.array([MEAS[x] for x in MEAS if abs(x - 0.5) > 1e-6]); p = np.array([res[f"{x:.2f}"]["dc"] for x in MEAS if abs(x - 0.5) > 1e-6])
            e = np.log10(p / m)
            rec[key] = dict(aT=at, res=res, rms=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()))
            print(f"  {key:62s} aT {at:5.3f}  " + " ".join(f"{res[f'{x:.2f}']['dc']:8.2f}" for x in MEAS) + f"   rms {rec[key]['rms']:.2f}")
    OUT["assessments"][name] = rec
    print(f"  elapsed {time.time() - t0:.0f} s", flush=True)
print("\nmeasured" + " " * 65 + " ".join(f"{MEAS[x]:8.2f}" for x in MEAS))
# rivals, same alloys
tl = np.array([TM_GE[x] for x in MEAS]); ratio = np.array([anchors(x)[1] / TM_GE[x] for x in MEAS]); trg = np.array([anchors(x)[0] / TM_GE[x] for x in MEAS]); lm = np.log10(list(MEAS.values()))
def loo(v):
    err = []
    for i in range(len(v)):
        k = np.arange(len(v)) != i
        c = np.polyfit(v[k], lm[k], 1); err.append(np.polyval(c, v[i]) - lm[i])
    return float(np.sqrt(np.mean(np.square(err))))
OUT["rivals"] = dict(ratio_loo=loo(ratio), trg_loo=loo(trg), constant=float(lm.std()))
print("rivals, rms log10 error: T*/Tl line", round(OUT["rivals"]["ratio_loo"], 3), " Tg/Tl line", round(OUT["rivals"]["trg_loo"], 3), " constant", round(OUT["rivals"]["constant"], 3))
# summary over assessments
print("\nSummary: median and range of rms log10 error across assessments; rank correlation of the median diameters with measured")
keys = sorted({k for a in OUT["assessments"].values() for k in a if "|" in k})
summ = {}
for k in keys:
    vals = [a[k]["rms"] for a in OUT["assessments"].values() if k in a and "rms" in a[k]]
    if not vals:
        continue
    med = {f"{x:.2f}": float(np.median([a[k]["res"][f"{x:.2f}"]["dc"] for a in OUT["assessments"].values() if k in a and "res" in a[k]])) for x in MEAS}
    rho = spearmanr([med[f"{x:.2f}"] for x in MEAS], list(MEAS.values()))[0]
    ats = [a[k]["aT"] for a in OUT["assessments"].values() if k in a and "aT" in a[k]]
    summ[k] = dict(rms_median=float(np.median(vals)), rms_min=float(min(vals)), rms_max=float(max(vals)), n=len(vals), med=med, rho=float(rho), aT=[float(min(ats)), float(max(ats))])
    print(f"  {k:62s} rms {np.median(vals):5.2f} ({min(vals):.2f}-{max(vals):.2f})  aT {min(ats):.2f}-{max(ats):.2f}  rank {rho:+.2f}  Cu64Zr36 median {med['0.36']:.3g} mm")
OUT["summary"] = summ
json.dump(OUT, open(os.path.join(HERE, sys.argv[2]), "w"), indent=1, default=float)
