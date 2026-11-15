#!/usr/bin/env python3
"""Supplementary Tables S6-S8 from the saved results, so no value in Sections S11 and S12 is typed by hand.

S6  configurational heat capacity of the liquids (cconf_table.json, spectrum_widths.json)
S7  three alloys with each crystal description (real_phase_kinetics.json)
S8  the binary series, every liquid with every crystal description (reality_final.json, rescore.json)
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = HERE
ld = lambda *p: json.load(open(os.path.join(WORK, *p)))
cal = lambda t: (55.979 - 0.01688 * t) - (25.5 + 0.00517 * t)

# ---- S6
cc = ld("cconf_table.json")
S = ld("spectrum_widths.json")
def mendelev(x, t):
    rows = S[f"{x:.2f}"]["rows"]; T = np.array([q["T"] for q in rows]); E = np.array([q["EIS"] for q in rows]); se = np.array([max(q["se"], 1.0) for q in rows])
    a = np.linalg.lstsq(np.column_stack([np.ones(len(T)), 1 / T, 1 / T ** 2]) / se[:, None], E / se, rcond=None)[0]
    return -a[1] / t ** 2 - 2 * a[2] / t ** 3
TT = ("800", "900", "1000", "1208")
print("Table S6")
print("| Liquid | " + " | ".join(f"{t} K" for t in TT) + " |\n|---|---|---|---|---|")
for comp, x, lab in (("cu64zr36", 0.36, "Cu~64~Zr~36~"), ("cu50zr50", 0.50, "Cu~50~Zr~50~"), ("cu46zr54", 0.54, "Cu~46~Zr~54~")):
    print(f"| {lab}, MACE-MP-0 | " + " | ".join(f"{cc[comp][t]['C']:.1f} ± {cc[comp][t]['cell_sd']:.1f}" for t in TT) + " |")
    print(f"| {lab}, Mendelev EAM | " + " | ".join(f"{mendelev(x, float(t)):.1f}" for t in TT) + " |")
print("| CuZr, calorimetry | " + " | ".join(f"{cal(float(t)):.1f}" for t in TT) + " |")

# ---- S7
r = ld("real_phase_kinetics.json")
kept = [k for k, v in r.items() if v.get("kept")]
CASES = [("mixture", "Equilibrium assemblage as one nucleus"), ("equilibrium phases, each its own nucleus", "Equilibrium phases, each its own nucleus"),
         ("fastest real phase", "Fastest single phase"), ("crystals observed first", "Crystals reported first")]
print("\nTable S7  assessments:", kept)
print("| Crystal description | α~T~ | Cu~64~Zr~36~ (mm) | Cu~46~Zr~54~ (mm) |\n|---|---|---|---|")
fmt = lambda v: f"{v:.2g}" if v < 10 else (f"{v:.0f}" if v < 1e4 else f"{v:.1e}".replace("e+0", " × 10^").replace("e+", " × 10^") + "^")
for key, lab in CASES:
    a = [r[k]["cases"][key]["aT"] for k in kept]
    print(f"| {lab} | {min(a):.2f}–{max(a):.2f} | " + ", ".join(fmt(r[k]['cases'][key]['dc']['Cu64Zr36']) for k in kept)
          + " | " + ", ".join(fmt(r[k]['cases'][key]['dc']['Cu46Zr54']) for k in kept) + " |")

# ---- S8
F = ld("reality_final.json"); M = ld("rescore.json")
meas = {float(k): v for k, v in F["measured"].items()}
def stats(recs):
    """median typical error (range), share within a factor of 3, median Cu64Zr36 and Cu46Zr54."""
    recs = [q for q in recs if "rms" in q]
    e = np.array([np.log10(q["res"][x]["dc"] / meas[float(x)]) for q in recs for x in q["res"] if x != "0.50"])
    rms = [q["rms"] for q in recs]
    med = lambda x: float(np.median([q["res"][x]["dc"] for q in recs]))
    return dict(rms=np.median(rms), lo=min(rms), hi=max(rms), w3=np.mean(np.abs(e) < np.log10(3)) * 100, c64=med("0.36"), c46=med("0.54"))
LIQ = [("assessed liquid", F, "Assessed"), ("calorimetric dCp of CuZr", F, "Calorimetric ΔC~p~ of CuZr"),
       ("measured spectrum", F, "Mendelev EAM, configurational"), ("Gaussian spectrum", F, "Gaussian spectrum"),
       ("MACE configurational", M, "MACE-MP-0, configurational"), ("MACE configurational + vibrational", M, "MACE-MP-0, configurational and vibrational")]
COLS = [("mixture", "mixture"), ("all real phases, attachment growth", None), ("all real phases, diffusion growth", None),
        ("rule: without B2 and Cu8Zr3, attachment growth", "rule, attachment growth"), ("rule: without B2 and Cu8Zr3, diffusion growth", "rule, diffusion growth")]
print("\nTable S8")
print("| Liquid | Equilibrium assemblage | All phases, attachment | All phases, diffusion | Rule, attachment | Rule, diffusion |\n|---|---|---|---|---|---|")
detail = {}
for key, src, lab in LIQ:
    cells = []
    for fk, mk in COLS:
        k = fk if src is F else mk
        if k is None:
            cells.append("not run"); continue
        s = stats([a[f"{key} | {k}"] for a in src["assessments"].values()])
        cells.append(f"{s['rms']:.2f} ({s['lo']:.2f}–{s['hi']:.2f})"); detail[(lab, fk)] = s
    print(f"| {lab} | " + " | ".join(cells) + " |")
print("\nrivals", {k: round(v, 3) for k, v in F["rivals"].items()})
for (lab, fk), s in detail.items():
    print(f"  {lab:45s} {fk:48s} within x3 {s['w3']:3.0f}%  Cu64Zr36 {s['c64']:.3g}  Cu46Zr54 {s['c46']:.3g}")
