#!/usr/bin/env python3
"""Validation step: after config2_campaign.py computes soft1, soft2, and dE, re-run the
engine with those computed values and compare the predicted casting diameters against the
declared baseline (soft2 = 0.80, dE = 6.0) and against experiment. Self-contained; reads
config2_results.json. Covers the two binaries the EAM campaign produces.

Two honest outcomes, both real results:
  - computed soft2/dE reproduce the diameters  -> genuine first-principles two-config Zentropy
  - they do not                                 -> the two-config reduction's limit, reported as such
Do NOT tune soft2 back toward 0.80 to save the prediction.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import zentropy_driving_force as z

R, NA, kB = z.R, z.NA, 1.380649e-23
eta0, lam, x_det = 4e-5, 2.8e-10, 1e-6
Cvft = np.log(1e12 / eta0)
ALLOY = {  # Tg, Tl, D*, xZr, bounding phases, measured dc
    "Cu50Zr50": dict(Tg=673, Tl=1208, D=13.6, x=0.50, bnd=("Cu10Zr7", "CuZr"), meas=2.0),
    "Cu64Zr36": dict(Tg=745, Tl=1230, D=13.6, x=0.36, bnd=("Cu8Zr3", "Cu10Zr7"), meas=2.0),
}

def visc(T, Tg, D):
    T0 = Cvft * Tg / (D + Cvft); return eta0 * np.exp(D * T0 / (T - T0))

def dG_of(T, x, lo, hi, dE_ico, soft1, soft2, dE_split):
    E0c, thc, _ = z.crystalline_assembly(x, lo, hi)
    Gc = z.G_crystal(T, E0c, thc)
    Gl, _ = z.G_liquid(T, E0c, thc, dE_ico, dE_split, soft1=soft1, soft2=soft2)
    return Gl - Gc

def calib(x, lo, hi, Tl, soft1, soft2, dE_split):
    from scipy.optimize import brentq
    return brentq(lambda dE: dG_of(Tl, x, lo, hi, dE, soft1, soft2, dE_split), 2e3, 60e3, xtol=1.0)

def dc(a, soft1, soft2, dE_split):
    x, lo, hi, Tl, Tg, D = a["x"], *a["bnd"], a["Tl"], a["Tg"], a["D"]
    dE_ico = calib(x, lo, hi, Tl, soft1, soft2, dE_split)
    _, _, Vat = z.crystalline_assembly(x, lo, hi); Vm = Vat * NA
    dGf = lambda T: dG_of(T, x, lo, hi, dE_ico, soft1, soft2, dE_split)
    dSf = -(dGf(Tl + 1) - dGf(Tl - 1)) / 2.0; sig = 0.50 * Tl * dSf / (NA ** (1 / 3) * Vm ** (2 / 3))
    T = np.linspace(Tg + 5, Tl - 3, 1400); tt = np.empty_like(T)
    for i, Ti in enumerate(T):
        e = visc(Ti, Tg, D); g = max(dGf(Ti), 1.0); gv = g / Vm
        dGs = 16 * np.pi * sig ** 3 / (3 * gv ** 2)
        I = (NA / Vm) * (kB * Ti / (3 * np.pi * lam ** 3 * e)) * np.exp(-dGs / (kB * Ti))
        U = max((kB * Ti / (3 * np.pi * lam ** 2 * e)) * (1 - np.exp(-g / (R * Ti))), 1e-30)
        tt[i] = (3 * x_det / (np.pi * max(I, 1e-300) * U ** 3)) ** 0.25
    j = np.argmin(tt); Rc = (Tl - T[j]) / tt[j]
    return 10 * np.sqrt(10 / Rc)

if __name__ == "__main__":
    rf = os.path.join(HERE, "config2_results.json")
    if not os.path.exists(rf):
        sys.exit("Run config2_campaign.py first (config2_results.json not found).")
    res = json.load(open(rf))
    print(f"{'alloy':>10} {'soft1':>6} {'soft2':>6} {'dE':>5} | {'dc declared':>11} {'dc computed':>11} {'meas':>5}")
    for comp, a in ALLOY.items():
        rc = {c: [r for r in res if r["comp"] == comp and r["config"] == c]
              for c in ("config1_slow", "config2_fast")}
        if not all(rc.values()):
            print(f"{comp:>10}  (campaign not complete for this alloy)"); continue
        s1 = np.mean([r["soft"] for r in rc["config1_slow"]])
        s2 = np.mean([r["soft"] for r in rc["config2_fast"]])
        e1 = np.mean([r["E0_eV"] for r in rc["config1_slow"]])
        e2 = np.mean([r["E0_eV"] for r in rc["config2_fast"]])
        dE_kJ = (e2 - e1) * 96.485
        dc_decl = dc(a, 0.80, 0.80, 6.0e3)
        dc_comp = dc(a, s1, s2, dE_kJ * 1e3)
        print(f"{comp:>10} {s1:>6.3f} {s2:>6.3f} {dE_kJ:>5.1f} | "
              f"{dc_decl:>10.2f}  {dc_comp:>10.2f}  {a['meas']:>5.1f}")
    print("\nIf 'dc computed' stays near 'dc declared' and the measured value, the computed "
          "configuration free energies reproduce the predictions. If it swings far, that is the "
          "honest limit of the two-configuration reduction -- report it, do not retune soft2.")
