#!/usr/bin/env python3
"""Viscosity anchored to measurements for each alloy, against the form the paper used.

The paper: eta0 = 4e-5 fixed, eta(Tg) = 1e12, D* = 13.6 (from the fragility index m = 62 near Tg, Wang 2005;
copied to Cu64Zr36) and 20.6 (ternary, attributed to Gangopadhyay 2017, whose Table 1 lists no D*).
Here: VFT through two measured anchors from one table (Gangopadhyay 2017 Table 1, levitated liquids):
eta(Tg) = 1e12 Pa s and eta(T*) = 0.1 Pa s, with a common prefactor. Checks against the fits to all data
of Ganorkar 2020 (Table I) and Castellero and Battezzati 2023.
"""
import os, sys
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import single_state_test as t
z = t.z; kB = t.kB; Rg = t.Rg; NA = t.NA

# Gangopadhyay et al., JCP 146 (2017) 154506, Table 1: Tg, Tl, T* (eta = 0.1 Pa s)
ANCHOR = {"Cu50Zr50": (675, 1222, 1064), "Cu47Zr45Al8": (681, 1163, 1112), "Cu64Zr36": (733, 1200, 1090)}
FITS = {"Cu50Zr50": [("Ganorkar 2020", 2.3e-4, 2874, 603), ("Castellero 2023", 1.78e-4, 3118, 590)],
        "Cu64Zr36": [("Ganorkar 2020", 6.04e-4, 1785, 698)]}


def vft_from_anchors(key, eta0):
    Tg, _, Ts = ANCHOR[key]
    a, b = np.log(1e12 / eta0), np.log(0.1 / eta0)      # a = B/(Tg-T0), b = B/(Ts-T0)
    T0 = (a * Tg - b * Ts) / (a - b)
    return eta0, b * (Ts - T0), T0


def eta_paper(key, T):
    x, lo, hi, Tl, Tg, D, _ = t.ALLOYS[key]
    T0 = t.Cvft * Tg / (D + t.Cvft)
    return t.eta0 * np.exp(D * T0 / (T - T0))


def predict(key, vft=None, aT=0.50, dG=None, dHf=None):
    """Paper's nucleation model; vft=None keeps the paper's viscosity."""
    x, lo, hi, Tl, Tg, D, meas = t.ALLOYS[key]
    base = t.predict(key)
    dG = dG or base["dGfn"]; dHf = dHf or base["dHf"]
    Vm = z.comp_props(x)[0] * NA
    sig = aT * dHf / (NA ** (1 / 3) * Vm ** (2 / 3))
    Ts = np.linspace(Tg + 5, Tl - 3, 1400); out = np.empty_like(Ts)
    with np.errstate(over="ignore", divide="ignore"):
        for i, Ti in enumerate(Ts):
            e = vft[0] * np.exp(vft[1] / (Ti - vft[2])) if vft else eta_paper(key, Ti)
            g = max(dG(Ti), 1.0); gv = g / Vm
            bar = 16 * np.pi * sig ** 3 / (3 * gv ** 2)
            I = (NA / Vm) * (kB * Ti / (3 * np.pi * t.lam ** 3 * e)) * np.exp(-bar / (kB * Ti))
            U = max((kB * Ti / (3 * np.pi * t.lam ** 2 * e)) * (1 - np.exp(-g / (Rg * Ti))), 1e-30)
            out[i] = (3 * t.x_det / (np.pi * max(I, 1e-300) * U ** 3)) ** 0.25
    j = int(np.argmin(out)); Rc = (Tl - Ts[j]) / out[j]
    return dict(Tn=Ts[j], tn=out[j], Rc=Rc, dc=10 * np.sqrt(10 / Rc), sig=sig * 1e3)


def main():
    print("Anchored VFT, common eta0 = 2.3e-4 Pa s, and comparison at the anchors and the nose (Pa s)")
    for key in ALLOYS_ORDER:
        e0, B, T0 = vft_from_anchors(key, 2.3e-4); Tg, _, Ts = ANCHOR[key]
        Tn = t.predict(key)["Tn"]
        print(f"\n{key}: B {B:.0f} K, T0 {T0:.0f} K, D* = B/T0 = {B / T0:.2f} (paper {t.ALLOYS[key][5]})")
        for T in (Ts, 1208, Tn):
            s = f"  {T:6.0f} K  anchored {e0 * np.exp(B / (T - T0)):10.3g}  paper {eta_paper(key, T):10.3g}  ratio {eta_paper(key, T) / (e0 * np.exp(B / (T - T0))):7.1f}"
            for name, a0, b0, t0 in FITS.get(key, []):
                s += f"  | {name} {a0 * np.exp(b0 / (T - t0)):9.3g}"
            print(s)

    print("\nDiameter (mm) with the paper's thermodynamics, aT 0.50")
    print("alloy         measured | paper viscosity | anchored eta0 1.78e-4 / 2.3e-4 / 6e-4")
    for key in ALLOYS_ORDER:
        r0 = predict(key)["dc"]
        rs = [predict(key, vft_from_anchors(key, e))["dc"] for e in (1.78e-4, 2.3e-4, 6e-4)]
        print(f"  {key:12s} {t.ALLOYS[key][6]:5.1f}   | {r0:8.3g}       | " + " / ".join(f"{v:8.3g}" for v in rs))

    print("\nTurnbull coefficient that recovers each measured diameter, anchored viscosity eta0 2.3e-4 (a diagnostic, not a fit to adopt)")
    for key in ALLOYS_ORDER:
        v = vft_from_anchors(key, 2.3e-4); meas = t.ALLOYS[key][6]
        a = brentq(lambda x: np.log(predict(key, v, x)["dc"] / meas), 0.35, 0.8, xtol=1e-3)
        r = predict(key, v, a)
        print(f"  {key:12s} aT {a:.3f}  sigma {r['sig']:6.1f} mJ/m2  Tnose {r['Tn']:.0f}  tnose {r['tn']:.3g} s")
    print("\nDiameter (mm) with anchored viscosity, one Turnbull coefficient for all three")
    for a in (0.50, 0.519, 0.53, 0.54, 0.55, 0.56, 0.58):
        print(f"  aT {a:.3f}: " + "  ".join(f"{k} {predict(k, vft_from_anchors(k, 2.3e-4), a)['dc']:8.3g}" for k in ALLOYS_ORDER))


ALLOYS_ORDER = ("Cu50Zr50", "Cu47Zr45Al8", "Cu64Zr36")
if __name__ == "__main__":
    main()
