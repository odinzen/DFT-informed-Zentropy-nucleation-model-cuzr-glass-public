#!/usr/bin/env python3
"""What the two-configuration liquid contributes: the same crystalline reference, liquidus
calibration and nucleation model run with a single-state liquid, and with other splits and
configuration counts. Also the populations of the two liquid configurations, the entropy of
fusion split into vibrational and configurational parts, and the configurational heat capacity
(the energy fluctuation between configurations). Prints only; no parameter is changed in the engine.
"""
import os, sys
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import zentropy_driving_force as z

kB = 1.380649e-23; Rg = z.R; NA = 6.02214076e23
eta0 = 4e-5; lam = 2.8e-10; x_det = 1e-6; Cvft = np.log(1e12 / eta0); aT = 0.50
ALLOYS = {"Cu50Zr50": (0.50, "Cu10Zr7", "CuZr", 1208, 673, 13.6, 2.0),
          "Cu47Zr45Al8": (0.38, "Cu8Zr3", "Cu10Zr7", 1163, 681, 20.6, 15.0),
          "Cu64Zr36": (0.36, "Cu8Zr3", "Cu10Zr7", 1230, 745, 13.6, 2.0)}


def G_liq(T, E0, th, dE, splits, softs):
    """Zentropy mixture over any number of configurations; one split entry gives a single state."""
    F = np.array([E0 + dE + s + z.Fvib_molar(T, f * th) for s, f in zip(splits, softs)])
    w = np.exp(-(F - F.min()) / (Rg * T)); p = w / w.sum()
    return float((p * F).sum() + Rg * T * (p * np.log(p)).sum()), p, F


def predict(key, splits=(0.0, 6.0e3), softs=(0.80, 0.80), form=None):
    x, lo, hi, Tl, Tg, D, _ = ALLOYS[key]
    E0, th, Vat = z.crystalline_assembly(x, lo, hi)
    dG0 = lambda T, dE: G_liq(T, E0, th, dE, splits, softs)[0] - z.G_crystal(T, E0, th)
    dE = brentq(lambda e: dG0(Tl, e), -40e3, 60e3, xtol=1.0)
    dG = lambda T: dG0(T, dE)
    Vm = Vat * NA
    dSf = -(dG(Tl + 1) - dG(Tl - 1)) / 2.0; dHf = Tl * dSf
    # same enthalpy of fusion and interfacial energy, classical forms of the driving force
    if form == "linear":
        dG = lambda T: dSf * (Tl - T)
    elif form == "thompson-spaepen":
        dG = lambda T: dHf * (Tl - T) / Tl * 2 * T / (Tl + T)
    sig = aT * dHf / (NA ** (1 / 3) * Vm ** (2 / 3))
    T0 = Cvft * Tg / (D + Cvft)
    Ts = np.linspace(Tg + 5, Tl - 3, 1400); t = np.empty_like(Ts)
    with np.errstate(over="ignore", divide="ignore"):
        for i, Ti in enumerate(Ts):
            e = eta0 * np.exp(D * T0 / (Ti - T0)); g = max(dG(Ti), 1.0); gv = g / Vm
            bar = 16 * np.pi * sig ** 3 / (3 * gv ** 2)
            I = (NA / Vm) * (kB * Ti / (3 * np.pi * lam ** 3 * e)) * np.exp(-bar / (kB * Ti))
            U = max((kB * Ti / (3 * np.pi * lam ** 2 * e)) * (1 - np.exp(-g / (Rg * Ti))), 1e-30)
            t[i] = (3 * x_det / (np.pi * max(I, 1e-300) * U ** 3)) ** 0.25
    j = int(np.argmin(t)); Rc = (Tl - Ts[j]) / t[j]; dc = 10 * np.sqrt(10 / Rc)
    gn = dG(Ts[j]); barn = 16 * np.pi * sig ** 3 / (3 * (gn / Vm) ** 2) / (kB * Ts[j])
    return dict(dSf=dSf, dHf=dHf, sig=sig * 1e3, Tn=Ts[j], tn=t[j], Rc=Rc, dc=dc, dE=dE, gn=gn, barkT=barn,
                dGfn=dG, E0=E0, th=th)


def row(name, r):
    print(f"  {name:34s} dSf {r['dSf']:6.2f}  dHf {r['dHf'] / 1e3:6.2f}  sigma {r['sig']:6.1f}  "
          f"Tnose {r['Tn']:5.0f}  barrier/kT {r['barkT']:6.1f}  Rc {r['Rc']:10.3g}  dc {r['dc']:10.3g} mm")


if __name__ == "__main__":
    print("Baseline must reproduce Table 2: 854 K/553/1.34, 832 K/3.4/17.2, 893 K/8.5/10.9")
    soft_eq = None
    for key in ALLOYS:
        print(f"\n{key} (measured dc {ALLOYS[key][6]} mm)")
        base = predict(key)
        row("two configurations (paper)", base)
        row("single state, same softening 0.80", predict(key, (0.0,), (0.80,)))
        # a single state can match the entropy of fusion only by softening further
        x, lo, hi, Tl, Tg, D, _ = ALLOYS[key]
        f_eq = np.exp(-base["dSf"] / (3 * Rg))
        row(f"single state, softening {f_eq:.3f}", predict(key, (0.0,), (f_eq,)))
        row("single state, softening 0.75", predict(key, (0.0,), (0.75,)))
        row("single state, softening 0.85", predict(key, (0.0,), (0.85,)))
        row("linear dG, same dHf and sigma", predict(key, form="linear"))
        row("Thompson-Spaepen dG, same dHf, sigma", predict(key, form="thompson-spaepen"))
        for dEs in (0.7e3, 1.8e3, 12e3, 24e3):
            row(f"two configurations, split {dEs / 1e3:g} kJ", predict(key, (0.0, dEs)))
        row("three configurations, 0/6/12 kJ", predict(key, (0.0, 6e3, 12e3), (0.80, 0.80, 0.80)))
        row("three configurations, 0/1.8/3.6 kJ", predict(key, (0.0, 1.8e3, 3.6e3), (0.80, 0.80, 0.80)))

    print("\nDriving force at the nose and across the window, Cu50Zr50 (kJ/mol)")
    a = predict("Cu50Zr50"); b = predict("Cu50Zr50", (0.0,), (0.80,)); c = predict("Cu50Zr50", (0.0,), (np.exp(-a["dSf"] / (3 * Rg)),))
    for T in (700, 800, 900, 1000, 1100, 1200):
        print(f"  {T} K  two-config {a['dGfn'](T) / 1e3:6.3f}   single 0.80 {b['dGfn'](T) / 1e3:6.3f}   single matched {c['dGfn'](T) / 1e3:6.3f}")

    print("\nPopulations of the two liquid configurations (the ones in G_liq), split 6 kJ")
    for key in ALLOYS:
        x, lo, hi, Tl, Tg, D, _ = ALLOYS[key]
        r = predict(key)
        for T in (Tg, Tl):
            _, p, F = G_liq(T, r["E0"], r["th"], r["dE"], (0.0, 6e3), (0.8, 0.8))
            Sc = -Rg * (p * np.log(p)).sum()
            Sv = 3 * Rg * np.log(1 / 0.8)  # high-temperature limit, checked below against dSf
            x_ = 6e3 / (Rg * T); C = Rg * x_ ** 2 * p[0] * p[1]
            print(f"  {key:12s} T={T:5d}  p1 {p[0]:.3f}  p2 {p[1]:.3f}  S_conf {Sc:5.2f} (R ln2={Rg * np.log(2):.2f})  "
                  f"C_conf {C:5.2f} J/mol/K")
    print("\nEntropy of fusion split at the liquidus (J/mol/K)")
    for key in ALLOYS:
        x, lo, hi, Tl, Tg, D, _ = ALLOYS[key]
        r = predict(key); s = predict(key, (0.0,), (0.80,))
        print(f"  {key:12s} total {r['dSf']:.2f}  vibrational (single state) {s['dSf']:.2f}  configurational {r['dSf'] - s['dSf']:.2f}  "
              f"share {100 * (r['dSf'] - s['dSf']) / r['dSf']:.0f}%")
    print("\nConfigurational entropy at Tl = 1208 K against the split")
    for dEs in (0.7e3, 1.8e3, 6e3, 12e3, 24e3):
        q = np.exp(-dEs / (Rg * 1208)); p = np.array([1, q]) / (1 + q)
        print(f"  split {dEs / 1e3:5.1f} kJ  p2 {p[1]:.3f}  S_conf {-Rg * (p * np.log(p)).sum():.2f}")
