#!/usr/bin/env python3
"""Nucleation kinetics on the measured viscosity, the single place the published diameters come from.

The viscosity is a Vogel-Fulcher curve through two anchors measured on each alloy (Gangopadhyay et al. 2017,
Table 1: eta(Tg) = 1e12 Pa s and eta(T*) = 0.1 Pa s) with the prefactor of the Cu50Zr50 fit of Ganorkar et al.
2020. The Turnbull coefficient follows from Castellero and Battezzati 2023 (sigma 0.1336 J/m2 on a measured
fusion enthalpy of 9.1 kJ/mol), who set that energy to the measured critical cooling rate of CuZr, so Cu50Zr50
is not an independent test of it.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

import zentropy_driving_force as z

kB = 1.380649e-23
Rg = z.R
NA = 6.02214076e23
ALPHA_T = 0.597
ETA0 = 2.3e-4
LAM = 2.8e-10
X_DET = 1e-6
SPLIT = 6.0e3
SOFT = 0.80


@dataclass(frozen=True)
class Alloy:
    label: str
    x: float            # Zr/(Cu+Zr) of the crystalline reference
    lo: str
    hi: str
    Tl: float
    Tg: float           # lower end of the supercooled range
    Tg_eta: float       # Tg of the viscosity anchor
    Tstar: float        # T at eta = 0.1 Pa s
    dc_meas: float


ALLOYS = {
    "Cu50Zr50": Alloy("Cu50Zr50", 0.50, "Cu10Zr7", "CuZr", 1208, 673, 675, 1064, 2.0),
    "Cu47Zr45Al8": Alloy("Cu47Zr45Al8", 45 / 92, "Cu10Zr7", "CuZr", 1163, 681, 681, 1112, 15.0),
    "Cu64Zr36": Alloy("Cu64Zr36", 0.36, "Cu8Zr3", "Cu10Zr7", 1230, 745, 733, 1090, 2.0),
}
ZR_RICH = dict(x=0.556, lo="CuZr", hi="CuZr2", Tl=1092, Tg=706)     # viscosity not anchored, see zr_rich()
z.PHASES.setdefault("CuZr2", dict(Hf=-12.665e3, B=111e9, xZr=2 / 3))


def vft_anchored(Tg_eta, Tstar, eta0=ETA0):
    """(B, T0) of eta = eta0 exp(B / (T - T0)) through eta(Tg) = 1e12 and eta(T*) = 0.1 Pa s."""
    a, b = np.log(1e12 / eta0), np.log(0.1 / eta0)
    T0 = (a * Tg_eta - b * Tstar) / (a - b)
    return b * (Tstar - T0), T0


def liquid(T, E0, th, dE, splits, softs):
    """Zentropy mixture over the configurations; returns G, populations, configuration free energies."""
    F = np.array([E0 + dE + s + z.Fvib_molar(T, f * th) for s, f in zip(splits, softs)])
    w = np.exp(-(F - F.min()) / (Rg * T))
    p = w / w.sum()
    return float((p * F).sum() + Rg * T * (p * np.log(p)).sum()), p, F


def thermo(a, splits=(0.0, SPLIT), softs=(SOFT, SOFT), dG_at_Tl=0.0, B_scale=1.0):
    """Driving force with the icosahedral offset calibrated so that dG(Tl) equals dG_at_Tl."""
    saved = {k: v["B"] for k, v in z.PHASES.items()}
    try:
        for k in z.PHASES:
            z.PHASES[k]["B"] = saved[k] * B_scale
        E0, th, Vat = z.crystalline_assembly(a.x, a.lo, a.hi)
    finally:
        for k in z.PHASES:
            z.PHASES[k]["B"] = saved[k]

    def raw(T, dE):
        return liquid(T, E0, th, dE, splits, softs)[0] - z.G_crystal(T, E0, th)

    dE = brentq(lambda e: raw(a.Tl, e) - dG_at_Tl, -40e3, 60e3, xtol=1.0)
    dG = lambda T: raw(T, dE)
    dSf = -(dG(a.Tl + 1) - dG(a.Tl - 1)) / 2.0
    return dict(dG=dG, dSf=dSf, dHf=a.Tl * dSf, dE=dE, E0=E0, th=th, Vm=Vat * NA)


def kinetics(a, dG, dHf, Vm, aT=ALPHA_T, eta0=ETA0, B=None, T0=None, lam=LAM, x_det=X_DET, sigma=None, n=1400):
    """TTT curve and the nose. sigma in J/m2 overrides the Turnbull relation."""
    if B is None:
        B, T0 = vft_anchored(a.Tg_eta, a.Tstar, eta0)
    sig = sigma if sigma is not None else aT * dHf / (NA ** (1 / 3) * Vm ** (2 / 3))
    Ts = np.linspace(a.Tg + 5, a.Tl - 3, n)
    t = np.empty_like(Ts)
    with np.errstate(over="ignore", divide="ignore"):
        for i, T in enumerate(Ts):
            eta = eta0 * np.exp(B / (T - T0))
            g = max(dG(T), 1.0)
            bar = 16 * np.pi * sig ** 3 / (3 * (g / Vm) ** 2)
            rate = (NA / Vm) * (kB * T / (3 * np.pi * lam ** 3 * eta)) * np.exp(-bar / (kB * T))
            U = max((kB * T / (3 * np.pi * lam ** 2 * eta)) * (1 - np.exp(-g / (Rg * T))), 1e-30)
            t[i] = (3 * x_det / (np.pi * max(rate, 1e-300) * U ** 3)) ** 0.25
    j = int(np.argmin(t))
    Rc = (a.Tl - Ts[j]) / t[j]
    return dict(T=Ts, t=t, Tn=float(Ts[j]), tn=float(t[j]), Rc=float(Rc), dc=float(10 * np.sqrt(10 / Rc)), sig=float(sig),
                B=float(B), T0=float(T0), dG_nose=float(dG(Ts[j])))


def predict(key, aT=ALPHA_T, **kw):
    a = ALLOYS[key]
    th = thermo(a, **{k: kw.pop(k) for k in ("splits", "softs", "dG_at_Tl", "B_scale") if k in kw})
    Vm = th["Vm"] * kw.pop("Vm_scale", 1.0)
    return {**th, **kinetics(a, th["dG"], th["dHf"], Vm, aT=aT, **kw)}


if __name__ == "__main__":
    for k in ALLOYS:
        r = predict(k)
        print(f"{k:12s} Tnose {r['Tn']:5.0f} K  tnose {r['tn']:8.3g} s  Rc {r['Rc']:8.3g} K/s  dc {r['dc']:6.2f} mm  (measured {ALLOYS[k].dc_meas})")
