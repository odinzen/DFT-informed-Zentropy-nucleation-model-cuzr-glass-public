#!/usr/bin/env python3
"""How much of the result is prediction. (1) Each description of the liquid with its coefficient set the way Castellero
set theirs, on Cu50Zr50 only, then carried to the other alloys. (2) Null models with no Debye or Zentropy content.
(3) Leave one out on the coefficient. (4) Input errors propagated by Monte Carlo. (5) Liquidus taken from the same
source as the viscosity. (6) The ternary composition input."""
import sys
import numpy as np
from dataclasses import replace
from scipy.optimize import brentq

import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import hyalos_kinetics as h
KEYS = list(h.ALLOYS); MEAS = {k: h.ALLOYS[k].dc_meas for k in KEYS}


def dc_of(a, dG, dHf, Vm, aT):
    return h.kinetics(a, dG, dHf, Vm, aT=aT, n=900)["dc"]


def fit_aT(a, dG, dHf, Vm, target):
    return brentq(lambda x: np.log(dc_of(a, dG, dHf, Vm, x) / target), 0.25, 1.6, xtol=1e-4)


def describe(name, make):
    """make(alloy) -> (dG, dHf, Vm). Coefficient set on Cu50Zr50 = 2 mm, then transferred."""
    a50 = h.ALLOYS["Cu50Zr50"]; at = fit_aT(a50, *make(a50), 2.0)
    out = {k: dc_of(h.ALLOYS[k], *make(h.ALLOYS[k]), at) for k in KEYS}
    print(f"  {name:44s} aT {at:5.3f}   " + "  ".join(f"{out[k]:7.2f}" for k in KEYS))
    return at, out


def th(**kw):
    def f(a):
        t = h.thermo(a, **kw); return t["dG"], t["dHf"], t["Vm"]
    return f


def linear(dS):
    def f(a):
        Vm = h.thermo(a)["Vm"]; return (lambda T: dS * (a.Tl - T)), a.Tl * dS, Vm
    return f


def ts(dS):
    def f(a):
        Vm = h.thermo(a)["Vm"]; H = a.Tl * dS
        return (lambda T: H * (a.Tl - T) / a.Tl * 2 * T / (a.Tl + T)), H, Vm
    return f


print("(1,2) coefficient set on Cu50Zr50 = 2 mm only, then transferred. Measured:           " + "  ".join(f"{MEAS[k]:7.2f}" for k in KEYS))
print("                                                                     " + "  ".join(f"{k[:7]:>7s}" for k in KEYS))
describe("two configurations, softening 0.80 (paper)", th())
describe("two configurations, split 1.8", th(splits=(0.0, 1.8e3)))
describe("three configurations", th(splits=(0.0, 6e3, 12e3), softs=(.8, .8, .8)))
describe("single state, softening 0.85", th(splits=(0.0,), softs=(.85,)))
describe("single state, softening 0.80", th(splits=(0.0,), softs=(.80,)))
describe("single state, softening 0.75", th(splits=(0.0,), softs=(.75,)))
describe("NULL linear dG, dSf 7.53 measured for CuZr", linear(7.53))
describe("NULL linear dG, dSf 8.3 Richards", linear(8.3))
describe("NULL Thompson-Spaepen, dSf 7.53", ts(7.53))
describe("NULL Thompson-Spaepen, dSf 8.3", ts(8.3))


def viscosity_only():
    """No thermodynamics at all: dc scales as eta at a fixed reduced temperature (t_nose ~ eta, dc ~ sqrt t)."""
    print("\n(2b) viscosity alone, eta at T/Tl = 0.67, dc ~ sqrt(eta), scaled to Cu50Zr50 = 2 mm")
    def eta(a):
        B, T0 = h.vft_anchored(a.Tg_eta, a.Tstar); return h.ETA0 * np.exp(B / (0.67 * a.Tl - T0))
    e50 = eta(h.ALLOYS["Cu50Zr50"])
    print("  " + "  ".join(f"{k} {2.0 * np.sqrt(eta(h.ALLOYS[k]) / e50):.2f}" for k in KEYS))


viscosity_only()

print("\n(3) leave one out, Turnbull coefficient fixed from one alloy, others predicted (mm)")
for src in KEYS:
    a = h.ALLOYS[src]; t = h.thermo(a); at = fit_aT(a, t["dG"], t["dHf"], t["Vm"], a.dc_meas)
    print(f"  aT from {src:12s} {at:.3f}: " + "  ".join(f"{k} {h.predict(k, aT=at)['dc']:.2f}" for k in KEYS))

print("\n(5) liquidus from the same table as the viscosity (Gangopadhyay: 1222 / 1163 / 1200 K; Kang's curve near 1210 K for Cu64Zr36)")
for k, Tls in {"Cu50Zr50": (1208, 1222), "Cu64Zr36": (1230, 1210, 1200), "Cu47Zr45Al8": (1163,)}.items():
    a = h.ALLOYS[k]
    print(f"  {k}: " + "  ".join(f"Tl {T}: {(lambda al: (lambda t: h.kinetics(al, t['dG'], t['dHf'], t['Vm'])['dc'])(h.thermo(al)))(replace(a, Tl=T)):.2f} mm" for T in Tls))

print("\n(6) ternary composition. Paper x = 0.38 with Cu8Zr3|Cu10Zr7. Zr/(Cu+Zr) is 45/92 = 0.489, between Cu10Zr7 and CuZr")
for x, lo, hi in ((0.38, "Cu8Zr3", "Cu10Zr7"), (45 / 92, "Cu10Zr7", "CuZr")):
    al = replace(h.ALLOYS["Cu47Zr45Al8"], x=x, lo=lo, hi=hi); t = h.thermo(al); r = h.kinetics(al, t["dG"], t["dHf"], t["Vm"])
    print(f"  x {x:.3f} {lo}|{hi}: theta {t['th']:.0f} dSf {t['dSf']:.2f} Vm {t['Vm'] * 1e6:.3f} sigma {r['sig'] * 1e3:.1f} nose {r['Tn']:.0f} dc {r['dc']:.2f} mm")

print("\n(4) Monte Carlo of the measured and literature inputs, 1-sigma: Tl 10 K, Tg 5 K, T* 10 K, soft 0.025, ln(eta0) from the fits (0.4), aT 0.01")
rng = np.random.default_rng(7)
N = 400
res = {k: [] for k in KEYS}
for i in range(N):
    dT = rng.normal(0, 10); dg = rng.normal(0, 5); ds = rng.normal(0, 10)
    e0 = h.ETA0 * np.exp(rng.normal(0, 0.4)); at = h.ALPHA_T + rng.normal(0, 0.01); sf = 0.80 + rng.normal(0, 0.025)
    for k in KEYS:
        a = h.ALLOYS[k]
        al = replace(a, Tl=a.Tl + rng.normal(0, 10), Tg_eta=a.Tg_eta + rng.normal(0, 5), Tstar=a.Tstar + rng.normal(0, 10))
        t = h.thermo(al, softs=(sf, sf))
        res[k].append(h.kinetics(al, t["dG"], t["dHf"], t["Vm"], aT=at, eta0=e0, n=500)["dc"])
for k in KEYS:
    v = np.array(res[k]); q = np.percentile(v, [2.5, 16, 50, 84, 97.5])
    print(f"  {k:12s} median {q[2]:7.2f}  68% {q[1]:7.2f}-{q[3]:7.2f}  95% {q[0]:7.2f}-{q[4]:7.2f}  factor (84/16)^0.5 {np.sqrt(q[3] / q[1]):.2f}  measured {MEAS[k]}")
for name, kw in (("Tl only", "Tl"), ("Tstar only", "Tstar"), ("aT only", "aT"), ("soft only", "soft")):
    a = h.ALLOYS["Cu47Zr45Al8"]
    if kw == "Tl":
        al = lambda s: replace(a, Tl=a.Tl + 10 * s); ex = {}
    elif kw == "Tstar":
        al = lambda s: replace(a, Tstar=a.Tstar + 10 * s); ex = {}
    if kw in ("Tl", "Tstar"):
        vals = [(lambda b: (lambda t: h.kinetics(b, t["dG"], t["dHf"], t["Vm"])["dc"])(h.thermo(b)))(al(s)) for s in (-1, 1)]
    elif kw == "aT":
        vals = [h.predict("Cu47Zr45Al8", aT=h.ALPHA_T + .01 * s)["dc"] for s in (-1, 1)]
    else:
        vals = [h.predict("Cu47Zr45Al8", softs=(.8 + .025 * s, .8 + .025 * s))["dc"] for s in (-1, 1)]
    print(f"    ternary, {name} +-1 sigma: {vals[0]:.1f} to {vals[1]:.1f} mm")
