#!/usr/bin/env python3
"""The oxygen term rebuilt on the measured viscosity. The 114 degree angle was inferred from Kang's levitation
undercooling with the old viscosity and aT 0.519, so it is rederived here on the same footing as the casting
prediction. Also: the site density, which the paper holds at its upper limit (every atom a site)."""
import sys
import numpy as np
from scipy.optimize import brentq

import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viscosity_measured as v
t = v.t; z = v.z; kB = v.kB; Rg = v.Rg; NA = v.NA
AT = 0.597; ETA0 = 2.3e-4
V_DROP, TAU = 8.2e-9, 1.0        # 2.5 mm sphere, 1 s residence, as in nucleation_model/heterogeneous.py
UNDERCOOL = {"Cu50Zr50": {"Kang 2014": 0.32, "Ganorkar 2020": 0.261}, "Cu64Zr36": {"Kang 2014": 0.24, "Ganorkar 2020": 0.206}}


def f_theta(deg):
    c = np.cos(np.radians(deg)); return (2 + c) * (1 - c) ** 2 / 4


def theta_of_f(f):
    return brentq(lambda th: f_theta(th) - f, 1e-3, 180.0)


class Alloy:
    def __init__(self, key, aT=AT, eta0=ETA0):
        self.key = key; self.x, _, _, self.Tl, self.Tg, _, self.meas = t.ALLOYS[key]
        b = t.predict(key); self.dG = b["dGfn"]; self.vft = v.vft_from_anchors(key, eta0)
        self.Vm = z.comp_props(self.x)[0] * NA
        self.sig = aT * b["dHf"] / (NA ** (1 / 3) * self.Vm ** (2 / 3))

    def eta(self, T):
        return self.vft[0] * np.exp(self.vft[1] / (T - self.vft[2]))

    def rate(self, T, f, xs):
        """Homogeneous plus heterogeneous nucleation: the catalysed channel has only the site fraction xs of the atoms."""
        g = max(self.dG(T), 1.0); gv = g / self.Vm
        bar = 16 * np.pi * self.sig ** 3 / (3 * gv ** 2)
        pre = (NA / self.Vm) * (kB * T / (3 * np.pi * t.lam ** 3 * self.eta(T)))
        hom = np.exp(-bar / (kB * T))
        return pre * hom if f >= 1.0 else pre * (hom + xs * np.exp(-f * bar / (kB * T)))

    def dc(self, f=1.0, xs=1.0):
        Ts = np.linspace(self.Tg + 5, self.Tl - 3, 1400); out = np.empty_like(Ts)
        with np.errstate(over="ignore", divide="ignore"):
            for i, T in enumerate(Ts):
                g = max(self.dG(T), 1.0)
                U = max((kB * T / (3 * np.pi * t.lam ** 2 * self.eta(T))) * (1 - np.exp(-g / (Rg * T))), 1e-30)
                out[i] = (3 * t.x_det / (np.pi * max(self.rate(T, f, xs), 1e-300) * U ** 3)) ** 0.25
        j = int(np.argmin(out)); Rc = (self.Tl - Ts[j]) / out[j]
        return 10 * np.sqrt(10 / Rc)

    def undercooling(self, f=1.0, xs=1.0):
        """Reduced undercooling at which one nucleus appears in the droplet within the residence time."""
        Ts = np.linspace(self.Tl - 3, self.Tg + 5, 900)
        with np.errstate(over="ignore", divide="ignore"):
            hit = [T for T in Ts if self.rate(T, f, xs) * V_DROP * TAU >= 1]
        return (self.Tl - hit[0]) / self.Tl if hit else None

    def f_from_undercooling(self, target, xs=1.0):
        h = self.undercooling(1.0, xs)
        if h is not None and h <= target:
            return None     # undercools no deeper than measured even with no catalysis
        return brentq(lambda f: (self.undercooling(f, xs) or 1.0) - target, 0.02, 1.0, xtol=1e-3)

    def f_from_dc(self, target, xs=1.0):
        if self.dc(1.0, xs) <= target:
            return None
        return brentq(lambda f: self.dc(f, xs) - target, 0.3, 1.0, xtol=1e-3)


def main():
    a50, a64, a47 = Alloy("Cu50Zr50"), Alloy("Cu64Zr36"), Alloy("Cu47Zr45Al8")
    print(f"homogeneous undercooling, aT {AT}, measured viscosity (a test of aT that uses no casting data)")
    for a in (a50, a64):
        u = a.undercooling()
        print(f"  {a.key}: predicted {'no nucleation above Tg' if u is None else round(u, 3)}  measured {UNDERCOOL[a.key]}")
    for a0 in (0.50, 0.53, 0.55, 0.57):
        print(f"    aT {a0}: Cu50Zr50 {Alloy('Cu50Zr50', aT=a0).undercooling()}  Cu64Zr36 {Alloy('Cu64Zr36', aT=a0).undercooling()}")
    f_need = a64.f_from_dc(2.0)
    print(f"\nCu64Zr36 homogeneous {a64.dc():.2f} mm. Barrier factor that gives 2 mm with every atom a site: f = {f_need:.3f}, theta = {theta_of_f(f_need):.1f} deg")
    print("\nsite fraction xs of the atoms (paper: 1). Cu64Zr36 angle from levitation at the same xs, then the casting diameter")
    print("   xs      source    f(ESL)  theta   dc Cu64Zr36 mm")
    for xs in (1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        for src, d in UNDERCOOL["Cu64Zr36"].items():
            f = a64.f_from_undercooling(d, xs)
            if f is None:
                print(f"  {xs:7.0e} {src:13s} none (no catalyst needed to match levitation)")
                continue
            print(f"  {xs:7.0e} {src:13s} {f:6.3f} {theta_of_f(f):6.1f}  {a64.dc(f, xs):8.2f}")
    print("\nsame, with 114 deg kept and only the site density changed: Cu64Zr36 mm")
    for xs in (1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        print(f"  xs {xs:7.0e}: {a64.dc(f_theta(114), xs):6.2f}")
    print("\nCu50Zr50 levitation: angle needed at xs = 1")
    for src, d in UNDERCOOL["Cu50Zr50"].items():
        f = a50.f_from_undercooling(d)
        print(f"  {src}: {'no catalysis needed' if f is None else f'f {f:.3f} theta {theta_of_f(f):.1f}'}")
    print("\nsite fraction from oxygen: xs = 7 c_O (6 a / d), oxide Zr4Cu2O (7 atoms per O), a 0.25 nm atom, particle diameter d")
    for c in (100, 300, 1000, 3000):
        print("  c_O", c, "ppm: " + "  ".join(f"d {d} nm xs {7 * c * 1e-6 * 6 * 0.25 / d:.1e} dc {a64.dc(f_theta(114), 7 * c * 1e-6 * 6 * 0.25 / d):.2f}" for d in (2, 5, 20)))
    print("\nthe map cos(theta) = r - delta/delta_inc with computed delta 1.8 % and delta_inc 7.0 %, Cu50Zr50 5.7 % and 7.4 %")
    for r in (0.0, 0.05, 0.1, 0.2):
        c64 = r - 1.8 / 7.0; c50 = r - 5.7 / 7.4
        th64, th50 = np.degrees(np.arccos(c64)), np.degrees(np.arccos(c50))
        print(f"  r {r:.2f}: theta64 {th64:.1f}  theta50 {th50:.1f}  dc64 {a64.dc(f_theta(th64)):.2f}  dc50 {a50.dc(f_theta(th50)):.2f}")


if __name__ == "__main__":
    main()
