#!/usr/bin/env python3
"""Kang 2014 Fig. 2 gives the time to crystallize a levitated Cu64Zr36 drop at fixed temperature, which is a
measured nucleation rate against temperature. A single contact angle fixes how steeply that rate moves, so the
slope tests the single-angle catalyst without any casting data. Points are read off the published inset."""
import sys
import numpy as np
from scipy.optimize import least_squares

import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oxygen_consistent as oc
t = oc.t; kB = oc.kB; Rg = oc.Rg; NA = oc.NA

# (hold time s, T/Tl), red stars of the inset
KANG = np.array([(15, .831), (17, .836), (22, .8375), (24, .839), (30, .844), (42, .841), (78, .849), (84, .848),
                 (180, .860), (420, .859), (440, .865), (950, .871), (1150, .873), (2000, .882)])
TL = 1200.0     # the reduced scale of the figure uses an average liquidus near this


def main():
    a = oc.Alloy("Cu64Zr36")
    tm, tr = KANG[:, 0], KANG[:, 1]
    slope, icept = np.polyfit(tr, np.log(tm), 1)
    print(f"measured: d ln(t) / d(T/Tl) = {slope:.0f} (ln t = {icept:.1f} + {slope:.0f} T/Tl), scatter {np.std(np.log(tm) - slope * tr - icept):.2f} in ln t")

    def lnt_model(T, f, xs):
        return -np.log(a.rate(T, f, xs) * oc.V_DROP)

    # slope a single barrier gives at this temperature, for each number of catalytic sites
    Trs = 0.855; T0 = Trs * TL
    n_at = NA / a.Vm * oc.V_DROP
    print(f"\ndroplet holds {n_at:.2e} atoms. Single contact angle, angle set to hit the measured time at T/Tl {Trs}")
    print("  sites in the drop    xs        f      theta   slope d ln t / d(T/Tl)")
    tgt = icept + slope * Trs
    for N in (n_at, n_at * 1e-6, n_at * 1e-12, 1e6, 1.0):
        xs = N / n_at
        lo, hi = 1e-4, 0.999
        fn = lambda f: lnt_model(T0, f, xs) - tgt
        if fn(lo) * fn(hi) > 0:
            print(f"  {N:12.2e} {xs:10.1e}   no angle reaches it")
            continue
        from scipy.optimize import brentq
        f = brentq(fn, lo, hi)
        s = (lnt_model(T0 + 6, f, xs) - lnt_model(T0 - 6, f, xs)) / (12 / TL)
        print(f"  {N:12.2e} {xs:10.1e} {f:7.4f} {oc.theta_of_f(f):7.1f} {s:10.0f}")

    # the same line extrapolated should also account for the cooling-curve recalescence
    for R in (5.0, 10.0, 20.0):
        Ts = np.linspace(TL * 0.99, TL * 0.60, 4000)
        rate = np.exp(-(icept + slope * Ts / TL))
        n = np.cumsum(rate) * (Ts[0] - Ts[1]) / R
        i = np.searchsorted(n, 1.0)
        print(f"line integrated on cooling at {R:3.0f} K/s: recalescence at reduced undercooling {1 - Ts[i] / TL:.3f} (Kang: 0.19 in Fig 1a, best 0.24)")

    # casting with the measured rate used directly, extrapolated, on top of homogeneous
    print("\ncasting diameter with the levitation rate as a volumetric nucleation channel")
    def dc_with(cap=None, scale=1.0):
        Ts = np.linspace(a.Tg + 5, a.Tl - 3, 1400); tx = np.empty_like(Ts)
        for i, T in enumerate(Ts):
            g = max(a.dG(T), 1.0)
            U = max((kB * T / (3 * np.pi * t.lam ** 2 * a.eta(T))) * (1 - np.exp(-g / (Rg * T))), 1e-30)
            trT = max(T / TL, cap) if cap else T / TL
            Iex = scale * np.exp(-(icept + slope * trT)) / oc.V_DROP
            tx[i] = (3 * t.x_det / (np.pi * (a.rate(T, 1.0, 1.0) + Iex) * U ** 3)) ** 0.25
        j = int(np.argmin(tx)); Rc = (a.Tl - Ts[j]) / tx[j]
        return 10 * np.sqrt(10 / Rc), Ts[j], Rc
    print("  homogeneous only: %.2f mm" % a.dc())
    for lab, cap, sc in [("line extrapolated to the nose", None, 1), ("rate held at its deepest measured value, T/Tl 0.831", 0.831, 1),
                         ("held at Kang's best undercooling, T/Tl 0.76", 0.76, 1), ("extrapolated, sites scale with surface, 2 mm rod vs 2.5 mm drop", None, 1.2)]:
        d, Tn, Rc = dc_with(cap, sc)
        print(f"  {lab}: {d:.2f} mm (nose {Tn:.0f} K, Rc {Rc:.0f} K/s)")

    print("\nCu64Zr36 sigma that alone gives 2 mm (no catalyst), and with 114 deg at each site fraction")
    from scipy.optimize import brentq
    for lab, f, xs in [("no catalyst", 1.0, 1.0), ("114 deg, xs 1", oc.f_theta(114), 1.0), ("114 deg, xs 1e-3", oc.f_theta(114), 1e-3)]:
        g = lambda at: oc.Alloy("Cu64Zr36", aT=at).dc(f, xs) - 2.0
        try:
            at = brentq(g, 0.54, 0.68, xtol=1e-3)
            print(f"  {lab}: aT {at:.3f} ({100 * (at / oc.AT - 1):+.1f} % on sigma)")
        except ValueError:
            print(f"  {lab}: no root in range")


if __name__ == "__main__":
    main()
