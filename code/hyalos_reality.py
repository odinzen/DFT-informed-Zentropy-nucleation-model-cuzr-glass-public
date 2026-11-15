#!/usr/bin/env python3
"""Crystallization of Cu-Zr liquids from the phases that exist and a liquid with a spectrum of configurations.

Replaces two assumptions of hyalos_kinetics for binary Cu-Zr:

Crystal. Each real phase is its own nucleus. Its driving force is the parallel tangent from the liquid of the alloy's
composition, its interfacial energy follows the Turnbull relation with its own enthalpy of fusion, and the rates add.
A crystal whose composition differs from the liquid grows by diffusion as well as attachment, 1/v = 1/U + R/(2 D lam^2),
with lam the sphere growth parameter of Zener in the form of Chen, Jeppsson and Agren (growth_parameter below).

Liquid. Instead of two states, a spectrum of inherent structures, read through its heat capacity C_conf = dE_IS/dT.
A Gaussian spectrum of width w would give C_conf = w^2/(R T^2); the measured E_IS(T) of these liquids falls faster below
about 1000 K, so a measured form is used as well. The assessed liquid is kept at the anchor temperature, where the phase
diagram fixes it, and below it the heat capacity difference is replaced by C_conf.
"""
from dataclasses import dataclass, field

import numpy as np
from pycalphad import Database, calculate
from scipy.optimize import brentq
from scipy.special import erfcx


def growth_parameter(omega):
    """Zener's sphere growth parameter, R = 2 lam sqrt(D t), from Omega = 2 lam^2 [1 - sqrt(pi) lam exp(lam^2) erfc(lam)]
    (Zener 1949; the form used by Chen, Jeppsson and Agren 2008). Solved numerically for 0 < Omega < 1."""
    f = lambda lam: 2 * lam ** 2 * (1 - np.sqrt(np.pi) * lam * erfcx(lam)) - omega
    return brentq(f, 1e-12, 1e3, xtol=1e-15, rtol=1e-14)


kB = 1.380649e-23
Rg = 8.314462
NA = 6.02214076e23
ETA0 = 2.3e-4
JUMP = 2.8e-10
X_DET = 1e-6
V_CU, V_ZR = 7.11e-6, 14.02e-6
XZR = {"CU51ZR14": 14 / 65, "CU8ZR3": 3 / 11, "CU10ZR7": 7 / 17, "CUZR": 0.5, "CUZR2": 2 / 3}
T_ANCHOR = 1250.0
TGRID = np.arange(640.0, 1400.0, 5.0)
XGRID = np.linspace(0.14, 0.76, 125)


def vm(x):
    return (1 - x) * V_CU + x * V_ZR


def vft(Tg, Ts, eta0=ETA0):
    a, b = np.log(1e12 / eta0), np.log(0.1 / eta0)
    T0 = (a * Tg - b * Ts) / (a - b)
    return b * (Ts - T0), T0


@dataclass
class Tables:
    """Assessed Gibbs energies on a T, x grid, J per mole of atoms."""
    name: str
    GL: np.ndarray
    Gc: dict

    @classmethod
    def from_tdb(cls, path, name):
        db = Database(path)
        phs = {}
        for p in db.phases:
            q = p.upper()
            for k in XZR:
                if q == k or (q.startswith(k) and not q[len(k)].isdigit()):
                    phs.setdefault(k, p)
        pts = np.column_stack([1 - XGRID, XGRID])
        GL = np.zeros((len(TGRID), len(XGRID)))
        Gc = {k: np.zeros(len(TGRID)) for k in phs}
        for i, T in enumerate(TGRID):
            r = calculate(db, ["CU", "ZR"], "LIQUID", P=101325, T=T, N=1, output="GM", points=pts)
            GL[i] = r.GM.values.squeeze()
            for k, p in phs.items():
                c = calculate(db, ["CU", "ZR"], p, P=101325, T=T, N=1, output="GM", pdens=50)
                xx = np.atleast_1d(c.X.sel(component="ZR").values.squeeze())
                ee = np.atleast_1d(c.GM.values.squeeze())
                Gc[k][i] = float(ee[np.argmin(np.abs(xx - XZR[k]))])
        return cls(name, GL, Gc)


def nk_difference(tb, x):
    """Assessed heat capacity difference of the liquid over the lever mixture of crystals at x, J/mol/K, on TGRID."""
    i = int(np.argmin(np.abs(XGRID - x)))
    ks = sorted(tb.Gc, key=lambda k: XZR[k])
    lo = max((k for k in ks if XZR[k] <= x), key=lambda k: XZR[k], default=ks[0])
    hi = min((k for k in ks if XZR[k] >= x), key=lambda k: XZR[k], default=ks[-1])
    f = 0.0 if lo == hi else (x - XZR[lo]) / (XZR[hi] - XZR[lo])
    d = tb.GL[:, i] - ((1 - f) * tb.Gc[lo] + f * tb.Gc[hi])
    return -TGRID * np.gradient(np.gradient(d, TGRID), TGRID)


def gaussian_spectrum(w_of_x):
    return lambda x, T: float(w_of_x(x)) ** 2 / (Rg * T ** 2)


def with_spectrum(tb, c_conf, anchor=T_ANCHOR):
    """New liquid table. Heat capacity difference below the anchor replaced by c_conf(x, T); G and S unchanged at the anchor."""
    GL = tb.GL.copy()
    ia = int(np.argmin(np.abs(TGRID - anchor)))
    for j, x in enumerate(XGRID):
        c = c_conf(x, TGRID) - nk_difference(tb, x)
        # cumulative integrals from the anchor
        dH = np.zeros(len(TGRID)); dS = np.zeros(len(TGRID))
        for i in range(ia - 1, -1, -1):
            h = TGRID[i] - TGRID[i + 1]
            dH[i] = dH[i + 1] + 0.5 * (c[i] + c[i + 1]) * h
            dS[i] = dS[i + 1] + 0.5 * (c[i] / TGRID[i] + c[i + 1] / TGRID[i + 1]) * h
        for i in range(ia + 1, len(TGRID)):
            dH[i] = dS[i] = 0.0
        GL[:, j] = tb.GL[:, j] + dH - TGRID * dS
    return Tables(tb.name + "+spectrum", GL, tb.Gc)


def tangent(tb, i, x0):
    d = np.gradient(tb.GL[i], XGRID)
    return float(np.interp(x0, XGRID, tb.GL[i])), float(np.interp(x0, XGRID, d)), d


def forces(tb, x0, phases):
    """Parallel tangent force per mole of atoms of nucleus, each phase, on TGRID."""
    out = {k: np.zeros(len(TGRID)) for k in phases}
    for i in range(len(TGRID)):
        g0, s0, _ = tangent(tb, i, x0)
        for k in phases:
            out[k][i] = g0 + (XZR[k] - x0) * s0 - tb.Gc[k][i]
    return out


def mixture_force(tb, x0):
    """Liquid minus the lowest lever mixture of real phases at x0, the construction of hyalos_kinetics."""
    out = np.zeros(len(TGRID))
    ks = sorted(tb.Gc, key=lambda k: XZR[k])
    for i in range(len(TGRID)):
        g0, _, _ = tangent(tb, i, x0)
        best = None
        for a in ks:
            for b in ks:
                if XZR[a] < x0 < XZR[b]:
                    f = (x0 - XZR[a]) / (XZR[b] - XZR[a]); e = (1 - f) * tb.Gc[a][i] + f * tb.Gc[b][i]
                    best = e if best is None else min(best, e)
            if abs(XZR[a] - x0) < 1e-9:
                best = tb.Gc[a][i] if best is None else min(best, tb.Gc[a][i])
        out[i] = g0 - best
    return out


def fusion_enthalpy(tb, k, T=1200.0):
    i = int(np.argmin(np.abs(TGRID - T)))
    g = np.array([np.interp(XZR[k], XGRID, tb.GL[t]) - tb.Gc[k][t] for t in range(len(TGRID))])
    return float(g[i] + T * (-np.gradient(g, TGRID)[i]))


def interface_liquid(tb, i, k, x0):
    """Liquid composition in equilibrium with phase k at TGRID[i], or None if there is none."""
    xk = XZR[k]
    if abs(xk - x0) < 1e-6:
        return None
    d = np.gradient(tb.GL[i], XGRID)
    def D(x):
        return np.interp(x, XGRID, tb.GL[i]) + (xk - x) * np.interp(x, XGRID, d) - tb.Gc[k][i]
    if D(x0) <= 0:
        return None
    # the growing crystal drains the liquid at its interface, so that liquid lies away from the crystal's composition
    far = XGRID[0] if xk > x0 else XGRID[-1]
    pts = np.linspace(x0, far, 81)
    vals = [D(p) for p in pts]
    for a, b, va, vb in zip(pts[:-1], pts[1:], vals[:-1], vals[1:]):
        if va > 0 >= vb:
            return float(brentq(D, a, b, xtol=1e-5))
    return None


def supersaturation(tb, i, k, x0):
    xi = interface_liquid(tb, i, k, x0)
    if xi is None:
        return None
    om = (x0 - xi) / (XZR[k] - xi)
    return float(om) if 0 < om < 1 else None


@dataclass
class Channel:
    name: str
    force: np.ndarray
    dHf: float
    x: float
    omega: np.ndarray = None     # None: no diffusion limit


def channels_real(tb, x0, phases, partition):
    out = []
    f = forces(tb, x0, phases)
    for k in phases:
        om = None
        if partition and abs(XZR[k] - x0) > 1e-6:
            om = np.array([supersaturation(tb, i, k, x0) if f[k][i] > 1.0 else np.nan for i in range(len(TGRID))], dtype=object)
            om = np.array([np.nan if v is None else v for v in om], dtype=float)
        out.append(Channel(k, f[k], fusion_enthalpy(tb, k), XZR[k], om))
    return out


def channel_mixture(tb, x0):
    f = mixture_force(tb, x0)
    i = int(np.argmin(np.abs(TGRID - 1200.0)))
    return [Channel("mixture", f, float(f[i] + 1200.0 * (-np.gradient(f, TGRID)[i])), x0, None)]


def radius_time(R, U, A):
    return R / U + (R ** 2 / (4 * A) if A else 0.0)


def fraction_integral(R, U, A):
    """Integral of R^3 ds to radius R, with s(R) = R/U + R^2/(4 A)."""
    return R ** 4 / (4 * U) + (R ** 5 / (10 * A) if A else 0.0)


def transformation_time(rates, Us, As):
    """Time for the summed extended fraction to reach X_DET. Every channel has its own rate, attachment speed, A = D lam^2."""
    def frac(t):
        total = 0.0
        for I, U, A in zip(rates, Us, As):
            if I <= 0 or U <= 0:
                continue
            if A:
                # R from t = R/U + R^2/(4A)
                R = 2 * t * U / (np.sqrt(1 + t * U ** 2 / A) + 1)
            else:
                R = U * t
            total += I * (4 * np.pi / 3) * fraction_integral(R, U, A)
        return total
    lo, hi = 1e-12, 1e16
    if frac(hi) < X_DET:
        return np.inf
    for _ in range(90):
        mid = np.sqrt(lo * hi)
        if frac(mid) < X_DET:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def ttt(chs, Tg, Ts, aT, sigma=None, n=260, Tmax=None):
    """TTT curve and nose. sigma overrides Turnbull. Returns dc in mm, nose, Rc and the liquidus of these channels."""
    B, T0 = vft(Tg, Ts)
    tl = max((TGRID[np.where(c.force > 0)[0].max()] if (c.force > 0).any() else 0.0) for c in chs)
    # refine the liquidus to the crossing
    for c in chs:
        idx = np.where(c.force > 0)[0]
        if len(idx) and idx.max() + 1 < len(TGRID) and TGRID[idx.max()] == tl:
            a, b = c.force[idx.max()], c.force[idx.max() + 1]
            tl = tl + (TGRID[1] - TGRID[0]) * a / (a - b)
    Ts_ = np.linspace(max(Tg + 5, TGRID[0]), min(tl - 3, Tmax or 1e9), n)
    best = (np.inf, None)
    curve = np.full(n, np.inf)
    for m, T in enumerate(Ts_):
        eta = ETA0 * np.exp(B / (T - T0))
        D = kB * T / (3 * np.pi * JUMP * eta)
        rates, Us, As = [], [], []
        for c in chs:
            g = float(np.interp(T, TGRID, c.force))
            if g <= 1.0:
                continue
            V = vm(c.x)
            s = sigma if sigma is not None else aT * c.dHf / (NA ** (1 / 3) * V ** (2 / 3))
            bar = 16 * np.pi * s ** 3 / (3 * (g / V) ** 2)
            rates.append((NA / V) * (kB * T / (3 * np.pi * JUMP ** 3 * eta)) * np.exp(-bar / (kB * T)))
            Us.append(D / JUMP * (1 - np.exp(-g / (Rg * T))))
            A = 0.0
            if c.omega is not None:
                om = float(np.interp(T, TGRID, np.nan_to_num(c.omega, nan=-1.0)))
                if 0 < om < 1:
                    lam = growth_parameter(om); A = D * lam ** 2
            As.append(A)
        if rates:
            curve[m] = transformation_time(rates, Us, As)
    j = int(np.argmin(curve))
    Rc = (tl - Ts_[j]) / curve[j]
    return dict(dc=float(10 * np.sqrt(10 / Rc)), Tn=float(Ts_[j]), tn=float(curve[j]), Rc=float(Rc), Tl=float(tl), T=Ts_, t=curve)


def fit_coefficient(chs, Tg, Ts, target_dc, lo=0.2, hi=1.6):
    return brentq(lambda a: np.log(ttt(chs, Tg, Ts, a)["dc"] / target_dc), lo, hi, xtol=2e-3)
