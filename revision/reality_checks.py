#!/usr/bin/env python3
"""Tests of hyalos_reality against things with known answers, each with a planted failure where that is meaningful."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import hyalos_reality as r
from hyalos_reality import growth_parameter
ROOT = os.environ.get("HALLSTEDT_TDB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdb"))
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + "  " + str(detail))
    fails += 0 if ok else 1


# 1. isothermal fraction reduces to the textbook form when there is no diffusion limit
I, U = 1e18, 1e-4
t = r.transformation_time([I], [U], [0.0])
want = (3 * r.X_DET / (np.pi * I * U ** 3)) ** 0.25
check("attachment limit equals (3x/pi I U^3)^(1/4)", abs(t / want - 1) < 1e-9, (t, want))
# 2. an enormous A behaves as no diffusion limit; a small A slows the transformation
tA = r.transformation_time([I], [U], [1e3])
check("large A recovers attachment control", abs(tA / want - 1) < 1e-6, tA / want)
tB = r.transformation_time([I], [U], [1e-14])
check("diffusion limit lengthens the time", tB > want, tB / want)
# 3. pure diffusion limit: R = 2 lam sqrt(D t), so x = I (4 pi/3) int R^3 ds = I (4 pi/3) (2 sqrt A)^3 t^(5/2) / (5/2)
A = 1e-16
t3 = r.transformation_time([I], [1e6], [A])
want3 = (r.X_DET / (I * (4 * np.pi / 3) * (2 * np.sqrt(A)) ** 3 / 2.5)) ** 0.4
check("diffusion limit has the t^(5/2) closed form", abs(t3 / want3 - 1) < 1e-3, t3 / want3)
# 4. growth parameter: lam -> sqrt(Omega/2) at small Omega
check("Zener lam at small Omega", abs(growth_parameter(1e-4) / np.sqrt(0.5e-4) - 1) < 2e-2, growth_parameter(1e-4) / np.sqrt(0.5e-4))
# 5. the same code path reproduces the calorimetric control of Castellero and Battezzati through hyalos_kinetics
Tf, dH = 1208.0, 9100.0
a, b = 55.979 - 25.5, -0.01688 - 0.00517
G = dH * (1 - r.TGRID / Tf) - (a * (Tf - r.TGRID) + b * (Tf ** 2 - r.TGRID ** 2) / 2) + r.TGRID * (a * np.log(Tf / r.TGRID) + b * (Tf - r.TGRID))
c = r.Channel("cal", G, dH, 0.5, None)
res = r.ttt([c], 675, 1064, 0.597, sigma=0.1336, n=500)
check("calorimetric control, 1.87 mm in revision_numbers", abs(res["dc"] / 1.87 - 1) < 0.05, round(res["dc"], 3))
check("  planted wrong sigma is caught", abs(r.ttt([c], 675, 1064, 0.597, sigma=0.14, n=200)["dc"] / 1.87 - 1) > 0.05)

tb = r.Tables.from_tdb(os.path.join(ROOT, "CuZr-16Hsi.tdb"), "16Hsi")
# 6. parallel tangent: a phase at the liquid composition has the plain difference
f = r.forces(tb, 0.5, ["CUZR"])["CUZR"]
i850 = int(np.argmin(np.abs(r.TGRID - 850)))
g0 = np.interp(0.5, r.XGRID, tb.GL[i850]) - tb.Gc["CUZR"][i850]
check("tangent force at own composition is G_L - G_k", abs(f[i850] - g0) < 1e-6, f[i850])
# 7. mixture force is the weighted mean of the real phase forces when the mixture bounds x0 (the point of the crystal fix)
F = r.forces(tb, 0.36, ["CU8ZR3", "CU10ZR7"])
fr = (0.36 - r.XZR["CU8ZR3"]) / (r.XZR["CU10ZR7"] - r.XZR["CU8ZR3"])
mean = (1 - fr) * F["CU8ZR3"][i850] + fr * F["CU10ZR7"][i850]
check("mixture force = weighted mean of tangent forces", abs(mean - r.mixture_force(tb, 0.36)[i850]) < 2.0 or r.mixture_force(tb, 0.36)[i850] > mean, (round(mean, 1), round(r.mixture_force(tb, 0.36)[i850], 1)))
# 8. spectrum: target heat capacity realized, G and S unchanged at the anchor
w = 7500.0
sp = r.with_spectrum(tb, r.gaussian_spectrum(lambda x: w))
j = int(np.argmin(np.abs(r.XGRID - 0.5)))
ia = int(np.argmin(np.abs(r.TGRID - r.T_ANCHOR)))
check("G unchanged at the anchor", abs(sp.GL[ia, j] - tb.GL[ia, j]) < 1e-6)
new = r.nk_difference(sp, 0.5)
idx = [int(np.argmin(np.abs(r.TGRID - T))) for T in (750, 900, 1100)]
want_c = [w ** 2 / (r.Rg * r.TGRID[k] ** 2) for k in idx]
check("heat capacity difference equals w^2/(R T^2)", all(abs(new[k] / wc - 1) < 0.02 for k, wc in zip(idx, want_c)), [(round(new[k], 2), round(wc, 2)) for k, wc in zip(idx, want_c)])
d0 = np.interp(0.5, r.XGRID, tb.GL[i850]) - np.interp(0.5, r.XGRID, sp.GL[i850])
check("spectrum lowers the liquid below the anchor", d0 > 0, round(d0))
# 9. Omega lies in (0,1) and grows on cooling for a partitioning phase
oms = [r.supersaturation(tb, int(np.argmin(np.abs(r.TGRID - T))), "CU10ZR7", 0.36) for T in (1100, 1000, 900)]
check("supersaturation of Cu10Zr7 from Cu64Zr36 grows on cooling", all(o is not None for o in oms) and oms[0] < oms[1] < oms[2], oms)
print("failures", fails)
sys.exit(1 if fails else 0)
