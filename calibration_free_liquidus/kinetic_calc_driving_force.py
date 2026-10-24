#!/usr/bin/env python3
"""Carry the calculated driving force of Appendix B through the nucleation model.

The published engine pins the icosahedral offset so that dG(T_l) = 0 at the measured liquidus. Two
replacements, both from crystal_dG.json (Table B1), for the two binaries Appendix B covers:
  anchored  dG(T_l) = the calculated, PBE-corrected difference instead of zero, so the engine's
            liquidus moves to the calculated one; its own entropy of fusion is kept.
  +entropy  the same anchor, and the slope at T_l rescaled to the calculated entropy of fusion
            (the engine's curvature below T_l is kept); the Turnbull interfacial energy follows,
            since it is built from T_l dS_f.
Each is run at the cumulant value and at +-1 standard error of dG(T_l). The engine's nucleation
physics is executed from nucleation_zentropy.py itself, up to its figure code, so it is unchanged.

Usage:  python kinetic_calc_driving_force.py      (run from any scratch directory)
"""
import json
import os
import sys
import types

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, "..", "code"))
TABLE = os.path.join(HERE, "ti_driver", "results_crystal", "crystal_dG.json")
sys.path.insert(0, ENGINE)
import zentropy_driving_force as z

src = open(os.path.join(ENGINE, "nucleation_zentropy.py"), encoding="utf-8").read()
nz = types.ModuleType("nz")
exec(compile(src[:src.index("res={k:ttt(v)")], "nucleation_zentropy.py", "exec"), nz.__dict__)

# the engine's Cu50Zr50 reference is B2 CuZr (lever rule between Cu10Zr7 and CuZr at x = 0.5)
COLUMN = {"Cu50Zr50": "Cu50Zr50 vs B2 CuZr", "Cu64Zr36": "Cu64Zr36 vs Cu8Zr3 + Cu10Zr7"}


def engine_dG(a, target=0.0):
    """Engine driving force with the offset pinned so that dG(T_l) = target (J/mol)."""
    x, lo, hi = a["x"], *a["bnd"]
    f = lambda dE: z.driving_force(a["Tl"], x, lo, hi, dE)[0] - target
    dE = brentq(f, 2.0e3, 40.0e3, xtol=1.0)
    return (lambda T: z.driving_force(T, x, lo, hi, dE)[0]), dE


def run(a, target=0.0, dS_calc=None):
    dG0, dE = engine_dG(a, target)
    Tl = a["Tl"]
    dS_eng = -(dG0(Tl + 1) - dG0(Tl - 1)) / 2.0
    if dS_calc is None:
        dG = dG0
    else:
        k = dS_calc / dS_eng
        dG = lambda T: target + k * (dG0(T) - target)
    nz.z = types.SimpleNamespace(make_dG_callable=lambda *_: (dG, z.comp_props(a["x"])[0], dE))
    return nz.ttt(a)


def main():
    table = json.load(open(TABLE))
    print("alloy      case                     dG(Tl) kJ/mol  dS_f J/(K mol)  T_nose K  R_c K/s   d_c mm")
    for name, a in nz.ALLOYS.items():
        base = run(a)
        print(f"{name:10s} engine (published)       {0:+6.2f}         {base['dSf']:6.2f}        "
              f"{base['Tn']:6.0f}  {base['Rc']:8.3g}  {base['dc']:6.2f}")
        if name not in COLUMN:
            continue
        col = table[COLUMN[name]]
        g, se = col["cumulant"]["dG_kJ"] * 1e3, np.hypot(col["cumulant"]["correction_se_kJ"], col["dG_mace_se_kJ"]) * 1e3
        for label, dS in (("anchored", None), ("anchored + entropy", col["dS_f_J"])):
            for tag, t in (("", g), (" -1 SE", g - se), (" +1 SE", g + se)):
                r = run(a, t, dS)
                print(f"{'':10s} {label + tag:24s} {t / 1e3:+6.2f}         {r['dSf']:6.2f}        "
                      f"{r['Tn']:6.0f}  {r['Rc']:8.3g}  {r['dc']:6.2f}")


if __name__ == "__main__":
    main()
