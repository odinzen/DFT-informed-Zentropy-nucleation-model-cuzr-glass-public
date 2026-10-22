#!/usr/bin/env python3
"""Calibration-free driving force from the switched liquid free energies (ti_driver/results_local).

dG(T) = F_liquid(T) - F_crystal(T), F_crystal the tie-line lever of MACE E0 + harmonic F_vib
(Cu50Zr50: Cu10Zr7 + CuZr2; Cu64Zr36: Cu8Zr3 + Cu10Zr7). The liquid is the mean of cells a, b, c,
with the standard error of that mean. The split into dH and dS uses the liquid's own <U> and
S = (<U> + 3/2 kT - F)/T, and the crystal's harmonic U and S_vib. The liquidus follows from a
straight-line fit of dG(T) through the three temperatures, extrapolated to dG = 0.
"""
import glob
import json
import os

import numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
FV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "fvib")
ENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "mlip_dE_ico.json")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ti_driver", "results_local")
EV_KJ, KB = 96.485332, 8.617333262e-5
X_ZR = {"Cu8Zr3": 3 / 11, "Cu10Zr7": 7 / 17, "CuZr2": 2 / 3}
# exact cell compositions (54/108, 39/108): a raw PBE energy moves ~1.3 keV per unit x, so the
# rounded 0.361 alone would shift the crystal lever by ~140 meV/atom
CASES = {"cu50zr50": (54 / 108, 1208.0, "Cu10Zr7", "CuZr2"), "cu64zr36": (39 / 108, 1230.0, "Cu8Zr3", "Cu10Zr7")}


def fvib(label, T, key):
    d = json.load(open(os.path.join(FV, f"cryst_{label}.json")))
    return float(np.interp(T, d["T"], d[key]))


def crystal(lo, hi, x, T):
    """(F, U, S) of the tie-line crystal per atom: eV, eV, J/(K mol)."""
    refs = {k: v["epa"] for k, v in json.load(open(ENE))["refs"].items()}
    f = (x - X_ZR[lo]) / (X_ZR[hi] - X_ZR[lo])
    w = {lo: 1 - f, hi: f}
    F = sum(w[p] * (refs[p] + fvib(p, T, "F_vib_kJ_per_mol_atom") / EV_KJ) for p in w)
    S = sum(w[p] * fvib(p, T, "S_vib_J_per_K_mol_atom") for p in w)
    return F, F + T * S / (EV_KJ * 1000), S


def main():
    out = {}
    for comp, (x, Tl, lo, hi) in CASES.items():
        rows = []
        for T in (Tl - 50, Tl, Tl + 50):
            runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RES, f"Fliquid_{comp}_*_{int(T)}K.json")))]
            F = np.array([r["F_liquid_eV"] for r in runs])
            U = np.array([r["epot_mean_eV"] + 1.5 * KB * T for r in runs])
            S = (U - F) / T * EV_KJ * 1000
            Fc, Uc, Sc = crystal(lo, hi, x, T)
            se = F.std(ddof=1) / np.sqrt(len(F))
            rows.append(dict(T=T, n=len(F), dG_kJ=(F.mean() - Fc) * EV_KJ, dG_se_kJ=se * EV_KJ,
                             dH_kJ=(U.mean() - Uc) * EV_KJ, dS_J=S.mean() - Sc,
                             spread_meV=1000 * np.ptp(F)))
        T, dG = np.array([r["T"] for r in rows]), np.array([r["dG_kJ"] for r in rows])
        slope, icpt = np.polyfit(T, dG, 1)
        out[comp] = dict(rows=rows, liquidus_K=-icpt / slope, measured_liquidus_K=Tl)
        print(f"{comp}  (measured liquidus {Tl:.0f} K)")
        for r in rows:
            print(f"  {r['T']:.0f} K  dG {r['dG_kJ']:+6.2f} +- {r['dG_se_kJ']:.2f} kJ/mol   dH {r['dH_kJ']:+5.2f} kJ/mol  "
                  f"dS {r['dS_J']:+5.2f} J/(K mol)   cells {r['n']}, spread {r['spread_meV']:.1f} meV")
        print(f"  dG = 0 at {-icpt / slope:.0f} K (straight line through the three points)")
    json.dump(out, open(os.path.join(RES, "liquid_dG.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
