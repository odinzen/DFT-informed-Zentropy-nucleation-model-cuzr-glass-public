#!/usr/bin/env python3
"""Heat capacities already contained in the Appendix B MACE-MP-0 runs: Cp = dH/dT with
H = <Epot> + (3/2)kT at zero pressure, for the liquids and crystals at the three temperatures of
each alloy. Compared with the calorimetric liquid Cp fit of Castellero and Battezzati (2023, Eq. 5).
"""
import glob, json, os
import numpy as np

kB_eV = 8.617333e-5; F = 96485.33  # J/mol per eV
HERE = os.path.dirname(os.path.abspath(__file__))
LIQ = os.path.join(HERE, "..", "calibration_free_liquidus", "ti_driver", "results_local")
CRY = os.path.join(HERE, "..", "calibration_free_liquidus", "ti_driver", "results_crystal")


def slope(Ts, Es):
    """Least-squares dH/dT in J/mol/K with its standard error; Es per atom in eV."""
    Ts = np.asarray(Ts, float); H = (np.asarray(Es) + 1.5 * kB_eV * Ts) * F
    A = np.vstack([Ts, np.ones_like(Ts)]).T
    c, res, *_ = np.linalg.lstsq(A, H, rcond=None)
    if len(Ts) > 2:
        s2 = ((H - A @ c) ** 2).sum() / (len(Ts) - 2)
        se = np.sqrt(s2 / ((Ts - Ts.mean()) ** 2).sum())
    else:
        se = float("nan")
    return c[0], se


def main():
    print("units check, first liquid file: epot_mean_eV =", json.load(open(glob.glob(LIQ + "/Fliquid_*.json")[0]))["epot_mean_eV"])
    for comp in ("cu50zr50", "cu64zr36"):
        rows = [json.load(open(f)) for f in glob.glob(f"{LIQ}/Fliquid_{comp}_*.json")]
        Ts = [r["T"] for r in rows]; Es = [r["epot_mean_eV"] for r in rows]
        c, se = slope(Ts, Es)
        print(f"liquid {comp}: {len(rows)} runs, T {min(Ts):.0f}-{max(Ts):.0f} K, Cp = {c:.1f} +- {se:.1f} J/mol/K")
        for cell in "abc":
            sub = [r for r in rows if r["cell"] == cell]
            cc, _ = slope([r["T"] for r in sub], [r["epot_mean_eV"] for r in sub])
            print(f"   cell {cell}: Cp = {cc:.1f}")
    for name in ("CuZr_B2", "Cu10Zr7", "CuZr2", "Cu8Zr3"):
        rows = [json.load(open(f)) for f in glob.glob(f"{CRY}/Fcrystal_{name}_*.json")]
        rows = [r for r in rows if "epot_mean_eV" in r]
        Ts = [r["T"] for r in rows]; Es = [r["epot_mean_eV"] for r in rows]
        c, se = slope(Ts, Es)
        print(f"crystal {name}: {len(rows)} runs, T {min(Ts):.0f}-{max(Ts):.0f} K, Cp = {c:.1f} +- {se:.1f} J/mol/K")
    print("\ncalorimetry, CuZr at 1208 K: liquid", round(55.979 - 0.01688 * 1208, 1), "crystal (Neumann-Kopp, read from their Fig. 3) 31.7")


if __name__ == "__main__":
    main()
