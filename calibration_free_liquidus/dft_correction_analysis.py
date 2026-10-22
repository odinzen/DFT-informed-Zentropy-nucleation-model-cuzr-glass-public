#!/usr/bin/env python3
"""DFT (PBE) correction to the MACE driving force, from the dft_correction/ outputs.

Liquid: free-energy perturbation from the MACE ensemble to PBE,
    dF = <dU> - (1/2kT) Var(dU),   dU = E_PBE - E_MACE per frame (whole cell),
with the Appendix A Gamma-point k-corrections (+0.7 meV/atom Cu50Zr50, +7.6 Cu64Zr36) added to E_PBE.
Crystal: E_PBE at the reference structure (the first SCF of a run) minus MACE at its own minimum, on
the same tie-lines as assemble_liquid_dG.py. Relaxing CuZr2 fully in PBE lowered it by only
0.6 meV/atom, so the reference structure stands in for the PBE minimum. The driving force moves
by (liquid correction - crystal correction); the crystal phonons stay MACE, so only the energy
surface is corrected.

Usage:  python dft_correction_analysis.py [dft_correction dir] [k222]
        k222: the frames are already k-converged (2x2x2), so no Gamma-point correction is added
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RY = 13.605693122994
KB = 8.617333262e-5
EV_KJ = 96.485332
K_CORR = {"cu50zr50": 0.0007, "cu64zr36": 0.0076}
# exact cell compositions (54/108, 39/108): a raw PBE energy moves ~1.3 keV per unit x, so the
# rounded 0.361 alone would shift the crystal lever by ~140 meV/atom
TIE = {"cu50zr50": (54 / 108, 1208.0, "Cu10Zr7", "CuZr2"), "cu64zr36": (39 / 108, 1230.0, "Cu8Zr3", "Cu10Zr7")}
X_ZR = {"Cu8Zr3": 3 / 11, "Cu10Zr7": 7 / 17, "CuZr2": 2 / 3}
MACE_REFS = os.path.join(HERE, "inputs", "mlip_dE_ico.json")
LIQUID_DG = os.path.join(HERE, "ti_driver", "results_local", "liquid_dG.json")


def total_energy(path, first=False):
    """'!' total energy (eV, last or first) and atom count; None until it exists. A liquid frame
    must have finished (JOB DONE); a crystal needs only its first SCF."""
    if not os.path.exists(path):
        return None
    text = open(path, errors="replace").read()
    e = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+) Ry", text, re.M)
    if not e or (not first and "JOB DONE" not in text):
        return None
    nat = int(re.search(r"number of atoms/cell\s+=\s+(\d+)", text).group(1))
    return float(e[0] if first else e[-1]) * RY, nat


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "dft_correction_k222")
    k_corr = {c: 0.0 for c in K_CORR} if "k222" in sys.argv[2:] else K_CORR
    mace_cryst = {k: v["epa"] for k, v in json.load(open(MACE_REFS))["refs"].items()}
    cryst = {}
    for name in ("CuZr2", "Cu8Zr3", "Cu10Zr7"):
        r = total_energy(os.path.join(d, f"cryst_{name}.out"), first=True)
        if r:
            cryst[name] = r[0] / r[1] - mace_cryst[name]      # eV/atom, PBE minus MACE
    rows = [l.split("\t") for l in open(os.path.join(d, "mace_energies.tsv")).read().splitlines()[1:]]
    out = {"crystal_dE_eV_per_atom": cryst}
    for comp, (x, T, lo, hi) in TIE.items():
        dU = []
        for name, e_mace, nat in rows:
            if comp not in name:
                continue
            r = total_energy(os.path.join(d, f"{name}.out"))
            if r:
                dU.append(r[0] + k_corr[comp] * int(nat) - float(e_mace))
        if not dU or lo not in cryst or hi not in cryst:
            print(f"{comp}: {len(dU)} liquid frames done; crystals {sorted(cryst)}; waiting")
            continue
        dU = np.array(dU)
        n = int(rows[0][2])
        var = dU.var(ddof=1) if len(dU) > 1 else 0.0       # second-order term needs two frames
        liq = (dU.mean() - var / (2 * KB * T)) / n
        f = (x - X_ZR[lo]) / (X_ZR[hi] - X_ZR[lo])
        cr = (1 - f) * cryst[lo] + f * cryst[hi]
        shift = (liq - cr) * EV_KJ
        se = np.sqrt(var / len(dU)) / n * EV_KJ
        out[comp] = dict(frames=len(dU), liquid_dF_eV_per_atom=liq, crystal_dE_eV_per_atom=cr,
                         dG_shift_kJ=shift, dG_shift_se_kJ=se,
                         dU_spread_meV_per_atom=1000 * np.sqrt(var) / n)
        print(f"{comp}: {len(dU)} frames  liquid PBE-MACE {1000 * liq:+.1f} meV/atom  "
              f"crystal {1000 * cr:+.1f} meV/atom  -> driving force shifts {shift:+.2f} +- {se:.2f} kJ/mol "
              f"(frame spread {1000 * np.sqrt(var) / n:.1f} meV/atom)")
        if len(dU) >= 4:
            out[comp].update(assess(dU, n, T, cr, comp))
    json.dump(out, open(os.path.join(d, "dft_correction.json"), "w"), indent=2)


def cumulant_kJ(dU, n, T):
    """Second-order FEP free energy per atom (kJ/mol) and its variance term."""
    v = dU.var(ddof=1) / (2 * KB * T)
    return (dU.mean() - v) / n * EV_KJ, -v / n * EV_KJ


def assess(dU, n, T, cr, comp, first=10, boots=4000, seed=7):
    """Bootstrap the corrected liquidus and test whether more frames are needed. The rule was fixed
    before batch 2 was read (2026-10-20): more frames if the bootstrap standard error of the
    liquidus exceeds 50 K, or if the variance term moves between the first ten frames and all of
    them by more than its own bootstrap error on the first ten."""
    g = json.load(open(LIQUID_DG))[comp]
    row = next(r for r in g["rows"] if r["T"] == T)
    rng = np.random.default_rng(seed)

    def liquidus(sample):
        return T + 1000 * (row["dG_kJ"] + cumulant_kJ(sample, n, T)[0] - cr * EV_KJ) / row["dS_J"]

    boot = np.array([liquidus(rng.choice(dU, len(dU))) for _ in range(boots)])
    var_all = cumulant_kJ(dU, n, T)[1]
    head = dU[:first]
    var_head = cumulant_kJ(head, n, T)[1]
    var_head_se = np.std([cumulant_kJ(rng.choice(head, len(head)), n, T)[1] for _ in range(boots)])
    zwanzig = (dU.min() - KB * T * np.log(np.mean(np.exp(-(dU - dU.min()) / (KB * T))))) / n * EV_KJ
    more = bool(boot.std() > 50 or (len(dU) > first and abs(var_all - var_head) > var_head_se))
    print(f"   liquidus {liquidus(dU):.0f} +- {boot.std():.0f} K (bootstrap); variance term {var_all:+.2f} "
          f"(first {first}: {var_head:+.2f} +- {var_head_se:.2f}) kJ/mol; exponential average differs from "
          f"the cumulant by {zwanzig - cumulant_kJ(dU, n, T)[0]:+.2f} kJ/mol -> "
          f"{'MORE FRAMES' if more else 'enough frames'}")
    return dict(liquidus_K=liquidus(dU), liquidus_boot_se_K=float(boot.std()), variance_term_kJ=var_all,
                variance_term_first_kJ=var_head, variance_term_first_se_kJ=float(var_head_se),
                zwanzig_minus_cumulant_kJ=zwanzig - cumulant_kJ(dU, n, T)[0], more_frames=more)


if __name__ == "__main__":
    main()
