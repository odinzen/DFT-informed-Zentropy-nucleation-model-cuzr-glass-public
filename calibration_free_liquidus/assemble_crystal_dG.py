#!/usr/bin/env python3
"""Table B1: calibration-free liquidus from the switched liquid (ti_driver/results_local) against
the switched crystals, both corrected to PBE by free-energy perturbation at the measured liquidus
(liquid: dft_correction_k222; crystals: the crystal frames).

At T_l, dG = F_liq - F_cryst from the switching runs, dH = <U_liq> - <U_cryst>, dS = (dH - dG)/T_l,
and the liquidus is T_l + dG_corrected/dS. Cu50Zr50 is referenced to the Cu10Zr7 + CuZr2 assembly
and to B2 CuZr, the phase that melts at 1208 K; Cu64Zr36 to Cu8Zr3 + Cu10Zr7. The PBE correction
is evaluated three ways (second-order cumulant, exponential average, first order), the liquid and
the crystals always on the same estimator. Standard errors bootstrap the crystal frames and the
liquid correction and include the spread of the three liquid cells. Also reports B2 against the
assembly at 1208 K.

Usage:  python assemble_crystal_dG.py <crystal results dir> [crystal dft dir]
Without the dft dir it reports the MACE-level columns only.
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIQ = os.path.join(HERE, "ti_driver", "results_local")
LIQ_DFT = os.path.join(HERE, "dft_correction_k222", "dft_correction.json")
RY, KB, EV_KJ = 13.605693122994, 8.617333262e-5, 96.485332
X_ZR = {"Cu8Zr3": 3 / 11, "Cu10Zr7": 7 / 17, "CuZr2": 2 / 3, "CuZr_B2": 1 / 2}
COLUMNS = [  # label, alloy, exact cell x, measured liquidus, crystal reference
    ("Cu50Zr50 vs Cu10Zr7 + CuZr2", "cu50zr50", 54 / 108, 1208, ("Cu10Zr7", "CuZr2")),
    ("Cu50Zr50 vs B2 CuZr", "cu50zr50", 54 / 108, 1208, ("CuZr_B2",)),
    ("Cu64Zr36 vs Cu8Zr3 + Cu10Zr7", "cu64zr36", 39 / 108, 1230, ("Cu8Zr3", "Cu10Zr7")),
]
ESTIMATORS = ("cumulant", "exponential", "first")
N_BOOT = 4000


def liquid(comp, T):
    """Mean F and <U> per atom (eV) over the three cells, and the standard error of F."""
    runs = [json.load(open(p)) for p in glob.glob(os.path.join(LIQ, f"Fliquid_{comp}_*_{T}K.json"))]
    F = np.array([r["F_liquid_eV"] for r in runs])
    U = np.array([r["epot_mean_eV"] + 1.5 * KB * T for r in runs])
    return F.mean(), U.mean(), F.std(ddof=1) / np.sqrt(len(F))


def crystal(cdir, name, T):
    r = json.load(open(os.path.join(cdir, f"Fcrystal_{name}_{T}K.json")))
    return r["F_crystal_eV"], r["epot_mean_eV"] + 1.5 * KB * T


def frame_differences(ddir):
    """E_PBE - E_MACE per frame (eV, whole cell) and atom count, by crystal tag."""
    out = {}
    for line in open(os.path.join(ddir, "mace_energies.tsv")).read().splitlines()[1:]:
        name, e_mace, nat = line.split("\t")
        text = open(os.path.join(ddir, "outputs", f"{name}.out"), errors="replace").read()
        e = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+) Ry", text, re.M)
        if "JOB DONE" not in text or not e:
            raise SystemExit(f"{name}: PBE frame not finished")
        tag = name[len("cryst_"):name.index("K_f")]
        out.setdefault(tag, ([], int(nat)))[0].append(float(e[-1]) * RY - float(e_mace))
    return {k: (np.array(v), n) for k, (v, n) in out.items()}


def fep(dU, T, how):
    """Free-energy shift of the whole cell from MACE to PBE."""
    kt = KB * T
    if how == "cumulant":
        return dU.mean() - dU.var(ddof=1) / (2 * kt)
    if how == "first":
        return dU.mean()
    return dU.min() - kt * np.log(np.mean(np.exp(-(dU - dU.min()) / kt)))


def mix(refs, x, values):
    if len(refs) == 1:
        return values[refs[0]]
    lo, hi = refs
    f = (x - X_ZR[lo]) / (X_ZR[hi] - X_ZR[lo])
    return (1 - f) * values[lo] + f * values[hi]


def correction(comp, x, Tl, refs, frames, liq, how, rng=None):
    """PBE correction to dG(T_l) in kJ/mol. The liquid's cumulant shift is stored; its first-order
    and exponential values follow from the stored second-order term and exponential offset."""
    lq = liq[comp]
    liquid_shift = lq["liquid_dF_eV_per_atom"] * EV_KJ + {
        "cumulant": 0.0, "first": -lq["variance_term_kJ"],
        "exponential": lq["zwanzig_minus_cumulant_kJ"]}[how]
    if rng is not None:
        liquid_shift += rng.normal(0, lq["dG_shift_se_kJ"])
    shifts = {}
    for c in refs:
        dU, n = frames[f"{c}_{Tl}"]
        if rng is not None:
            dU = rng.choice(dU, len(dU))
        shifts[c] = fep(dU, Tl, how) / n
    return liquid_shift - mix(refs, x, shifts) * EV_KJ


def main():
    cdir = sys.argv[1]
    ddir = sys.argv[2] if len(sys.argv) > 2 else None
    frames = frame_differences(ddir) if ddir else None
    liq = json.load(open(LIQ_DFT))
    rng = np.random.default_rng(20261024)
    out = {}
    for label, comp, x, Tl, refs in COLUMNS:
        F_l, U_l, se_l = liquid(comp, Tl)
        Fc = {c: crystal(cdir, c, Tl) for c in refs}
        dG = (F_l - mix(refs, x, {c: v[0] for c, v in Fc.items()})) * EV_KJ
        dH = (U_l - mix(refs, x, {c: v[1] for c, v in Fc.items()})) * EV_KJ
        dS = (dH - dG) / Tl * 1000
        col = dict(measured_K=Tl, dG_mace_kJ=dG, dG_mace_se_kJ=se_l * EV_KJ, dH_f_kJ=dH, dS_f_J=dS)
        print(f"{label} (measured {Tl} K)\n  MACE-MP-0  dG {dG:+.2f} +- {se_l * EV_KJ:.2f} kJ/mol  "
              f"dH_f {dH:.2f} kJ/mol  dS_f {dS:.2f} J/(K mol)")
        if frames:
            for how in ESTIMATORS:
                corr = correction(comp, x, Tl, refs, frames, liq, how)
                col[how] = dict(correction_kJ=corr, dG_kJ=dG + corr, liquidus_K=Tl + 1000 * (dG + corr) / dS)
            boot = np.array([[correction(comp, x, Tl, refs, frames, liq, "cumulant", rng),
                              rng.normal(0, se_l * EV_KJ)] for _ in range(N_BOOT)])
            col["cumulant"]["correction_se_kJ"] = boot[:, 0].std()
            col["cumulant"]["liquidus_se_K"] = (Tl + 1000 * (dG + boot.sum(axis=1)) / dS).std()
            second = (correction(comp, x, Tl, refs, frames, liq, "cumulant")
                      - correction(comp, x, Tl, refs, frames, liq, "first"))
            col["second_order_kJ"] = second
            c = col["cumulant"]
            print(f"  PBE correction {c['correction_kJ']:+.2f} +- {c['correction_se_kJ']:.2f} kJ/mol "
                  f"(second-order terms {second:+.2f})  corrected dG {c['dG_kJ']:+.2f} kJ/mol")
            print("  liquidus  " + "  ".join(
                f"{how} {col[how]['liquidus_K']:.0f}" + (f" +- {c['liquidus_se_K']:.0f}" if how == "cumulant" else "")
                + " K" for how in ESTIMATORS))
        out[label] = col
    F = {c: crystal(cdir, c, 1208)[0] for c in ("CuZr_B2", "Cu10Zr7", "CuZr2")}
    pair = ("Cu10Zr7", "CuZr2")
    b2 = {"mace": (F["CuZr_B2"] - mix(pair, 0.5, F)) * EV_KJ}
    if frames:
        def pbe(how, r=None):
            s = {}
            for c in F:
                dU, n = frames[f"{c}_1208"]
                s[c] = fep(dU if r is None else r.choice(dU, len(dU)), 1208, how) / n
            return b2["mace"] + (s["CuZr_B2"] - mix(pair, 0.5, s)) * EV_KJ
        b2.update({how: pbe(how) for how in ESTIMATORS})
        b2["cumulant_se"] = np.std([pbe("cumulant", rng) for _ in range(N_BOOT)])
    out["B2_minus_pair_1208K_kJ"] = b2
    print("B2 CuZr minus Cu10Zr7 + CuZr2 at 1208 K (kJ/mol): "
          + "  ".join(f"{k} {v:+.2f}" for k, v in b2.items()) + "  (negative: B2 the more stable)")
    json.dump(out, open(os.path.join(cdir, "crystal_dG.json" if frames else "crystal_dG_mace.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
