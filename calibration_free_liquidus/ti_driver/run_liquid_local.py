#!/usr/bin/env python3
"""Local-GPU twin of sol/run_liquid_sol.py: F_liquid(T) for both Cu-Zr compositions by
UF -> MACE switching, plus a sampled stretch of each real liquid that Hermes reads afterwards
(hermes_compare.py).

Runs cell a for both compositions before cells b and c, so a first look at hysteresis and the
Hermes comparison comes after a third of the campaign. Resumable: finished (cell, T) results
are skipped.

Usage:  python run_liquid_local.py [cu50zr50|cu64zr36 ...]      the full campaign, or
        python run_liquid_local.py cu64zr36_b_1180 ...           just these (composition_cell_T) runs
Env:    TI_DEVICE (cuda), TI_DTYPE (float32), TI_CUEQ (1), TI_T_EQ_FS (20000),
        TI_T_SWITCH_FS (50000), TI_T_SAMPLE_FS (30000), TI_T_RELAX_FS (20000), TI_OUT, TI_GLASS,
        CALPHY_PYTHON.
"""
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from liquid_ti import liquid_free_energy, relax_volume

DEVICE = os.environ.get("TI_DEVICE", "cuda")
DTYPE = os.environ.get("TI_DTYPE", "float32")
CUEQ = os.environ.get("TI_CUEQ", "1") == "1"
T_EQ = int(os.environ.get("TI_T_EQ_FS", "20000"))
T_SW = int(os.environ.get("TI_T_SWITCH_FS", "50000"))
T_SAMPLE = int(os.environ.get("TI_T_SAMPLE_FS", "30000"))
T_RELAX = int(os.environ.get("TI_T_RELAX_FS", "20000"))
# sigma 1.5 (calphy's default) puts the UF repulsion near 3 A, the Cu-Zr neighbour distance; at
# 2.0 the reference fluid is squeezed to the real density and the switch dissipates more
P, SIGMA = 50, 1.5
GLASS = os.environ.get("TI_GLASS") or os.path.normpath(os.path.join(
    HERE, "..", "glass"))
LIQUIDUS = {"cu50zr50": 1208.0, "cu64zr36": 1230.0}
OUT = os.environ.get("TI_OUT", os.path.join(HERE, "results_local"))


def get_calc():
    from mace.calculators import mace_mp
    return mace_mp(model="medium", dispersion=False, default_dtype=DTYPE, device=DEVICE,
                   enable_cueq=CUEQ)


def run_one(calc, comp, cell, T):
    tag = f"{comp}_{cell}_{int(T)}K"
    dst = os.path.join(OUT, f"Fliquid_{tag}.json")
    if os.path.exists(dst):
        print(f"[skip] {tag}", flush=True)
        return
    t0 = time.time()

    def log(msg):
        print(f"  {time.strftime('%H:%M:%S')} {tag}: {msg}", flush=True)

    log("relax volume")
    g0 = read(os.path.join(GLASS, f"scf_glass_reos_{comp}_{cell}_v2.in"), format="espresso-in")
    g = relax_volume(g0, calc, T, t_fs=T_RELAX, log=log)
    r = liquid_free_energy(g, calc, T, p=P, sigma=SIGMA, t_equilibrate_fs=T_EQ,
                           t_switch_fs=T_SW, t_sample_fs=T_SAMPLE, log=log)
    s = r.pop("samples")
    np.savez_compressed(os.path.join(OUT, f"frames_{tag}.npz"), **s)
    r.update(cell=cell, composition=comp, device=DEVICE, dtype=DTYPE, cueq=CUEQ,
             volume_A3_per_atom=g.get_volume() / len(g), t_sample_fs=T_SAMPLE,
             epot_mean_eV=float(s["epot_eV"].mean()) / len(g),
             pressure_mean_GPa=float(s["pressure_GPa"].mean()), wall_s=time.time() - t0)
    json.dump(r, open(dst, "w"), indent=2)
    print(f"[{tag}] F_liquid={r['F_liquid_eV']:.4f} eV/atom  hyst={r['hysteresis_eV']*1000:.1f} meV  "
          f"P={r['pressure_mean_GPa']:+.2f} GPa  ({r['wall_s']/3600:.1f} h)", flush=True)


def main():
    args = sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    tagged = [a for a in args if a.count("_") == 2]
    if tagged:
        calc = get_calc()
        for tag in tagged:
            comp, cell, T = tag.split("_")
            run_one(calc, comp, cell, float(T))
        print("JOB DONE ->", OUT, flush=True)
        return
    comps = args or list(LIQUIDUS)
    calc = get_calc()
    for cell in ("a", "b", "c"):
        for comp in comps:
            Tl = LIQUIDUS[comp]
            for T in (Tl - 50, Tl, Tl + 50):
                run_one(calc, comp, cell, T)
    print("JOB DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
