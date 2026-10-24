#!/usr/bin/env python3
"""Crystal free energies for the calibration-free liquidus: one Frenkel-Ladd run per (crystal, T),
MACE-MP-0 on the GPU, with a sampled stretch of the real crystal for the PBE correction.

Crystals and temperatures: B2 CuZr, CuZr2 and Cu10Zr7 at the Cu50Zr50 liquidus and 50 K either side
(1158, 1208, 1258 K); Cu8Zr3 and Cu10Zr7 at the Cu64Zr36 ones (1180, 1230, 1280 K). Each run is a
tag <crystal>_<T>, for example CuZr_B2_1208. Finished tags are skipped.

Usage:  python run_crystal.py CuZr_B2_1208 [more tags]
Env:    TI_DEVICE (cuda), TI_DTYPE (float32), TI_CUEQ (0), TI_STRUCT (../structures), TI_OUT,
        TI_T_RELAX_FS (20000), TI_T_MSD_FS (10000), TI_T_EQ_FS (10000), TI_T_SWITCH_FS (50000),
        TI_T_SAMPLE_FS (20000), CALPHY_PYTHON.
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
from crystal_ti import crystal_free_energy, supercell

DEVICE = os.environ.get("TI_DEVICE", "cuda")
DTYPE = os.environ.get("TI_DTYPE", "float32")
CUEQ = os.environ.get("TI_CUEQ", "0") == "1"
STRUCT = os.environ.get("TI_STRUCT") or os.path.join(HERE, "..", "structures")
OUT = os.environ.get("TI_OUT", os.path.join(HERE, "results_crystal"))
# environment names TI_T_RELAX_FS etc., as in the liquid runner
LENGTHS = {k: int(os.environ.get(f"TI_{k.upper()}", v)) for k, v in
           dict(t_relax_fs=20000, t_msd_fs=10000, t_equilibrate_fs=10000, t_switch_fs=50000,
                t_sample_fs=20000).items()}


# explicit repeats keep every cell at 96-176 atoms, which the 2x2x2 PBE stage can afford; a 10 A
# minimum width alone would take Cu10Zr7 to 272 atoms
REPEAT = {"CuZr_B2": (4, 4, 4), "Cu10Zr7": (1, 2, 2), "Cu8Zr3": (2, 2, 1)}


def get_calc():
    from mace.calculators import mace_mp
    return mace_mp(model="medium", dispersion=False, default_dtype=DTYPE, device=DEVICE,
                   enable_cueq=CUEQ)


def run_one(calc, tag):
    name, T = tag.rsplit("_", 1)
    dst = os.path.join(OUT, f"Fcrystal_{tag}K.json")
    if os.path.exists(dst):
        print(f"[skip] {tag}", flush=True)
        return
    t0 = time.time()

    def log(msg):
        print(f"  {time.strftime('%H:%M:%S')} {tag}: {msg}", flush=True)

    prim = read(os.path.join(STRUCT, f"{name}.vasp"))
    ideal = prim.repeat(REPEAT[name]) if name in REPEAT else supercell(prim)
    log(f"{len(ideal)} atoms, cell {np.round(ideal.cell.lengths(), 2)}")
    r = crystal_free_energy(ideal, calc, float(T), log=log, **LENGTHS)
    s = r.pop("samples")
    np.savez_compressed(os.path.join(OUT, f"frames_{tag}K.npz"), **s)
    r.update(crystal=name, device=DEVICE, dtype=DTYPE, cueq=CUEQ, lengths_fs=LENGTHS,
             volume_A3_per_atom=float(abs(np.linalg.det(s["cell"]))) / len(ideal),
             epot_mean_eV=float(s["epot_eV"].mean()) / len(ideal),
             pressure_mean_GPa=float(s["pressure_GPa"].mean()), wall_s=time.time() - t0)
    json.dump(r, open(dst, "w"), indent=2)
    F = "reference pending" if r["F_crystal_eV"] is None else f"{r['F_crystal_eV']:.4f} eV/atom"
    print(f"[{tag}] F_crystal={F}  dF_switch={r['dF_switch_eV']:.4f} eV/atom  hyst={r['hysteresis_eV']*1000:.1f} meV  "
          f"P={r['pressure_mean_GPa']:+.2f} GPa  max MSD {r['msd_sample_max_A2']:.3f} A^2  ({r['wall_s']/3600:.1f} h)",
          flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    calc = get_calc()
    for tag in sys.argv[1:]:
        run_one(calc, tag)
    print("JOB DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
