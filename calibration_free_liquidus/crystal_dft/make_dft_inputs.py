#!/usr/bin/env python3
"""Stage 2 inputs: PBE single points on crystal frames sampled at each liquidus (2x2x2 mesh, the
settings of the liquid correction). Nine frames per run, 2.5 ps apart. Run on the login node after
stage 1, from the kit root. Writes dft/inputs/*.in, dft/run_list.txt and dft/mace_energies.tsv.

`python sol/make_dft_inputs.py topup` adds the frames between those, every 0.5 ps, for the crystals
in TOPUP. run_list.txt then holds only the new jobs (the first list is kept as run_list_stage2a.txt)
and mace_energies.tsv lists every frame.
"""
import os
import sys

import numpy as np

AT_LIQUIDUS = ("CuZr_B2_1208", "CuZr2_1208", "Cu10Zr7_1208", "Cu8Zr3_1230", "Cu10Zr7_1230")
EVERY = 25                                  # frames are 0.1 ps apart
# the pre-set rule (bootstrap SE > 50 K) flagged the Cu50Zr50 pair; MACE energies decorrelate within
# 5 frames, so 0.5 ps spacing still gives independent samples
TOPUP, TOPUP_EVERY = ("CuZr2_1208", "Cu10Zr7_1208"), 5
TEMPLATE = """&control
  calculation='scf', prefix='{name}', pseudo_dir='./pseudo', outdir='./tmp_{name}',
  tprnfor=.false., tstress=.false.
/
&system
  ibrav=0, nat={nat}, ntyp=2, ecutwfc=50, ecutrho=400,
  occupations='smearing', smearing='cold', degauss=0.02
/
&electrons
  conv_thr=1.0d-8, mixing_mode='local-TF', mixing_beta=0.2, mixing_ndim=16, diago_david_ndim=4,
  electron_maxstep=300
/
ATOMIC_SPECIES
  Cu 63.546 Cu.pbe-dn-kjpaw_psl.1.0.0.UPF
  Zr 91.224 Zr.pbe-spn-kjpaw_psl.1.0.0.UPF
"""


def frames(tag, n, topup):
    """Frame indices of the first batch and, in top-up mode, the new ones between them."""
    first = list(range(0, n, EVERY))
    if not topup or tag not in TOPUP:
        return first, []
    return first, [i for i in range(0, n, TOPUP_EVERY) if i % EVERY]


def main():
    topup = sys.argv[1:] == ["topup"]
    os.makedirs("dft/inputs", exist_ok=True)
    if topup and not os.path.exists("dft/run_list_stage2a.txt"):
        os.replace("dft/run_list.txt", "dft/run_list_stage2a.txt")
    jobs, energies = [], ["name\tE_mace_eV\tnatoms"]
    for tag in AT_LIQUIDUS:
        path = f"results_crystal/frames_{tag}K.npz"
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}: stage 1 has not finished for {tag}")
        f = np.load(path)
        first, extra = frames(tag, len(f["positions"]), topup)
        for index in sorted(first + extra):
            name = f"cryst_{tag}K_f{index}"
            lines = [TEMPLATE.format(name=name, nat=len(f["symbols"])), "CELL_PARAMETERS angstrom"]
            lines += [" ".join(f"{v:.10f}" for v in row) for row in f["cell"]]
            lines.append("ATOMIC_POSITIONS angstrom")
            lines += [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(f["symbols"], f["positions"][index])]
            lines.append("K_POINTS automatic\n2 2 2 0 0 0")
            open(f"dft/inputs/{name}.in", "w", newline="\n").write("\n".join(lines) + "\n")
            if index in extra or not topup:
                jobs.append(name)
            energies.append(f"{name}\t{float(f['epot_eV'][index]):.6f}\t{len(f['symbols'])}")
    open("dft/run_list.txt", "w", newline="\n").write("\n".join(jobs) + "\n")
    open("dft/mace_energies.tsv", "w", newline="\n").write("\n".join(energies) + "\n")
    print(f"{len(jobs)} PBE jobs written to dft/inputs (submit with: bash sol/submit.sh dft)")


if __name__ == "__main__":
    main()
