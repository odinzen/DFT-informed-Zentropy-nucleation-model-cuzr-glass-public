#!/usr/bin/env python3
"""Known-answer gate for the liquid switching driver: UF -> Mendelev EAM on a Cu-Zr liquid,
against calphy (LAMMPS) on the same structure, potential, temperature and UF reference.

calphy runs first (minutes in LAMMPS); this script then switches the same cell, rescaled to
calphy's mean NPT volume, through the ASE driver (matscipy EAM, CPU) and compares F. It passes
within 5 meV/atom, about the 0.5 kJ/mol the liquidus test can tolerate. On 2026-10-12 the
108-atom Cu50Zr50 cell at 2300 K gave calphy F = -6.65023 eV/atom (864 atoms: -6.65131).

Usage:  python gate_eam_liquid.py <calphy report.yaml> <structure.data> <eam.fs>
Env:    GATE_T_EQ_FS (25000), GATE_T_SWITCH_FS (50000), GATE_T_SAMPLE_FS (20000), GATE_OUT,
        CALPHY_PYTHON.
"""
import json
import os
import sys
import time

import numpy as np
import yaml
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from liquid_ti import liquid_free_energy

TOL_EV = 0.005
KB_eV = 8.617333262e-5


def main():
    from matscipy.calculators.eam import EAM as MatscipyEAM
    report_path, structure = sys.argv[1], sys.argv[2]
    eam = sys.argv[3]          # the Mendelev et al. 2009 Cu-Zr eam.fs, obtained separately
    report = yaml.safe_load(open(report_path))
    T = float(report["input"]["temperature"])
    F_ref = report["results"]["free_energy"]
    vol = report["average"]["vol_atom"]

    atoms = read(structure, format="lammps-data", style="atomic", Z_of_type={1: 29, 2: 40})
    atoms.set_cell(atoms.cell.array * (vol * len(atoms) / atoms.get_volume()) ** (1 / 3), scale_atoms=True)
    t0 = time.time()
    r = liquid_free_energy(atoms, MatscipyEAM(eam, kind="eam/fs"), T, p=50, sigma=1.5,
                           t_equilibrate_fs=int(os.environ.get("GATE_T_EQ_FS", 25000)),
                           t_switch_fs=int(os.environ.get("GATE_T_SWITCH_FS", 50000)),
                           t_sample_fs=int(os.environ.get("GATE_T_SAMPLE_FS", 20000)))
    s = r.pop("samples")
    out = os.environ.get("GATE_OUT", HERE)
    np.savez_compressed(os.path.join(out, "gate_eam_frames.npz"), **s)
    U = float(s["epot_eV"].mean()) / len(atoms)
    diff = r["F_liquid_eV"] - F_ref
    r.update(U_eV=U, S_J_K_mol=(U + 1.5 * KB_eV * T - r["F_liquid_eV"]) / T * 96485.33212,
             calphy_report=os.path.abspath(report_path), calphy_F_eV=F_ref, calphy_vol_A3=vol,
             difference_eV=diff, passed=bool(abs(diff) <= TOL_EV),
             pressure_mean_GPa=float(s["pressure_GPa"].mean()), wall_s=time.time() - t0)
    json.dump(r, open(os.path.join(out, "gate_eam_liquid.json"), "w"), indent=2)
    print(f"F driver {r['F_liquid_eV']:.5f}  calphy {F_ref:.5f}  diff {diff*1000:+.2f} meV/atom  "
          f"hyst {r['hysteresis_eV']*1000:.2f} meV  P {r['pressure_mean_GPa']:+.2f} GPa")
    print("GATE", "PASS" if r["passed"] else "FAIL", f"({r['wall_s']/3600:.1f} h)")


if __name__ == "__main__":
    main()
