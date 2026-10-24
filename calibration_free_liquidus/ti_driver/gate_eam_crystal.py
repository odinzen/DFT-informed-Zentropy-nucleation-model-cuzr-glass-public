#!/usr/bin/env python3
"""Known-answer gate for the crystal switching driver: Einstein crystal -> Mendelev EAM on 128-atom
B2 CuZr, against calphy (LAMMPS) on the same cell, potential, temperature and volume. Two elements
of unequal mass test the per-element springs and the fixed centre of mass. On 2026-10-22 calphy
gave F = -5.41839 eV/atom at 900 K. Passes within 5 meV/atom, as the liquid gate does.

Usage:  python gate_eam_crystal.py <calphy report.yaml> <structure.data> <eam.fs>
Env:    GATE_OUT, CALPHY_PYTHON.
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
from crystal_ti import crystal_free_energy

TOL_EV = 0.005


def main():
    from matscipy.calculators.eam import EAM as MatscipyEAM
    report_path, structure, eam = sys.argv[1:4]
    report = yaml.safe_load(open(report_path))
    T = float(report["input"]["temperature"])
    F_ref = report["results"]["free_energy"]
    vol = report["average"]["vol_atom"]
    atoms = read(structure, format="lammps-data", style="atomic", Z_of_type={1: 29, 2: 40})
    atoms.set_cell(atoms.cell.array * (vol * len(atoms) / atoms.get_volume()) ** (1 / 3), scale_atoms=True)
    t0 = time.time()
    r = crystal_free_energy(atoms, MatscipyEAM(eam, kind="eam/fs"), T, t_relax_fs=0,
                            log=lambda m: print(f"  {time.strftime('%H:%M:%S')} {m}", flush=True))
    r.pop("samples")
    diff = r["F_crystal_eV"] - F_ref
    r.update(calphy_report=os.path.basename(report_path), calphy_F_eV=F_ref, calphy_vol_A3=vol,
             calphy_springs=report["average"].get("spring_constant"), difference_eV=diff,
             passed=bool(abs(diff) <= TOL_EV), wall_s=time.time() - t0)
    out = os.environ.get("GATE_OUT", HERE)
    json.dump(r, open(os.path.join(out, "gate_eam_crystal.json"), "w"), indent=2)
    print(f"F driver {r['F_crystal_eV']:.5f}  calphy {F_ref:.5f}  diff {diff*1000:+.2f} meV/atom  "
          f"hyst {r['hysteresis_eV']*1000:.2f} meV  springs {r['k_eV_A2']} (calphy {report['average'].get('spring_constant')})  "
          f"F_cm {r['F_cm_eV']*1000:+.2f} meV")
    print("GATE", "PASS" if r["passed"] else "FAIL", f"({r['wall_s']/3600:.1f} h)")


if __name__ == "__main__":
    main()
