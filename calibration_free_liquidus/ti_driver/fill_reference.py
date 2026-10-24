#!/usr/bin/env python3
"""Evaluate the Einstein reference for crystal runs that deferred it (no calphy where they ran) and
complete F_crystal. Needs calphy importable here, or CALPHY_PYTHON naming an interpreter that has it.

Usage:  python fill_reference.py <results_crystal dir>
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crystal_ti import einstein_from_inputs


def main():
    for path in sorted(glob.glob(os.path.join(sys.argv[1], "Fcrystal_*.json"))):
        r = json.load(open(path))
        if r.get("F_crystal_eV") is not None:
            continue
        r["F_reference_eV"], r["F_einstein_eV"], r["F_cm_eV"] = einstein_from_inputs(r["reference_inputs"])
        r["F_crystal_eV"] = r["F_reference_eV"] + r["dF_switch_eV"]
        json.dump(r, open(path, "w"), indent=2)
        print(f"{os.path.basename(path)}: F_crystal {r['F_crystal_eV']:.5f} eV/atom")


if __name__ == "__main__":
    main()
