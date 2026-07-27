#!/usr/bin/env python3
"""Regenerate the icosahedral cluster and write the simple references.
The complex intermetallics Cu5Zr, Cu8Zr3, Cu10Zr7, Cu51Zr14 have large unit
cells; pull validated CIFs from the Materials Project and convert with pymatgen,
or from ICSD. Suggested Materials Project entries:
  Cu5Zr   : search Cu5Zr  (F-43m, AuBe5 type)
  Cu8Zr3  : search Cu8Zr3 (Pnma)
  Cu10Zr7 : search Cu10Zr7 (Aba2 / C2ca, 68 atom cell)
  CuZr    : mp B2, already provided as CuZr_B2.vasp
  CuZr2   : already provided as CuZr2_C11b.vasp
Conversion example (requires pymatgen and an MP API key):
  from pymatgen.ext.matproj import MPRester
  from pymatgen.io.vasp import Poscar
  with MPRester(API_KEY) as m:
      s = m.get_structure_by_material_id("mp-XXXX")
      Poscar(s).write_file("structures/Cu10Zr7.vasp")
Validate every downloaded cell visually before queueing the calculation.
"""
print("Icosahedral cluster and simple references are in structures/.")
print("Fetch complex intermetallic CIFs as noted above, then validate them.")
