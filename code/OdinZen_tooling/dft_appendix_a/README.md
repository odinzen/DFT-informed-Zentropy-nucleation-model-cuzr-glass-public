# Appendix A | Static DFT package for the Cu Zr Zentropy nucleation prediction

This package computes the one input the prediction still draws from literature:
the icosahedral cluster energy that sets the Zentropy driving force. Running it
replaces the calibrated placeholder with a first principles number.

## Environment
Use your Ubuntu shell under conda. No graphical step and no PowerShell are needed.
```
conda create -n cuzr python=3.11 numpy scipy
conda activate cuzr
```
A DFT engine must be on your PATH. The inputs are provided for both VASP (vasp/)
and Quantum ESPRESSO (qe/).

## Structures
- structures/icosahedron_CuZr.vasp : the Cu centred Cu8Zr5 cluster in a 16 Angstrom box. This is the centrepiece calculation.
- structures/CuZr_B2.vasp, structures/CuZr2_C11b.vasp : exact simple references.
- Complex intermetallics (Cu8Zr3, Cu10Zr7, Cu5Zr): fetch validated CIFs as noted in build_structures.py, then validate visually.

## Order of work
1. Relax each cell (vasp/INCAR.relax or a QE relax run).
2. Scan volumes across +/- 8 percent and run a static calculation at each (vasp/INCAR.static, ISMEAR -5; or QE scf).
3. Write each scan to ev_<name>.dat as two columns: volume per atom, energy per atom.
4. Fit and get the Debye temperature: `python fit_eos_debye.py <name> ev_<name>.dat <mean_mass_amu>`.
5. Hand off: `python zentropy_handoff.py Cu8Zr5 Cu10Zr7`.
6. Substitute the printed dE_ico and theta ratio into zentropy_driving_force.py, then rerun nucleation_zentropy.py.

## What closes
The crystalline free energies already use published DFT bulk moduli; this run
makes them yours and, more importantly, computes the icosahedral offset directly,
removing the last literature anchored quantity from the prediction.
