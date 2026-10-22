# Zentropy nucleation model for Cu-Zr / Zr-Cu-Al glass-forming ability

Figures and runnable code accompanying a Zentropy-based nucleation model that predicts
glass-forming ability in Cu-Zr and Zr-Cu-Al alloys. The model couples a Debye-Grueneisen
crystalline free energy (from published DFT bulk moduli) to a two-configuration Zentropy
free energy for the supercooled liquid, and feeds the resulting driving force into a
classical nucleation model to return a TTT diagram, a critical cooling rate, and a critical
casting diameter.

Odinzen LLC, in collaboration with Arizona State University. The code (everything under
`code/`, and the `.py` and `.sh` files under `calibration_free_liquidus/`) is under the MIT
License, see `LICENSE`; the figures (everything under `figures/`) and the data under
`calibration_free_liquidus/` are under CC BY 4.0, see `LICENSE-DATA`. Cite the paper if you use
any of them.

## Contents

- `code/` - the runnable package
  - `zentropy_driving_force.py` - Debye-Grueneisen crystalline free energy, two-configuration
    Zentropy supercooled liquid, driving force calibrated to the measured liquidus
  - `nucleation_zentropy.py` - classical nucleation model; TTT nose, critical cooling rate,
    critical casting diameter, and the figures
  - `sensitivity.py` - sensitivity of the prediction to each input
  - `dft_appendix_a/` - structures, VASP and Quantum ESPRESSO inputs, a Birch-Murnaghan
    equation-of-state and Debye fitter, and the handoff script for the one first-principles
    input the model otherwise takes from the literature
  - `config2/` - the config-2 free-energy campaign that computes the second configuration's
    softening (soft2) and energy split from a classical-potential melt-quench and phonon
    density of states; the licensed EAM potential is not redistributed (see config2/README.md)
  - `MANIFEST.txt` - description of each file
- `calibration_free_liquidus/` - Appendix B: the free energy of the liquid calculated with no
  calibrated quantity, by thermodynamic integration with the MACE-MP-0 potential and a
  correction to density functional theory; code, the 18 liquid runs, the Quantum ESPRESSO
  inputs and outputs, and the analysis (see its README)
- `figures/` - the figures and graphical abstract

## Requirements

Python 3 with `numpy`, `scipy`, and `matplotlib`. No production DFT or molecular dynamics is
needed at the point of prediction; `code/dft_appendix_a` documents how to compute the
remaining first-principles input on your own machine.

## Reproducing the figures

```
cd code
python nucleation_zentropy.py
python sensitivity.py
```
