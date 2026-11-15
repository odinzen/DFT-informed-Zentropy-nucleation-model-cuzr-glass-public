# Hyalos: Zentropy nucleation kinetics for Cu-Zr and Zr-Cu-Al glass-forming ability

Code, data and figures accompanying the paper "Glass-forming ability in Cu-Zr and Zr-Cu-Al with Hyalos:
liquidus-calibrated Zentropy nucleation kinetics". Hyalos couples a Debye-Grueneisen crystalline free energy
(from published DFT bulk moduli) to a two-configuration Zentropy free energy for the supercooled liquid, and
feeds the resulting driving force into a classical nucleation model on the measured viscosity to return a TTT
diagram, a critical cooling rate, and a critical casting diameter.

The code (the `.py` and `.sh` files under `code/`, `calibration_free_liquidus/` and `revision/`) is under the
MIT License, see `LICENSE`. The figures (`figures/`) and the data (everything else under
`calibration_free_liquidus/` and `revision/`) are under CC BY 4.0, see `LICENSE-DATA`. Cite the paper if you
use any of them.

## Contents

- `code/` - the runnable package
  - `zentropy_driving_force.py` - Debye-Grueneisen crystalline free energy, two-configuration
    Zentropy supercooled liquid, driving force calibrated to the measured liquidus
  - `hyalos_kinetics.py` - the viscosity through two measured points per alloy, the Turnbull
    coefficient of 0.597, and the classical nucleation model from which the published diameters come
  - `nucleation_zentropy.py` - TTT nose, critical cooling rate, critical casting diameter, and figures
  - `sensitivity.py` - sensitivity of the prediction to each input
  - `hyalos_reality.py` - assessed crystal phases as separate nuclei and a liquid with a spectrum of
    configurations (Supplementary Section S12)
  - `dft_appendix_a/` - structures, VASP and Quantum ESPRESSO inputs, a Birch-Murnaghan
    equation-of-state and Debye fitter, and the handoff script for the one first-principles
    input the model otherwise takes from the literature
  - `config2/` - the config-2 free-energy campaign that computes the second configuration's
    softening (soft2) and energy split from a classical-potential melt-quench and phonon
    density of states; the licensed EAM potential is not redistributed (see config2/README.md)
  - `MANIFEST.txt` - description of each file
- `calibration_free_liquidus/` - Appendix B: the free energies of the liquid and the crystals
  calculated with no calibrated quantity, by thermodynamic integration with the MACE-MP-0
  potential and a correction to density functional theory; code, the 18 liquid and 15 crystal
  runs, the Quantum ESPRESSO inputs and outputs, and the analysis (see its README)
- `revision/` - the calculations of the revised manuscript, with their outputs and data: the numbers
  and figures of the main text, the retrospective check and the blind predictions, and Supplementary
  Sections S9 to S13 (see its README)
- `figures/` - the figures of the paper, numbered as in the paper, and the graphical abstract

## Requirements

Python 3 with `numpy`, `scipy`, and `matplotlib`. No production DFT or molecular dynamics is
needed at the point of prediction; `code/dft_appendix_a` documents how to compute the
remaining first-principles input on your own machine. `revision/README.md` lists the further
packages some of the revision calculations use.

## Reproducing the results

```
cd code
python nucleation_zentropy.py
python sensitivity.py
cd ../revision
python revision_numbers.py
python blind_predictions.py
python regen_figures_rev.py
```

`revision_numbers.py` and `blind_predictions.py` rewrite the JSON files included in `revision/`, and
`regen_figures_rev.py` writes Figures 4 to 8, S1 and S2 to `revision/figures_out/`.

## Releases

- v1.4.0 - the revised manuscript: `hyalos_kinetics.py`, `hyalos_reality.py`, `revision/`, and the figures
  renumbered as in the paper. The blind predictions of Supplementary Section S13 are fixed by this release.
- v1.3.0 and earlier - the submitted manuscript.
