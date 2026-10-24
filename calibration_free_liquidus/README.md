# Calibration-free liquidus (Appendix B)

Code and data for Appendix B of the paper: the free energies of the Cu-Zr liquid and of the
competing crystals calculated with no calibrated quantity, by thermodynamic integration with the
MACE-MP-0 foundation potential and a free-energy-perturbation correction of both phases to density
functional theory (PBE).

| | Cu50Zr50, vs Cu10Zr7 + CuZr2 | Cu50Zr50, vs B2 CuZr | Cu64Zr36, vs Cu8Zr3 + Cu10Zr7 |
|---|---|---|---|
| Calculated liquidus (second-order cumulant) | 1227 +- 28 K | 1213 +- 43 K | 1169 +- 28 K |
| Measured liquidus | 1208 K | 1208 K | 1230 K |

## Reproduce the numbers (numpy only)

```
python assemble_liquid_dG.py                                   # MACE-MP-0 liquid vs harmonic crystals, writes ti_driver/results_local/liquid_dG.json
python dft_correction_analysis.py dft_correction_k222 k222     # PBE correction of the liquid, writes dft_correction_k222/dft_correction.json
python assemble_crystal_dG.py ti_driver/results_crystal crystal_dft   # Table B1, writes ti_driver/results_crystal/crystal_dG.json
```

The first two steps are the earlier harmonic-crystal treatment; the third uses the switched
crystals and their PBE frames together with the liquid correction from the second, and gives every
entry of Table B1.

## Contents

- `ti_driver/` - the switching driver: Uhlenbeck-Ford reference fluid (`uhlenbeck_ford.py`), the
  liquid switching and sampling (`liquid_ti.py`, `frenkel_ladd.py`), the crystal switching from an
  Einstein crystal (`crystal_ti.py`), the runners (`run_liquid_local.py`, `run_crystal.py`),
  `fill_reference.py`, which evaluates the Einstein reference for runs made where calphy was not
  installed, and the known-answer checks against calphy on an EAM liquid and an EAM B2 crystal
  (`gate_eam_liquid.py`, `gate_eam_crystal.py`).
  - `results_local/` - the 18 liquid runs (two alloys, three 108-atom cells, the measured liquidus
    and 50 K either side): free energy, switching work, pressure, and the sampled frames
    (`frames_*.npz`, positions, velocities and MACE energies every 0.1 ps).
  - `results_crystal/` - the 15 crystal runs (B2 CuZr, CuZr2 and Cu10Zr7 at 1158, 1208 and 1258 K;
    Cu8Zr3 and Cu10Zr7 at 1180, 1230 and 1280 K): free energy, spring constants, switching work,
    volume, and 20 ps of sampled frames each, and the result (`crystal_dG.json`).
  - `gate_local/`, `gate_crystal_local/` - the calphy reference runs and the driver's result on the
    same cells.
- `dft_correction_k222/` - Quantum ESPRESSO inputs and outputs for the 78 liquid frames (2x2x2
  mesh), two 3x3x3 checks, the three crystals at their reference structures, the MACE-MP-0 energy
  of each frame (`mace_energies.tsv`), and the result (`dft_correction.json`).
- `crystal_dft/` - Quantum ESPRESSO inputs and outputs for the 109 crystal frames (2x2x2 mesh;
  9 per crystal 2.5 ps apart, 41 for CuZr2 and Cu10Zr7 at 1208 K), their MACE-MP-0 energies, and
  `make_dft_inputs.py`, which wrote the inputs from the crystal frames.
- `inputs/` - MACE-MP-0 crystal ground-state energies and harmonic vibrational free energies used
  by the harmonic treatment.
- `structures/` - the four crystal unit cells; `glass/` - the six starting liquid cells.
- `assemble_liquid_dG.py`, `dft_correction_analysis.py`, `assemble_crystal_dG.py` - the analysis.
- `dft_queue.sh` - runs a list of pw.x jobs one after another on a workstation.

## Settings

Liquid switching: Uhlenbeck-Ford p = 50, sigma = 1.5 A; 50 ps switches each way after 20 ps
equilibration; Langevin, 1 fs; zero-pressure volume from 20 ps NPT; MACE-MP-0 medium in float32.
Crystal switching: Einstein crystal with one spring constant per element from the mean-square
displacement, center of mass fixed; 128 (B2 CuZr), 96 (CuZr2), 136 (Cu10Zr7) and 176 (Cu8Zr3)
atoms; zero-pressure volume from 20 ps NPT; 50 ps switches each way; 20 ps sampled.
DFT: Quantum ESPRESSO 7.3.1 and 6.7 (identical Gamma-point energies on the same frame), PBE,
PSLibrary 1.0.0 PAW, 50/400 Ry, cold smearing 0.02 Ry.

Not redistributed (obtain separately):
- `Cu.pbe-dn-kjpaw_psl.1.0.0.UPF` (md5 cfffd1f38ee05704b3e17f46f1730a6f) and
  `Zr.pbe-spn-kjpaw_psl.1.0.0.UPF` (md5 02112c41713a0468aff2bb39ef194e36), from
  https://pseudopotentials.quantum-espresso.org/upf_files/
- the Mendelev et al. (2009) Cu-Zr EAM potential used only by the two gate scripts.

Re-running the switching needs ase, mace-torch, matscipy and calphy; re-running the DFT needs
Quantum ESPRESSO. The analysis above needs numpy only.

Code under the MIT License (`LICENSE`), data under CC BY 4.0 (`LICENSE-DATA`).
