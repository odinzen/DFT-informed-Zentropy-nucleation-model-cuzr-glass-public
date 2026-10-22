# Calibration-free liquidus (Appendix B)

Code and data for Appendix B of the paper: the free energy of the Cu-Zr liquid calculated with no
calibrated quantity, by thermodynamic integration with the MACE-MP-0 foundation potential and a
free-energy-perturbation correction to density functional theory (PBE).

| | Cu50Zr50 | Cu64Zr36 |
|---|---|---|
| Calculated liquidus (second-order cumulant) | 1347 +- 10 K | 1214 +- 16 K |
| Measured liquidus | 1208 K | 1230 K |

## Reproduce the numbers (numpy only)

```
python assemble_liquid_dG.py                                   # MACE-MP-0 liquid vs crystal, writes ti_driver/results_local/liquid_dG.json
python dft_correction_analysis.py dft_correction_k222 k222     # PBE correction and the calibration-free liquidus
```

## Contents

- `ti_driver/` - the switching driver: Uhlenbeck-Ford reference fluid (`uhlenbeck_ford.py`), the
  switching and sampling (`liquid_ti.py`, `frenkel_ladd.py`), the campaign runner
  (`run_liquid_local.py`), and the known-answer check against calphy on an EAM liquid
  (`gate_eam_liquid.py`).
  - `results_local/` - the 18 liquid runs (two alloys, three 108-atom cells, the measured liquidus
    and 50 K either side): free energy, switching work, pressure, and the sampled frames
    (`frames_*.npz`, positions, velocities and MACE energies every 0.1 ps).
  - `gate_local/` - the calphy reference runs and the driver's result on the same cell.
- `dft_correction_k222/` - Quantum ESPRESSO inputs and outputs for the 78 liquid frames (2x2x2
  mesh), two 3x3x3 checks, the three crystals (Cu8Zr3, Cu10Zr7, CuZr2), the MACE-MP-0 energy of
  each frame (`mace_energies.tsv`), and the result (`dft_correction.json`).
- `inputs/` - MACE-MP-0 crystal ground-state energies and harmonic vibrational free energies used
  for the crystalline reference.
- `glass/` - the six starting cells.
- `assemble_liquid_dG.py`, `dft_correction_analysis.py` - the analysis.
- `dft_queue.sh` - runs a list of pw.x jobs one after another on a workstation.

## Settings

Switching: Uhlenbeck-Ford p = 50, sigma = 1.5 A; 50 ps switches each way after 20 ps
equilibration; Langevin, 1 fs; zero-pressure volume from 20 ps NPT; MACE-MP-0 medium in float32.
DFT: Quantum ESPRESSO 7.3.1 and 6.7 (identical Gamma-point energies on the same frame), PBE,
PSLibrary 1.0.0 PAW, 50/400 Ry, cold smearing 0.02 Ry.

Not redistributed (obtain separately):
- `Cu.pbe-dn-kjpaw_psl.1.0.0.UPF` (md5 cfffd1f38ee05704b3e17f46f1730a6f) and
  `Zr.pbe-spn-kjpaw_psl.1.0.0.UPF` (md5 02112c41713a0468aff2bb39ef194e36), from
  https://pseudopotentials.quantum-espresso.org/upf_files/
- the Mendelev et al. (2009) Cu-Zr EAM potential used only by `gate_eam_liquid.py`.

Re-running the switching needs ase, mace-torch, matscipy and calphy; re-running the DFT needs
Quantum ESPRESSO. The analysis above needs numpy only.

Code under the MIT License (`LICENSE`), data under CC BY 4.0 (`LICENSE-DATA`).
