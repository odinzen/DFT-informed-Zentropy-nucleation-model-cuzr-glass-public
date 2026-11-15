# Calculations of the revised manuscript

These scripts produce the numbers, tables and figures added or changed in the revision of the paper
(November 2026). Each writes its result to a JSON file of the same name, and the JSON files from the run
reported in the paper are included, so every value can be checked without running anything. The scripts
import the model from `../code`.

Run them from this folder with Python 3, `numpy`, `scipy` and `matplotlib`. `pycalphad` is needed for the
assessed crystal phases (Section S12), and `ase` with `mace-torch` for the simulations (Sections S10 to S12).

## Main text

| Script | Output | Reproduces |
|---|---|---|
| `revision_numbers.py` | `revision_numbers.json` | Tables 2, 3 and 5, Supplementary Tables S1 to S3, and the numbers of Sections 3.1 to 3.7 and Appendix B |
| `single_state_test.py` | printed | single state liquid and other configuration counts (Table 3), used by the scripts below |
| `predictive_tests.py` | printed | Table 4, the null models, and the Monte Carlo propagation of input uncertainty (Section 3.4) |
| `measured_check.py` | printed | the calorimetric control with the measurements of Castellero and Battezzati (Table 5) |
| `blind_predictions.py` | `blind_predictions.json` | Table 6 and Supplementary Table S9. The JSON key `rival` is the baseline of the paper |
| `regen_figures_rev.py` | `figures_out/` | Figures 4 to 8, S1 and S2, from `revision_numbers.json` |
| `heat_capacity_from_runs.py` | printed | heat capacities of the liquid and B2 CuZr from the Appendix B runs (Section 3.3) |

`viscosity_measured.py`, `oxygen_consistent.py` and `levitation_rate_test.py` are imported by the scripts above.
`levitation_rate_test.py` also reproduces Section S9.

## Supplementary information

| Script | Output | Reproduces |
|---|---|---|
| `jobc_local.py`, `jobc_decompose.py` | `jobc_local.json` | Section S10 and Table S5, oxide and crystal interface energies with MACE-MP-0 |
| `sol_spectrum_package/` | `data/sol_spectrum/` | Section S11, the MACE-MP-0 liquid and crystal runs as submitted on one GPU |
| `checks.py` | `checks.json` | Section S11, checks on those runs (no crystallization, quench convergence, pressure) |
| `cconf_table.py` | `cconf_table.json` | Table S6, configurational heat capacity of the liquids |
| `sconf.py` | `sconf.json` | Section S11, configurational entropy at the liquidus and where it reaches zero |
| `run_spectrum.py`, `analyse_spectrum.py` | `spectrum_widths.json` | Table S6, the Mendelev EAM values (LAMMPS) |
| `assessed_phases.py`, `real_phase_kinetics.py` | `real_phase_kinetics.json` | Table S7 |
| `reality_run.py` | `reality_final.json` | Table S8, assessed, calorimetric, Mendelev and Gaussian liquids |
| `rescore.py` | `rescore.json` | Table S8, the MACE-MP-0 liquids |
| `si_s11_s12_tables.py` | printed | Tables S6 to S8 as they appear in the supplement, from the JSON files |
| `liquid_structure.py` | `liquid_structure.json` | Section S12, Voronoi polyhedra and site energies of the melt-quenched cells |
| `reality_checks.py` | printed | tests of `../code/hyalos_reality.py` against cases with known answers |

The model of Section S12 is `../code/hyalos_reality.py`. Its Zener growth parameter is solved in that file from
the published equation.

Usage notes:

- `checks.py`, `cconf_table.py`, `sconf.py` and `rescore.py` read `data/sol_spectrum` unless another results
  folder is given as the first argument. `sconf.py` also reads the liquid free energies of
  `../calibration_free_liquidus/ti_driver/results_local`.
- `reality_run.py` takes the Mendelev spectrum and an output name, as
  `python reality_run.py spectrum_widths.json reality_final.json`.
- `analyse_spectrum.py` takes the results folder and an output name, as
  `python analyse_spectrum.py data/mendelev_spectrum spectrum_widths.json`.
- `assessed_phases.py`, `real_phase_kinetics.py`, `reality_run.py`, `rescore.py` and `reality_checks.py` need the
  Cu-Zr assessments compiled by Hallstedt (Calphad 89 (2025) 102833, supplementary data, files `CuZr-*.tdb`).
  They are not redistributed here. Set `HALLSTEDT_TDB_DIR` to the folder that holds them, or place them in `tdb/`.
- `sol_spectrum_package/spectrum_run.py` needs the MACE-MP-0 medium model of 2023-12-03
  (`20231203mace128L1_epoch199model`, MIT License) in `sol_spectrum_package/weights/`. The cluster job scripts
  are not included; each case runs as `python spectrum_run.py case TAG RESULTS`.
- `run_spectrum.py` needs LAMMPS and the Mendelev Cu-Zr EAM/FS potential, which is not redistributed.

## Data

- `data/sol_spectrum/` holds every temperature rung of the 13 MACE-MP-0 cases (inherent structure energies, volumes
  and Gamma-point frequencies) and the GPU check.
- `data/mendelev_spectrum/` holds the Mendelev inherent structure energies of six compositions.
- `data/glass_structures/` and `data/interface_structures/` hold the structures read by `liquid_structure.py` and
  the interface scripts.
