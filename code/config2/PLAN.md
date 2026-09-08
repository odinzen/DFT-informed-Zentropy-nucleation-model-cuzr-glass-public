# Config-2 free-energy campaign — grounding soft2 and δE from first principles

Purpose: replace the two remaining declared parameters in Table 1, soft2 = 0.80 and
δE = 6.0 kJ mol⁻¹, with computed values, so the two-configuration Zentropy free energy
is genuine rather than "Zentropy-style." This is the campaign that answers the central
review objection (the softened-Debye approximation and the ungrounded soft2). soft1 is
already computed (Appendix A); this closes the other two.

## Physical definition of the two configurations (by quench rate)

Appendix A already records that the computed amorphous energy falls with slower cooling
over three decades of rate. That spread is the configurational energy range, so:

- Config 1 (icosahedral ground): SLOW quench → lower energy, more icosahedral order.
  soft1 already computed there, θ_D,glass ≈ 247–257 K → soft1 ≈ 0.75–0.85, adopted 0.80.
- Config 2 (defective): FAST quench → higher energy, looser packing.
- δE = E₀(config 2) − E₀(config 1), the configurational energy split.
- soft_k = θ_D,config_k(VDOS) / θ_D,cryst(engine).

This is a modeling choice (a continuous liquid mapped to two discrete configurations),
defensible and the same reduction the paper already uses; the campaign only replaces the
two declared numbers with computed ones, it does not change the two-config assumption.

## Method (identical to the config-1 / soft1 pipeline)

- Potential: Mendelev 2009 Cu–Zr EAM/FS, `Cu-Zr_2.eam.fs` (the potential that made the
  config-1 glasses). Cu–Zr only, so this EAM campaign covers the two binaries; the Al
  ternary is out of scope for the EAM and would need MACE separately.
- Melt-quench: ase + EAM Langevin MD. Melt at 2200 K, quench to 300 K. Config 1 slow
  (≈150 K/ps), config 2 fast (≈1500 K/ps, ~10× faster). Same melt, two quench rates.
- Relax: LBFGS with cell relaxation, as `run_vdos.py` does.
- E₀: EAM potential energy per atom of the relaxed cell.
- VDOS: finite-displacement Γ-point Hessian (`phonon_vdos.py`), entropy-weighted
  ω_ln → θ_D,eff, exactly as soft1 was computed.
- θ_D,cryst: the engine's crystalline Debye temperature (from the DFT bulk modulus). The
  EAM makes B2 CuZr dynamically unstable, so the crystal reference is taken from the engine,
  as in the config-1 work — same denominator for soft1 and soft2, so the ratio is consistent.

## Cells

- Compositions: Cu₅₀Zr₅₀ and Cu₆₄Zr₃₆ (the two with existing config-1 data).
- 108 atoms, matching config 1.
- n = 5 independent replicas per config per composition → 2 × 2 × 5 = 20 supercells.
- Regenerating config 1 fresh at the slow rate also re-checks soft1 (should land 0.75–0.85).

## What gets recomputed in Table 1

| Table 1 row | now | after |
|---|---|---|
| soft1, soft2 | 0.80, 0.80 (soft1 phonon DOS; soft2 declared) | soft1 0.80, soft2 = computed; both "EAM phonon DOS" |
| δE (config split) | 6.0 kJ mol⁻¹, declared modeling choice | computed E₀ split (config 2 − config 1) |

## Critical validation step (do not skip)

After computing soft2 and δE, re-run the engine with the computed values and check the
predictions. Two honest outcomes, both publishable:

- The computed soft2/δE reproduce the validated diameters (17.2 / 1.34 …) → genuine
  first-principles two-configuration Zentropy that works. The strong result.
- They do not (soft2 is one of the most sensitive inputs; the sensitivity table and the
  soft2 sweep show dc swings from ~6 to ~85 mm across the plausible range, so a computed
  soft2 far from 0.80 can break the predictions) → an honest limit of the two-config
  reduction with computed parameters. Also a real result, and it bounds the model.

Do not tune soft2 back toward 0.80 to save the predictions; report whichever the
computation gives.

## What this does NOT fix

The absolute liquidus (the offset δE_ico calibration) is a separate, harder problem. The
DFT amorphous energy is only good to ~±24 %, and substituting it shifts the melting point
by ~200 K (Appendix A). Grounding soft2 and δE does not close that; it needs DFT-accurate
absolute energetics, a separate Sol campaign, and remains the stated open problem.

## Timing and compute

Mendelev EAM is classical, ~ms per force call, so each melt-quench is tens of seconds and
each VDOS (≈648 force evaluations on 108 atoms) under a second. Twenty supercells run in
roughly minutes to an hour, single-core, locally. No HPC. Can run alongside the ensemble
(EAM is light) or right after it.

## Launch

`config2_campaign.py`, resume-safe (checkpoints per cell to `config2_results.json`).
Launch after the ensemble melt-quench finishes, or alongside it, with one command.
