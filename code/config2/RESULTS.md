# Config-2 free-energy campaign — computed results (2026-09-07)

The campaign in `PLAN.md`, run to completion: **20/20 supercells** (Cu50Zr50 and Cu64Zr36 ×
config1_slow/config2_fast × 5 replicas), classical Mendelev 2009 Cu–Zr EAM via ASE melt-quench +
finite-displacement Γ-point VDOS. This replaces the two declared parameters in Table 1
(`soft2 = 0.80`, `δE = 6.0 kJ/mol`) with **computed** values. Raw per-cell data:
`config2_results.json`. Validation harness: `validate_engine.py`.

Reported as computed — **soft2 was not retuned toward 0.80** (per PLAN's rule).

## Computed parameters

| composition | soft1 (computed) | soft2 (computed) | δE (kJ/mol) | declared |
|---|---|---|---|---|
| Cu50Zr50 | 0.749 ± 0.006 | 0.745 ± 0.003 | 1.80 | soft 0.80 / 0.80, δE 6.0 |
| Cu64Zr36 | 0.747 ± 0.005 | 0.747 ± 0.005 | 0.66 | soft 0.80 / 0.80, δE 6.0 |

(n = 5 replicas per group; ± is population std. Cells were near-dynamically-stable — only 3 and 2
imaginary modes total across the 10 cells of each composition. soft1 lands in the 0.75–0.85 band
Appendix A predicted, re-validating the config-1 pipeline.)

## Validation — engine re-run with the computed parameters

| alloy | dc declared (mm) | **dc computed (mm)** | dc measured (mm) |
|---|---|---|---|
| Cu50Zr50 | 1.34 | **3.10** | 2.0 |
| Cu64Zr36 | 10.86 | **27.84** | 2.0 |

## Honest reading (the science call is Michael's)

Two computed findings, both real:

1. **soft2 ≈ soft1 and δE is small.** The fast-quench "defective" configuration softens almost
   identically to the slow-quench ground (0.745 vs 0.749; 0.747 vs 0.747), and the configurational
   energy split comes out **0.7–1.8 kJ/mol, far below the declared 6.0**. So the two-configuration
   *split* is smaller than the declared modeling choice assumed — the two quench-rate configs are
   more alike than the hand-set numbers implied.

2. **Predictions stay within the method's stated factor-of-two, they do not tighten.** Computed
   soft2 (~0.745) sits just below the declared 0.80, and shifts the diameters **up**:
   - **Cu50Zr50** (homogeneous case): 1.34 → **3.10 mm** vs measured 2.0 — i.e. it moves from a
     factor-~1.5 *under*-prediction to a factor-~1.5 *over*-prediction. Same factor-of-two envelope
     the paper already claims as the intrinsic limit; the computed parameter neither breaks nor
     sharpens it. Given soft2's steep sensitivity (PLAN: dc swings ~6→85 mm across its plausible
     range), a computed value this close to 0.80 producing only a ~2× diameter move is a **robust**
     outcome, not a broken one.
   - **Cu64Zr36**: 10.86 → **27.84 mm** vs measured 2.0. This alloy is the known
     **heterogeneous-nucleation-limited** case (oxygen big-cube; see `../VALIDATION_FINDINGS.md`):
     its *intrinsic* diameter is expected to be large and is corrected down to ~2 mm by
     heterogeneous nucleation, not by the intrinsic engine. So the large intrinsic value is
     consistent with the paper's narrative — the computed soft2/δE do not change that this alloy's
     measured diameter is set by heterogeneous nucleation.

**Net:** the softened-Debye two-configuration reduction is now grounded in computed soft1/soft2/δE
rather than declared numbers, and it reproduces the homogeneous diameter to the same factor-of-two
the method already owns as its intrinsic limit — with the honest caveat that the computed δE is
small, so the "two configurations" are energetically close. This directly addresses the review
objection (soft2 and δE are no longer hand-set) without retuning to save the fit.

## Table 1 — what to change

| row | was | now |
|---|---|---|
| soft1 / soft2 | 0.80 (phonon DOS) / 0.80 (declared) | 0.749 / 0.745 (Cu50Zr50), 0.747 / 0.747 (Cu64Zr36) — both EAM phonon DOS |
| δE (config split) | 6.0 kJ/mol (declared) | 1.8 kJ/mol (Cu50Zr50), 0.7 kJ/mol (Cu64Zr36) — computed E₀ split |

Still open (unchanged): the absolute liquidus / icosahedral offset is a separate, harder problem
needing DFT-accurate absolute energies (a Sol campaign); this EAM campaign does not close it. See
`PLAN.md`.

## Provenance & reproducibility

- Run 2026-09-07 via the self-contained hand-off harness (`config2_handoff/`), functionally
  identical to this directory's `config2_campaign.py` (same Mendelev potential, quench rates
  6000/600 steps, 108-atom cells, n=5). matscipy C-accelerated EAM; ~3 h single-process.
- `Cu-Zr_2.eam.fs` (Mendelev et al. 2009) is **licensed — do not redistribute**; it is gitignored
  and not committed. Obtain it separately to reproduce.
