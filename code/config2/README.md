# Config-2 free-energy campaign: computing soft2 and dE

Everything needed to compute the second configuration's free energy (soft2 and the
configurational split dE) from first principles, on a clean machine, except the licensed
potential (see below). This is the campaign
that turns the two-configuration Zentropy free energy from "-style" into genuine, by
replacing the two declared parameters in the paper's Table 1 with computed ones. The
science and design are in **PLAN.md**; this file is how to run it.

## No MLIP needed

This campaign uses a **classical EAM potential** (Mendelev 2009 Cu-Zr), run through
matscipy's fast C-accelerated calculator. It needs **no MACE, no PyTorch, no GPU**. If you
were expecting to install an MLIP, you do not need one for this. (An MLIP would only be
needed to extend the campaign to the aluminum ternary, which the Cu-Zr EAM cannot do.)

## Contents

| file | what it is |
|---|---|
| `config2_campaign.py` | the campaign: melt-quench at two rates, relax, phonon DOS, energy |
| `smoke_test.py` | ~1-2 min self-test of the environment and pipeline; run this first |
| `validate_engine.py` | re-runs the engine with the computed soft2/dE vs the declared values |
| `phonon_vdos.py` | the finite-displacement phonon-DOS code (same as config 1) |
| `zentropy_driving_force.py` | the engine, used for the crystalline Debye temperature and validation |
| `environment.yml` / `requirements.txt` | the conda environment / pip fallback |
| `PLAN.md` | the scientific design (quench rates, cells, Table 1 changes, honest limits) |
| `RESULTS.md` / `config2_results.json` | the computed soft1/soft2/dE and the raw per-cell data |

**The EAM potential is not included.** `Cu-Zr_2.eam.fs` (Mendelev et al. 2009) is licensed and
is not redistributed here; obtain it separately (for example from the NIST Interatomic
Potentials Repository or the original authors) and place it in this folder before running the
campaign.

Only Python and four common packages (numpy, scipy, ase, matscipy) are needed from the
network; with the potential in place, everything else is in this folder. If matscipy will not
install, the campaign still runs on ase's built-in EAM, just ~9x slower.

## Setup (once, on the clean machine)

Install Miniconda/Miniforge if needed, then:

```
conda env create -f environment.yml
conda activate config2
```

That gives Python 3.11 with numpy, scipy, ase, and matscipy. Nothing else. If conda is not
available, any Python 3.10+ works with:  `pip install -r requirements.txt`.

## Verify first (~1-2 min)

Before the multi-hour run, confirm the environment, the potential, and the pipeline all work,
and get a runtime estimate for this machine:

```
python smoke_test.py
```

It should end with `SMOKE TEST PASSED`. If it reports matscipy is unavailable it will still
pass on ase's EAM, but the estimate will be ~1 day instead of ~2-3 h -- install matscipy first
if you can.

## Run the campaign

```
conda activate config2
python config2_campaign.py
```

It runs 20 supercells (2 binaries x 2 configurations x 5 replicas), printing each as it
finishes and checkpointing to `config2_results.json`. **It is resume-safe**: if the machine
sleeps or you stop it, just run it again and it skips what is done. Expect roughly **2 to 3
hours** total (classical EAM, single process; the slow-quench configs dominate). It ends with
a summary of soft1, soft2, and dE per composition against the declared soft2 = 0.80, dE = 6.0.

## Validate against the predictions

```
python validate_engine.py
```

This recalibrates the engine with the computed soft1, soft2, and dE and prints the casting
diameter next to the declared-parameter value and the measured value, for the two binaries.
This is the decisive step:

- If the computed values keep the diameters near experiment, you have genuine
  first-principles two-configuration Zentropy that works.
- If they swing far (soft2 is one of the most sensitive inputs), that is the honest limit of
  the two-configuration reduction. **Report whichever you get; do not retune soft2 toward
  0.80 to save the prediction.**

## What to bring back

`config2_results.json` (the raw per-cell results) and the two summary printouts. Those give
the computed soft1, soft2, and dE that replace the declared rows in Table 1, plus the
validation outcome. Send those back and the paper's Table 1 and the associated text update
from there.

## Notes

- The EAM potential is licensed (Mendelev et al. 2009); it is here so you can run, but do
  not post it publicly.
- The absolute liquidus (the icosahedral offset) is a **separate, harder** problem that this
  campaign does not solve; it needs DFT-accurate absolute energies (a Sol job) and stays the
  paper's stated open problem. See PLAN.md.
- To extend to the Cu-Zr-Al ternary you would swap the EAM for MACE (an MLIP), which then
  does need PyTorch; that is a later step, not part of this hand-off.
