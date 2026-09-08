#!/usr/bin/env bash
# Appendix A driver. Run inside the conda environment that exposes your DFT code.
# Edit ENGINE and the volume scale list, then execute step by step.
set -e
ENGINE=${1:-vasp}          # vasp or qe
SCALES="0.94 0.96 0.98 1.00 1.02 1.04 1.06"   # +/- 8 percent EOS scan
echo "1) relax each structure in structures/ (INCAR.relax or pw relax)"
echo "2) for each relaxed cell, scan volumes ${SCALES} and run a static calculation"
echo "3) collect volume,energy per atom into ev_<name>.dat"
echo "4) python fit_eos_debye.py <name> ev_<name>.dat <mean_mass_amu>"
echo "5) python zentropy_handoff.py Cu8Zr5 Cu10Zr7"
echo "Validate every cell visually before queueing. No PowerShell needed; Ubuntu shell only."
