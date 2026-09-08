#!/usr/bin/env python3
"""Fast self-test (~1-2 min). Verifies the environment, the EAM potential, and the full
pipeline (melt-quench -> relax -> phonon DOS) on a tiny run before you commit to the
multi-hour campaign, and estimates the full runtime on this machine. Run:  python smoke_test.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config2_campaign as C
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

def main():
    import ase, scipy
    print(f"[ok] numpy {np.__version__} | scipy {scipy.__version__} | ase {ase.__version__}")
    calc = C.make_calc()
    d = C.COMPS["Cu50Zr50"]
    at = C.build(dict(Cu=d["Cu"], Zr=d["Zr"]), 2000); at.calc = calc
    at.get_forces()
    MaxwellBoltzmannDistribution(at, temperature_K=2200)
    dyn = Langevin(at, 2 * units.fs, temperature_K=2200, friction=0.02)
    t = time.time(); dyn.run(50); dt = (time.time() - t) / 50
    print(f"[ok] one MD force call ~{dt:.3f} s")
    for k in range(0, 100, 50):
        dyn.set_temperature(temperature_K=2200 + (300 - 2200) * k / 100); dyn.run(50)
    C.relax(at, calc)
    freq = C.pv.hessian_frequencies(at, calc); m = C.pv.spectrum_metrics(freq)
    thc = C.theta_cryst(0.50, "Cu10Zr7", "CuZr")
    print(f"[ok] pipeline runs: theta_glass={m['theta_D_eff']:.0f} K  "
          f"soft={m['theta_D_eff']/thc:.3f}  n_imag={m['n_imag_extra']}")
    est = (5 * (1600 + 7000 + 2 * 400 + 2 * 648) * dt * 2) / 3600
    print(f"[ok] estimated full campaign on this machine: ~{est:.1f} h")
    print("\nSMOKE TEST PASSED -- ready to run:  python config2_campaign.py")

if __name__ == "__main__":
    main()
