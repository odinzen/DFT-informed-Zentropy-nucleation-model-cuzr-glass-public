#!/usr/bin/env python3
"""Replace the placeholder icosahedral energy offset with the computed value.
Reads results_*.json from the DFT runs: the crystalline assembly endpoints and
the icosahedral cluster. Computes dE_ico = E0(icosahedron) - E0(crystal reference)
and prints the value to substitute into zentropy_driving_force.py, then the
prediction follows with no further fitting."""
import json, glob, sys
def load(name):
    return json.load(open(f"results_{name}.json"))
def main(cluster="Cu8Zr5", cryst="Cu10Zr7"):
    ic=load(cluster); cr=load(cryst)
    # eV/atom to kJ/mol-atom
    dE = (ic["E0_eV"] - cr["E0_eV"])*96.485
    print(f"Computed icosahedral offset dE_ico = {dE:.2f} kJ/mol-atom")
    print(f"Cluster thetaD = {ic['thetaD_K']:.0f} K ; crystal thetaD = {cr['thetaD_K']:.0f} K")
    print("Substitute dE_ico above into make_dG_callable (replacing the calibrated placeholder),")
    print("set soft1 = thetaD_cluster/thetaD_crystal, then rerun nucleation_zentropy.py.")
if __name__=="__main__":
    main(*(sys.argv[1:3] or []))
