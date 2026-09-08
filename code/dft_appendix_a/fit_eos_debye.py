#!/usr/bin/env python3
"""Fit a third order Birch Murnaghan equation of state to an energy volume scan
and compute the Debye temperature. Reads ev.dat with two columns: volume per atom
(Angstrom^3) and energy per atom (eV). Writes results.json for the Zentropy handoff."""
import json, sys, numpy as np
from scipy.optimize import curve_fit
def bm3(V,E0,V0,B0,Bp):
    eta=(V0/V)**(2/3)
    return E0 + 9*V0*B0/16*( (eta-1)**3*Bp + (eta-1)**2*(6-4*eta) )
def main(name, ev_file, mass_amu):
    V,E=np.loadtxt(ev_file,unpack=True)
    p0=[E.min(),V[np.argmin(E)],1.0,4.0]
    (E0,V0,B0_eV,Bp),_=curve_fit(bm3,V,E,p0=p0,maxfev=20000)
    B0=B0_eV*160.2176        # eV/A^3 -> GPa
    # Debye temperature from bulk modulus, G ~ 0.4 B
    hbar=1.054571e-34; kB=1.380649e-23; NA=6.022e23
    Vm3=V0*1e-30; rho=(mass_amu*1e-3/NA)/Vm3; G=0.4*B0*1e9
    vl=np.sqrt((B0*1e9+4*G/3)/rho); vt=np.sqrt(G/rho)
    vm=((1/3)*(1/vl**3+2/vt**3))**(-1/3)
    thetaD=(hbar/kB)*(6*np.pi**2/Vm3)**(1/3)*vm
    out=dict(name=name,E0_eV=float(E0),V0_A3=float(V0),B0_GPa=float(B0),Bp=float(Bp),thetaD_K=float(thetaD))
    json.dump(out,open(f"results_{name}.json","w"),indent=2)
    print(out); return out
if __name__=="__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
