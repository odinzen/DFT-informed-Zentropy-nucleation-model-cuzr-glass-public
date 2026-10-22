#!/usr/bin/env python3
"""Uhlenbeck-Ford (UF) fluid reference for liquid free-energy TI.

The UF model is a soft, purely-repulsive pair fluid whose absolute free energy is known
analytically, so it is the standard reference for a liquid (you cannot pin a liquid to
lattice sites, so Frenkel-Ladd's Einstein reference does not apply). We switch the liquid
from UF -> MACE by adiabatic switching to get F_liquid = F_UF + dF_switch.

Potential (LAMMPS `pair_style ufm` convention, calphy): phi(r) = -eps ln[1 - exp(-(r/sigma)^2)],
eps = p kB T, cutoff = 5 sigma. Analytic reference free energy is taken from calphy's validated
implementation (pure numpy; F_UF = ideal gas + UF excess), not re-derived here.
"""
import json
import os
import subprocess

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
try:
    # same "ijdD" contract as ASE's, compiled; the 10 A UF cutoff makes ASE's the bottleneck
    from matscipy.neighbours import neighbour_list as neighbor_list
except ImportError:
    from ase.neighborlist import neighbor_list

KB_eV = 8.617333262e-5
# calphy pulls in pyscal3, which does not install cleanly next to the CUDA MACE stack, so the
# GPU venv can point at a separate interpreter that has calphy.
CALPHY_PYTHON = os.environ.get("CALPHY_PYTHON")


class UFCalculator(Calculator):
    """Uhlenbeck-Ford pair potential. eps = p*kB*T (temperature-dependent by construction)."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, p, sigma, T):
        super().__init__()
        self.p, self.sigma, self.T = float(p), float(sigma), float(T)
        self.eps = self.p * KB_eV * self.T
        self.cutoff = 5.0 * self.sigma

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        i, j, d, D = neighbor_list("ijdD", atoms, self.cutoff)
        s2 = self.sigma ** 2
        u = (d * d) / s2
        e_m = np.exp(-u)
        omE = 1.0 - e_m
        e_pair = -self.eps * np.log(omE)                 # per directed pair
        energy = 0.5 * float(np.sum(e_pair))
        dphi_dd = -self.eps * (e_m / omE) * (2.0 * d / s2)   # phi'(d) < 0 (repulsive)
        fvec = (dphi_dd / d)[:, None] * D                # force on i from (i,j): phi'(d) D/d
        forces = np.zeros((len(atoms), 3))
        np.add.at(forces, i, fvec)
        self.results["energy"] = energy
        self.results["forces"] = forces


def reference_free_energy_eV_per_atom(atoms, T, p, sigma):
    """Absolute UF-fluid free energy per atom (eV) = ideal-gas + UF-excess, from calphy.
    Requires calphy importable, or CALPHY_PYTHON naming an interpreter that has it."""
    rho = len(atoms) / atoms.get_volume()               # atoms / Ang^3
    syms = atoms.get_chemical_symbols()
    species = sorted(set(syms))
    conc = [syms.count(s) / len(syms) for s in species]
    mass = [next(a.mass for a in atoms if a.symbol == s) for s in species]   # aligned with conc
    f_ig, f_uf = _calphy_reference(T, rho, len(atoms), mass, conc, p, sigma)
    return f_ig + f_uf, dict(f_ideal_gas=f_ig, f_uf_excess=f_uf, rho=rho,
                             species=species, concentration=conc)


_REFERENCE_SCRIPT = """
import json, sys
from calphy.integrators import get_ideal_gas_fe, get_uhlenbeck_ford_fe
a = json.loads(sys.argv[1])
print(json.dumps([float(get_ideal_gas_fe(a["T"], a["rho"], a["n"], a["mass"], a["conc"])),
                  float(get_uhlenbeck_ford_fe(a["T"], a["rho"], a["p"], a["sigma"]))]))
"""


def _calphy_reference(T, rho, n, mass, conc, p, sigma):
    """(ideal gas, UF excess) free energy per atom in eV, from calphy."""
    try:
        from calphy.integrators import get_ideal_gas_fe, get_uhlenbeck_ford_fe
    except ImportError:
        if not CALPHY_PYTHON:
            raise
        args = json.dumps(dict(T=float(T), rho=float(rho), n=int(n), mass=[float(m) for m in mass],
                               conc=[float(c) for c in conc], p=float(p), sigma=float(sigma)))
        out = subprocess.run([CALPHY_PYTHON, "-c", _REFERENCE_SCRIPT, args],
                             capture_output=True, text=True, check=True).stdout
        f_ig, f_uf = json.loads(out.strip().splitlines()[-1])
        return f_ig, f_uf
    return get_ideal_gas_fe(T, rho, n, mass, conc), get_uhlenbeck_ford_fe(T, rho, p, sigma)
