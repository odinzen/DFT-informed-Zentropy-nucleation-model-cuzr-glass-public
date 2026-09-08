#!/usr/bin/env python3
"""EAM vibrational density of states and the entropy-consistent Debye temperature.

soft1 = theta_D,amorph / theta_D,cryst is defined here from the full phonon
spectrum rather than the static bulk modulus, so it carries the excess
low-frequency (boson-peak) weight of the glass that an elastic estimate misses.

Method: finite-displacement Gamma-point Hessian of a (super)cell with the
Mendelev 2009 Cu-Zr EAM/FS potential (the potential that generated the glasses).
The glass uses its native 108-atom cell; crystals use a supercell large enough
to converge the spectrum. The entropy-relevant Debye temperature is
    theta_D,eff = (hbar/kB) * omega_ln * exp(1/3),   omega_ln = exp(<ln omega>),
which is the theta_D a Debye model needs to reproduce the true high-temperature
vibrational entropy of a given density of states. In the ratio soft1 the exp(1/3)
cancels, so soft1 = omega_ln(amorph) / omega_ln(cryst).
"""
import numpy as np
from ase.calculators.eam import EAM

hbar = 1.054571817e-34; kB = 1.380649e-23
# eV/(Ang^2 * amu) -> (rad/s)^2
OMEGA2_SI = 1.602176634e-19 / (1e-20 * 1.66053906660e-27)

def get_calc(potential):
    return EAM(potential=potential, form="fs")

def make_supercell(atoms, rep):
    return atoms.repeat(rep) if rep != (1, 1, 1) else atoms.copy()

def hessian_frequencies(atoms, calc, delta=0.015, progress=None):
    """Gamma-point phonon frequencies (THz) of `atoms` by central finite differences.
    Applies the acoustic sum rule so the three translational modes sit at zero.
    `progress`: optional path; the fraction of displacements done is written there."""
    at = atoms.copy(); at.calc = calc
    N = len(at); m = at.get_masses()
    pos0 = at.get_positions()
    Phi = np.zeros((3*N, 3*N))
    for i in range(N):
        for a in range(3):
            fp = _forces_at(at, pos0, i, a, +delta)
            fm = _forces_at(at, pos0, i, a, -delta)
            Phi[3*i+a, :] = -(fp - fm).ravel() / (2*delta)   # d^2E/dr dr = -dF/dr
        if progress and i % 10 == 0:
            open(progress, "w").write(f"hessian {i+1}/{N} atoms displaced\n")
    Phi = 0.5*(Phi + Phi.T)                                   # symmetrize
    # acoustic sum rule: self-block = -sum of couplings to other atoms
    for i in range(N):
        for a in range(3):
            for b in range(3):
                s = sum(Phi[3*i+a, 3*j+b] for j in range(N) if j != i)
                Phi[3*i+a, 3*i+b] = -s
    Phi = 0.5*(Phi + Phi.T)
    minv = np.repeat(1.0/np.sqrt(m), 3)
    D = Phi * np.outer(minv, minv)                           # mass-weighted dynamical matrix
    D = 0.5*(D + D.T)
    lam = np.linalg.eigvalsh(D)                              # eV/(Ang^2 amu)
    omega = np.sign(lam)*np.sqrt(np.abs(lam)*OMEGA2_SI)      # rad/s (negative = imaginary)
    return omega/(2*np.pi)/1e12                              # THz

def _forces_at(at, pos0, i, a, d):
    p = pos0.copy(); p[i, a] += d
    at.set_positions(p)
    return at.get_forces()

def spectrum_metrics(freq_THz, drop_lowest=3):
    """omega_ln, theta_ln, theta_D,eff (K) from THz frequencies, dropping the three
    acoustic zero modes. Reports any imaginary modes beyond those three."""
    f = np.sort(freq_THz)
    imag = int(np.sum(f < -1e-3))                            # genuine imaginary (THz)
    pos = f[drop_lowest:]                                    # drop 3 acoustic ~0
    pos = pos[pos > 1e-4]
    omega = pos*1e12*2*np.pi                                 # rad/s
    ln_omega = np.mean(np.log(omega))
    omega_ln = np.exp(ln_omega)
    theta_ln = hbar*omega_ln/kB
    theta_D = theta_ln*np.exp(1/3)
    return dict(theta_D_eff=theta_D, theta_ln=theta_ln, omega_ln_THz=omega_ln/2/np.pi/1e12,
                n_modes=len(pos), n_imag_extra=max(0, imag), f_max_THz=float(f[-1]))
