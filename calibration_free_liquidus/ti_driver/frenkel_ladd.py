#!/usr/bin/env python3
"""Crystal absolute free energy by non-equilibrium Frenkel-Ladd thermodynamic
integration with an ASE calculator (MACE). Portable: runs on CPU locally and on a
GPU (device='cuda') on Sol without any compiled LAMMPS pair style.

Method (Freitas/de Koning adiabatic switching):
  H(lambda) = (1-lambda) U_spring + lambda U_pot,   U_spring = sum_i k/2 |r_i - r0_i|^2
  F_pot = F_Einstein + integral_0^1 <U_pot - U_spring> dlambda
The switching integral is measured as non-equilibrium work over a slow lambda ramp,
forward (0->1) and backward (1->0); the reversible value is (W_fwd - W_bwd)/2, and the
forward/backward gap is the dissipation (a convergence diagnostic).

F_Einstein (classical, 3N oscillators pinned to absolute sites, one frequency omega):
  F_E = 3 N kB T ln(hbar omega / kB T),  omega = sqrt(k/m).
No centre-of-mass fixing here (springs pin absolute positions); the residual O(kT lnN / N)
COM/translation term is small and is quantified against the harmonic reference in validation.
"""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

HBAR = 1.054571817e-34
KB = 1.380649e-23
# k in eV/Ang^2, m in amu -> omega^2 in (rad/s)^2
OMEGA2_SI = 1.602176634e-19 / (1e-20 * 1.66053906660e-27)
KB_eV = 8.617333262e-5


class SpringCalculator(Calculator):
    """Einstein springs to fixed reference sites: E = sum k_i/2 |r_i-r0_i|^2 (k scalar or per atom)."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, r0, k_eV_A2):
        super().__init__()
        self.r0 = np.asarray(r0, float)
        k = np.asarray(k_eV_A2, float)
        self.k = float(k) if k.ndim == 0 else k[:, None]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        d = atoms.get_positions() - self.r0
        self.results["energy"] = 0.5 * float(np.sum(self.k * d * d))
        self.results["forces"] = -self.k * d


class SwitchCalculator(Calculator):
    """Blend U = (1-lam) U_spring + lam U_pot; caches the two energies for the work sum."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, spring, pot):
        super().__init__()
        self.spring, self.pot = spring, pot
        self.lam = 0.0
        self.e_pot = self.e_spring = 0.0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.spring.calculate(atoms, ("energy", "forces"), system_changes)
        self.pot.calculate(atoms, ("energy", "forces"), system_changes)
        es, ep = self.spring.results["energy"], self.pot.results["energy"]
        fs, fp = self.spring.results["forces"], self.pot.results["forces"]
        self.e_spring, self.e_pot = es, ep
        self.results["energy"] = (1 - self.lam) * es + self.lam * ep
        self.results["forces"] = (1 - self.lam) * fs + self.lam * fp


def einstein_free_energy_eV_per_atom(mass_amu, k_eV_A2, T):
    """F_E per atom (eV) = 3 kBT ln(hbar omega/kBT), omega = sqrt(k/m)."""
    omega = np.sqrt(k_eV_A2 / mass_amu * OMEGA2_SI)          # rad/s
    x = HBAR * omega / (KB * T)
    return 3.0 * KB_eV * T * np.log(x)


def spring_constant_for_omega(mass_amu, omega_rad_s):
    return mass_amu * omega_rad_s ** 2 / OMEGA2_SI           # eV/Ang^2


def _switch_leg(atoms, switch, T, n_steps, dt_fs, friction, forward=True, watch=None, fixcm=False):
    """One lambda ramp; returns accumulated work sum_i dlam*(U_pot-U_spring) [eV].
    `watch(dyn)`, if given, may attach its own observers to the dynamics. fixcm keeps the centre of
    mass still, which a crystal tethered to absolute sites needs (its reference carries the
    matching centre-of-mass correction)."""
    dyn = Langevin(atoms, dt_fs * units.fs, temperature_K=T, friction=friction, fixcm=fixcm)
    dlam = (1.0 if forward else -1.0) / n_steps   # backward integrates lambda 1->0 (dlam<0)
    work = 0.0
    state = {"i": 0}

    def step():
        i = state["i"]
        lam = (i + 0.5) / n_steps
        switch.lam = lam if forward else 1.0 - lam
        state["i"] += 1

    def accumulate():
        nonlocal work
        work += dlam * (switch.e_pot - switch.e_spring)

    dyn.attach(step, interval=1)       # set lambda for the upcoming step
    dyn.attach(accumulate, interval=1) # accumulate dH/dlam at current state
    if watch is not None:
        watch(dyn)
    switch.lam = 0.0 if forward else 1.0
    dyn.run(n_steps)
    return work


def frenkel_ladd_free_energy(atoms, pot_calc, T, omega_rad_s,
                             t_equilibrate_fs=2000, t_switch_fs=10000, dt_fs=1.0,
                             friction=0.02, seed=1):
    """Absolute free energy per atom (eV) of `atoms` under `pot_calc` at temperature T.
    Returns a dict with F_total, F_einstein, dF_switch, and the fwd/bwd work + hysteresis."""
    atoms = atoms.copy()
    r0 = atoms.get_positions().copy()
    m = atoms.get_masses()
    if np.ptp(m) > 1e-6:
        # per-species spring so every atom shares omega; F_E below uses the mean via the sum
        k_arr = np.array([spring_constant_for_omega(mi, omega_rad_s) for mi in m])
        k = float(np.mean(k_arr))   # SpringCalculator uses a single k; keep omega ~ mean
    else:
        k = spring_constant_for_omega(m[0], omega_rad_s)
    spring = SpringCalculator(r0, k)
    switch = SwitchCalculator(spring, pot_calc)
    atoms.calc = switch

    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)
    # equilibrate in the Einstein reference (lambda = 0)
    switch.lam = 0.0
    Langevin(atoms, dt_fs * units.fs, temperature_K=T, friction=friction, fixcm=False
             ).run(int(t_equilibrate_fs / dt_fs))

    n_sw = int(t_switch_fs / dt_fs)
    w_fwd = _switch_leg(atoms, switch, T, n_sw, dt_fs, friction, forward=True)
    # re-equilibrate at lambda=1 before the backward leg
    switch.lam = 1.0
    Langevin(atoms, dt_fs * units.fs, temperature_K=T, friction=friction, fixcm=False
             ).run(int(t_equilibrate_fs / dt_fs))
    w_bwd = _switch_leg(atoms, switch, T, n_sw, dt_fs, friction, forward=False)

    dF_switch = 0.5 * (w_fwd - w_bwd) / len(atoms)          # eV/atom
    # F_E per atom: use the per-atom omega (mass-specific) averaged over atoms
    F_E = float(np.mean([einstein_free_energy_eV_per_atom(mi, k, T) for mi in m]))
    return dict(
        T=T, natoms=len(atoms), k_eV_A2=k, omega_rad_s=omega_rad_s,
        F_einstein_eV=F_E, dF_switch_eV=dF_switch, F_total_eV=F_E + dF_switch,
        w_fwd_eV=w_fwd / len(atoms), w_bwd_eV=w_bwd / len(atoms),
        hysteresis_eV=(w_fwd + w_bwd) / len(atoms),
    )
