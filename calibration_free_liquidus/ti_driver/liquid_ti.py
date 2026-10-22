#!/usr/bin/env python3
"""Liquid absolute free energy by UF -> MACE adiabatic switching (reuses the validated
switch machinery from frenkel_ladd). F_liquid(T) = F_UF(T) + dF_switch, F_UF from calphy's
analytic reference. Run at a few T around the liquidus to get F_liquid(T) and its slope.

Intended for Sol A100 (device='cuda'); on CPU it is too slow for production but the code path
is identical. Keep the liquid at fixed volume (the melt-quench cell) during switching; pre-relax
the volume at NPT(P~0,T) first for a true zero-pressure liquid (see relax_volume()).
"""
import time

import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from frenkel_ladd import SwitchCalculator, _switch_leg
from uhlenbeck_ford import UFCalculator, neighbor_list, reference_free_energy_eV_per_atom


class _Watch:
    """Stops a switching run that has gone unphysical, and logs its progress.

    MACE-MP-0 has a deep hole at short range (a Cu-Zr pair at 0.5 A is -570 eV, Zr-Zr at 0.3 A
    -8e6 eV) behind a wall of tens to hundreds of eV. A pair that gets through collapses, the run
    stops making progress, and its work is meaningless, so a pair closer than d_stop or a
    non-finite energy ends the run with an error rather than a silent spin."""

    def __init__(self, atoms, switch, log, every=1000, log_every=10000, d_stop=1.0):
        self.atoms, self.switch, self.log = atoms, switch, log
        self.every, self.log_every, self.d_stop = every, log_every, d_stop

    def __call__(self, dyn, stage):
        last = {"t": time.time(), "n": dyn.nsteps}

        def check():
            d = neighbor_list("d", self.atoms, 2.5)
            dmin = float(d.min()) if len(d) else 2.5
            e = (self.switch.e_pot, self.switch.e_spring)
            if dmin < self.d_stop or not np.all(np.isfinite(e)):
                raise RuntimeError(f"{stage}: closest pair {dmin:.2f} A, energies {e} at step "
                                   f"{dyn.nsteps}, lambda {self.switch.lam:.3f}")
            if dyn.nsteps and dyn.nsteps % self.log_every == 0:
                ms = 1000 * (time.time() - last["t"]) / max(1, dyn.nsteps - last["n"])
                last.update(t=time.time(), n=dyn.nsteps)
                self.log(f"  {stage} step {dyn.nsteps} lambda {self.switch.lam:.3f} closest {dmin:.2f} A "
                         f"E_pot {self.switch.e_pot / len(self.atoms):.3f} eV/atom {ms:.0f} ms/step")

        dyn.attach(check, interval=self.every)


def relax_volume(atoms, mace_calc, T, t_fs=10000, dt_fs=2.0, P_bar=0.0, log=None):
    """NPT(P,T) equilibration to the zero-pressure liquid volume. Returns the last configuration
    rescaled to the mean cell of the second half: a 108-atom cell's volume fluctuates by a few
    percent, and switching at a snapshot volume would carry that pressure into F."""
    a = atoms.copy(); a.calc = mace_calc
    MaxwellBoltzmannDistribution(a, temperature_K=T)
    dyn = NPTBerendsen(a, timestep=dt_fs * units.fs, temperature_K=T,
                       pressure_au=P_bar * units.bar, taut=100 * units.fs, taup=1000 * units.fs,
                       compressibility_au=5e-7 / units.bar)
    if log is not None:
        dyn.attach(lambda: log(f"  relax volume step {dyn.nsteps} V {a.get_volume() / len(a):.3f} A^3/atom"),
                   interval=2500)
    n = int(t_fs / dt_fs)
    dyn.run(n // 2)
    cells = []
    dyn.attach(lambda: cells.append(a.cell.array.copy()), interval=10)
    dyn.run(n - n // 2)
    a.set_cell(np.mean(cells, axis=0), scale_atoms=True)
    return a


def liquid_free_energy(atoms, mace_calc, T, p=50, sigma=2.0,
                       t_equilibrate_fs=5000, t_switch_fs=20000, dt_fs=1.0,
                       friction=0.05, seed=1, t_sample_fs=0, sample_every_fs=100, log=None):
    """Absolute liquid free energy per atom (eV) at fixed volume of `atoms`, temperature T.

    With t_sample_fs > 0 the real liquid (lambda = 1) is also sampled for that long before the
    backward switch, and the result carries the frames under "samples": positions, velocities,
    the potential energy of each frame, and a pressure every tenth frame. The entropy then follows
    from the same run as S = (<U> + 3/2 kT - F) / T, which is what an entropy estimate from the
    frames (Hermes) is checked against. `log`, if given, is called with a line at each stage."""
    log = log or (lambda msg: None)
    atoms = atoms.copy()
    uf = UFCalculator(p, sigma, T)
    switch = SwitchCalculator(uf, mace_calc)      # spring<-UF, pot<-MACE
    atoms.calc = switch
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)

    watch = _Watch(atoms, switch, log)

    def equilibrate(stage):
        dyn = Langevin(atoms, dt_fs * units.fs, temperature_K=T, friction=friction, fixcm=False)
        watch(dyn, stage)
        dyn.run(int(t_equilibrate_fs / dt_fs))

    switch.lam = 0.0                               # equilibrate the UF fluid
    log("equilibrate UF")
    equilibrate("equilibrate UF")
    n_sw = int(t_switch_fs / dt_fs)
    log("forward switch")
    w_fwd = _switch_leg(atoms, switch, T, n_sw, dt_fs, friction, forward=True,
                        watch=lambda dyn: watch(dyn, "forward switch"))
    log(f"equilibrate real liquid (w_fwd {w_fwd / len(atoms):.4f} eV/atom)")
    switch.lam = 1.0                               # equilibrate the real (MACE) liquid
    equilibrate("equilibrate real liquid")
    log("sample real liquid")
    samples = _sample(atoms, mace_calc, T, t_sample_fs, sample_every_fs, dt_fs, friction,
                      watch=lambda dyn: watch(dyn, "sample")) if t_sample_fs > 0 else None
    log("backward switch")
    w_bwd = _switch_leg(atoms, switch, T, n_sw, dt_fs, friction, forward=False,
                        watch=lambda dyn: watch(dyn, "backward switch"))
    log(f"reference free energy (w_bwd {w_bwd / len(atoms):.4f} eV/atom)")

    dF_switch = 0.5 * (w_fwd - w_bwd) / len(atoms)
    F_uf, info = reference_free_energy_eV_per_atom(atoms, T, p, sigma)
    out = dict(
        T=T, natoms=len(atoms), p=p, sigma=sigma,
        F_uf_ref_eV=F_uf, dF_switch_eV=dF_switch, F_liquid_eV=F_uf + dF_switch,
        F_uf_ideal_gas=info["f_ideal_gas"], F_uf_excess=info["f_uf_excess"], rho=info["rho"],
        w_fwd_eV=w_fwd / len(atoms), w_bwd_eV=w_bwd / len(atoms),
        hysteresis_eV=(w_fwd + w_bwd) / len(atoms),
    )
    if samples is not None:
        out["samples"] = samples
    return out


def _sample(atoms, pot_calc, T, t_fs, every_fs, dt_fs, friction, watch=None):
    """Langevin at lambda = 1 (atoms.calc is the switch, already at 1), recording frames."""
    switch = atoms.calc
    pos, vel, epot, press = [], [], [], []

    def record():
        pos.append(atoms.get_positions())
        vel.append(atoms.get_velocities())
        epot.append(switch.e_pot)
        if len(pos) % 10 == 1:
            probe = atoms.copy(); probe.calc = pot_calc
            # MACE stress is the virial only; the kinetic part is added for the liquid's pressure
            stress = probe.get_stress(voigt=True) - \
                len(atoms) * units.kB * T / atoms.get_volume() * np.array([1, 1, 1, 0, 0, 0])
            press.append(-stress[:3].mean() / units.GPa)

    dyn = Langevin(atoms, dt_fs * units.fs, temperature_K=T, friction=friction, fixcm=False)
    dyn.attach(record, interval=max(1, int(every_fs / dt_fs)))
    if watch is not None:
        watch(dyn)
    dyn.run(int(t_fs / dt_fs))
    return dict(positions=np.array(pos), velocities=np.array(vel), epot_eV=np.array(epot),
                pressure_GPa=np.array(press), cell=atoms.cell.array.copy(),
                symbols=np.array(atoms.get_chemical_symbols()), masses=atoms.get_masses(),
                every_fs=every_fs, dt_fs=dt_fs)
