#!/usr/bin/env python3
"""Crystal absolute free energy by Frenkel-Ladd switching, the crystalline twin of liquid_ti.

The crystal is tethered to its lattice sites by one Einstein spring constant per element, set from
the mean-square displacement of that element in the real crystal (k = 3 kT / <dr^2>, as calphy
does), and switched to the interatomic potential with the centre of mass held still. The reference
free energy, including the centre-of-mass correction that the fixed centre of mass calls for, is
calphy's get_einstein_crystal_fe, called in calphy's own environment (CALPHY_PYTHON) because calphy
does not install beside the CUDA potential.

Unlike the harmonic phonon free energy, this keeps the anharmonicity near melting, and it handles
B2 CuZr, which is dynamically unstable at 0 K but stable at the measured liquidus.
"""
import json
import math
import subprocess

import numpy as np
from ase import units
from ase.constraints import FixCom
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from frenkel_ladd import SpringCalculator, SwitchCalculator, _switch_leg
from liquid_ti import _Watch, relax_volume
from uhlenbeck_ford import CALPHY_PYTHON

KB_eV = 8.617333262e-5
MELT_MSD_A2 = 0.5          # a crystal near melting stays well below this; a liquid passes it in ps


def supercell(prim, min_width=10.0):
    """Repeat the cell until every perpendicular width is at least min_width (A)."""
    a = prim.cell.array
    vol = abs(np.linalg.det(a))
    widths = [vol / np.linalg.norm(np.cross(a[(i + 1) % 3], a[(i + 2) % 3])) for i in range(3)]
    return prim.repeat([max(1, math.ceil(min_width / w)) for w in widths])


def reference_inputs(atoms, T, k_by_species):
    """Everything the Einstein reference depends on, so it can also be evaluated after the run."""
    species = sorted(set(atoms.get_chemical_symbols()))
    syms = atoms.get_chemical_symbols()
    return dict(T=float(T), vol=float(atoms.get_volume()), element=species,
                count=[syms.count(s) for s in species],
                mass=[float(next(a.mass for a in atoms if a.symbol == s)) for s in species],
                k=[float(k_by_species[s]) for s in species])


def einstein_from_inputs(args):
    """(F_ref, F_e, F_cm) in eV per atom from reference_inputs, via calphy."""
    script = """
import json, sys
from calphy.integrators import get_einstein_crystal_fe
a = json.loads(sys.argv[1])
class Calc: pass
c = Calc(); c._temperature = a["T"]; c.element = a["element"]
c._element_dict = {e: {"count": n, "mass": m} for e, n, m in zip(a["element"], a["count"], a["mass"])}
fe, fcm = get_einstein_crystal_fe(c, a["vol"], a["k"], cm_correction=True, return_contributions=True)
print(json.dumps([float(fe), float(fcm)]))
"""
    out = subprocess.run([CALPHY_PYTHON or "python", "-c", script, json.dumps(args)],
                         capture_output=True, text=True, check=True).stdout
    fe, fcm = json.loads(out.strip().splitlines()[-1])
    return fe + fcm, fe, fcm


def _msd_by_species(atoms, r0):
    d2 = np.sum((atoms.get_positions() - r0) ** 2, axis=1)
    syms = np.array(atoms.get_chemical_symbols())
    return {s: float(d2[syms == s].mean()) for s in sorted(set(syms))}, float(d2.mean())


def crystal_free_energy(ideal, pot, T, t_relax_fs=20000, t_msd_fs=10000, t_equilibrate_fs=10000,
                        t_switch_fs=50000, t_sample_fs=20000, sample_every_fs=100, dt_fs=1.0,
                        friction=0.05, seed=1, log=None):
    """Absolute free energy per atom (eV) of the crystal `ideal` (sites at their ideal positions) at
    temperature T and zero pressure. Returns the result with the sampled frames under "samples"."""
    log = log or (lambda msg: None)
    atoms = ideal.copy()
    if t_relax_fs > 0:                                # t_relax_fs = 0 keeps the given cell
        log("relax volume")
        g = relax_volume(ideal, pot, T, t_fs=t_relax_fs, log=log)
        atoms.set_cell(g.cell, scale_atoms=True)      # ideal sites at the mean zero-pressure cell
    r0 = atoms.get_positions().copy()
    # springs with per-element k on unequal masses can push the centre of mass, which the
    # thermostat's fixcm does not undo; the constraint holds it, as calphy does in LAMMPS
    atoms.set_constraint(FixCom())
    rng = np.random.default_rng(seed)

    log("spring constants")
    atoms.calc = pot
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)
    Stationary(atoms)
    dyn = Langevin(atoms, dt_fs * units.fs, temperature_K=T, friction=friction, fixcm=True)
    dyn.run(int(t_msd_fs / dt_fs / 2))                  # settle, then measure
    acc = []
    dyn.attach(lambda: acc.append(_msd_by_species(atoms, r0)[0]), interval=50)
    dyn.run(int(t_msd_fs / dt_fs / 2))
    msd = {s: float(np.mean([a[s] for a in acc])) for s in acc[0]}
    if max(msd.values()) > MELT_MSD_A2:
        raise RuntimeError(f"crystal melted while measuring springs: MSD {msd}")
    k_by_species = {s: 3 * KB_eV * T / v for s, v in msd.items()}
    karr = np.array([k_by_species[s] for s in atoms.get_chemical_symbols()])
    log("springs " + ", ".join(f"{s} {k:.3f} eV/A^2 (MSD {msd[s]:.4f} A^2)" for s, k in k_by_species.items()))

    switch = SwitchCalculator(SpringCalculator(r0, karr), pot)
    atoms.set_positions(r0)
    atoms.calc = switch
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)
    Stationary(atoms)
    watch = _Watch(atoms, switch, log)

    def equilibrate(stage):
        d = Langevin(atoms, dt_fs * units.fs, temperature_K=T, friction=friction, fixcm=True)
        watch(d, stage)
        d.run(int(t_equilibrate_fs / dt_fs))

    switch.lam = 0.0
    log("equilibrate Einstein crystal")
    equilibrate("equilibrate Einstein")
    n_sw = int(t_switch_fs / dt_fs)
    log("forward switch")
    w_fwd = _switch_leg(atoms, switch, T, n_sw, dt_fs, friction, forward=True,
                        watch=lambda d: watch(d, "forward switch"), fixcm=True)
    switch.lam = 1.0
    log(f"equilibrate real crystal (w_fwd {w_fwd / len(atoms):.4f} eV/atom)")
    equilibrate("equilibrate real crystal")

    log("sample real crystal")
    pos, epot, msd_t, press = [], [], [], []

    def record():
        pos.append(atoms.get_positions())
        epot.append(switch.e_pot)
        m = _msd_by_species(atoms, r0)[1]
        msd_t.append(m)
        if m > MELT_MSD_A2:
            raise RuntimeError(f"crystal melted during sampling: MSD {m:.3f} A^2")
        if len(pos) % 10 == 1:
            probe = atoms.copy(); probe.calc = pot
            stress = probe.get_stress(voigt=True) - \
                len(atoms) * units.kB * T / atoms.get_volume() * np.array([1, 1, 1, 0, 0, 0])
            press.append(-stress[:3].mean() / units.GPa)

    d = Langevin(atoms, dt_fs * units.fs, temperature_K=T, friction=friction, fixcm=True)
    d.attach(record, interval=max(1, int(sample_every_fs / dt_fs)))
    watch(d, "sample")
    d.run(int(t_sample_fs / dt_fs))

    log("backward switch")
    w_bwd = _switch_leg(atoms, switch, T, n_sw, dt_fs, friction, forward=False,
                        watch=lambda d: watch(d, "backward switch"), fixcm=True)
    log(f"reference free energy (w_bwd {w_bwd / len(atoms):.4f} eV/atom)")
    ref_in = reference_inputs(atoms, T, k_by_species)
    try:
        F_ref, F_e, F_cm = einstein_from_inputs(ref_in)
    except (OSError, subprocess.CalledProcessError) as e:
        # no calphy where this ran (Sol): keep the inputs, fill_reference.py evaluates them later
        log(f"reference deferred ({type(e).__name__}); inputs stored")
        F_ref = F_e = F_cm = None
    dF = 0.5 * (w_fwd - w_bwd) / len(atoms)
    return dict(
        T=T, natoms=len(atoms), k_eV_A2=k_by_species, msd_A2=msd,
        F_einstein_eV=F_e, F_cm_eV=F_cm, F_reference_eV=F_ref, dF_switch_eV=dF,
        F_crystal_eV=None if F_ref is None else F_ref + dF, reference_inputs=ref_in,
        w_fwd_eV=w_fwd / len(atoms), w_bwd_eV=w_bwd / len(atoms),
        hysteresis_eV=(w_fwd + w_bwd) / len(atoms), msd_sample_max_A2=float(max(msd_t)),
        samples=dict(positions=np.array(pos), epot_eV=np.array(epot), pressure_GPa=np.array(press),
                     cell=atoms.cell.array.copy(), symbols=np.array(atoms.get_chemical_symbols()),
                     masses=atoms.get_masses(), sites=r0, every_fs=sample_every_fs, dt_fs=dt_fs),
    )
