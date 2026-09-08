#!/usr/bin/env python3
"""Config-2 free-energy campaign (self-contained hand-off version).

Computes soft2 and the configurational split dE from first principles using the config-1
pipeline: Mendelev Cu-Zr EAM (via matscipy, C-accelerated), ase relaxation, and a
finite-displacement phonon DOS, at two quench rates. Config 1 = slow quench (icosahedral
ground), config 2 = fast quench (defective). soft_k = theta_D,k(VDOS) / theta_D,cryst(engine);
dE = E0(config 2) - E0(config 1). Two binaries, n=5 replicas each, checkpointed and
resume-safe (config2_results.json).

Everything it needs is in this folder: Cu-Zr_2.eam.fs, phonon_vdos.py, zentropy_driving_force.py.
See README.md for setup. Run:  python config2_campaign.py
"""
import os, sys, json, time
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zentropy_driving_force as z
import phonon_vdos as pv

POT = os.path.join(HERE, "Cu-Zr_2.eam.fs")
RESULTS = os.path.join(HERE, "config2_results.json")
Vat = {"Cu": 11.81, "Zr": 23.28}                 # A^3
N = 108
NRUNS = 5
COMPS = {
    "Cu50Zr50": dict(Cu=54, Zr=54, xZr=0.50, bnd=("Cu10Zr7", "CuZr")),
    "Cu64Zr36": dict(Cu=69, Zr=39, xZr=0.36, bnd=("Cu8Zr3", "Cu10Zr7")),
}
QUENCH = {"config1_slow": 6000, "config2_fast": 600}   # ~158 and ~1580 K/ps

def theta_cryst(xZr, lo, hi):
    _, th, _ = z.crystalline_assembly(xZr, lo, hi); return th

def build(counts, seed):
    rng = np.random.default_rng(seed)
    syms = []
    for k in ("Cu", "Zr"):
        syms += [k] * counts[k]
    rng.shuffle(syms)
    frac = {k: counts[k] / N for k in ("Cu", "Zr")}
    L = (sum(frac[k] * Vat[k] for k in frac) * N) ** (1 / 3)
    s = int(np.ceil(N ** (1 / 3))); pts = []
    for i in range(s):
        for j in range(s):
            for k in range(s):
                if len(pts) < N: pts.append((i, j, k))
    pos = np.array(pts, float) / s * L
    return Atoms(syms, positions=pos, cell=[L, L, L], pbc=True)

def melt_quench(atoms, calc, quench_steps, seed):
    np.random.seed(seed)
    atoms = atoms.copy(); atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=2200)
    dyn = Langevin(atoms, 2 * units.fs, temperature_K=2200, friction=0.02)
    dyn.run(1000)
    T0, T1 = 2200, 300
    for k in range(0, quench_steps, 50):
        dyn.set_temperature(temperature_K=T0 + (T1 - T0) * k / quench_steps); dyn.run(50)
    return atoms

def relax(atoms, calc):
    atoms.calc = calc
    LBFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=2e-4, steps=400)
    return atoms

def one_config(counts, bnd, seed, quench_steps, calc):
    atoms = melt_quench(build(counts, seed), calc, quench_steps, seed)
    relax(atoms, calc)
    E0 = atoms.get_potential_energy() / len(atoms)
    freq = pv.hessian_frequencies(atoms, calc)
    m = pv.spectrum_metrics(freq)
    thc = theta_cryst(counts["xZr"], *bnd)
    return dict(E0_eV=float(E0), theta_glass=float(m["theta_D_eff"]),
                omega_ln_THz=float(m["omega_ln_THz"]), n_imag=int(m["n_imag_extra"]),
                theta_cryst=float(thc), soft=float(m["theta_D_eff"] / thc))

def make_calc():
    """matscipy EAM (C-accelerated, ~2-3 h campaign) if available, else ase's pure-Python
    EAM (correct but ~9x slower, ~1 day). Same Mendelev potential either way."""
    if not os.path.exists(POT):
        sys.exit(f"EAM potential not found: {POT}")
    try:
        from matscipy.calculators.eam import EAM
        c = EAM(POT, kind="eam/fs")
        print("[calc] matscipy EAM (fast) -- campaign ~2-3 h", flush=True)
        return c
    except Exception as e:
        from ase.calculators.eam import EAM
        print(f"[calc] matscipy unavailable ({str(e)[:50]}); using ase EAM (slow, ~1 day). "
              f"Install matscipy for a ~9x speedup.", flush=True)
        return EAM(potential=POT, form="fs")

if __name__ == "__main__":
    calc = make_calc()
    results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else []
    done = {(r["comp"], r["config"], r["seed"]) for r in results}
    total = len(COMPS) * len(QUENCH) * NRUNS
    for comp, d in COMPS.items():
        for cfg, qsteps in QUENCH.items():
            for i in range(NRUNS):
                seed = 2000 + i
                if (comp, cfg, seed) in done:
                    print(f"[skip] {comp} {cfg} seed{seed}", flush=True); continue
                print(f"[run ] {comp} {cfg} seed{seed} ({len(results)}/{total}) ...", flush=True)
                t = time.time()
                try:
                    r = one_config(dict(Cu=d["Cu"], Zr=d["Zr"], xZr=d["xZr"]),
                                   d["bnd"], seed, qsteps, calc)
                except Exception as e:
                    print(f"[FAIL] {comp} {cfg} seed{seed}: {e}", flush=True); continue
                r.update(comp=comp, config=cfg, seed=seed, seconds=round(time.time() - t, 1))
                results.append(r)
                json.dump(results, open(RESULTS, "w"), indent=1)
                print(f"[done] {comp} {cfg} seed{seed}: E0={r['E0_eV']:.4f} eV/at  "
                      f"soft={r['soft']:.3f}  theta_g={r['theta_glass']:.0f}K  ({r['seconds']}s)", flush=True)

    print("\n=== CONFIG-2 CAMPAIGN SUMMARY (mean +/- SEM) ===", flush=True)
    for comp in COMPS:
        rc = {c: [r for r in results if r["comp"] == comp and r["config"] == c] for c in QUENCH}
        if all(rc[c] for c in QUENCH):
            s1 = np.array([r["soft"] for r in rc["config1_slow"]])
            s2 = np.array([r["soft"] for r in rc["config2_fast"]])
            e1 = np.array([r["E0_eV"] for r in rc["config1_slow"]]).mean()
            e2 = np.array([r["E0_eV"] for r in rc["config2_fast"]]).mean()
            dE_kJ = (e2 - e1) * 96.485
            print(f"  {comp}:  soft1={s1.mean():.3f}+/-{s1.std(ddof=1)/len(s1)**.5:.3f}  "
                  f"soft2={s2.mean():.3f}+/-{s2.std(ddof=1)/len(s2)**.5:.3f}  "
                  f"dE={dE_kJ:.2f} kJ/mol  (declared: soft2=0.80, dE=6.0)", flush=True)
