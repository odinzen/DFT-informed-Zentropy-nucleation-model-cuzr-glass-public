#!/usr/bin/env python3
"""The configuration spectrum of the Cu-Zr liquids with MACE-MP-0 on one GPU.

Each liquid case cools one 108-atom cell down a temperature ladder at zero pressure, keeps the energy and
positions at every rung, then quenches saved frames to their inherent structures, at the rung's cell and then with
the cell relaxed to zero pressure, and takes Gamma-point phonons of some of them. The slope of the zero pressure
inherent energy is the configurational heat capacity. Each crystal case does the energy ladder only. Stages resume from a checkpoint, so a job cut off
at the walltime loses at most one chunk; rerunning submit.sh continues it.

    python spectrum_run.py check RESULTS
    python spectrum_run.py case TAG RESULTS [--smoke]
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from ase import units
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights" / "20231203mace128L1_epoch199model"
DT_FS = 2.0
FRICTION = 0.002 / units.fs        # weak, so the saved trajectories keep the liquid's own relaxation
CHUNK_PS = 25.0
FRAME_PS = 1.0
EPOT_FS = 100.0
PRESSURE_PS = 1.0
MELT_K, MELT_PS = 1500.0, 20.0
DSTOP = 1.0                        # MACE-MP-0 has a deep hole below 1 A behind its repulsive wall

# (T in K, equilibration ps, sampling ps). Long enough to outlast the structural relaxation time estimated from
# measured viscosity (eta / 30 GPa): 1 ps at the liquidus, 0.14 ns at 900 K, about 1 ns at 850 K.
COLD = [(1100, 200, 200), (1050, 250, 250), (1000, 300, 300), (950, 500, 500), (900, 1000, 1000), (850, 2000, 2000)]
LIQUID = {"cu50zr50": [(1400, 100, 100), (1300, 100, 100), (1208, 150, 150)] + COLD,
          "cu64zr36": [(1400, 100, 100), (1300, 100, 100), (1230, 150, 150)] + COLD,
          "cu46zr54": [(1400, 100, 100), (1300, 100, 100), (1198, 150, 150)] + COLD}
CRYSTAL_LADDER = [(1250, 40, 100), (1150, 40, 100), (1050, 40, 100), (950, 40, 100), (850, 40, 100), (750, 40, 100)]
CRYSTALS = {"CuZr_B2": ((4, 4, 4), 128), "CuZr2": ((4, 4, 2), 96), "Cu10Zr7": ((1, 2, 2), 136), "Cu8Zr3": ((2, 2, 1), 176)}
QUENCH_FRAMES, PHONON_FRAMES = 60, 10
EXTRA_ZR = 4                             # Cu46Zr54 cells: a Cu50Zr50 cell with this many Cu made Zr, melted before use
V_CU, V_ZR = 11.81, 23.28                # A^3/atom, for the starting volume only; the first rung relaxes it
FORCE_TOL, ENERGY_TOL = 0.005, 0.2e-3    # eV/A, eV/atom: GPU against the local reference frame


def mace(dtype="float32", cueq=True):
    from mace.calculators import mace_mp
    return mace_mp(model=str(WEIGHTS), dispersion=False, default_dtype=dtype, device="cuda", enable_cueq=cueq)


def load_case(tag):
    if tag in CRYSTALS:
        rep, n = CRYSTALS[tag]
        a = read(HERE / "structures" / f"{tag}.vasp").repeat(rep)
        if len(a) != n:
            raise SystemExit(f"{tag}: {len(a)} atoms, expected {n}")
        return a, CRYSTAL_LADDER, False
    comp, cell = tag.rsplit("_", 1)
    src = "cu50zr50" if comp == "cu46zr54" else comp
    a = read(HERE / "structures" / f"scf_glass_reos_{src}_{cell}_v2.in", format="espresso-in")
    if len(a) != 108:
        raise SystemExit(f"{tag}: {len(a)} atoms, expected 108")
    if comp == "cu46zr54":
        sym = np.array(a.get_chemical_symbols())
        cu = np.where(sym == "Cu")[0]
        sym[np.random.default_rng(sum(map(ord, tag))).choice(cu, EXTRA_ZR, replace=False)] = "Zr"
        a.set_chemical_symbols(sym.tolist())
        a.set_cell(a.cell.array * (1 + EXTRA_ZR * (V_ZR - V_CU) / (54 * (V_CU + V_ZR))) ** (1 / 3), scale_atoms=True)
    return a, LIQUID[comp], True


def pressure_gpa(atoms, T):
    """Virial plus kinetic part; MACE's stress is the virial only."""
    s = atoms.get_stress(voigt=True) - len(atoms) * units.kB * T / atoms.get_volume() * np.array([1, 1, 1, 0, 0, 0])
    return float(-s[:3].mean() / units.GPa)


def guard(atoms, stage):
    d = neighbor_list("d", atoms, 2.5)
    dmin = float(d.min()) if len(d) else 2.5
    if dmin < DSTOP:
        raise RuntimeError(f"{stage}: closest pair {dmin:.2f} A, the cell has gone unphysical")


class Case:
    def __init__(self, tag, results, smoke):
        self.tag, self.smoke = tag, smoke
        self.dir = Path(results) / "runs" / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.atoms, ladder, self.liquid = load_case(tag)
        self.ladder = ladder[:2] if smoke else ladder
        self.ckpt = self.dir / "checkpoint.npz"
        # a checkpoint counts rungs, so it is only valid for the ladder that wrote it
        stamp = self.dir / "ladder.json"
        now = json.dumps(dict(ladder=self.ladder, quench=QUENCH_FRAMES, phonon=PHONON_FRAMES))
        if stamp.exists() and stamp.read_text() != now:
            raise SystemExit(f"{tag}: {self.dir} was run with a different ladder; it has to be moved aside first")
        stamp.write_text(now)
        self.st = dict(rung=0, stage=0, done=0, melted=not self.liquid, cells=[], epot=[], press=[], frames=[])
        self.atoms.calc = mace()
        np.random.seed(sum(map(ord, tag)))   # reproducible per case; a resumed run continues from its velocities
        fresh = not self.ckpt.exists()
        if fresh:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=(MELT_K if self.liquid else self.ladder[0][0]))
        else:
            self.restore()

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {self.tag}: {msg}"
        print(line, flush=True)
        with open(self.dir / "progress.log", "a") as f:
            f.write(line + "\n")

    def steps(self, ps):
        return 200 if self.smoke else int(round(ps * 1000 / DT_FS))

    def save(self):
        tmp = self.dir / "checkpoint.tmp.npz"
        np.savez(tmp, positions=self.atoms.get_positions(), velocities=self.atoms.get_velocities(),
                 cell=self.atoms.cell.array, rung=self.st["rung"], stage=self.st["stage"], done=self.st["done"],
                 melted=self.st["melted"], cells=np.array(self.st["cells"]), epot=np.array(self.st["epot"]),
                 press=np.array(self.st["press"]), frames=np.array(self.st["frames"], dtype=np.float32))
        os.replace(tmp, self.ckpt)

    def restore(self):
        c = np.load(self.ckpt)
        self.atoms.set_cell(c["cell"], scale_atoms=False)
        self.atoms.set_positions(c["positions"]); self.atoms.set_velocities(c["velocities"])
        self.st.update(rung=int(c["rung"]), stage=int(c["stage"]), done=int(c["done"]), melted=bool(c["melted"]),
                       cells=list(c["cells"]), epot=list(c["epot"]), press=list(c["press"]), frames=list(c["frames"]))
        self.log(f"resumed at rung {self.st['rung']} stage {self.st['stage']} step {self.st['done']}")

    def advance(self, make_dyn, n, every, name):
        """Run n steps in chunks, checkpointing after each, with `every` hooks recording into the state."""
        chunk = 200 if self.smoke else int(CHUNK_PS * 1000 / DT_FS)
        while self.st["done"] < n:
            m = min(chunk, n - self.st["done"])
            dyn = make_dyn()
            for fn, interval in every:
                dyn.attach(fn, interval=interval)
            t0 = time.time()
            dyn.run(m)
            self.st["done"] += m
            guard(self.atoms, name)
            self.save()
            self.log(f"{name} {self.st['done']}/{n}  {1000 * (time.time() - t0) / m:.1f} ms/step  "
                     f"V {self.atoms.get_volume() / len(self.atoms):.3f} A^3/atom")
        self.st["done"] = 0

    def npt(self, T):
        a = self.atoms
        return lambda: NPTBerendsen(a, timestep=DT_FS * units.fs, temperature_K=T, pressure_au=0.0,
                                    taut=100 * units.fs, taup=1000 * units.fs, compressibility_au=5e-7 / units.bar)

    def nvt(self, T):
        a = self.atoms
        return lambda: Langevin(a, DT_FS * units.fs, temperature_K=T, friction=FRICTION, fixcm=False)

    def run(self):
        if not self.st["melted"]:
            # the cells come from a fast EAM quench; melting them first erases that history
            self.advance(self.nvt(MELT_K), self.steps(MELT_PS), [], f"melt {MELT_K:.0f} K")
            self.st["melted"] = True; self.save()
        for i, (T, eq, sample) in enumerate(self.ladder):
            if i < self.st["rung"]:
                continue
            self.rung(T, eq, sample)
            self.st.update(rung=i + 1, stage=0, done=0, cells=[], epot=[], press=[], frames=[])
            self.save()
        (self.dir / "CASE_DONE").write_text("done\n")
        self.log("CASE_DONE")

    def rung(self, T, eq, sample):
        a = self.atoms
        name = f"{T} K"
        if self.st["stage"] == 0:
            # half the equilibration at constant pressure; the second half of that sets the volume
            n = self.steps(eq / 2)
            def cell():
                if self.st["done"] >= n // 2:
                    self.st["cells"].append(a.cell.array.copy())
            self.advance(self.npt(T), n, [(cell, 25)], f"{name} NPT")
            if self.st["cells"]:
                a.set_cell(np.mean(self.st["cells"], axis=0), scale_atoms=True)
            self.st.update(stage=1, cells=[]); self.save()
        if self.st["stage"] == 1:
            self.advance(self.nvt(T), self.steps(eq / 2), [], f"{name} NVT equilibrate")
            self.st["stage"] = 2; self.save()
        if self.st["stage"] == 2:
            def e():
                self.st["epot"].append(a.get_potential_energy())
            def p():
                self.st["press"].append(pressure_gpa(a, T))
            def fr():
                self.st["frames"].append(a.get_positions().astype(np.float32))
            hooks = [(e, max(1, int(EPOT_FS / DT_FS)))]
            hooks += [(p, max(1, int(PRESSURE_PS * 1000 / DT_FS))), (fr, max(1, int(FRAME_PS * 1000 / DT_FS)))]
            if self.smoke:
                hooks = [(e, 10), (p, 50), (fr, 50)]
            self.advance(self.nvt(T), self.steps(sample), hooks, f"{name} sample")
            self.st["stage"] = 3; self.save()
        self.finish_rung(T)

    def finish_rung(self, T):
        a = self.atoms
        n = len(a)
        epot = np.array(self.st["epot"], float) / n
        out = dict(T=float(T), natoms=n, cell=a.cell.array, symbols=np.array(a.get_chemical_symbols()),
                   masses=a.get_masses(), epot_eV=epot, pressure_GPa=np.array(self.st["press"], float),
                   frames=np.array(self.st["frames"], dtype=np.float32), dt_fs=DT_FS, frame_ps=FRAME_PS)
        if self.liquid:
            out.update(self.inherent(T, out["frames"]))
        np.savez_compressed(self.dir / f"rung_{int(T)}K.npz", **out)
        self.log(f"{T} K done: <Epot> {epot.mean():.5f} eV/atom, <P> {out['pressure_GPa'].mean():+.3f} GPa")

    def inherent(self, T, frames):
        """Quench saved frames to inherent structures, at the rung's cell and then at zero pressure; phonons of the first few."""
        n_q = 2 if self.smoke else QUENCH_FRAMES
        n_p = 1 if self.smoke else PHONON_FRAMES
        calc = mace("float64", cueq=False)      # double precision: the phonons difference small forces
        idx = np.unique(np.linspace(0, len(frames) - 1, n_q).astype(int))
        probe = self.atoms.copy(); probe.calc = calc
        cell = self.atoms.cell.array.copy()
        e, fmax, e0, v0, fmax0, pos, cells, freqs = [], [], [], [], [], [], [], []
        for k, i in enumerate(idx):
            probe.set_cell(cell, scale_atoms=False)
            probe.set_positions(frames[i].astype(float))
            FIRE(probe, logfile=None).run(fmax=0.01, steps=2000)
            e.append(probe.get_potential_energy() / len(probe))
            fmax.append(float(np.abs(probe.get_forces()).max()))
            guard(probe, f"{T} K quench")
            FIRE(FrechetCellFilter(probe, hydrostatic_strain=True), logfile=None).run(fmax=0.01, steps=2000)
            e0.append(probe.get_potential_energy() / len(probe))
            v0.append(probe.get_volume() / len(probe))
            fmax0.append(float(np.abs(probe.get_forces()).max()))
            guard(probe, f"{T} K zero pressure quench")
            pos.append(probe.get_positions().astype(np.float32))
            cells.append(probe.cell.array.copy())
            if k < n_p:
                freqs.append(gamma_modes(probe))
            self.log(f"{T} K quench {k + 1}/{len(idx)}: e_IS {e[-1]:.5f}, at zero pressure {e0[-1]:.5f} eV/atom, "
                     f"V {v0[-1]:.3f} A^3/atom, fmax {fmax0[-1]:.4f}")
        return dict(is_index=idx, is_energy_eV=np.array(e), is_fmax=np.array(fmax), is_energy0_eV=np.array(e0),
                    is_volume0=np.array(v0), is_fmax0=np.array(fmax0), is_positions=np.array(pos),
                    is_cells=np.array(cells), is_freq_THz=np.array(freqs))


def gamma_modes(atoms, delta=0.02):
    """Gamma-point frequencies (THz, negative for imaginary) by central differences of the forces."""
    n = len(atoms); p0 = atoms.get_positions()
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(3):
            p = p0.copy(); p[i, j] += delta; atoms.set_positions(p); fp = atoms.get_forces().ravel()
            p = p0.copy(); p[i, j] -= delta; atoms.set_positions(p); fm = atoms.get_forces().ravel()
            H[3 * i + j] = -(fp - fm) / (2 * delta)
    atoms.set_positions(p0)
    H = 0.5 * (H + H.T)
    m = np.repeat(atoms.get_masses(), 3)
    w2 = np.linalg.eigvalsh(H / np.sqrt(np.outer(m, m)))
    # eV/A^2/amu to THz: sqrt(eV/(A^2 amu)) = 15.633 THz (angular/2pi)
    return np.sign(w2) * np.sqrt(np.abs(w2)) * 15.6330


def reference_frame():
    return read(HERE / "structures" / "scf_glass_reos_cu50zr50_a_v2.in", format="espresso-in")


def check(results):
    import torch
    ref = np.load(HERE / "reference_forces.npz")
    a = reference_frame()
    if not np.allclose(a.get_positions(), ref["positions"], atol=1e-6):
        raise SystemExit("the reference frame differs from the one computed locally")
    res = {"gpu": torch.cuda.get_device_name(0)}
    for name, (dtype, cueq) in {"float64": ("float64", False), "float32_cueq": ("float32", True)}.items():
        a.calc = mace(dtype, cueq)
        df = float(np.abs(a.get_forces() - ref["forces"]).max())
        de = float(abs(a.get_potential_energy() - float(ref["energy"])) / len(a))
        res[name] = dict(max_force_difference_eV_A=df, energy_difference_eV_atom=de)
    # timing and a short run: this also proves the MD integrators run on this GPU
    MaxwellBoltzmannDistribution(a, temperature_K=1400)
    dyn = Langevin(a, DT_FS * units.fs, temperature_K=1400, friction=FRICTION, fixcm=False)
    t0 = time.time(); dyn.run(200)
    res["ms_per_step"] = 1000 * (time.time() - t0) / 200
    ok = all(res[k]["max_force_difference_eV_A"] <= FORCE_TOL and res[k]["energy_difference_eV_atom"] <= ENERGY_TOL
             for k in ("float64", "float32_cueq"))
    res["verdict"] = "PASS" if ok else "FAIL"
    Path(results, "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return ok


def main():
    mode = sys.argv[1]
    if mode == "check":
        return 0 if check(sys.argv[2]) else 1
    tag, results = sys.argv[2], sys.argv[3]
    c = Case(tag, results, "--smoke" in sys.argv)
    if (c.dir / "CASE_DONE").exists():
        print(f"{tag} already finished", flush=True)
        return 0
    c.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
