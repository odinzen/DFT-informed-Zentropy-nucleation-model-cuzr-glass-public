"""Local run of the oxide|crystal interface energy (Job C decks). Relaxes bulks and both interfaces with
MACE-MP-0. float32 on the laptop GPU to fit 4 GB, CPU fallback if the 880-atom cell does not fit."""
import json, sys, time
import numpy as np
import torch
from ase.io import read
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from mace.calculators import MACECalculator

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "interface_structures") + os.sep
MODEL = sys.argv[1]
OUT = sys.argv[2]
EV_M2 = 16.021766


def calc(device):
    return MACECalculator(model_paths=MODEL, device=device, default_dtype="float32")


def relax(atoms, device, cell=False, fmax=0.05, steps=500):
    a = atoms.copy(); a.calc = calc(device)
    dyn = LBFGS(FrechetCellFilter(a) if cell else a, logfile=None)
    dyn.run(fmax=fmax, steps=steps)
    return a, dyn.nsteps


res = {}
mu = {}
for name, f in [("bigcube", "bulk_bigcube.xyz"), ("Cu10Zr7", "bulk_Cu10Zr7.xyz"), ("B2", "bulk_B2.xyz")]:
    t0 = time.time()
    a, n = relax(read(ROOT + f), "cuda", cell=True)
    mu[name] = a.get_potential_energy() / len(a)
    print(f"bulk {name}: {len(a)} atoms mu {mu[name]:.4f} eV/atom, {n} steps, {time.time()-t0:.0f} s", flush=True)
res["mu"] = mu

for tag in ("B2", "Cu10Zr7"):
    t0 = time.time()
    atoms = read(ROOT + f"iface_bigcube_{tag}.xyz")
    try:
        a, n = relax(atoms, "cuda")
        dev = "cuda"
    except RuntimeError as e:
        print("GPU failed, CPU", str(e)[:80], flush=True)
        torch.cuda.empty_cache()
        a, n = relax(atoms, "cpu")
        dev = "cpu"
    sym = a.get_chemical_symbols()
    n_ox = sym.count("O") * 7
    n_cr = len(a) - n_ox
    c = a.get_cell()
    A = np.linalg.norm(np.cross(c[0], c[1]))
    E = a.get_potential_energy()
    g = (E - n_ox * mu["bigcube"] - n_cr * mu[tag]) / (2 * A) * EV_M2
    fmax = np.abs(a.get_forces()).max()
    res[tag] = dict(n_ox=n_ox, n_cr=n_cr, area=float(A), gamma=float(g), steps=int(n), fmax=float(fmax), device=dev)
    print(f"bigcube|{tag}: N_ox {n_ox} N_cr {n_cr} A {A:.1f} A2 gamma {g:.3f} J/m2 steps {n} fmax {fmax:.3f} {dev} {time.time()-t0:.0f} s", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
print("done", flush=True)
