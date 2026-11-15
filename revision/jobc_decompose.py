"""Split the oxide|crystal energy into the strain the crystal slab carries and the interface proper."""
import itertools, json, sys
import numpy as np
from ase.io import read
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from mace.calculators import MACECalculator

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "interface_structures") + os.sep
calc = MACECalculator(model_paths=sys.argv[1], device="cuda", default_dtype="float32")
res = json.load(open(sys.argv[2]))
EV_M2 = 16.021766
STRAIN = {"B2": (-0.053, -0.053), "Cu10Zr7": (-0.026, -0.002)}

for tag in ("B2", "Cu10Zr7"):
    a = read(ROOT + f"iface_bigcube_{tag}.xyz")
    s = np.array(a.get_chemical_symbols()); c = a.cell.lengths()
    print(tag, "iface cell", np.round(c, 2), {e: int((s == e).sum()) for e in sorted(set(s))})
    zO = a.positions[s == "O", 2]
    print("  O z", round(zO.min(), 2), round(zO.max(), 2), "of c", round(c[2], 2))
    nO = int((s == "O").sum()); nox = 7 * nO; ncr = len(a) - nox
    nCu = int((s == "Cu").sum()) - 2 * nO; nZr = int((s == "Zr").sum()) - 4 * nO
    print(f"  crystal slab if the oxide is stoichiometric: Cu {nCu} Zr {nZr}  (ideal ratio {'1.000' if tag == 'B2' else '1.429'}, here {nCu / nZr:.3f})")
    bulk = read(ROOT + f"bulk_{tag}.xyz")
    base = bulk.copy(); base.calc = calc
    LBFGS(FrechetCellFilter(base), logfile=None).run(fmax=0.02, steps=300)
    mu0 = base.get_potential_energy() / len(base)
    # in-plane strain on each pair of bulk axes, normal axis relaxed by a scan; the lowest pair is the generous bound
    best = None
    for i, j in itertools.permutations(range(3), 2):
        k = 3 - i - j
        for fac in np.linspace(0.97, 1.12, 16):
            b = base.copy(); cell = np.array(b.cell)
            cell[i] *= 1 + STRAIN[tag][0]; cell[j] *= 1 + STRAIN[tag][1]; cell[k] *= fac
            b.set_cell(cell, scale_atoms=True); b.calc = calc
            LBFGS(b, logfile=None).run(fmax=0.03, steps=60)
            e = b.get_potential_energy() / len(b)
            if best is None or e < best[0]:
                best = (e, i, j, fac)
    d = best[0] - mu0
    A = res[tag]["area"]; n = res[tag]["n_cr"]; g = res[tag]["gamma"]
    gs = n * d / (2 * A) * EV_M2
    print(f"  strained crystal costs {d * 1000:.1f} meV/atom (axes {best[1]},{best[2]}, normal x{best[3]:.2f}), over this slab {gs:.3f} J/m2 per interface")
    print(f"  total {g:.3f} J/m2, interface proper {g - gs:.3f} J/m2")
    res[tag].update(strain_mev=d * 1000, gamma_strain=gs, gamma_interface=g - gs)
json.dump(res, open(sys.argv[2], "w"), indent=1)
