#!/usr/bin/env python3
"""What the liquid and glass look like against the two configurations the model assumes. Voronoi index of every atom in
the melt-quenched cells of Appendix A, by composition, and the MACE-MP-0 site energy of each atom after relaxation.
If two states existed the site energies would be bimodal and two polyhedra would dominate. Needs ase and mace-torch."""
import glob, json, os, sys
from collections import Counter
import numpy as np
from ase.io import read
from ase.optimize import LBFGS
from scipy.spatial import Voronoi
from mace.calculators import MACECalculator

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "glass_structures") + os.sep
MODEL = sys.argv[1]
OUT = sys.argv[2]
calc = MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64")


def voronoi_index(atoms, min_area_frac=0.01):
    """<n3,n4,n5,n6> per atom with periodic images. Faces under 1 % of the cell surface are dropped, the usual filter."""
    pos = atoms.get_positions(); cell = atoms.get_cell(); n = len(atoms)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    all_pos = np.vstack([pos + s @ np.array(cell) for s in shifts])
    home = n * list(map(tuple, shifts)).index((0, 0, 0))
    v = Voronoi(all_pos)
    faces = {i: [] for i in range(n)}
    for (a, b), verts in zip(v.ridge_points, v.ridge_vertices):
        if -1 in verts:
            continue
        for p in (a, b):
            if home <= p < home + n:
                faces[p - home].append(verts)
    idx = []
    for i in range(n):
        areas = []
        for verts in faces[i]:
            pts = v.vertices[verts]; c = pts.mean(axis=0)
            nrm = np.cross(pts[1] - pts[0], pts[2] - pts[0]); nrm /= np.linalg.norm(nrm) + 1e-12
            ang = np.arctan2(np.dot(np.cross(pts - c, np.tile(pts[0] - c, (len(pts), 1))), nrm), np.sum((pts - c) * (pts[0] - c), axis=1))
            o = np.argsort(ang); p2 = pts[o]
            a = 0.5 * np.abs(np.dot(sum(np.cross(p2[j] - c, p2[(j + 1) % len(p2)] - c) for j in range(len(p2))), nrm))
            areas.append((a, len(verts)))
        tot = sum(a for a, _ in areas)
        cnt = Counter(m for a, m in areas if a > min_area_frac * tot)
        idx.append((cnt.get(3, 0), cnt.get(4, 0), cnt.get(5, 0), cnt.get(6, 0)))
    return idx


res = {}
for comp in ("cu64zr36", "cu50zr50"):
    tally = Counter(); byel = {"Cu": Counter(), "Zr": Counter()}; E = {"Cu": [], "Zr": []}; Eico = []; Eother = []; totals = []
    files = sorted(glob.glob(ROOT + f"scf_glass_reos_{comp}_*_v2.in"))
    for f in files:
        a = read(f, format="espresso-in")
        a.calc = calc
        LBFGS(a, logfile=None).run(fmax=0.02, steps=300)
        idx = voronoi_index(a)
        calc.calculate(a, properties=["energy", "forces", "node_energy"])
        ne = calc.results.get("node_energy")
        sym = a.get_chemical_symbols()
        totals.append(a.get_potential_energy() / len(a))
        for s, i, e in zip(sym, idx, ne):
            tally[i] += 1; byel[s][i] += 1; E[s].append(float(e))
            if s == "Cu":
                (Eico if i == (0, 0, 12, 0) else Eother).append(float(e))
    n = sum(tally.values()); ncu = sum(byel["Cu"].values())
    top = tally.most_common(8)
    hist = np.histogram(E["Cu"], bins=12)
    e = np.array(E["Cu"]); z = (e - e.mean()) / e.std()
    res[comp] = dict(cells=len(files), atoms=n, top=[(str(k), v / n) for k, v in top], ico_all=tally[(0, 0, 12, 0)] / n, ico_Cu=byel["Cu"][(0, 0, 12, 0)] / max(ncu, 1), distinct=len(tally),
                     E_Cu_mean=float(e.mean()), E_Cu_sd=float(e.std()), E_Cu_skew=float((z ** 3).mean()), E_Cu_kurt=float((z ** 4).mean() - 3), bimodality=float(((z ** 3).mean() ** 2 + 1) / ((z ** 4).mean())),
                     E_ico=float(np.mean(Eico)) if Eico else None, E_other=float(np.mean(Eother)), sd_ico=float(np.std(Eico)) if Eico else None, sd_other=float(np.std(Eother)), n_ico=len(Eico), hist=[int(x) for x in hist[0]], edges=[float(x) for x in hist[1]], per_cell_E=totals)
    r = res[comp]
    print(f"\n{comp}: {r['cells']} cells, {n} atoms, {r['distinct']} distinct Voronoi types")
    print("  most common: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in r["top"]))
    print(f"  full icosahedra <0,0,12,0>: {r['ico_all'] * 100:.1f}% of atoms, {r['ico_Cu'] * 100:.1f}% of Cu")
    print(f"  Cu site energy: sd {r['E_Cu_sd'] * 1000:.0f} meV ({r['E_Cu_sd'] * 96.485:.1f} kJ/mol), skew {r['E_Cu_skew']:+.2f}, excess kurtosis {r['E_Cu_kurt']:+.2f}, bimodality coefficient {r['bimodality']:.2f} (above 0.56 suggests two modes)")
    print(f"  icosahedral Cu mean {r['E_ico']} ({r['n_ico']}) vs other Cu {r['E_other']:.3f}; difference {(r['E_ico'] - r['E_other']) * 96.485 if r['E_ico'] else float('nan'):+.2f} kJ/mol; within-type sd {r['sd_ico'] * 96.485 if r['sd_ico'] else float('nan'):.1f} and {r['sd_other'] * 96.485:.1f} kJ/mol")
    print("  Cu site energy histogram: " + " ".join(map(str, r["hist"])))
json.dump(res, open(OUT, "w"), indent=1, default=float)
