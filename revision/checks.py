#!/usr/bin/env python3
"""Checks on the Sol spectrum runs before any number is used: pressure and its PV term per rung, a smooth total heat
capacity from all rungs at once, crystallization per cell (inherent energy, volume and local order at 850 K against the
trend), quench convergence, and the crystal heat capacities against harmonic and Neumann-Kopp values."""
import glob, json, os, sys
import numpy as np

R_ = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sol_spectrum"))
F = 96485.33; R = 8.314462; kB = 8.617333e-5
NK = lambda T: 25.5 + 0.00517 * T      # Castellero and Battezzati, crystal CuZr, read from their Fig. 3
CAL = lambda T: (55.979 - 0.01688 * T) - NK(T)


def rungs(tag):
    out = []
    for p in sorted(glob.glob(os.path.join(R_, "runs", tag, "rung_*K.npz"))):
        out.append(dict(np.load(p)))
    return sorted(out, key=lambda r: float(r["T"]))


def q6(pos, cell):
    """Mean local q6 with a 3.4 A cutoff; crystals give 0.35-0.55, these liquids about 0.25."""
    from scipy.special import sph_harm_y
    n = len(pos); inv = np.linalg.inv(cell)
    vals = []
    for i in range(n):
        d = pos - pos[i]; f = d @ inv; f -= np.round(f); d = f @ cell
        r = np.linalg.norm(d, axis=1); m = (r > 0.1) & (r < 3.4)
        v = d[m] / r[m, None]
        th = np.arccos(np.clip(v[:, 2], -1, 1)); ph = np.arctan2(v[:, 1], v[:, 0])
        q = np.array([sph_harm_y(6, mm, th, ph).mean() for mm in range(-6, 7)])
        vals.append(np.sqrt(4 * np.pi / 13 * (np.abs(q) ** 2).sum()))
    return float(np.mean(vals))


out = {}
for comp in ("cu50zr50", "cu64zr36", "cu46zr54"):
    print(f"\n== {comp}")
    cells = {c: rungs(f"{comp}_{c}") for c in "abc"}
    print("  per cell: T, <P> GPa, PV term kJ/mol, e_IS0 eV/atom, V_IS, max fmax0, q6 of the last IS")
    for c, rs in cells.items():
        for r in rs:
            T = float(r["T"]); V = abs(np.linalg.det(r["cell"])) / int(r["natoms"]); P = float(r["pressure_GPa"].mean())
            qq = q6(r["is_positions"][-1].astype(float), r["is_cells"][-1]) if T <= 900 or T >= 1400 else float("nan")
            print(f"   {c} {T:5.0f}  P {P:+6.2f}  PV {P * V / 160.21766 * F / 1e3:+6.2f}  e_IS0 {r['is_energy0_eV'].mean():.5f} +- {r['is_energy0_eV'].std(ddof=1) / np.sqrt(len(r['is_energy0_eV'])):.5f}"
                  f"  V_IS {r['is_volume0'].mean():.3f}  fmax0 {r['is_fmax0'].max():.4f}  q6 {qq:.3f}")
    # smooth heat capacity from E_pot only, fitted over every rung of every cell
    T = np.array([float(r["T"]) for rs in cells.values() for r in rs])
    U = np.array([(r["epot_eV"].mean() + 1.5 * kB * float(r["T"])) * F for rs in cells.values() for r in rs])
    PV = np.array([float(r["pressure_GPa"].mean()) * abs(np.linalg.det(r["cell"])) / int(r["natoms"]) / 160.21766 * F for rs in cells.values() for r in rs])
    E = np.array([r["is_energy0_eV"].mean() * F for rs in cells.values() for r in rs])
    res = {}
    for lab, y in (("liquid, E + 3/2 kT (no PV)", U), ("liquid, with PV", U + PV), ("inherent structures", E)):
        X = np.column_stack([np.ones_like(T), T, T ** 2]); p, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ p
        cp = {t: float(p[1] + 2 * p[2] * t) for t in (850, 900, 1000, 1100, 1208, 1300)}
        res[lab] = dict(cp=cp, rms_resid=float(resid.std()))
        print(f"  quadratic fit, {lab:28s} dH/dT at 900/1000/1100/1208 K: " + " ".join(f"{cp[t]:5.1f}" for t in (900, 1000, 1100, 1208)) + f"   residual {resid.std():.0f} J/mol")
    # local slope of the inherent energy for the a+b/T+c/T^2 form is in collect; here: vibrational part
    vib = {t: res["liquid, E + 3/2 kT (no PV)"]["cp"][t] - res["inherent structures"]["cp"][t] for t in (900, 1000, 1100, 1208)}
    print("  liquid heat capacity minus configurational part (vibrational, incl. 3/2 R kinetic): " + " ".join(f"{t} K {v:.1f}" for t, v in vib.items()))
    out[comp] = res
print("\n== crystals: dU/dT from a linear fit over the six rungs (harmonic classical value 3R = 24.94)")
for name in ("CuZr_B2", "Cu10Zr7", "CuZr2", "Cu8Zr3"):
    rs = rungs(name); T = np.array([float(r["T"]) for r in rs]); U = np.array([(r["epot_eV"].mean() + 1.5 * kB * float(r["T"])) * F for r in rs])
    P = [float(r["pressure_GPa"].mean()) for r in rs]
    p = np.polyfit(T, U, 1); q = np.polyfit(T, U, 2)
    print(f"  {name:8s} linear {p[0]:5.2f}  quadratic at 900/1200 K {q[1] + 2 * q[0] * 900:5.2f} {q[1] + 2 * q[0] * 1200:5.2f}  <P> {min(P):+.2f} to {max(P):+.2f} GPa"
          + (f"   Neumann-Kopp estimate {NK(900):.1f} {NK(1200):.1f}" if name == "CuZr_B2" else ""))
    out[name] = dict(linear=float(p[0]))
print("\n== CuZr calorimetric difference liquid - crystal: " + " ".join(f"{t} K {CAL(t):.1f}" for t in (900, 1000, 1100, 1208)))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checks.json"), "w"), indent=1, default=float)
