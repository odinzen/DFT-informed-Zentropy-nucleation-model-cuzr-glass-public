#!/usr/bin/env python3
"""Which crystal, from assessed Gibbs energies. The paper's reference is a lever mixture nucleating as if it were one
structure at the liquid composition. Here every published Cu-Zr assessment gives, for each real phase, the parallel
tangent driving force from a liquid of the alloy's composition (the force on an actual nucleus of that phase), the
mixture value the paper's construction corresponds to, and the temperature at which each vanishes."""
import glob, json, os, sys, warnings
import numpy as np
from pycalphad import Database, calculate, variables as v
warnings.filterwarnings("ignore")

ROOT = os.environ.get("HALLSTEDT_TDB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdb"))
FILES = sorted(glob.glob(os.path.join(ROOT, "CuZr-*.tdb")))
ALLOYS = {"Cu64Zr36": 0.36, "Cu50Zr50": 0.50, "Cu46Zr54": 0.54}
TEMPS = np.arange(700.0, 1321.0, 10.0)
XZR = {"CU9ZR2": 2 / 11, "CU5ZR": 1 / 6, "CU51ZR14": 14 / 65, "CU8ZR3": 3 / 11, "CU10ZR7": 7 / 17, "CUZR": 0.5, "CUZR2": 2 / 3}


def phase_G(db, name, T, sp):
    r = calculate(db, sp, name, P=101325, T=T, N=1, output="GM", pdens=20)
    return r


def liquid_curve(db, T):
    xs = np.linspace(0.20, 0.75, 111)
    out = []
    for x in xs:
        r = calculate(db, ["CU", "ZR"], "LIQUID", P=101325, T=T, N=1, output="GM", points=np.array([[1 - x, x]]))
        out.append(float(r.GM.values.squeeze()))
    return xs, np.array(out)


def run_one(path):
    db = Database(path)
    name = os.path.basename(path)[5:-4]
    phs = {}
    for p in db.phases:
        q = p.upper().replace("_", "")
        for k in XZR:
            if q == k or q == k + "S" or q.startswith(k):
                phs[k] = p
    res = {}
    table = {}
    for T in TEMPS:
        xs, GL = liquid_curve(db, T)
        d = np.gradient(GL, xs)
        G = {}
        for k, p in phs.items():
            r = calculate(db, ["CU", "ZR"], p, P=101325, T=T, N=1, output="GM", pdens=50)
            X = r.X.sel(component="ZR").values.squeeze(); E = r.GM.values.squeeze()
            X = np.atleast_1d(X); E = np.atleast_1d(E)
            i = np.argmin(np.abs(X - XZR[k])); G[k] = float(E[i])
        table[float(T)] = dict(xs=xs, GL=GL, d=d, G=G)
    for key, x0 in ALLOYS.items():
        rows = {}
        for T, t in table.items():
            g0 = np.interp(x0, t["xs"], t["GL"]); s0 = np.interp(x0, t["xs"], t["d"])
            D = {k: g0 + (XZR[k] - x0) * s0 - val for k, val in t["G"].items()}      # J per mole of atoms of the nucleus
            # lower envelope of the real phases at x0, the mixture the paper's reference stands for
            ks = sorted(t["G"], key=lambda k: XZR[k])
            best = None
            for i in ks:
                for j in ks:
                    if XZR[i] <= x0 <= XZR[j] and i != j:
                        f = (x0 - XZR[i]) / (XZR[j] - XZR[i]); e = (1 - f) * t["G"][i] + f * t["G"][j]
                        if best is None or e < best[0]:
                            best = (e, i, j)
                    elif XZR[i] == x0 and i == j:
                        if best is None or t["G"][i] < best[0]:
                            best = (t["G"][i], i, i)
            rows[T] = dict(D=D, mix=g0 - best[0], pair=best[1:])
        Ts = sorted(rows)
        def vanish(fn):
            val = [fn(rows[T]) for T in Ts]
            for a, b, ta, tb in zip(val[:-1], val[1:], Ts[:-1], Ts[1:]):
                if a > 0 >= b:
                    return ta + (tb - ta) * a / (a - b)
            return float("nan")
        tl = {k: vanish(lambda r, k=k: r["D"][k]) for k in phs}
        res[key] = dict(rows={T: {"D": rows[T]["D"], "mix": rows[T]["mix"], "pair": rows[T]["pair"]} for T in Ts}, Tzero=tl, Tzero_mix=vanish(lambda r: r["mix"]))
    return name, sorted(phs), res


OUT = {}
for f in FILES:
    try:
        n, ph, r = run_one(f)
        OUT[n] = r
        print("\n" + n, "phases:", ph)
        for key in ALLOYS:
            q = r[key]; tz = q["Tzero"]
            print(f"  {key}: T(dG=0) mix {q['Tzero_mix']:.0f} K; phases " + ", ".join(f"{k} {tz[k]:.0f}" for k in tz if tz[k] == tz[k]))
            for T in (850.0, 1000.0):
                x = q["rows"][T]
                top = sorted(x["D"].items(), key=lambda a: -a[1])[:3]
                print(f"     {T:.0f} K: mix ({'+'.join(x['pair'])}) {x['mix'] / 1e3:5.2f} kJ; nuclei: " + ", ".join(f"{k} {val / 1e3:5.2f}" for k, val in top))
    except Exception as e:
        print(os.path.basename(f), "FAILED", repr(e)[:160])
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assessed_phases.json"), "w"), default=float)
