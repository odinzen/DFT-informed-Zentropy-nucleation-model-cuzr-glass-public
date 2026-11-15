"""Cu50Zr50: the calorimetric thermodynamics of Castellero and Battezzati (JNCS 610, 2023, 122311) through
the paper's nucleation model. dHf = 9100 J/mol at 1208 K, liquid Cp linear fit (their Eq. 5); crystal Cp read
from their Fig. 3 (28.6 at 600 K, 31.7 at 1200 K)."""
import sys
import numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import single_state_test as t
z = t.z; kB = t.kB; Rg = t.Rg; NA = t.NA

Tf, dHf = 1208.0, 9100.0
a, b = 55.979 - 25.5, -0.01688 - 0.00517        # dCp = a + b T


def dG_meas(T):
    I1 = a * (Tf - T) + b * (Tf ** 2 - T ** 2) / 2
    I2 = a * np.log(Tf / T) + b * (Tf - T)
    return dHf * (1 - T / Tf) - I1 + T * I2


def run(dG, sig_mJ, key="Cu50Zr50"):
    x, lo, hi, Tl, Tg, D, _ = t.ALLOYS[key]
    Vm = z.comp_props(x)[0] * NA; sig = sig_mJ * 1e-3
    T0 = t.Cvft * Tg / (D + t.Cvft); Ts = np.linspace(Tg + 5, Tl - 3, 1400); out = np.empty_like(Ts)
    with np.errstate(over="ignore", divide="ignore"):
        for i, Ti in enumerate(Ts):
            e = t.eta0 * np.exp(D * T0 / (Ti - T0)); g = max(dG(Ti), 1.0); gv = g / Vm
            bar = 16 * np.pi * sig ** 3 / (3 * gv ** 2)
            I = (NA / Vm) * (kB * Ti / (3 * np.pi * t.lam ** 3 * e)) * np.exp(-bar / (kB * Ti))
            U = max((kB * Ti / (3 * np.pi * t.lam ** 2 * e)) * (1 - np.exp(-g / (Rg * Ti))), 1e-30)
            out[i] = (3 * t.x_det / (np.pi * max(I, 1e-300) * U ** 3)) ** 0.25
    j = int(np.argmin(out)); Rc = (Tl - Ts[j]) / out[j]
    return Ts[j], out[j], Rc, 10 * np.sqrt(10 / Rc), dG(Ts[j])


base = t.predict("Cu50Zr50")
print("planted check: paper dG, paper sigma ->", [round(float(v), 3) for v in run(base["dGfn"], base["sig"])])
print("\nDriving force kJ/mol        model   calorimetry  ratio")
for T in (700, 800, 854, 900, 1000, 1100):
    m, c = base["dGfn"](T) / 1e3, dG_meas(T) / 1e3
    print(f"  {T:5d} K              {m:6.2f}   {c:6.2f}      {m / c:5.2f}")
print("\ndCp calorimetry J/mol/K at 700, 1000, 1208 K:", [round(a + b * T, 1) for T in (700, 1000, 1208)])
sigT = base["sig"] * dHf / base["dHf"]
print(f"\nTurnbull sigma with measured dHf (aT 0.50): {sigT:.1f} mJ/m2; model {base['sig']:.1f}; Castellero fit 133.6; aT that gives 133.6: {0.5 * 133.6 / sigT:.3f}")
for name, dG, s in [("paper model", base["dGfn"], base["sig"]),
                    ("calorimetric dG, Turnbull sigma aT=0.50", dG_meas, sigT),
                    ("calorimetric dG, sigma 133.6 (their fit)", dG_meas, 133.6),
                    ("calorimetric dG, paper sigma 162.8", dG_meas, base["sig"]),
                    ("paper dG, Turnbull sigma on measured dHf", base["dGfn"], sigT)]:
    Tn, tn, Rc, dc, g = run(dG, s)
    print(f"  {name:44s} Tnose {Tn:5.0f}  dG {g / 1e3:5.2f}  Rc {Rc:10.3g}  dc {dc:9.3g} mm")
print("\nTwo-configuration variants that land on the measured dSf = 7.53 J/mol/K")
for f in (0.93, 0.95):
    r = t.predict("Cu50Zr50", (0.0, 1.8e3), (f, f))
    t.row(f"split 1.8 kJ, softening {f}", r)
for key in t.ALLOYS:
    r = t.predict(key, (0.0, 1.8e3), (0.93, 0.93))
    print(f"  {key:12s} split 1.8, softening 0.93: dSf {r['dSf']:.2f}  dc {r['dc']:.3g} mm  (measured {t.ALLOYS[key][6]})")
