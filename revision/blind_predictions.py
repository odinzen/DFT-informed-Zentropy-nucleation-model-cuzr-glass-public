#!/usr/bin/env python3
"""Retrospective check and blind predictions, written to blind_predictions.json. Inputs are Tg, Tl, T* of Gangopadhyay
et al. 2017 Table 1 for every alloy, with the method unchanged. The rival is one straight line of log10(dc) on T*/Tl,
fitted to the retrospective alloys, so the method has to beat a single measured ratio."""
import json, os, sys
from dataclasses import replace
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import hyalos_kinetics as h

# name: Tg, Tl, T*
IN = {"Cu64Zr36": (733, 1200, 1090), "Cu50Zr50": (675, 1222, 1064), "Cu46Zr54": (633, 1198, 1030), "Cu50Zr45Al5": (687, 1173, 1092), "Cu49Zr45Al6": (705, 1177, 1085),
      "Cu47Zr47Al6": (697, 1172, 1110), "Cu47Zr45Al8": (681, 1163, 1112), "Cu50Zr40Ti10": (661, 1168, 1075), "Cu50Zr42.5Ti7.5": (669, 1152, 1031), "Cu60Zr20Ti20": (693, 1127, 1068),
      "Ti40Zr10Cu36Pd14": (675, 1185, 1060), "Ti40Zr10Cu30Pd20": (690, 1189, 1083), "Zr60Ni25Al15": (692, 1248, 1202), "Zr64Ni25Al11": (669, 1212, 1176), "Ti51Cu36Ni4Al9": (697, 1157, 1129),
      "Zr62Ni8Cu20Al10": (655, 1152, 1134), "Zr56Co28Al16": (730, 1241, 1210), "Ni59.5Nb40.5": (891, 1448, 1414)}
RETRO = {"Cu50Zr50": 2.0, "Cu47Zr45Al8": 15.0, "Cu64Zr36": 2.0, "Cu46Zr54": 2.0, "Cu50Zr42.5Ti7.5": 5.0, "Cu49Zr45Al6": 12.0, "Zr56Co28Al16": 18.0, "Ti40Zr10Cu36Pd14": 6.0}
BLIND = ["Cu47Zr47Al6", "Cu50Zr45Al5", "Cu50Zr40Ti10", "Cu60Zr20Ti20", "Ti40Zr10Cu30Pd20", "Zr60Ni25Al15", "Zr64Ni25Al11", "Zr62Ni8Cu20Al10", "Ti51Cu36Ni4Al9", "Ni59.5Nb40.5"]
BASE = h.ALLOYS["Cu50Zr50"]     # the barrier does not depend on the reference phases or molar volume


def calc(tg, tl, ts, **kw):
    a = replace(BASE, Tl=tl, Tg=tg, Tg_eta=tg, Tstar=ts)
    sf = kw.pop("soft", 0.8)
    t = h.thermo(a, softs=(sf, sf))
    return h.kinetics(a, t["dG"], t["dHf"], t["Vm"], n=700, **kw)


OUT = {"inputs": IN, "retro": {}, "blind": {}}
rng = np.random.default_rng(11)
N = 250
draws = [dict(sf=0.80 + rng.normal(0, 0.025), at=h.ALPHA_T + rng.normal(0, 0.01), e0=h.ETA0 * np.exp(rng.normal(0, 0.4))) for _ in range(N)]
for name, (tg, tl, ts) in IN.items():
    r = calc(tg, tl, ts)
    v = []
    for d in draws:
        v.append(calc(tg + rng.normal(0, 5), tl + rng.normal(0, 10), ts + rng.normal(0, 10), soft=d["sf"], aT=d["at"], eta0=d["e0"])["dc"])
    q = np.percentile(v, [16, 84])
    OUT.setdefault("all", {})[name] = dict(dc=r["dc"], Rc=r["Rc"], Tn=r["Tn"], lo=float(q[0]), hi=float(q[1]), ratio=ts / tl, trg=tg / tl, T0=r["T0"], B=r["B"])

A = OUT["all"]
x = np.array([A[k]["ratio"] for k in RETRO]); m = np.log10(list(RETRO.values())); p = np.log10([A[k]["dc"] for k in RETRO])
c = np.polyfit(x, m, 1)
OUT["rival"] = dict(slope=float(c[0]), intercept=float(c[1]))
e = p - m
loo = []
for i in range(len(x)):
    kk = np.arange(len(x)) != i
    cc = np.polyfit(x[kk], m[kk], 1); loo.append(np.polyval(cc, x[i]) - m[i])
cn = np.polyfit(np.array([A[k]["trg"] for k in RETRO]), m, 1)
loo2 = []
for i in range(len(x)):
    kk = np.arange(len(x)) != i
    tr = np.array([A[k]["trg"] for k in RETRO]); cc = np.polyfit(tr[kk], m[kk], 1); loo2.append(np.polyval(cc, tr[i]) - m[i])
OUT["stats"] = dict(n=len(x), rms=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()), within2=int((abs(e) < np.log10(2)).sum()), within3=int((abs(e) < np.log10(3)).sum()),
                    rank_model=float(spearmanr(p, m)[0]), rank_ratio=float(spearmanr(x, m)[0]), rival_loo_rms=float(np.sqrt(np.mean(np.square(loo)))), trg_loo_rms=float(np.sqrt(np.mean(np.square(loo2)))), spread_rms=float(m.std()),
                    rms_no64=float(np.sqrt((e[[i for i, k in enumerate(RETRO) if k != "Cu64Zr36"]] ** 2).mean())))
print("RETROSPECTIVE  (name, Tg/Tl, T*/Tl, dc, 68%, measured, rival leave-one-out)")
for i, k in enumerate(RETRO):
    a = A[k]; OUT["retro"][k] = {**a, "meas": RETRO[k], "rival": float(10 ** (loo[i] + m[i]))}
    print(f"  {k:20s} {a['trg']:.3f} {a['ratio']:.3f} {a['dc']:7.2f} ({a['lo']:6.2f}-{a['hi']:7.2f})  meas {RETRO[k]:5.1f}  rival {10 ** (loo[i] + m[i]):6.2f}  in band {a['lo'] <= RETRO[k] <= a['hi']}")
print("stats", {k: round(v, 3) for k, v in OUT["stats"].items()}, "rival", OUT["rival"])
print("BLIND (name, dc, 68%, rival)")
for k in BLIND:
    a = A[k]; riv = 10 ** np.polyval(c, a["ratio"]); OUT["blind"][k] = {**a, "rival": float(riv)}
    print(f"  {k:20s} Tg {IN[k][0]} Tl {IN[k][1]} T* {IN[k][2]} dc {a['dc']:7.2f} ({a['lo']:6.2f}-{a['hi']:7.2f}) Rc {a['Rc']:8.3g} rival {riv:6.2f}")
json.dump(OUT, open(os.path.join(HERE, "blind_predictions.json"), "w"), indent=1, default=float)
