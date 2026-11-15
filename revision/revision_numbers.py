#!/usr/bin/env python3
"""Every number of the November 2026 revision, written to revision_numbers.json. The manuscript, SI and figure
scripts read from that file. Checks at the top reproduce the submitted numbers on the old viscosity first, so a
broken refactor cannot pass."""
import json, os, sys
from dataclasses import replace
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import hyalos_kinetics as h
z = h.z
KEYS = list(h.ALLOYS)
OUT = {}


def old_eta(a, D):
    """Submitted form: eta0 4e-5, eta(Tg) 1e12, strength D*. Returns the arguments kinetics() takes."""
    C = np.log(1e12 / 4e-5); T0 = C * a.Tg / (D + C)
    return dict(eta0=4e-5, B=D * T0, T0=T0)


# ---- planted checks against the submitted Table 2, Table S1 baseline and B.6
OLD_D = {"Cu50Zr50": 13.6, "Cu47Zr45Al8": 20.6, "Cu64Zr36": 13.6}
for k, want in {"Cu50Zr50": 1.34, "Cu47Zr45Al8": 17.2, "Cu64Zr36": 10.9}.items():
    got = h.predict(k, aT=0.50, **old_eta(h.ALLOYS[k], OLD_D[k]))["dc"]
    assert abs(got / want - 1) < 0.01, (k, got, want)
got = h.predict("Cu50Zr50", aT=0.50, dG_at_Tl=40.0, **old_eta(h.ALLOYS["Cu50Zr50"], 13.6))["dc"]
print("B.6 check, old footing: Cu50Zr50 +0.04 kJ:", round(got, 2), "(submitted 1.19)")
got = h.predict("Cu64Zr36", aT=0.50, dG_at_Tl=-590.0, **old_eta(h.ALLOYS["Cu64Zr36"], 13.6))["dc"]
print("B.6 check, old footing: Cu64Zr36 -0.59 kJ:", round(got, 1), "(submitted 112)")

# ---- main table
OUT["alloys"] = {}
for k in KEYS:
    r = h.predict(k); a = h.ALLOYS[k]
    B, T0 = r["B"], r["T0"]
    OUT["alloys"][k] = dict(Tl=a.Tl, Tg=a.Tg, Tg_eta=a.Tg_eta, Tstar=a.Tstar, B=B, T0=T0, thetaD=r["th"], dSf=r["dSf"], dHf=r["dHf"] / 1e3,
                            dEico=r["dE"] / 1e3, sig=r["sig"] * 1e3, Vm=r["Vm"] * 1e6, Tn=r["Tn"], tn=r["tn"], Rc=r["Rc"], dc=r["dc"],
                            dc_meas=a.dc_meas, ratio=r["dc"] / a.dc_meas, dG_nose=r["dG_nose"] / 1e3,
                            eta_nose=h.ETA0 * np.exp(B / (r["Tn"] - T0)),
                            aT_recover=brentq(lambda x: np.log(h.predict(k, aT=x)["dc"] / a.dc_meas), 0.40, 0.8, xtol=1e-5))
    bar = 16 * np.pi * r["sig"] ** 3 / (3 * (r["dG_nose"] / r["Vm"]) ** 2) / (h.kB * r["Tn"])
    OUT["alloys"][k]["barrier_kT"] = bar

# ---- what the configurations contribute: single state, classical forms, more configurations
OUT["config"] = {}
for k in KEYS:
    a = h.ALLOYS[k]; b = h.thermo(a); e = {}
    s1 = h.thermo(a, splits=(0.0,), softs=(0.80,))
    e["two"] = h.kinetics(a, b["dG"], b["dHf"], b["Vm"])["dc"]
    e["single"] = h.kinetics(a, s1["dG"], s1["dHf"], s1["Vm"])["dc"]
    e["single_sameSigma"] = h.kinetics(a, s1["dG"], b["dHf"], s1["Vm"])["dc"]
    e["single_dSf"] = s1["dSf"]; e["two_dSf"] = b["dSf"]; e["single_dHf"] = s1["dHf"] / 1e3
    e["conf_share"] = 100 * (b["dSf"] - s1["dSf"]) / b["dSf"]
    feq = np.exp(-b["dSf"] / (3 * h.Rg))
    se = h.thermo(a, splits=(0.0,), softs=(feq,))
    e["soft_eq"] = feq; e["single_eq"] = h.kinetics(a, se["dG"], se["dHf"], se["Vm"])["dc"]
    for f in (0.75, 0.85):
        sf = h.thermo(a, splits=(0.0,), softs=(f,))
        e[f"single_{f}"] = h.kinetics(a, sf["dG"], sf["dHf"], sf["Vm"])["dc"]
    t3 = h.thermo(a, splits=(0.0, 6e3, 12e3), softs=(0.8,) * 3)
    e["three"] = h.kinetics(a, t3["dG"], t3["dHf"], t3["Vm"])["dc"]; e["three_dSf"] = t3["dSf"]
    TS = lambda T, dHf=b["dHf"], Tl=a.Tl: dHf * (Tl - T) / Tl * 2 * T / (Tl + T)
    lin = lambda T, dSf=b["dSf"], Tl=a.Tl: dSf * (Tl - T)
    rt = h.kinetics(a, TS, b["dHf"], b["Vm"]); rl = h.kinetics(a, lin, b["dHf"], b["Vm"]); r0 = h.kinetics(a, b["dG"], b["dHf"], b["Vm"])
    e["TS"] = rt["dc"]; e["linear"] = rl["dc"]
    e["dG_two_nose"] = b["dG"](r0["Tn"]) / 1e3; e["dG_TS_nose"] = TS(r0["Tn"]) / 1e3; e["dG_lin_nose"] = lin(r0["Tn"]) / 1e3; e["dG_single_nose"] = s1["dG"](r0["Tn"]) / 1e3
    for dEs in (0.7e3, 1.8e3, 12e3, 24e3):
        d = h.thermo(a, splits=(0.0, dEs))
        e[f"split_{dEs/1e3:g}"] = h.kinetics(a, d["dG"], d["dHf"], d["Vm"])["dc"]; e[f"split_{dEs/1e3:g}_dSf"] = d["dSf"]
    # populations and the configurational heat capacity (the energy fluctuation between configurations)
    pops = {}
    for lab, T in (("Tg", a.Tg), ("Tl", a.Tl)):
        _, p, F = h.liquid(T, b["E0"], b["th"], b["dE"], (0.0, 6e3), (0.8, 0.8))
        xx = 6e3 / (h.Rg * T)
        pops[lab] = dict(p1=p[0], p2=p[1], Sconf=-h.Rg * (p * np.log(p)).sum(), Cconf=h.Rg * xx ** 2 * p[0] * p[1])
    e["pops"] = pops
    OUT["config"][k] = e

# ---- two-configuration ceiling on the heat capacity: max over x of x^2 p(1-p) with p = 1/(1+e^-x)
x = np.linspace(0.05, 10, 4000); pp = 1 / (1 + np.exp(-x)); val = x ** 2 * pp * (1 - pp)
OUT["ceiling_R"] = float(val.max()); OUT["ceiling"] = float(val.max() * h.Rg); OUT["ceiling_at_split_over_RT"] = float(x[val.argmax()])
OUT["split_at_ceiling_1000K"] = float(x[val.argmax()] * h.Rg * 1000 / 1e3)

# ---- calorimetry of Castellero and Battezzati 2023 through the same kinetics
Tf, dHm = 1208.0, 9100.0
ca, cb = 55.979 - 25.5, -0.01688 - 0.00517


def dG_cal(T):
    return dHm * (1 - T / Tf) - (ca * (Tf - T) + cb * (Tf ** 2 - T ** 2) / 2) + T * (ca * np.log(Tf / T) + cb * (Tf - T))


a = h.ALLOYS["Cu50Zr50"]; b = h.thermo(a); Vm = b["Vm"]
ctrl = h.kinetics(a, dG_cal, dHm, Vm, sigma=0.1336)
mod = h.kinetics(a, b["dG"], b["dHf"], Vm)
aT_cal = 0.1336 / (dHm / (h.NA ** (1 / 3) * Vm ** (2 / 3)))
barm = mod["sig"] ** 3 / b["dG"](854.0) ** 2; barc = 0.1336 ** 3 / dG_cal(854.0) ** 2
T_list = (700, 800, 854, 900, 1000, 1100)
OUT["calorimetry"] = dict(aT_from_cal=aT_cal, control_dc=ctrl["dc"], control_Tn=ctrl["Tn"], control_Rc=ctrl["Rc"], dSf_meas=dHm / Tf, dHf_meas=dHm / 1e3,
                          dG_model={T: b["dG"](T) / 1e3 for T in T_list}, dG_meas={T: dG_cal(T) / 1e3 for T in T_list},
                          dCp_meas={T: ca + cb * T for T in (700, 1000, 1208)},
                          barrier_ratio_model_over_meas=barm / barc, sig_model=mod["sig"] * 1e3)

# ---- Cu64Zr36 limitation: size of the gap
g = lambda at: np.log(h.predict("Cu64Zr36", aT=at)["dc"] / 2.0)
at64 = brentq(g, 0.45, 0.62, xtol=1e-5)
f_theta = lambda th: (2 + np.cos(np.radians(th))) * (1 - np.cos(np.radians(th))) ** 2 / 4
sig0 = h.predict("Cu64Zr36")["sig"]
fneed = brentq(lambda f: np.log(h.predict("Cu64Zr36", sigma=sig0 * f ** (1 / 3))["dc"] / 2.0), 0.5, 1.0, xtol=1e-4)
OUT["cu64"] = dict(aT_needed=at64, sigma_pct=100 * (at64 / h.ALPHA_T - 1), sig_now=sig0 * 1e3, sig_needed=sig0 * 1e3 * at64 / h.ALPHA_T,
                   f_barrier_only=fneed, theta_barrier_only=brentq(lambda th: f_theta(th) - fneed, 1, 180))

# ---- soft1 and split ranges (Appendix A), other alloys
OUT["soft1_range"] = {}
for k in KEYS:
    OUT["soft1_range"][k] = {f: h.predict(k, softs=(f, 0.80))["dc"] for f in (0.75, 0.80, 0.85)}
OUT["softboth_075"] = {k: h.predict(k, softs=(0.75, 0.75))["dc"] for k in KEYS}
OUT["split_range"] = {k: {s: h.predict(k, splits=(0.0, s * 1e3))["dc"] for s in (0.7, 1.8, 6.0)} for k in KEYS}

# ---- sensitivity, Cu47Zr45Al8, +-20 %
k = "Cu47Zr45Al8"; a = h.ALLOYS[k]; B0, T00 = h.vft_anchored(a.Tg_eta, a.Tstar)
base = h.predict(k)["dc"]
cases = {
    "aT": lambda s: h.predict(k, aT=h.ALPHA_T * s),
    "soft2": lambda s: h.predict(k, softs=(0.8, 0.8 * s)),
    "soft1": lambda s: h.predict(k, softs=(0.8 * s, 0.8)),
    "B": lambda s: h.predict(k, B=B0 * s, T0=T00),
    "lam": lambda s: h.predict(k, lam=h.LAM * s),
    "eta0": lambda s: h.predict(k, eta0=h.ETA0 * s),
    "x": lambda s: h.predict(k, x_det=h.X_DET * s),
    "Vm": lambda s: h.predict(k, Vm_scale=s),
    "split": lambda s: h.predict(k, splits=(0.0, h.SPLIT * s)),
    "B0": lambda s: h.predict(k, B_scale=s),
}
OUT["sensitivity"] = dict(base=base, rows=sorted(({"key": n, "lo": f(0.8)["dc"], "hi": f(1.2)["dc"]} for n, f in cases.items()),
                                                 key=lambda r: -abs(r["hi"] - r["lo"])))
OUT["eta0_range"] = {k2: {e: h.predict(k2, eta0=e)["dc"] for e in (1.78e-4, 2.3e-4, 6e-4)} for k2 in KEYS}
OUT["aT_050"] = {k2: h.predict(k2, aT=0.50)["dc"] for k2 in KEYS}
OUT["aT_pm"] = {k2: {at: h.predict(k2, aT=at)["dc"] for at in (0.58, 0.597, 0.61)} for k2 in KEYS}

# ---- Appendix B.6 on the new footing
DS = {"Cu50Zr50": (40.0, 8.00), "Cu64Zr36": (-590.0, 9.66)}
OUT["B6"] = {}
for k, (dG0, dS) in DS.items():
    a = h.ALLOYS[k]; rr = {}
    rr["cal"] = h.predict(k)["dc"]
    rr["calc_dG"] = h.predict(k, dG_at_Tl=dG0)["dc"]
    rr["lo"] = h.predict(k, dG_at_Tl=dG0 - 300)["dc"]; rr["hi"] = h.predict(k, dG_at_Tl=dG0 + 300)["dc"]
    th = h.thermo(a)
    fn = lambda T, dG0=dG0, dS=dS, Tl=a.Tl: dG0 + dS * (Tl - T)
    rr["calc_dG_dS"] = h.kinetics(a, fn, a.Tl * dS, th["Vm"])["dc"]
    # how well dG(Tl) must be known for a factor of two
    fac = {}
    for d in (100, 150, 200, 250, 300):
        fac[d] = (h.predict(k, dG_at_Tl=d)["dc"] / rr["cal"], h.predict(k, dG_at_Tl=-d)["dc"] / rr["cal"])
    rr["factor_vs_error_J"] = fac
    OUT["B6"][k] = rr

# ---- Zr-rich test; no anchor exists for this alloy, so T* is bracketed by the Cu-Zr-Al anchors
Z = h.ZR_RICH
OUT["zr"] = {}
for Ts in (1030, 1064, 1112):
    a = h.Alloy("Zr50Cu40Al10", Z["x"], Z["lo"], Z["hi"], Z["Tl"], Z["Tg"], Z["Tg"], Ts, 10.0)
    th = h.thermo(a); r = h.kinetics(a, th["dG"], th["dHf"], th["Vm"])
    OUT["zr"][Ts] = dict(thetaD=th["th"], dSf=th["dSf"], dHf=th["dHf"] / 1e3, dEico=th["dE"] / 1e3, sig=r["sig"] * 1e3, Vm=th["Vm"] * 1e6, Tn=r["Tn"], tn=r["tn"], Rc=r["Rc"], dc=r["dc"], B=r["B"], T0=r["T0"])
a = h.Alloy("Zr50Cu40Al10", Z["x"], Z["lo"], Z["hi"], Z["Tl"], Z["Tg"], Z["Tg"], 1064, 10.0)
save = {n: z.PHASES[n]["Hf"] for n in ("CuZr", "CuZr2")}
OUT["zr_shift"] = {}
for sh in (0, -3e3, -6e3, 3e3):
    for n in save:
        z.PHASES[n]["Hf"] = save[n] + sh
    th = h.thermo(a); OUT["zr_shift"][sh / 1e3] = h.kinetics(a, th["dG"], th["dHf"], th["Vm"])["dc"]
for n in save:
    z.PHASES[n]["Hf"] = save[n]
gv = {}
allz = dict(h.ALLOYS); allz["Zr50Cu40Al10"] = a
for n, al in allz.items():
    th = h.thermo(al); gv[n] = {d: th["dG"](al.Tl - d) / th["Vm"] / 1e6 for d in (150, 250, 350)}
OUT["zr_dGv"] = gv
OUT["viscosity_old_vs_new"] = {}
for k in KEYS:
    a = h.ALLOYS[k]; e = old_eta(a, OLD_D[k]); r = h.predict(k)
    OUT["viscosity_old_vs_new"][k] = dict(ratio_nose=float(e["eta0"] * np.exp(e["B"] / (r["Tn"] - e["T0"])) / (h.ETA0 * np.exp(r["B"] / (r["Tn"] - r["T0"])))))

json.dump(OUT, open(os.path.join(HERE, "revision_numbers.json"), "w"), indent=1, default=float)
pr = lambda *a: print(*a)
pr("\nMAIN"); [pr(k, {m: (round(v, 3) if isinstance(v, float) else v) for m, v in OUT["alloys"][k].items()}) for k in KEYS]
pr("\nCONFIG"); [pr(k, {m: (round(v, 3) if isinstance(v, float) else v) for m, v in OUT["config"][k].items() if m != "pops"}, OUT["config"][k]["pops"]) for k in KEYS]
for s in ("ceiling_R", "ceiling", "ceiling_at_split_over_RT", "split_at_ceiling_1000K", "calorimetry", "cu64", "soft1_range", "softboth_075", "split_range", "sensitivity", "eta0_range", "aT_050", "aT_pm", "B6", "zr", "zr_shift", "zr_dGv", "viscosity_old_vs_new"):
    pr("\n" + s, json.dumps(OUT[s], default=float)[:1400])
