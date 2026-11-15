#!/usr/bin/env python3
"""Result figures of the November 2026 revision, from hyalos_kinetics and revision_numbers.json. Nothing is baked
on the images, values live in the captions. Printed values let each caption be checked against the plot."""
import json, os, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import hyalos_kinetics as h
OUT = os.path.join(HERE, "figures_out") + os.sep
os.makedirs(OUT, exist_ok=True)
N = json.load(open(os.path.join(HERE, "revision_numbers.json")))

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 1.0, "axes.edgecolor": "black",
                     "savefig.dpi": 600, "figure.dpi": 150, "legend.frameon": False, "mathtext.default": "regular",
                     "xtick.direction": "out", "ytick.direction": "out"})
BLUE, ORANGE, AQUA, PLUM, BROWN, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#a3449e", "#7f3b08", "#52514e"
COL = {"Cu50Zr50": BLUE, "Cu47Zr45Al8": ORANGE, "Cu64Zr36": AQUA}
LAB = {"Cu50Zr50": "Cu$_{50}$Zr$_{50}$", "Cu47Zr45Al8": "Cu$_{47}$Zr$_{45}$Al$_{8}$", "Cu64Zr36": "Cu$_{64}$Zr$_{36}$"}
KEYS = list(h.ALLOYS)


def fig_validation():
    fig, ax = plt.subplots(figsize=(4.6, 4.3)); lim = [0.5, 50]
    ax.fill_between(lim, [x / 3 for x in lim], [x * 3 for x in lim], color="#e4e4e4", zorder=0)
    ax.plot(lim, lim, color=NEUT, lw=1.0, ls=(0, (4, 2)))
    style = {"Cu50Zr50": dict(marker="D", facecolor="white", edgecolor=BLUE, label=LAB["Cu50Zr50"]),
             "Cu47Zr45Al8": dict(marker="o", facecolor=ORANGE, edgecolor=ORANGE, label=LAB["Cu47Zr45Al8"]),
             "Cu64Zr36": dict(marker="s", facecolor=AQUA, edgecolor=AQUA, label=LAB["Cu64Zr36"])}
    for k in KEYS:
        a = N["alloys"][k]
        ax.scatter([a["dc_meas"]], [a["dc"]], s=80, linewidths=1.6, zorder=5, **style[k])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xticks([1, 10]); ax.set_yticks([1, 10]); ax.set_xticklabels(["1", "10"]); ax.set_yticklabels(["1", "10"])
    ax.set_xlabel("Measured critical diameter, mm"); ax.set_ylabel("Predicted critical diameter, mm")
    ax.legend(loc="upper left", fontsize=10, handletextpad=0.3)
    fig.tight_layout(pad=0.4); fig.savefig(OUT + "Figure_4_validation_corrected.png"); plt.close(fig)


def fig_ttt():
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for k, ls in zip(KEYS, ("-", (0, (6, 2)), (0, (2, 1.5)))):
        r = h.predict(k)
        ax.plot(r["t"], r["T"], color=COL[k], lw=2.0, ls=ls, label=LAB[k])
        ax.scatter([r["tn"]], [r["Tn"]], s=46, color=COL[k], edgecolor="white", linewidths=0.8, zorder=6)
        print(f"TTT {k}: nose {r['Tn']:.0f} K {r['tn']:.3g} s Rc {r['Rc']:.3g} dc {r['dc']:.2f}")
    ax.set_xscale("log"); ax.set_xlim(1e-1, 1e6); ax.set_ylim(660, 1240)
    ax.set_xlabel("Time, s"); ax.set_ylabel("Temperature, K"); ax.legend(loc="upper right")
    fig.tight_layout(pad=0.4); fig.savefig(OUT + "Figure_4_TTT.png"); plt.close(fig)


def dG_cal(T):
    a, b = 55.979 - 25.5, -0.01688 - 0.00517; Tf, dH = 1208.0, 9100.0
    return dH * (1 - T / Tf) - (a * (Tf - T) + b * (Tf ** 2 - T ** 2) / 2) + T * (a * np.log(Tf / T) + b * (Tf - T))


def fig_driving_force():
    a = h.ALLOYS["Cu50Zr50"]; b = h.thermo(a); s = h.thermo(a, splits=(0.0,), softs=(0.8,))
    T = np.linspace(a.Tg, a.Tl, 300); Tl = a.Tl
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(T, [b["dG"](x) / 1e3 for x in T], color=BLUE, lw=2.2, label="Two configurations (Zentropy)")
    ax.plot(T, [s["dG"](x) / 1e3 for x in T], color=ORANGE, lw=2.0, ls=(0, (6, 2)), label="Single state, softening 0.80")
    ax.plot(T, b["dHf"] * (Tl - T) / Tl * 2 * T / (Tl + T) / 1e3, color=PLUM, lw=1.6, ls=(0, (3, 1.5)), label="Thompson-Spaepen form")
    ax.plot(T, b["dSf"] * (Tl - T) / 1e3, color=NEUT, lw=1.4, ls=(0, (1, 1.5)), label=r"Linear, $\Delta S_f(T_l - T)$")
    ax.plot(T, dG_cal(T) / 1e3, color=AQUA, lw=2.2, ls="-.", label="Calorimetry")
    ax.set_xlim(a.Tg, Tl); ax.set_ylim(0, 6.6)
    ax.set_xlabel("Temperature, K"); ax.set_ylabel(r"Driving force $\Delta G$, kJ mol$^{-1}$")
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.4); fig.savefig(OUT + "Figure_5_driving_force.png"); plt.close(fig)
    for x in (800, 810, 854, 900):
        ratio = (b["dG"](x) / dG_cal(x)); br = (N["alloys"]["Cu50Zr50"]["sig"] / 133.6) ** 3 / ratio ** 2
        print(f"dG {x} K model {b['dG'](x)/1e3:.2f} cal {dG_cal(x)/1e3:.2f} TS {b['dHf']*(Tl-x)/Tl*2*x/(Tl+x)/1e3:.2f} barrier model/cal {br:.3f}")


def fig_populations():
    a = h.ALLOYS["Cu50Zr50"]; b = h.thermo(a); T = np.linspace(a.Tg, a.Tl, 300)
    P = np.array([h.liquid(x, b["E0"], b["th"], b["dE"], (0.0, 6e3), (0.8, 0.8))[1] for x in T])
    Sc = -h.Rg * (P * np.log(P)).sum(axis=1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 3.3))
    a1.plot(T, P[:, 0], color=BLUE, lw=2.2, label="Icosahedral, k = 1")
    a1.plot(T, P[:, 1], color=ORANGE, lw=2.2, ls=(0, (6, 2)), label="Defective, k = 2")
    a1.set_ylim(0, 1); a1.set_xlim(a.Tg, a.Tl); a1.set_xlabel("Temperature, K"); a1.set_ylabel("Configuration population $p^{(k)}$")
    a1.legend(loc="center right", fontsize=9, bbox_to_anchor=(1.0, 0.50))
    a2.plot(T, Sc, color=PLUM, lw=2.2)
    a2.axhline(h.Rg * np.log(2), color=NEUT, lw=1.1, ls=(0, (1, 1.5)))
    a2.text(a.Tg + 15, h.Rg * np.log(2) - 0.05, "R ln 2", fontsize=9, va="top", color="black")
    a2.set_xlim(a.Tg, a.Tl); a2.set_ylim(4, 6)
    a2.set_xlabel("Temperature, K"); a2.set_ylabel(r"$S_{conf}$, J mol$^{-1}$ K$^{-1}$")
    for ax, lab in ((a1, "(a)"), (a2, "(b)")):
        ax.text(-0.02, 1.03, lab, transform=ax.transAxes, fontsize=10, va="bottom", ha="left")
    fig.tight_layout(pad=0.4, w_pad=1.2); fig.savefig(OUT + "Figure_6_zentropy_crossover.png"); plt.close(fig)
    print(f"pops Tg {P[0]} Tl {P[-1]}  Sconf {Sc[0]:.2f}->{Sc[-1]:.2f}")


def fig_heat_capacity():
    a = h.ALLOYS["Cu50Zr50"]; b = h.thermo(a); T = np.linspace(a.Tg + 2, a.Tl - 2, 260)
    d = 4.0; dG = b["dG"]
    dCp = np.array([-x * (dG(x + d) - 2 * dG(x) + dG(x - d)) / d ** 2 for x in T])
    P = np.array([h.liquid(x, b["E0"], b["th"], b["dE"], (0.0, 6e3), (0.8, 0.8))[1] for x in T])
    xx = 6e3 / (h.Rg * T); Cc = h.Rg * xx ** 2 * P[:, 0] * P[:, 1]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(T, 55.979 - 25.5 + (-0.01688 - 0.00517) * T, color=AQUA, lw=2.2, ls="-.", label="Calorimetry")
    ax.plot(T, dCp, color=BLUE, lw=2.2, label="Two configurations, total")
    ax.plot(T, Cc, color=ORANGE, lw=1.8, ls=(0, (6, 2)), label="Two configurations, configurational part")
    ax.axhline(N["ceiling"], color=NEUT, lw=1.2, ls=(0, (1, 1.5)), label="Two-state maximum, 0.44R")
    ax.errorbar([1208], [7.9], yerr=[2.2], fmt="s", color=PLUM, ms=6, capsize=3, label="MACE-MP-0 simulation", zorder=6)
    ax.set_xlim(a.Tg, 1240); ax.set_ylim(0, 16.5)
    ax.set_xlabel("Temperature, K"); ax.set_ylabel(r"$\Delta C_p$, J mol$^{-1}$ K$^{-1}$"); ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(pad=0.4); fig.savefig(OUT + "Figure_8_heat_capacity.png"); plt.close(fig)
    print(f"dCp model {dCp[0]:.2f} at {T[0]:.0f}, {dCp[-1]:.2f} at {T[-1]:.0f}; Cconf {Cc[0]:.2f} -> {Cc[-1]:.2f}")


def fig_sensitivity():
    S = N["sensitivity"]; names = {"aT": r"Turnbull coefficient $\alpha_T$", "soft2": "Softening factor soft2", "soft1": "Softening factor soft1",
                                   "B": "Vogel-Fulcher B", "lam": r"Jump distance $\lambda$", "eta0": r"Viscosity prefactor $\eta_0$", "x": "Detectable fraction x",
                                   "Vm": r"Molar volume $V_m$", "split": r"Configuration split $\delta E$", "B0": r"Bulk modulus $B_0$ (DFT)"}
    rows = S["rows"][::-1]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, r in enumerate(rows):
        a, b = sorted((r["lo"], r["hi"]))
        ax.barh(i, b - a, left=a, height=0.62, color="#c9dcf3", edgecolor="black", lw=0.9)
    ax.axvspan(5, 45, color="#e4e4e4", zorder=0)
    ax.axvline(S["base"], color="black", lw=1.4, ls=(0, (5, 2)))
    ax.axvline(15, color=ORANGE, lw=1.4, ls=(0, (1, 1.5)))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([names[r["key"]] for r in rows], fontsize=9.5)
    ax.set_xscale("log"); ax.set_xlim(0.08, 1e4); ax.set_xlabel("Predicted critical diameter, mm")
    fig.tight_layout(pad=0.4); fig.savefig(OUT + "Figure_S1_sensitivity.png"); plt.close(fig)


def fig_hold_time():
    sys.path.insert(0, HERE)
    import levitation_rate_test as lv
    import oxygen_consistent as oc
    from scipy.optimize import brentq
    al = oc.Alloy("Cu64Zr36"); K = lv.KANG; slope, ic = np.polyfit(K[:, 1], np.log(K[:, 0]), 1); TL = lv.TL
    tr = np.linspace(0.825, 0.885, 120)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.scatter(K[:, 0], K[:, 1], s=46, color=AQUA, edgecolor="black", linewidths=0.6, zorder=5, label="Measured (Kang et al.)")
    ax.plot(np.exp(ic + slope * tr), tr, color=AQUA, lw=1.5, label=f"Fit to the measured times")
    nat = oc.NA / al.Vm * oc.V_DROP
    tgt = ic + slope * 0.855
    for lab, xs, c, ls in (("Single angle, every atom a site", 1.0, BLUE, (0, (6, 2))), ("Single angle, one site in the drop", 1.0 / nat, ORANGE, (0, (2, 1.5)))):
        f = brentq(lambda f: -np.log(al.rate(0.855 * TL, f, xs) * oc.V_DROP) - tgt, 1e-4, 0.999)
        ax.plot([np.exp(-np.log(al.rate(x * TL, f, xs) * oc.V_DROP)) for x in tr], tr, color=c, lw=1.9, ls=ls, label=lab)
    ax.set_xscale("log"); ax.set_xlim(1, 3e4); ax.set_ylim(0.825, 0.885)
    ax.set_xlabel("Time to crystallization, s"); ax.set_ylabel(r"$T/T_l$"); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(pad=0.4); fig.savefig(OUT + "Figure_S2_hold_time.png"); plt.close(fig)
    print("hold-time slope", round(slope, 1))


fig_validation(); fig_ttt(); fig_driving_force(); fig_populations(); fig_heat_capacity(); fig_sensitivity(); fig_hold_time()
print("sensitivity", [(r["key"], round(r["lo"], 2), round(r["hi"], 2)) for r in N["sensitivity"]["rows"]])
