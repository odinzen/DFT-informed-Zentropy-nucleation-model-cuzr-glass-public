#!/usr/bin/env python3
"""Nucleation TTT driven by the DFT informed Zentropy driving force.
Replaces the Thompson Spaepen term of the earlier model with dG(T) from
zentropy_driving_force.py. Regenerates Fig 4 (TTT), Fig 5 (validation),
and adds Fig 6 (driving force comparison)."""
import numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
import zentropy_driving_force as z

mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.linewidth":1.1,
    "axes.edgecolor":"black","savefig.dpi":300,"figure.dpi":150})
OUT="./figures/"
import os; os.makedirs(OUT, exist_ok=True)
GR=dict(d="#2b2b2b",m="#6e6e6e",l="#a9a9a9",vl="#d9d9d9")
def box(ax):
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(1.1); s.set_color("black")
    ax.tick_params(direction="out",length=4,width=1.0,colors="black"); ax.set_facecolor("white")

kB=1.380649e-23; Rg=8.314462; NA=6.02214076e23
eta0=4e-5; lam=2.8e-10; x_det=1e-6; Cvft=np.log(1e12/eta0)
def visc(T,Tg,D): T0=Cvft*Tg/(D+Cvft); return eta0*np.exp(D*T0/(T-T0))

# alloy table: Tg, Tl, D*, xZr, (lo,hi crystal bounds), alphaT, dc_meas, anomaly
ALLOYS = {
 "Cu50Zr50":   dict(Tg=673,Tl=1208,D=13.6,x=0.50,bnd=("Cu10Zr7","CuZr"),aT=0.50,dc=2.0,anom=False,lab=r"Cu$_{50}$Zr$_{50}$"),
 "Cu47Zr45Al8":dict(Tg=681,Tl=1163,D=20.6,x=0.38,bnd=("Cu8Zr3","Cu10Zr7"),aT=0.50,dc=15.0,anom=False,lab=r"Cu$_{47}$Zr$_{45}$Al$_{8}$"),
 "Cu64Zr36":   dict(Tg=745,Tl=1230,D=13.6,x=0.36,bnd=("Cu8Zr3","Cu10Zr7"),aT=0.50,dc=2.0,anom=True,lab=r"Cu$_{64}$Zr$_{36}$"),  # uniform aT=0.50 (no per-alloy tuning); dc_pred 7.3 mm
}

def build(a):
    dG, Vat, dE = z.make_dG_callable(a["Tl"], a["x"], *a["bnd"])
    Vm = Vat*NA                                   # molar volume m^3/mol
    dSf = -(dG(a["Tl"]+1)-dG(a["Tl"]-1))/2.0
    dHf = a["Tl"]*dSf                              # DFT informed fusion enthalpy J/mol
    sig = a["aT"]*dHf/(NA**(1/3)*Vm**(2/3))        # Turnbull interfacial energy
    return dG, Vm, dHf, sig, dE, dSf

def ttt(a):
    dG,Vm,dHf,sig,dE,dSf = build(a)
    T=np.linspace(a["Tg"]+5,a["Tl"]-3,1400); t=np.empty_like(T)
    for i,Ti in enumerate(T):
        e=visc(Ti,a["Tg"],a["D"]); g=max(dG(Ti),1.0); gv=g/Vm
        dGs=16*np.pi*sig**3/(3*gv**2)
        I=(NA/Vm)*(kB*Ti/(3*np.pi*lam**3*e))*np.exp(-dGs/(kB*Ti))
        U=max((kB*Ti/(3*np.pi*lam**2*e))*(1-np.exp(-g/(Rg*Ti))),1e-30)
        t[i]=(3*x_det/(np.pi*max(I,1e-300)*U**3))**0.25
    j=np.argmin(t); Tn,tn=T[j],t[j]
    Rc=(a["Tl"]-Tn)/tn; dc=10*np.sqrt(10/Rc)
    return dict(T=T,t=t,Tn=Tn,tn=tn,Rc=Rc,dc=dc,sig=sig,dHf=dHf,dE=dE,dSf=dSf,Vm=Vm)

res={k:ttt(v) for k,v in ALLOYS.items()}

# ---------------- FIG 4 : TTT (binary vs ternary) ----------------
def fig4():
    fig,ax=plt.subplots(figsize=(8.2,5.6))
    fig.suptitle("NUCLEATION TTT DRIVEN BY DFT INFORMED ZENTROPY FREE ENERGY",y=0.975,fontsize=11,fontweight="bold")
    box(ax)
    for key,ls in [("Cu50Zr50","-"),("Cu47Zr45Al8",(0,(5,2)))]:
        r=res[key]; a=ALLOYS[key]
        ax.plot(r["t"],r["T"],color="black",lw=1.9,ls=ls,label=a["lab"]+(" BINARY" if key=="Cu50Zr50" else " TERNARY"))
        ax.scatter([r["tn"]],[r["Tn"]],s=46,color="black",zorder=6)
        ax.plot([1e-3,r["tn"]],[a["Tl"],r["Tn"]],color=GR["m"],lw=1.0,ls=(0,(1,1.8)))
        ax.annotate(f"nose {r['Tn']:.0f} K, {r['tn']:.2g} s",(r["tn"],r["Tn"]),xytext=(r["tn"]*3,r["Tn"]-32),
                    fontsize=8,arrowprops=dict(arrowstyle="->",lw=0.9,color="black"))
    ax.axhline(ALLOYS["Cu50Zr50"]["Tl"],color=GR["l"],lw=0.8); ax.axhline(ALLOYS["Cu47Zr45Al8"]["Tl"],color=GR["l"],lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(1e-3,1e5); ax.set_ylim(640,1260)
    ax.set_xlabel("Time, s"); ax.set_ylabel("Temperature, K")
    ax.legend(loc="upper right",frameon=True,edgecolor="black",fontsize=9)
    b=res["Cu50Zr50"]; tn=res["Cu47Zr45Al8"]
    ax.text(0.025,0.04,f"BINARY   R$_c$={b['Rc']:.0f} K s$^{{-1}}$, d$_c$={b['dc']:.1f} mm\n"
                       f"TERNARY  R$_c$={tn['Rc']:.0f} K s$^{{-1}}$, d$_c$={tn['dc']:.1f} mm",
            transform=ax.transAxes,fontsize=8.2,va="bottom",bbox=dict(boxstyle="square,pad=0.35",fc="white",ec="black",lw=0.9))
    fig.text(0.5,0.022,"Driving force from Debye Grueneisen crystalline free energy (DFT bulk moduli, Du 2014) and a two configuration Zentropy liquid; "
             "viscosity from measured fragility.",ha="center",fontsize=6.6)
    fig.subplots_adjust(left=0.092,right=0.965,top=0.9,bottom=0.135)
    fig.savefig(OUT+"fig4_ttt_model.png"); plt.close(fig)

# ---------------- FIG 5 : validation ----------------
def fig5():
    fig,ax=plt.subplots(figsize=(6.6,5.8))
    fig.suptitle("PREDICTION versus EXPERIMENT | CRITICAL CASTING DIAMETER",y=0.975,fontsize=11,fontweight="bold")
    box(ax); lim=[0.4,60]
    ax.plot(lim,lim,color=GR["m"],lw=1.0,ls=(0,(4,2)))
    ax.fill_between(lim,[x/3 for x in lim],[x*3 for x in lim],color=GR["vl"],alpha=0.5,zorder=0)
    for key,a in ALLOYS.items():
        r=res[key]
        if a["anom"]: ax.scatter([a["dc"]],[r["dc"]],s=95,facecolor="white",edgecolor="black",marker="s",lw=1.4,zorder=5)
        else: ax.scatter([a["dc"]],[r["dc"]],s=85,facecolor="black",edgecolor="black",marker="o",zorder=5)
        ax.annotate(a["lab"],(a["dc"],r["dc"]),xytext=(a["dc"]*1.16,r["dc"]*0.72),fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Measured critical diameter, mm"); ax.set_ylabel("Predicted critical diameter, mm")
    ax.text(0.97,0.06,"shaded band: agreement within factor 3",transform=ax.transAxes,ha="right",fontsize=7.6,color=GR["d"])
    ax.scatter([],[],s=85,facecolor="black",edgecolor="black",marker="o",label="thermodynamics and kinetics aligned")
    ax.scatter([],[],s=85,facecolor="white",edgecolor="black",marker="s",label="documented anomaly, overpredicted")
    ax.legend(loc="upper left",frameon=True,edgecolor="black",fontsize=8)
    fig.text(0.5,0.022,"Predicted from the Zentropy driven nucleation model; measured by copper mold casting.",ha="center",fontsize=6.6)
    fig.subplots_adjust(left=0.12,right=0.96,top=0.9,bottom=0.12)
    fig.savefig(OUT+"fig5_validation.png"); plt.close(fig)

# ---------------- FIG 6 : driving force comparison ----------------
def fig6():
    a=ALLOYS["Cu50Zr50"]; dG,Vm,dHf,sig,dE,dSf=build(a)
    Tl=a["Tl"]; T=np.linspace(a["Tg"],Tl,300)
    gz=np.array([dG(Ti) for Ti in T])/1e3
    # Thompson Spaepen with the same DFT informed dHf for a like for like comparison
    gts=dHf*( (Tl-T)/Tl )*(2*T/(Tl+T))/1e3
    glin=dSf*(Tl-T)/1e3
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    fig.suptitle("CRYSTALLIZATION DRIVING FORCE | ZENTROPY versus CLASSICAL FORMS",y=0.975,fontsize=11,fontweight="bold")
    box(ax)
    ax.plot(T,gz,color="black",lw=2.0,label="Zentropy, DFT informed")
    ax.plot(T,gts,color="black",lw=1.6,ls=(0,(5,2)),label="Thompson Spaepen")
    ax.plot(T,glin,color=GR["m"],lw=1.4,ls=(0,(1,1.8)),label="linear, ΔS$_f$ (T$_l$ − T)")
    ax.set_xlim(a["Tg"],Tl); ax.set_ylim(0,max(gz.max(),gts.max())*1.08)
    ax.set_xlabel("Temperature, K"); ax.set_ylabel(r"Driving force, kJ mol$^{-1}$ atom$^{-1}$")
    ax.legend(loc="upper right",frameon=True,edgecolor="black",fontsize=9)
    ax.text(0.03,0.9,f"Cu$_{{50}}$Zr$_{{50}}$\nΔS$_f$ = {dSf:.1f} J mol$^{{-1}}$K$^{{-1}}$\nΔH$_f$ = {dHf/1e3:.1f} kJ mol$^{{-1}}$",
            transform=ax.transAxes,fontsize=8.5,va="top",bbox=dict(boxstyle="square,pad=0.35",fc="white",ec="black",lw=0.9))
    fig.text(0.5,0.022,"Zentropy curve from the calibrated two configuration model; classical forms shown with the same DFT informed fusion enthalpy and entropy.",
             ha="center",fontsize=6.6)
    fig.subplots_adjust(left=0.1,right=0.965,top=0.9,bottom=0.13)
    fig.savefig(OUT+"fig6_driving_force.png"); plt.close(fig)

fig4(); fig5(); fig6()
print("alloy        Tnose  tnose     Rc       dc_pred  dc_meas  sigma(mJ/m2) dHf(kJ/mol) dE_ico(kJ)")
for k,a in ALLOYS.items():
    r=res[k]
    print(f"{k:12s} {r['Tn']:5.0f} {r['tn']:9.2e} {r['Rc']:8.1f} {r['dc']:7.2f}  {a['dc']:5.1f}   "
          f"{r['sig']*1e3:8.1f}    {r['dHf']/1e3:7.2f}    {r['dE']/1e3:6.2f}")
