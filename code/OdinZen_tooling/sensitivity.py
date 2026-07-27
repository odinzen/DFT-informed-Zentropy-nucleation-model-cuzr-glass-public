#!/usr/bin/env python3
"""Sensitivity of the predicted critical diameter to each independent input,
for the ternary Cu47Zr45Al8. Each input varied by plus or minus 20 percent.
Produces a greyscale tornado figure and prints the spread."""
import numpy as np, copy
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
# ternary baseline
TG,TL,DSTAR,XZR,LO,HI = 681,1163,20.6,0.38,"Cu8Zr3","Cu10Zr7"
BASE=dict(aT=0.50, lam=2.8e-10, D=20.6, x_det=1e-6, eta0=4e-5, dE_split=6.0e3,
          Bscale=1.0, Vmscale=1.0, soft1=0.85, soft2=0.80)
_orig_G_liquid=z.G_liquid   # soft1/soft2 are G_liquid defaults, not threaded through
                            # make_dG_callable; monkeypatch so calibration+prediction see them

def predict_dc(p):
    # scale crystalline bulk moduli (affects Debye T and entropy of fusion)
    saved={k:z.PHASES[k]["B"] for k in z.PHASES}
    for k in z.PHASES: z.PHASES[k]["B"]=saved[k]*p["Bscale"]
    z.G_liquid=lambda T,E0,th,dEi,dEs,a=p["soft1"],b=p["soft2"]: _orig_G_liquid(T,E0,th,dEi,dEs,a,b)
    try:
        dG,Vat,dE = z.make_dG_callable(TL,XZR,LO,HI,p["dE_split"])
        Vm=Vat*NA*p["Vmscale"]
        dSf=-(dG(TL+1)-dG(TL-1))/2.0; dHf=TL*dSf
        sig=p["aT"]*dHf/(NA**(1/3)*Vm**(2/3))
        Nv=NA/Vm; Cvft=np.log(1e12/p["eta0"]); T0=Cvft*TG/(p["D"]+Cvft)
        T=np.linspace(TG+5,TL-3,1200); t=np.empty_like(T)
        for i,Ti in enumerate(T):
            eta=p["eta0"]*np.exp(p["D"]*T0/(Ti-T0))
            g=max(dG(Ti),1.0); gv=g/Vm
            dGs=16*np.pi*sig**3/(3*gv**2)
            I=Nv*(kB*Ti/(3*np.pi*p["lam"]**3*eta))*np.exp(-dGs/(kB*Ti))
            U=max((kB*Ti/(3*np.pi*p["lam"]**2*eta))*(1-np.exp(-g/(Rg*Ti))),1e-30)
            t[i]=(3*p["x_det"]/(np.pi*max(I,1e-300)*U**3))**0.25
        j=np.argmin(t); Rc=(TL-T[j])/t[j]; dc=10*np.sqrt(10/Rc)
    finally:
        for k in z.PHASES: z.PHASES[k]["B"]=saved[k]
        z.G_liquid=_orig_G_liquid
    return dc

base_dc=predict_dc(BASE); print(f"baseline ternary dc = {base_dc:.2f} mm")
LABELS={"aT":"Turnbull coefficient  $\\alpha_T$","lam":"jump distance  $\\lambda$",
        "D":"fragility  $D^*$","x_det":"detectable fraction  $x$","eta0":"viscosity prefactor  $\\eta_0$",
        "dE_split":"configuration split  $\\delta E$","Bscale":"crystal bulk modulus  $B_0$","Vmscale":"molar volume  $V_m$",
        "soft1":"softening factor  soft$_1$","soft2":"softening factor  soft$_2$"}
rows=[]
for k in BASE:
    lo=copy.deepcopy(BASE); hi=copy.deepcopy(BASE)
    lo[k]=BASE[k]*0.8; hi[k]=BASE[k]*1.2
    dlo=predict_dc(lo); dhi=predict_dc(hi)
    rows.append((LABELS[k],dlo,dhi)); print(f"{k:9s} -20% -> {dlo:5.2f} mm   +20% -> {dhi:5.2f} mm")

# order by total swing
rows.sort(key=lambda r: abs(r[2]-r[1]))
fig,ax=plt.subplots(figsize=(7.6,4.8))
fig.suptitle("SENSITIVITY OF PREDICTED CRITICAL DIAMETER | Cu$_{47}$Zr$_{45}$Al$_{8}$, EACH INPUT ±20%",
             y=0.975,fontsize=10.5,fontweight="bold")
box(ax)
y=np.arange(len(rows))
for i,(lab,dlo,dhi) in enumerate(rows):
    a,b=sorted((dlo,dhi))
    ax.barh(i,b-a,left=a,height=0.62,color=GR["vl"],edgecolor="black",lw=1.0,hatch="////")
    ax.plot([dlo,dlo],[i-0.31,i+0.31],color="black",lw=1.2)
    ax.plot([dhi,dhi],[i-0.31,i+0.31],color="black",lw=1.2)
ax.axvline(base_dc,color="black",lw=1.4,ls=(0,(5,2)))
ax.text(base_dc,len(rows)-0.4,f"baseline {base_dc:.1f} mm",fontsize=8,va="center",ha="center")
ax.axvspan(5,45,color=GR["l"],alpha=0.18,zorder=0)
ax.text(15,0.0,"measured\n15 mm",fontsize=7.6,ha="center",va="center",color=GR["d"])
ax.axvline(15,color=GR["m"],lw=1.0,ls=(0,(1,2)))
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=9)
ax.set_xlabel("Predicted critical diameter, mm  (log scale)")
ax.set_xscale("log"); ax.set_xlim(0.3,700)
fig.text(0.5,0.02,"Bars span the diameter as each input varies by ±20%; ticks mark the extremes. Shaded band: measurement within a factor three.",ha="center",fontsize=6.8)
fig.subplots_adjust(left=0.30,right=0.965,top=0.9,bottom=0.16)
fig.savefig(OUT+"fig7_sensitivity.png"); plt.close(fig)
swing=[(r[0],abs(r[2]-r[1])) for r in rows]
print("\nlargest swings:", sorted(swing,key=lambda s:-s[1])[:3])
