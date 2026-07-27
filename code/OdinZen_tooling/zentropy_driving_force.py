#!/usr/bin/env python3
"""DFT informed Zentropy crystallization driving force.

Crystalline side: Debye Grueneisen quasiharmonic free energy seeded by published
DFT bulk moduli and formation enthalpies (Du, Wen, Melnik, Kawazoe 2014).
Supercooled liquid side: two configuration Zentropy mixture (icosahedral ground
liquid configuration plus a softer defective configuration), with the icosahedral
energy offset anchored to the DFT convex hull distance as the placeholder that the
user's own cluster DFT (Appendix A) replaces.

Outputs dG(T) = G_liquid - G_crystal, the entropy and enthalpy of fusion, and a
callable used by the nucleation model.
"""
import numpy as np
from scipy.integrate import quad

R = 8.314462          # J/mol/K
hbar = 1.054571e-34
kB = 1.380649e-23
NA = 6.02214076e23
# elemental atomic volumes (m^3) and masses (kg) for composition weighting
V_Cu, V_Zr = 11.81e-30, 23.28e-30
M_Cu, M_Zr = 63.546e-3/NA, 91.224e-3/NA

def debye_D(x):
    """Debye function D(x) = (3/x^3) integral_0^x t^3/(e^t-1) dt."""
    if x < 1e-6: return 1.0
    val,_ = quad(lambda t: t**3/np.expm1(t), 0, x, limit=100)
    return 3.0*val/x**3

def theta_debye(B0_Pa, Vatom_m3, Matom_kg, poisson_G_over_B=0.4):
    """Debye temperature from bulk modulus via mean sound velocity (G ~ 0.4 B)."""
    rho = Matom_kg/Vatom_m3
    G = poisson_G_over_B*B0_Pa
    vl = np.sqrt((B0_Pa+4*G/3)/rho)
    vt = np.sqrt(G/rho)
    vm = ((1/3)*(1/vl**3 + 2/vt**3))**(-1/3)
    n = 1.0/Vatom_m3
    return (hbar/kB)*(6*np.pi**2*n)**(1/3)*vm

def Fvib_molar(T, thetaD):
    """Debye vibrational Helmholtz free energy per mole of atoms (J/mol)."""
    x = thetaD/T
    return (9/8)*R*thetaD + 3*R*T*np.log(1-np.exp(-x)) - R*T*debye_D(x)

def Svib_molar(T, thetaD):
    x = thetaD/T
    return R*(4*debye_D(x) - 3*np.log(1-np.exp(-x)))

# composition helper: x = atomic fraction Zr
def comp_props(xZr):
    Vat = (1-xZr)*V_Cu + xZr*V_Zr
    Mat = (1-xZr)*M_Cu + xZr*M_Zr
    return Vat, Mat

# DFT data (Du 2014): formation enthalpy (J/mol-atom), bulk modulus (Pa), xZr
PHASES = {
 "Cu8Zr3":  dict(Hf=-16.31e3, B=133e9, xZr=3/11),
 "Cu10Zr7": dict(Hf=-16.11e3, B=126e9, xZr=7/17),
 "CuZr":    dict(Hf=-6.06e3,  B=121e9, xZr=0.50),
 "CuZr2":   dict(Hf=-12.665e3, B=111e9, xZr=2/3),  # Du 2014 [8]; B0=Voigt avg of Cij (1000/9 GPa)
}
def phase_theta(name):
    p=PHASES[name]; Vat,Mat=comp_props(p["xZr"]); return theta_debye(p["B"],Vat,Mat)

def crystalline_assembly(xZr, lo, hi):
    """Lever rule mixture of bounding intermetallics lo, hi at composition xZr.
    Returns (E0 J/mol-atom, thetaD K, Vatom m^3)."""
    a,b = PHASES[lo], PHASES[hi]
    f = (xZr-a["xZr"])/(b["xZr"]-a["xZr"]); f=min(max(f,0),1)
    E0 = (1-f)*a["Hf"] + f*b["Hf"]
    th = (1-f)*phase_theta(lo) + f*phase_theta(hi)
    Vat = comp_props(xZr)[0]
    return E0, th, Vat

def G_crystal(T, E0, thetaD):
    return E0 + Fvib_molar(T, thetaD)

def G_liquid(T, E0_cryst, thetaD_cryst, dE_ico, dE_split, soft1=0.85, soft2=0.80):
    """Two configuration Zentropy supercooled liquid.
    config 1 (icosahedral ground liquid): E0_cryst + dE_ico, theta = soft1*theta_cryst
    config 2 (defective looser):          E0_cryst + dE_ico + dE_split, theta = soft2*theta_cryst
    Zentropy molar free energy G = sum p_k F_k + R T sum p_k ln p_k.
    """
    F1 = E0_cryst + dE_ico            + Fvib_molar(T, soft1*thetaD_cryst)
    F2 = E0_cryst + dE_ico + dE_split + Fvib_molar(T, soft2*thetaD_cryst)
    Fmin = min(F1,F2)
    z1 = np.exp(-(F1-Fmin)/(R*T)); z2 = np.exp(-(F2-Fmin)/(R*T))
    Z = z1+z2; p1,p2 = z1/Z, z2/Z
    G = p1*F1 + p2*F2 + R*T*(p1*np.log(p1)+p2*np.log(p2))
    return G, (p1,p2)

def driving_force(T, xZr, lo, hi, dE_ico=5.0e3, dE_split=6.0e3):
    E0c, thc, Vat = crystalline_assembly(xZr, lo, hi)
    Gc = G_crystal(T, E0c, thc)
    Gl,_ = G_liquid(T, E0c, thc, dE_ico, dE_split)
    return Gl - Gc, Vat            # J/mol-atom, atomic volume

from scipy.optimize import brentq
def calibrate_dE_ico(Tl, xZr, lo, hi, dE_split=6.0e3):
    """Pin the icosahedral energy offset so the driving force vanishes at the
    measured liquidus, dG(Tl)=0. Tl is a measured input; the DFT Debye model and
    the Zentropy configurational term then set the curvature of dG(T) below Tl."""
    f = lambda dE: driving_force(Tl, xZr, lo, hi, dE, dE_split)[0]
    return brentq(f, 2.0e3, 40.0e3, xtol=1.0)

def make_dG_callable(Tl, xZr, lo, hi, dE_split=6.0e3):
    """Returns dG(T) (J/mol-atom, positive below Tl), atomic volume, calibrated offset."""
    dE_ico = calibrate_dE_ico(Tl, xZr, lo, hi, dE_split)
    Vat = comp_props(xZr)[0]
    def dG(T): return driving_force(T, xZr, lo, hi, dE_ico, dE_split)[0]
    return dG, Vat, dE_ico

# ---------------- diagnostics ----------------
if __name__ == "__main__":
    print("Debye temperatures from DFT bulk moduli:")
    for n in PHASES: print(f"  {n:8s} thetaD = {phase_theta(n):.0f} K")
    print()
    # crystalline reference per alloy
    setups = {
      "Cu64Zr36":   (0.36, "Cu8Zr3","Cu10Zr7"),
      "Cu50Zr50":   (0.50, "Cu10Zr7","CuZr"),
      "Cu47Zr45Al8":(0.38, "Cu8Zr3","Cu10Zr7"),   # crystal ref ~ Cu rich eutectic assembly
    }
    Tls = {"Cu64Zr36":1230,"Cu50Zr50":1208,"Cu47Zr45Al8":1163}
    for name,(x,lo,hi) in setups.items():
        E0c,thc,Vat = crystalline_assembly(x,lo,hi)
        Tl = Tls[name]
        dG, Vat, dE = make_dG_callable(Tl, x, lo, hi)
        dSf = -(dG(Tl+1)-dG(Tl-1))/2.0
        print(f"{name:12s} thetaD={thc:.0f}K  dE_ico(cal)={dE/1e3:5.2f} kJ/mol  "
              f"dSf={dSf:5.2f} J/mol/K  dHf={Tl*dSf/1e3:5.2f} kJ/mol  "
              f"dG(Tl)={dG(Tl)/1e3:+.2f}  dG(900)={dG(900)/1e3:+.2f}  dG(800)={dG(800)/1e3:+.2f} kJ/mol")
