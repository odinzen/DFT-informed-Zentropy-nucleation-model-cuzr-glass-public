#!/usr/bin/env python3
"""Inherent structure spectrum of Cu-Zr liquids, Mendelev EAM/FS potential in LAMMPS, one composition per call.

The liquid is stepped down in temperature at zero pressure. At each rung, after equilibration, copies are drawn at equal
intervals and each is quenched to its inherent structure with the cell relaxed to zero pressure. E_IS(T) is the measured
quantity: its slope is the configurational heat capacity and needs no thermodynamic model.

Usage: run_spectrum.py xZr cells outdir potential nproc lmp [quick]
"""
import json, os, subprocess, sys

xZr, cells, outdir, pot, nproc, lmp = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
quick = len(sys.argv) > 7
os.makedirs(outdir, exist_ok=True)
DT = 0.002
if quick:
    LADDER, EQ, SAMPLE, N_IS = [1400, 1200], {1400: 2, 1200: 2}, 2, 2
else:
    LADDER = [1500, 1400, 1300, 1200, 1150, 1100, 1050, 1000, 950, 900]
    EQ = {T: (400 if T >= 1100 else 1000 if T >= 1000 else 2000) for T in LADDER}
    SAMPLE, N_IS = 500, 50
SEED = 4928 + int(xZr * 1000)
every = int(SAMPLE / DT / N_IS)
SETUP = f"""pair_style eam/fs
pair_coeff * * {pot} Cu Zr
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
timestep {DT}
"""
parts = [f"""units metal
atom_style atomic
boundary p p p
lattice fcc 3.90
region box block 0 {cells} 0 {cells} 0 {cells}
create_box 2 box
create_atoms 1 box
mass 1 63.546
mass 2 91.224
set type 1 type/fraction 2 {xZr} {SEED}
{SETUP}thermo 20000
velocity all create 2000 {SEED} mom yes rot yes dist gaussian
fix md all npt temp 2000 2000 0.2 iso 0.0 0.0 2.0
run {5000 if quick else 50000}
fix md all npt temp 2000 {LADDER[0]} 0.2 iso 0.0 0.0 2.0
run {5000 if quick else 50000}
unfix md
write_restart {outdir}/state.restart
"""]
for T in LADDER:
    parts.append(f"""
clear
read_restart {outdir}/state.restart
{SETUP}fix md all npt temp {T} {T} 0.2 iso 0.0 0.0 2.0
run {int(EQ[T] / DT)}
unfix md
write_restart {outdir}/state.restart
variable k loop {N_IS}
label loop{T}
clear
read_restart {outdir}/state.restart
{SETUP}fix md all npt temp {T} {T} 0.2 iso 0.0 0.0 2.0
run {every}
variable p0 equal pe/atoms
variable v0 equal vol/atoms
variable t0 equal temp
variable a equal $(v_p0:%.8f)
variable b equal $(v_v0:%.6f)
variable c equal $(v_t0:%.2f)
unfix md
write_restart {outdir}/state.restart
fix br all box/relax iso 0.0 vmax 0.001
min_style cg
minimize 1.0e-10 1.0e-6 4000 40000
unfix br
variable p1 equal pe/atoms
variable v1 equal vol/atoms
variable p2 equal press
print "{T} $k $a $b $c $(v_p1:%.8f) $(v_v1:%.6f) $(v_p2:%.1f)" append {outdir}/spectrum.dat screen no
write_dump all custom {outdir}/is_{T}_$k.dump id type x y z modify sort id
variable a delete
variable b delete
variable c delete
next k
jump SELF loop{T}
variable k delete
""")
inp = os.path.join(outdir, "in.spectrum")
open(inp, "w").write("\n".join(parts))
open(os.path.join(outdir, "spectrum.dat"), "w").close()
json.dump(dict(xZr=xZr, cells=cells, natoms=4 * cells ** 3, ladder=LADDER, eq_ps=EQ, sample_ps=SAMPLE, n_is=N_IS, dt=DT, potential=os.path.basename(pot),
               columns="T k PE_md V_md T_md E_IS V_IS P_IS"), open(os.path.join(outdir, "params.json"), "w"), indent=1)
env = dict(os.environ, OMP_NUM_THREADS="1")
subprocess.run(f"mpirun -np {nproc} --oversubscribe --bind-to none {lmp} -in {inp} -log {outdir}/log.lammps -screen none", shell=True, check=True, env=env)
print("done", xZr)
