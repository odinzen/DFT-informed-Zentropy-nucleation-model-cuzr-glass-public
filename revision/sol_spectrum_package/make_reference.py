"""Forces and energy of the check frame on the local GPU in double precision; the Sol check job compares against these."""
import numpy as np
import torch
import spectrum_run as sr

a = sr.reference_frame()
a.calc = sr.mace("float64", cueq=False)
np.savez("reference_forces.npz", positions=a.get_positions(), forces=a.get_forces(), energy=a.get_potential_energy(),
         device=torch.cuda.get_device_name(0))
print("energy", a.get_potential_energy() / len(a), "eV/atom on", torch.cuda.get_device_name(0))
