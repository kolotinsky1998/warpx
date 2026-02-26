#!/usr/bin/env python3

import sys

import numpy as np
import yt

filename = sys.argv[1]
ds = yt.load(filename)
ad = ds.all_data()

ex = ad["electrons", "particle_position_x"].to_ndarray()
ez = ad["electrons", "particle_position_z"].to_ndarray()
ix = ad["ions", "particle_position_x"].to_ndarray()
iz = ad["ions", "particle_position_z"].to_ndarray()

assert ex.size == 1
assert ez.size == 1
assert ix.size == 1
assert iz.size == 1

electron_disp = np.sqrt((ex[0] - 0.0) ** 2 + (ez[0] - 0.0) ** 2)
ion_disp = np.sqrt((ix[0] - 0.0) ** 2 + (iz[0] - 0.2) ** 2)

print(f"electron_disp = {electron_disp}")
print(f"ion_disp = {ion_disp}")

assert electron_disp < 1.0e-9
assert ion_disp > 1.0e-6
