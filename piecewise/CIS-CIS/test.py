import numpy as np
from get_radius_cis_piecewise import get_radius_cis


rng = np.random.default_rng(0)
d2 = rng.chisquare(df=3, size=50)        # some squared distances
L  = rng.exponential(scale=1, size=50)   # likelihood ratios

for delta in [0.05, 0.1, 0.2, 0.4]:
    print(delta, get_radius_cis(d2, L, delta**2))