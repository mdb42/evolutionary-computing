"""
A toy Earth-GEO 2-impulse transfer problem expressed as a PyGMO UDP.

Objectives
1. Minimise total delta-V (km/s)
2. Minimise time-of-flight (s)

"""

import numpy as np
import pygmo as pg # conda install -c conda-forge pygmo

class SimpleTransferUDP:
    def __init__(self):
        self.nobj = 2
        self.dim = 7  # [t0, dt1, dv1x, dv1y, dv1z, dv2x, dv2y]

    # Mandatory UDP hooks

    def get_nobj(self):
        return self.nobj

    def get_bounds(self):
        # Time bounds in seconds, delta-V components in km/s
        lb = [0, 0, -2, -2, -2, -2, -2]
        ub = [6*3600, 2*3600,  2,  2,  2,  2,  2]
        return (lb, ub)

    def fitness(self, x):
        # TODO: real Lambert solve
        dv1 = np.linalg.norm(x[2:5])
        dv2 = np.linalg.norm(x[5:7])
        total_dv = dv1 + dv2
        tof = x[0] + x[1]
        return [total_dv, tof]

    def get_name(self):
        return "Toy Earth-GEO transfer (2-impulse)"

if __name__ == "__main__":
    prob = pg.problem(SimpleTransferUDP())
    algo = pg.algorithm(pg.sga(gen=100))
    pop = pg.population(prob, size=40, seed=42)
    pop = algo.evolve(pop)

    print("Champion fitness:", pop.champion_f)
