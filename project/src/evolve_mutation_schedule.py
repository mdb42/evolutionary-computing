"""
Skeleton: evolve a GP-defined mutation-rate schedule.

    python -m src.evolve_mutation_schedule --quick
"""

import argparse
import numpy as np
import pygmo as pg # conda install -c conda-forge pygmo
from gplearn.genetic import SymbolicRegressor

from src.simple_transfer_udp import SimpleTransferUDP

def build_population(size=40, seed=0):
    prob = pg.problem(SimpleTransferUDP())
    return pg.population(prob, size=size, seed=seed)

def evaluate_gp_expression(expr, n_runs=5):
    """
    Evaluate a GP expression that maps optimisation state -> mutation_rate.
    """
    obj_scores = []
    for seed in range(n_runs):
        pop = build_population(seed=seed)
        algo = pg.algorithm(pg.sga(gen=100))

        # Patch the mutation rate schedule by overriding 'evolve'
        original_evolve = algo.evolve

        def custom_evolve(pop):
            generations = algo.extract(pg.sga).get_gen()
            for g in range(generations):
                # placeholders for state vars
                stagnation = 0
                diversity = 1
                mut_rate = float(expr(g, stagnation, diversity))
                # clamp
                mut_rate = max(1e-3, min(mut_rate, 1.0))
                algo.set_mutation_rate(mut_rate)
                pop = original_evolve(pop)
            return pop

        pop = custom_evolve(pop)
        obj_scores.append(pop.champion_f)

    obj_scores = np.array(obj_scores)
    mean_f = np.mean(obj_scores, axis=0)
    std_f = np.std(obj_scores, axis=0)
    return mean_f.tolist() + std_f.tolist()

# ---------------------------------------------------------------------------

def quick_test():
    est = SymbolicRegressor(
        population_size=100,
        generations=5,
        metric=lambda y_true, y_pred: 0.0,   # dummy
        verbose=1,
        random_state=0
    )
    # Fake dataset: state vectors -> target mutation rate
    X = np.random.rand(200, 3)
    y = np.random.rand(200)
    est.fit(X, y)
    print("Evolved expression:", est._program)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a 30-second smoke test")
    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        print("TODO: implement full GP evolution loop...")
