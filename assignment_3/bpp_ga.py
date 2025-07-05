"""
CSC 742 Evolutionary Computing
Assignment 3: Genetic Algorithms - Bin Packing Problem
Author: Matthew Branson
Date: 2025-07-05

Project Description:
SwiftShip, an e-commerce company, processes hundreds of customer orders daily.
Each order consists of items with varying weights, and each shipping box has a
maximum weight capacity of 10 kg and can carry multiple items (orders). Due to
rising packaging and courier costs, SwiftShip wants to minimize the total
number of boxes used, while ensuring that no box exceeds its weight limit...
"""
import random
import os
import matplotlib.pyplot as plt
import numpy as np


CROSSOVER_PROBABILITY = 0.90
LOG_PATH = "bpp_output.log"
IMAGES_PATH = "bpp_images"

# Q4: Test different GA configurations
GA_CONFIGS = [
    {'pop_size': 10, 'generations': 50, 'name': 'small_pop'},
    {'pop_size': 20, 'generations': 50, 'name': 'baseline'},
    {'pop_size': 40, 'generations': 50, 'name': 'large_pop'},
    {'pop_size': 20, 'generations': 25, 'name': 'short_run'},
    {'pop_size': 20, 'generations': 100, 'name': 'long_run'},
]

BPP_CONFIGS = {
    10: {'name': '10 Orders', 'n_items': 10},
    25: {'name': '25 Orders', 'n_items': 25},
    50: {'name': '50 Orders', 'n_items': 50},
    100: {'name': '100 Orders', 'n_items': 100}
}

def log(message):
    print(message)
    with open(LOG_PATH, 'a') as f:
        f.write(message + '\n')

class BinPackingProblem:
    def __init__(self, n_items, weight_range=(0, 2), bin_capacity=10):
        self.n_items = n_items
        self.bin_capacity = bin_capacity
        self.weights = [random.uniform(*weight_range) for _ in range(n_items)]

class BinPackingGA:
    def __init__(self, problem):
        self.problem = problem
        self.n_items = problem.n_items
        self.chromosome_length = self.n_items  # Each gene is a bin assignment
    
    def generate_population(self, size):
        return [[random.randint(0, self.n_items - 1) for _ in range(self.n_items)] 
                for _ in range(size)]
    
    def evaluate(self, chromosome):
        bin_weights = {}
        for item_idx, bin_idx in enumerate(chromosome):
            weight = self.problem.weights[item_idx]
            bin_weights.setdefault(bin_idx, 0.0)
            bin_weights[bin_idx] += weight
        
        f = len(bin_weights)
        g = sum(1 for w in bin_weights.values() if w > self.problem.bin_capacity)
        return (f, g)
    
    def tournament_selection(self, population, fitness_values):
        idx1, idx2 = random.sample(range(len(population)), 2)
        
        if self._compare_fitness(fitness_values[idx1], fitness_values[idx2]):
            return population[idx1].copy()
        else:
            return population[idx2].copy()
    
    def two_point_crossover(self, parent1, parent2):
        if random.random() < CROSSOVER_PROBABILITY:
            point1 = random.randint(1, self.chromosome_length - 2)
            point2 = random.randint(point1 + 1, self.chromosome_length - 1)
            
            offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            
            return offspring1, offspring2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutation(self, chromosome):
        mutation_probability = 1.0 / self.n_items
        
        for i in range(self.chromosome_length):
            if random.random() < mutation_probability:
                chromosome[i] = random.randint(0, self.n_items - 1)
        
        return chromosome
    
    def _compare_fitness(self, fitness1, fitness2):
        # Helper to avoid repeating logic in evaluation and run
        f1, g1 = fitness1
        f2, g2 = fitness2
        
        if g1 == 0 and g2 == 0:
            return f1 < f2  # Both feasible, prefer fewer bins
        elif g1 == 0:
            return True  # First is feasible
        elif g2 == 0:
            return False  # Second is feasible
        else:
            return g1 < g2  # Both infeasible, prefer fewer violations
    
    def run(self, generations, pop_size):
        # Initialize
        population = self.generate_population(size=pop_size)
        
        # Track statistics
        best_fitness_history = []
        avg_fitness_history = []
        feasible_count_history = []
        best_overall = None
        best_overall_fitness = (float('inf'), float('inf'))  # (f, g) tuple
        
        # Evolution loop
        for gen in range(generations):
            # Evaluate current population
            fitness_values = []
            for chrom in population:
                fitness = self.evaluate(chrom)
                fitness_values.append(fitness)
            
            # Extract feasible solutions
            feasible_fitnesses = [f for f, g in fitness_values if g == 0]
            feasible_count = len(feasible_fitnesses)
            feasible_count_history.append(feasible_count)
            
            # Track best and average fitness
            if feasible_fitnesses:
                best_f = min(feasible_fitnesses)
                avg_f = sum(feasible_fitnesses) / feasible_count
            else:
                best_f = float('inf')
                avg_f = float('inf')
            
            best_fitness_history.append(best_f)
            avg_fitness_history.append(avg_f)
            
            # Update best overall
            for i, fitness in enumerate(fitness_values):
                if self._compare_fitness(fitness, best_overall_fitness):
                    best_overall = population[i].copy()
                    best_overall_fitness = fitness
            
            # Create next generation
            new_population = []
            while len(new_population) < pop_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                
                # Crossover
                child1, child2 = self.two_point_crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            # Ensure exact population size
            population = new_population[:pop_size]
        
        log(f"Feasible solutions in final generation: {feasible_count}/{pop_size}")
        return best_overall, best_overall_fitness, best_fitness_history, avg_fitness_history, feasible_count_history

def create_plot(n_items, config_name, best_fit, best_hist, avg_hist, feasible_hist):
    # Create plot per configuration
    best_hist = [np.nan if v == float('inf') else v for v in best_hist]
    avg_hist = [np.nan if v == float('inf') else v for v in avg_hist]
    plt.figure(figsize=(10, 6))
    generations = range(len(best_hist))
    
    plt.plot(generations, best_hist, 'b-', label='Best Fitness', linewidth=2)
    plt.plot(generations, avg_hist, 'r--', label='Average Fitness', linewidth=2)
    
    plt.xlabel('Generation')
    plt.ylabel('Number of Bins')
    plt.title(f'Bin Packing GA - {n_items} Orders ({config_name})\nBest: {best_fit[0]} bins (g={best_fit[1]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"{IMAGES_PATH}/bpp_{n_items}items_{config_name}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"Plot saved: {filename}")


def main():
    # Clear log
    open(LOG_PATH, 'w').close()
    
    # Create images directory if it doesn't exist
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    
    log("Q4: Bin Packing GA Results\n")
    log(f"Crossover Probability: {CROSSOVER_PROBABILITY}")
    log("Mutation Probability: 1/n_items\n")
    
    # Test each configuration
    for config in GA_CONFIGS:
        pop_size = config['pop_size']
        generations = config['generations']
        config_name = config['name']
        
        log(f"\n{'='*60}")
        log(f"CONFIGURATION: {config_name}")
        log(f"Population Size: {pop_size}, Generations: {generations}")
        log('='*60)
        
        # Test different problem sizes
        for n_items in [10, 25, 50, 100]:
            log(f"\n{'-'*40}")
            log(f"{n_items} ORDERS")
            log('-'*40 + '\n')
            
            # Create problem instance
            problem = BinPackingProblem(n_items)
            ga = BinPackingGA(problem)
            
            # Log problem info
            log(f"Item weights: {[f'{w:.2f}' for w in problem.weights[:10]]}{'...' if n_items > 10 else ''}")
            log(f"Bin capacity: {problem.bin_capacity} kg")
            
            # Game faces, everyone! Time to evolve!
            best_chrom, best_fit, best_hist, avg_hist, feasible_hist = ga.run(
                generations=generations, pop_size=pop_size
            )
            
            # Log results
            log(f"\nBest solution found:")
            log(f"  Bins used: {best_fit[0]}")
            log(f"  Constraint violations: {best_fit[1]}")
            log(f"  Feasible: {'Yes' if best_fit[1] == 0 else 'No'}")
            
            bins = {}
            for idx, bin_id in enumerate(best_chrom):
                bins.setdefault(bin_id, []).append((idx, problem.weights[idx]))

            for bin_id, items in sorted(bins.items()):
                total_weight = sum(w for _, w in items)
                item_str = ', '.join(f'#{i}:{w:.2f}' for i, w in items)
                log(f"  Bin {bin_id}: [{item_str}] -> {total_weight:.2f}kg")
            
            # Create plot
            create_plot(n_items, config_name, best_fit, best_hist, avg_hist, feasible_hist)
    
    log(f"\nAll results saved to {LOG_PATH}")
    log(f"All plots saved to {IMAGES_PATH}/")


if __name__ == "__main__":
    # random.seed(773) # Uncomment for reproducibility
    main()