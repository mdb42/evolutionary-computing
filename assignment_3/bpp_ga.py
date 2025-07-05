"""
CSC 742 Evolutionary Computing
Assignment 3: Genetic Algorithms
Author: Matthew Branson
Date: 2025-07-05

Project Description:
Adapting for assignment 3. Not functional yet.
"""
import random
import os
import matplotlib.pyplot as plt


DEFAULT_POPULATION_SIZE = 20
MAX_GENERATIONS = 50
CROSSOVER_PROBABILITY = 0.90
LOG_PATH = "bpp_output.log"
IMAGES_PATH = "bpp_images"

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
    
    def generate_population(self, size=DEFAULT_POPULATION_SIZE):
        return [[random.randint(0, self.n_items - 1) for _ in range(self.n_items)] 
                for _ in range(size)]
    
    def evaluate(self, chromosome):
        pass
    
    def tournament_selection(self, population, fitness_values):
        pass
    
    def two_point_crossover(self, parent1, parent2):
        if random.random() < CROSSOVER_PROBABILITY:
            point1 = random.randint(1, self.chromosome_length - 2)
            point2 = random.randint(point1 + 1, self.chromosome_length - 1)
            
            offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            
            return offspring1, offspring2
        else:
            # No crossover - return copies of parents
            return parent1.copy(), parent2.copy()
    
    def mutation(self, chromosome):
        mutation_probability = 1.0 / self.n_items
        
        for i in range(self.chromosome_length):
            if random.random() < mutation_probability:
                # TODO: Reassign to random bin
                pass
        
        return chromosome
    
    def run(self, generations=MAX_GENERATIONS, pop_size=DEFAULT_POPULATION_SIZE):
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
                
                # TODO: Need to compare (f, g) tuples
            
            # TODO: Track best f, average f, number of feasible solutions
            
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
        
        return best_overall, best_overall_fitness, best_fitness_history, avg_fitness_history, feasible_count_history

def create_plot(n_items, best_fit, best_hist, avg_hist, feasible_hist):
    # TODO Update plotting for bin packing
    pass


def main():
    # Clear log
    open(LOG_PATH, 'w').close()
    
    # Create images directory if it doesn't exist
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    
    log("Q4: Bin Packing GA Results\n")
    log(f"Population Size: {DEFAULT_POPULATION_SIZE}")
    log(f"Generations: {MAX_GENERATIONS}")
    log(f"Crossover Probability: {CROSSOVER_PROBABILITY}")
    log("Mutation Probability: 1/n_items\n")
    
    # Test different problem sizes
    for n_items in [10, 25, 50, 100]:
        log(f"\n{'='*50}")
        log(f"{n_items} ORDERS")
        log('='*50 + '\n')
        
        # Create problem instance
        problem = BinPackingProblem(n_items)
        ga = BinPackingGA(problem)
        
        # Log problem info
        log(f"Item weights: {[f'{w:.2f}' for w in problem.weights[:10]]}{'...' if n_items > 10 else ''}")
        log(f"Bin capacity: {problem.bin_capacity} kg")
        
        # Game faces, everyone! Time to evolve!
        best_chrom, best_fit, best_hist, avg_hist, feasible_hist = ga.run()
        
        # TODO Q4: Log results and best configuration
        
        # Create plot
        create_plot(n_items, best_fit, best_hist, avg_hist, feasible_hist)
    
    log(f"\nAll results saved to {LOG_PATH}")
    log(f"All plots saved to {IMAGES_PATH}/")


if __name__ == "__main__":
    # random.seed(773) # Uncomment for reproducibility
    main()