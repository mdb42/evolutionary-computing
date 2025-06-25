"""
CSC 742 Evolutionary Computing
Assignment 2: Genetic Algorithms
Author: Matthew Branson
Date: 2025-06-24

Project Description:
This module implements a binary-encoded genetic algorithm to optimize four De Jong test functions.
"""
import random
import math
import os
import matplotlib.pyplot as plt


DEFAULT_POPULATION_SIZE = 20
MAX_GENERATIONS = 50
CROSSOVER_PROBABILITY = 0.90
LOG_PATH = "ga_output.log"
IMAGES_PATH = "ga_images"

DEJONG_CONFIGS = {
    'f1': {'name': 'Sphere Model', 'n_vars': 3, 'bounds': (-5.12, 5.12)},
    'f2': {'name': 'Weighted Sphere Model', 'n_vars': 2, 'bounds': (-2.048, 2.048)},
    'f3': {'name': 'Step Function', 'n_vars': 4, 'bounds': (-5.12, 5.12)},
    'f4': {'name': 'Noisy Quartic', 'n_vars': 4, 'bounds': (-1.28, 1.28)}
}

# De Jong Test Function Evaluations
def f1_sphere(x):
    """
    Evaluate the Sphere Model function.

    Args:
        x (list): Real-valued coordinates
        
    Returns:
        float: Function value at x
    """
    return sum(xi**2 for xi in x)

def f2_weighted_sphere(x):
    """
    Evaluate the Weighted Sphere Model function.
    
    Args:
        x (list): Real-valued coordinates
    Returns:
        float: Function value at x
    """
    return 100 * (x[0]**2 - x[1])**2 + (1 - x[0])**2

def f3_step(x):
    """
    Evaluate the Step Function.
    
    Args:
        x (list): Real-valued coordinates

    Returns:
        float: Function value at x
    """
    return sum(math.floor(xi) for xi in x)

def f4_noisy_quartic(x):
    """
    Evaluate the Noisy Quartic function.

    Args:
        x (list): Real-valued coordinates

    Returns:
        float: Function value at x
    """
    return sum((i+1) * xi**4 for i, xi in enumerate(x)) + random.random()

# Map function IDs to their evaluation functions
DEJONG_FUNCTIONS = {
    'f1': f1_sphere,
    'f2': f2_weighted_sphere,
    'f3': f3_step,
    'f4': f4_noisy_quartic
}

def log(message):
    """Log messages to console and file.
    
    Args:
        message (str): Message to log
    """
    print(message)
    with open(LOG_PATH, 'a') as f:
        f.write(message + '\n')

class BinaryGA:
    """
    Binary Genetic Algorithm for optimizing De Jong test functions.
    
    Attributes:
        function_id (str): ID of the De Jong function to optimize
        bits_per_variable (int): Number of bits per variable in the chromosome
        n_vars (int): Number of variables in the function
        bounds (tuple): Bounds for the function variables
        chromosome_length (int): Total length of the binary chromosome
        precision (float): Precision of the real-valued representation
    """
    def __init__(self, function_id, bits_per_variable):
        """
        Initialize the BinaryGA with the specified function ID and bits per variable.
        
        Args:
            function_id (str): ID of the De Jong function to optimize
            bits_per_variable (int): Number of bits to represent each variable
        """
        config = DEJONG_CONFIGS[function_id]
        self.function_id = function_id
        self.n_vars = config['n_vars']
        self.bounds = config['bounds']
        self.bits_per_variable = bits_per_variable
        self.chromosome_length = self.n_vars * bits_per_variable
        
        # Calculate precision
        lower, upper = self.bounds
        self.precision = (upper - lower) / (2**bits_per_variable - 1)
    
    def generate_population(self, size=DEFAULT_POPULATION_SIZE):
        """
        Generate an initial population of random binary chromosomes.

        Args:
            size (int): Number of chromosomes to generate

        Returns:
            list: List of binary chromosomes, each represented as a list of bits
        """
        return [[random.randint(0, 1) for _ in range(self.chromosome_length)] 
                for _ in range(size)]
    
    def decode(self, chromosome):
        """
        Decode a binary chromosome into real-valued variables.
        
        Args:
            chromosome (list): Binary chromosome to decode

        Returns:
            list: List of real-valued variables decoded from the chromosome
        """
        real_values = []
        for i in range(self.n_vars):
            # Extract bits for this variable
            start = i * self.bits_per_variable
            bits = chromosome[start:start + self.bits_per_variable]
            
            # Convert to decimal
            decimal = sum(bit * (2 ** (self.bits_per_variable - 1 - j)) 
                         for j, bit in enumerate(bits))
            
            # Map to real value
            lower, upper = self.bounds
            real = lower + (upper - lower) * decimal / (2**self.bits_per_variable - 1)
            real_values.append(real)
        
        return real_values
    
    def fitness_proportionate_selection(self, population, fitness_values):
        """
        Perform fitness proportionate selection (roulette wheel selection) on the population.

        Args:
            population (list): Current population of chromosomes
            fitness_values (list): Fitness values corresponding to the population

        Returns:
            list: Selected parent chromosome for reproduction
        """
        # Shift fitness values to ensure all are positive
        # Necessary since the step function can yield negative fitness
        min_fitness = min(fitness_values)

        if min_fitness < 0:
            shifted_fitness = [f - min_fitness + 1.0 for f in fitness_values]
        else:
            shifted_fitness = fitness_values
        
        # Transform for minimization
        transformed_fitness = [1.0 / (f + 1.0) for f in shifted_fitness]
        total_fitness = sum(transformed_fitness)
        probabilities = [f / total_fitness for f in transformed_fitness]
        
        # Roulette wheel
        r = random.random()
        cumulative_probability = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_probability += prob
            if r <= cumulative_probability:
                return population[i].copy()
        
        # Fallback (shouldn't reach here)
        return population[-1].copy()
    
    def two_point_crossover(self, parent1, parent2):
        """
        Perform two-point crossover between two parent chromosomes.

        Args:
            parent1 (list): First parent chromosome
            parent2 (list): Second parent chromosome
        
        Returns:
            tuple: Offspring chromosomes resulting from crossover
        """
        if random.random() < CROSSOVER_PROBABILITY:
            point1 = random.randint(1, self.chromosome_length - 2)
            point2 = random.randint(point1 + 1, self.chromosome_length - 1)
            
            offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            
            return offspring1, offspring2
        else:
            # No crossover - return copies of parents
            return parent1.copy(), parent2.copy()
    
    def bitwise_mutation(self, chromosome):
        """
        Perform bitwise mutation on a binary chromosome.

        Args:
            chromosome (list): Binary chromosome to mutate
        Returns:
            list: Mutated chromosome
        """
        mutation_probability = 1.0 / self.chromosome_length
        
        for i in range(self.chromosome_length):
            if random.random() < mutation_probability:
                # Flip bit
                chromosome[i] = 1 - chromosome[i]
        
        return chromosome
    
    def run(self, generations=MAX_GENERATIONS, pop_size=DEFAULT_POPULATION_SIZE):
        """
        Run the genetic algorithm for a specified number of generations.

        Args:
            generations (int): Number of generations to evolve
            pop_size (int): Size of the population
        
        Returns:
            tuple: Best overall chromosome, its fitness, best fitness history, average fitness history
        """
        # Initialize
        population = self.generate_population(size=pop_size)
        eval_func = DEJONG_FUNCTIONS[self.function_id]
        
        # Track statistics
        best_fitness_history = []
        avg_fitness_history = []
        best_overall = None
        best_overall_fitness = float('inf')
        
        # Evolution loop
        for gen in range(generations):
            # Evaluate current population
            fitness_values = []
            for chrom in population:
                decoded = self.decode(chrom)
                fitness = eval_func(decoded)
                fitness_values.append(fitness)
                
                # Track best overall
                if fitness < best_overall_fitness:
                    best_overall_fitness = fitness
                    best_overall = chrom.copy()
            
            # Record statistics
            best_gen_fitness = min(fitness_values)
            avg_gen_fitness = sum(fitness_values) / len(fitness_values)
            best_fitness_history.append(best_gen_fitness)
            avg_fitness_history.append(avg_gen_fitness)
            
            # Create next generation
            new_population = []
            while len(new_population) < pop_size:
                # Selection
                parent1 = self.fitness_proportionate_selection(population, fitness_values)
                parent2 = self.fitness_proportionate_selection(population, fitness_values)
                
                # Crossover
                child1, child2 = self.two_point_crossover(parent1, parent2)
                
                # Mutation
                child1 = self.bitwise_mutation(child1)
                child2 = self.bitwise_mutation(child2)
                
                new_population.extend([child1, child2])
            
            # Ensure exact population size
            population = new_population[:pop_size]
        
        return best_overall, best_overall_fitness, best_fitness_history, avg_fitness_history


def create_plot(fid, config, bits, best_fit, best_hist, avg_hist):
    """Create and save fitness plot for a given function and encoding
    
    Args:
        fid (str): Function ID
        config (dict): Configuration for the function
        bits (int): Number of bits per variable
        best_fit (float): Best fitness found
        best_hist (list): History of best fitness values
        avg_hist (list): History of average fitness values
    """
    plt.figure(figsize=(10, 6))
    generations = range(len(best_hist))
    
    plt.plot(generations, best_hist, 'b-', label='Best Fitness', linewidth=2)
    plt.plot(generations, avg_hist, 'r--', label='Average Fitness', linewidth=2)
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'{config["name"]} - {bits}-bit Encoding\nBest Fitness: {best_fit:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    filename = f"{IMAGES_PATH}/{fid}_{bits}bit_fitness.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"Plot saved: {filename}")


def main():
    """
    Main function to run the genetic algorithm and log results.
    """
    # Clear log
    open(LOG_PATH, 'w').close()
    
    # Create images directory if it doesn't exist
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    
    log("Q4: GA Execution Results\n")
    log(f"Population Size: {DEFAULT_POPULATION_SIZE}")
    log(f"Generations: {MAX_GENERATIONS}")
    log(f"Crossover Probability: {CROSSOVER_PROBABILITY}")
    log("Mutation Probability: 1/chromosome_length\n")
    
    # Both 8-bit and 16-bit encoding
    for bits in [8, 16]:
        log(f"\n{'='*50}")
        log(f"{bits}-BIT ENCODING")
        log('='*50 + '\n')
        
        # Test each De Jong function
        for fid in ['f1', 'f2', 'f3', 'f4']:
            ga = BinaryGA(fid, bits_per_variable=bits)
            config = DEJONG_CONFIGS[fid]
            
            log(f"\n{fid}: {config['name']}")
            log("-" * 30)
            
            # Game faces, everyone! Time to evolve!
            best_chrom, best_fit, best_hist, avg_hist = ga.run()
            
            # Decode and log best solution
            best_decoded = ga.decode(best_chrom)
            log(f"Best Fitness: {best_fit:.6f}")
            log(f"Best Solution (decoded): {[f'{x:.4f}' for x in best_decoded]}")
            
            # Create plot
            create_plot(fid, config, bits, best_fit, best_hist, avg_hist)
    
    log(f"\nAll results saved to {LOG_PATH}")
    log(f"All plots saved to {IMAGES_PATH}/")


if __name__ == "__main__":
    # random.seed(773) # Uncomment for reproducibility
    main()