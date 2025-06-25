"""
CSC 742 Evolutionary Computing
Assignment 2: Genetic Algorithms with Binary Encoding for De Jong Test Functions
Author: Matthew D. Branson
Date: 2025-06-24
"""
import random
import math


DEFAULT_POPULATION_SIZE = 50
CROSSOVER_PROBABILITY = 0.90
LOG_PATH = "ga_output.log"
DEJONG_CONFIGS = {
    'f1': {'name': 'Sphere Model', 'n_vars': 3, 'bounds': (-5.12, 5.12)},
    'f2': {'name': 'Weighted Sphere Model', 'n_vars': 2, 'bounds': (-2.048, 2.048)},
    'f3': {'name': 'Step Function', 'n_vars': 4, 'bounds': (-5.12, 5.12)},
    'f4': {'name': 'Noisy Quartic', 'n_vars': 4, 'bounds': (-1.28, 1.28)}
}

# De Jong Test Function Evaluations
def f1_sphere(x):
    return sum(xi**2 for xi in x)

def f2_weighted_sphere(x):
    return 100 * (x[0]**2 - x[1])**2 + (1 - x[0])**2

def f3_step(x):
    return sum(math.floor(xi) for xi in x)

def f4_noisy_quartic(x):
    return sum((i+1) * xi**4 for i, xi in enumerate(x)) + random.random()

# Map function IDs to their evaluation functions
DEJONG_FUNCTIONS = {
    'f1': f1_sphere,
    'f2': f2_weighted_sphere,
    'f3': f3_step,
    'f4': f4_noisy_quartic
}

def log(message):
    print(message)
    with open(LOG_PATH, 'a') as f:
        f.write(message + '\n')

class BinaryGA:
    def __init__(self, function_id, bits_per_variable):
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
        return [[random.randint(0, 1) for _ in range(self.chromosome_length)] 
                for _ in range(size)]
    
    def decode(self, chromosome):
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
        # Add small constant to avoid division by zero
        transformed_fitness = [1.0 / (f + 1.0) for f in fitness_values]
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
        mutation_probability = 1.0 / self.chromosome_length
        
        for i in range(self.chromosome_length):
            if random.random() < mutation_probability:
                # Flip bit
                chromosome[i] = 1 - chromosome[i]
        
        return chromosome

def main():
    # Clear log
    open(LOG_PATH, 'w').close()
    
    log("Binary Encoding, Initialization, and Evaluation\n")
    
    # Both 8-bit and 16-bit encoding
    for bits in [8, 16]:
        log(f"\n{'='*50}")
        log(f"{bits}-BIT ENCODING")
        log('='*50 + '\n')
        
        # Test each De Jong function
        for fid in ['f1', 'f2', 'f3', 'f4']:
            ga = BinaryGA(fid, bits_per_variable=bits)
            config = DEJONG_CONFIGS[fid]
            
            log(f"{fid}: {config['name']} ({config['n_vars']} vars, bounds {config['bounds']})")
            log(f"  Precision: {ga.precision:.6f}")
            
            # Generate small sample population
            pop = ga.generate_population(size=20)
            
            # Show first individual
            chromosome = pop[0]
            binary_str = ''.join(map(str, chromosome))
            real_vals = ga.decode(chromosome)
            
            # Get the mapped evaluation function
            eval_func = DEJONG_FUNCTIONS[fid]
            fitness = eval_func(real_vals)
            
            # For 16-bit, truncate binary display for readability
            if bits == 16:
                display_binary = binary_str[:32] + '...' if len(binary_str) > 32 else binary_str
            else:
                display_binary = binary_str
                
            log(f"  Sample: {display_binary}")
            log(f"  Decoded: {[f'{x:.4f}' for x in real_vals]}")
            log(f"  f({fid}) = {fitness:.6f}\n")
    
    # Testing out the GA Operations
    log("\n" + "="*50)
    log("Q3: GA Operations Demo (using 8-bit f1)")
    log("="*50 + "\n")
    
    ga = BinaryGA('f1', bits_per_variable=8)
    eval_func = DEJONG_FUNCTIONS['f1']
    
    pop = ga.generate_population(size=20)
    fitness_values = [eval_func(ga.decode(chrom)) for chrom in pop]
    
    #Initialization
    log("Initial Population:")
    for i, (chrom, fit) in enumerate(zip(pop, fitness_values)):
        log(f"  {i+1}: {''.join(map(str, chrom))} fitness={fit:.4f}")
    
    # Selection
    log("\nRoulette Wheel Selection:")
    selected = ga.fitness_proportionate_selection(pop, fitness_values)
    log(f"  Selected: {''.join(map(str, selected))}")
    
    # Crossover
    parent1 = pop[0]
    parent2 = pop[1]
    log(f"\nTwo-point Crossover (p={CROSSOVER_PROBABILITY}):")
    log(f"  Parent 1: {''.join(map(str, parent1))}")
    log(f"  Parent 2: {''.join(map(str, parent2))}")
    child1, child2 = ga.two_point_crossover(parent1.copy(), parent2.copy())
    log(f"  Child 1:  {''.join(map(str, child1))}")
    log(f"  Child 2:  {''.join(map(str, child2))}")
    
    # Mutation
    log(f"\nMutation (p=1/{ga.chromosome_length}):")
    original = pop[0].copy()
    mutated = ga.bitwise_mutation(original.copy())
    log(f"  Original: {''.join(map(str, original))}")
    log(f"  Mutated:  {''.join(map(str, mutated))}")

    log(f"\nPopulation size: {DEFAULT_POPULATION_SIZE}")
    log(f"Crossover probability: {CROSSOVER_PROBABILITY}")
    log(f"Mutation probability: 1/chromosome_length")


if __name__ == "__main__":
    # random.seed(42)
    main()