"""
CSC 742 Evolutionary Computing
Assignment 2: Genetic Algorithms with Binary Encoding for De Jong Test Functions
Author: Matthew D. Branson
Date: 2025-06-24
"""
import random
import math


DEFAULT_POPULATION_SIZE = 50
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

    log(f"Population size: {DEFAULT_POPULATION_SIZE}")


if __name__ == "__main__":
    random.seed(42)
    main()