from base_ga import GeneticAlgorithm
import random
import math
import matplotlib.pyplot as plt
import os


def log(message, log_path="ga_output.log"):
    print(message)
    with open(log_path, 'a') as f:
        f.write(message + '\n')


DEJONG_CONFIGS = {
    'f1': {'name': 'Sphere Model', 'n_vars': 3, 'bounds': (-5.12, 5.12)},
    'f2': {'name': 'Weighted Sphere Model', 'n_vars': 2, 'bounds': (-2.048, 2.048)},
    'f3': {'name': 'Step Function', 'n_vars': 4, 'bounds': (-5.12, 5.12)},
    'f4': {'name': 'Noisy Quartic', 'n_vars': 4, 'bounds': (-1.28, 1.28)}
}


def f1_sphere(x):
    return sum(xi**2 for xi in x)


def f2_weighted_sphere(x):
    return 100 * (x[0]**2 - x[1])**2 + (1 - x[0])**2


def f3_step(x):
    return sum(math.floor(xi) for xi in x)


def f4_noisy_quartic(x):
    return sum((i+1) * xi**4 for i, xi in enumerate(x)) + random.random()


DEJONG_FUNCTIONS = {
    'f1': f1_sphere,
    'f2': f2_weighted_sphere,
    'f3': f3_step,
    'f4': f4_noisy_quartic
}


class BinaryGA(GeneticAlgorithm):
    def __init__(self, function_id, bits_per_variable, **kwargs):
        super().__init__(**kwargs)
        
        config = DEJONG_CONFIGS[function_id]
        self.function_id = function_id
        self.function_name = config['name']
        self.n_vars = config['n_vars']
        self.bounds = config['bounds']
        self.bits_per_variable = bits_per_variable
        self.chromosome_length = self.n_vars * bits_per_variable
        
        # Calculate precision
        lower, upper = self.bounds
        self.precision = (upper - lower) / (2**bits_per_variable - 1)
        
        # Store evaluation function
        self.eval_func = DEJONG_FUNCTIONS[function_id]
    
    def generate_individual(self):
        return [random.randint(0, 1) for _ in range(self.chromosome_length)]
    
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
    
    def evaluate(self, individual):
        decoded = self.decode(individual)
        return self.eval_func(decoded)
    
    def is_better(self, fitness1, fitness2):
        return fitness1 < fitness2
    
    def select_parents(self, population, fitness_values, n_parents):
        # Select n_parents
        parents = []
        for _ in range(n_parents):
            parent = self.fitness_proportionate_select_one(population, fitness_values)
            parents.append(parent)
        return parents
    
    def fitness_proportionate_select_one(self, population, fitness_values):
        # Shift fitness values to ensure all are positive
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
        
        # Fallback
        return population[-1].copy()
    
    def crossover(self, parents):
        # Two-point crossover
        parent1, parent2 = parents[0], parents[1]
        point1 = random.randint(1, self.chromosome_length - 2)
        point2 = random.randint(point1 + 1, self.chromosome_length - 1)
        
        offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return [offspring1, offspring2]
    
    def mutate(self, individual):
        # Bitwise mutation
        mutation_probability = 1.0 / self.chromosome_length
        
        for i in range(self.chromosome_length):
            if random.random() < mutation_probability:
                individual[i] = 1 - individual[i]
        
        return individual
    
    def get_best_solution_info(self):
        if self.best_overall:
            decoded = self.decode(self.best_overall)
            return decoded, self.best_overall_fitness
        return None, None


def main():
    log_path = "ga_output.log"
    images_path = "ga_images"
    
    open(log_path, 'w').close()
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    
    log("Q4: GA Execution Results\n", log_path)
    log("Population Size: 20", log_path)
    log("Generations: 50", log_path)
    log("Crossover Probability: 0.9", log_path)
    log("Mutation Probability: 1/chromosome_length\n", log_path)
    
    # Test both 8-bit and 16-bit encoding
    for bits in [8, 16]:
        log(f"\n{'='*50}", log_path)
        log(f"{bits}-BIT ENCODING", log_path)
        log('='*50 + '\n', log_path)
        
        # Test each De Jong function
        for fid in ['f1', 'f2', 'f3', 'f4']:
            ga = BinaryGA(fid, bits_per_variable=bits, elitism_count=2)
            
            log(f"\n{fid}: {ga.function_name}", log_path)
            log("-" * 30, log_path)
            
            # Run GA
            best_chrom, best_fit = ga.run()
            
            # Get decoded solution
            best_decoded, _ = ga.get_best_solution_info()
            log(f"Best Fitness: {best_fit:.6f}", log_path)
            log(f"Best Solution (decoded): {[f'{x:.4f}' for x in best_decoded]}", log_path)
            
            # Create and save plot
            fig = ga.plot_fitness(title=f'{ga.function_name} - {bits}-bit Encoding')
            filename = f"{images_path}/{fid}_{bits}bit_fitness.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            log(f"Plot saved: {filename}", log_path)
    
    log(f"\nAll results saved to {log_path}", log_path)
    log(f"All plots saved to {images_path}/", log_path)


if __name__ == "__main__":
    main()