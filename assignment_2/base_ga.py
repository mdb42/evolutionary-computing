from abc import ABC, abstractmethod
import random
import matplotlib.pyplot as plt


class GeneticAlgorithm(ABC):
    def __init__(self, pop_size=20, max_generations=50, crossover_prob=0.9, 
                 elitism_count=0):
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.elitism_count = elitism_count
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_overall = None
        self.best_overall_fitness = None
        
    @abstractmethod
    def generate_individual(self):
        pass
    
    @abstractmethod
    def evaluate(self, individual):
        pass
    
    @abstractmethod
    def select_parents(self, population, fitness_values, n_parents):
        pass
    
    @abstractmethod
    def crossover(self, parents):
        pass
    
    @abstractmethod
    def mutate(self, individual):
        pass
    
    @abstractmethod
    def is_better(self, fitness1, fitness2):
        pass

    def generate_population(self):
        return [self.generate_individual() for _ in range(self.pop_size)]
    
    def evolve_generation(self, population, fitness_values):
        new_population = []
        
        # Elitism
        if self.elitism_count > 0:
            elite_indices = self.get_elite_indices(fitness_values, self.elitism_count)
            for idx in elite_indices:
                new_population.append(self.copy_individual(population[idx]))
        
        # Fill rest of population
        while len(new_population) < self.pop_size:
            # Select parents (default: 2, but overridable)
            parents = self.select_parents(population, fitness_values, 2)
            
            # Crossover
            if random.random() < self.crossover_prob:
                offspring = self.crossover(parents)
            else:
                offspring = [self.copy_individual(p) for p in parents]
            
            # Mutation
            offspring = [self.mutate(child) for child in offspring]
            
            new_population.extend(offspring)
        
        # Ensure exact population size
        return new_population[:self.pop_size]
    
    def get_elite_indices(self, fitness_values, n_elite):
        indexed_fitness = list(enumerate(fitness_values))
        indexed_fitness.sort(key=lambda x: x[1], reverse=False)
        
        # Sort
        for i in range(len(indexed_fitness)):
            for j in range(i + 1, len(indexed_fitness)):
                if self.is_better(indexed_fitness[j][1], indexed_fitness[i][1]):
                    indexed_fitness[i], indexed_fitness[j] = indexed_fitness[j], indexed_fitness[i]
        
        return [idx for idx, _ in indexed_fitness[:n_elite]]
    
    def run(self):
        # Initialize
        population = self.generate_population()
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_overall = None
        self.best_overall_fitness = None
        
        # Evolution loop
        for gen in range(self.max_generations):
            # Evaluate current population
            fitness_values = []
            for individual in population:
                fitness = self.evaluate(individual)
                fitness_values.append(fitness)
                
                # Track best overall
                if self.best_overall_fitness is None or self.is_better(fitness, self.best_overall_fitness):
                    self.best_overall_fitness = fitness
                    self.best_overall = self.copy_individual(individual)
            
            # Record statistics
            self.record_statistics(fitness_values, gen)
            
            # Create next generation
            population = self.evolve_generation(population, fitness_values)
        
        return self.best_overall, self.best_overall_fitness
    
    def record_statistics(self, fitness_values, generation):
        best_gen_fitness = self.get_best_fitness(fitness_values)
        avg_gen_fitness = sum(fitness_values) / len(fitness_values)
        self.best_fitness_history.append(best_gen_fitness)
        self.avg_fitness_history.append(avg_gen_fitness)
    
    def get_best_fitness(self, fitness_values):
        best = fitness_values[0]
        for f in fitness_values[1:]:
            if self.is_better(f, best):
                best = f
        return best
    
    def copy_individual(self, individual):
        if isinstance(individual, list):
            return individual.copy()
        else:
            return individual
    
    
    def plot_fitness(self, title="GA Fitness Progress"):
        plt.figure(figsize=(10, 6))
        generations = range(len(self.best_fitness_history))
        
        plt.plot(generations, self.best_fitness_history, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(generations, self.avg_fitness_history, 'r--', label='Average Fitness', linewidth=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()