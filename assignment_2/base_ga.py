from abc import ABC, abstractmethod
import random
import matplotlib.pyplot as plt
import os


class GeneticAlgorithm(ABC):
    def __init__(self, pop_size=20, max_generations=50, crossover_prob=0.9, log_path=None, images_path=None):
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.log_path = log_path or "ga_output.log"
        self.images_path = images_path or "ga_images"
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
    def crossover(self, parent1, parent2):
        pass
    
    @abstractmethod
    def mutate(self, individual):
        pass
    
    @abstractmethod
    def select(self, population, fitness_values):
        pass
    
    @abstractmethod
    def is_better(self, fitness1, fitness2):
        # minimization: fitness1 < fitness2
        # maximization: fitness1 > fitness2
        pass
    
    def generate_population(self):
        return [self.generate_individual() for _ in range(self.pop_size)]
    
    def log(self, message):
        print(message)
        with open(self.log_path, 'a') as f:
            f.write(message + '\n')
    
    def initialize_logging(self):
        open(self.log_path, 'w').close()
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)
    
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
            best_gen_fitness = self.get_best_fitness(fitness_values)
            avg_gen_fitness = sum(fitness_values) / len(fitness_values)
            self.best_fitness_history.append(best_gen_fitness)
            self.avg_fitness_history.append(avg_gen_fitness)
            
            # Log progress
            if gen % 10 == 0 or gen == self.max_generations - 1:
                self.log(f"Generation {gen}: Best = {best_gen_fitness:.6f}, Avg = {avg_gen_fitness:.6f}")
            
            # Create next generation
            new_population = []
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.select(population, fitness_values)
                parent2 = self.select(population, fitness_values)
                
                # Crossover
                if random.random() < self.crossover_prob:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = self.copy_individual(parent1), self.copy_individual(parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Ensure exact population size
            population = new_population[:self.pop_size]
        
        return self.best_overall, self.best_overall_fitness
    
    def get_best_fitness(self, fitness_values):
        best = fitness_values[0]
        for f in fitness_values[1:]:
            if self.is_better(f, best):
                best = f
        return best
    
    def copy_individual(self, individual):
        if isinstance(individual, list):
            return individual.copy()
    
    def plot_fitness(self, title="GA Fitness Progress"):
        plt.figure(figsize=(10, 6))
        generations = range(len(self.best_fitness_history))
        
        plt.plot(generations, self.best_fitness_history, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(generations, self.avg_fitness_history, 'r--', label='Average Fitness', linewidth=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'{title}\nBest Fitness: {self.best_overall_fitness:.6f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()