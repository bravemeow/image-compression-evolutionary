import numpy as np
import cv2
from Individual import Individual

class GA:
    def __init__(self, original_image, compressed_shape, 
                 population_size=100, crossover_rate=0.8, mutation_rate=0.1):
        self.original_image = original_image
        self.compressed_shape = compressed_shape
        self.original_shape = original_image.shape
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []

    def initialize_population(self, baseline):
        #create population of individuals with random noise
        self.population = []
        for _ in range(self.population_size):
            noisy = baseline.copy().astype(np.int16)
            noisy += np.random.randint(-15, 16, baseline.shape)
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.population.append(Individual(self.compressed_shape, noisy))

    def evaluate_fitness(self):
        #evaluate fitness of each individual
        for individual in self.population:
            upscaled = cv2.resize(individual.imgArray, 
                                 (self.original_shape[1], self.original_shape[0]),
                                 interpolation=cv2.INTER_CUBIC)
            mse = np.mean((self.original_image - upscaled) ** 2)
            individual.fitness = -mse  

    def tournament_selection(self, tournament_size=3):
        #select best individual from tournament
        tournament = np.random.choice(self.population, size=tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1, parent2):
        #crossover two parents to create two children
        child1 = parent1.individualCopy()
        child2 = parent2.individualCopy()
        
        mid = self.compressed_shape[0] // 2
        child1.imgArray[mid:] = parent2.imgArray[mid:]
        child2.imgArray[mid:] = parent1.imgArray[mid:]
        
        return child1, child2

    def mutate(self, individual):
        #mutate individual by adding random noise
        mutated = individual.individualCopy()
        mask = np.random.random(mutated.imgArray.shape) < self.mutation_rate
        noise = np.random.randint(-10, 11, mutated.imgArray.shape)
        mutated.imgArray[mask] = np.clip(mutated.imgArray[mask] + noise[mask], 0, 255)
        return mutated

    def evolve(self, generations=500, baseline=None):
        #evolve population for generations
        if baseline is None:
            baseline = cv2.resize(self.original_image, 
                                (self.compressed_shape[1], self.compressed_shape[0]),
                                interpolation=cv2.INTER_CUBIC)
        
        self.initialize_population(baseline)
        
        for gen in range(generations):
            #evaluate fitness of each individual
            self.evaluate_fitness()
            
            best = max(self.population, key=lambda x: x.fitness)
            print(f"Gen {gen}: Best MSE = {-best.fitness:.2f}")

            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.individualCopy(), parent2.individualCopy()
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.population_size]
        
        self.evaluate_fitness()
        return max(self.population, key=lambda x: x.fitness)