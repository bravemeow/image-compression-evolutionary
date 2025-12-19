import numpy as np
import cv2
from Individual import Individual

class GA:
    def __init__(self, original_image, compressed_shape, 
                 population_size=100, crossover_rate=0.8, mutation_rate=0.1, crossover='one_point'):
        self.original_image = original_image
        self.compressed_shape = compressed_shape
        self.original_shape = original_image.shape
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []
        self.crossover=crossover

    def initialize_population(self, baseline):
        #create population of individuals with random noise
        self.population = []
        for _ in range(self.population_size):
            noisy = baseline.copy().astype(np.int16)
            noisy += np.random.randint(-1, 2, baseline.shape)
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.population.append(Individual(self.compressed_shape, noisy))

    def evaluate_fitness(self):
        #evaluate fitness of each individual
        for individual in self.population:
            individual.individual_fitness(self.original_image, self.original_shape)

            # upscaled = cv2.resize(individual.imgArray, 
            #                      (self.original_shape[1], self.original_shape[0]),
            #                      interpolation=cv2.INTER_CUBIC)
            # mse = np.mean((self.original_image.astype(np.float32) - upscaled.astype(np.float32)) ** 2)
            # individual.fitness = -mse  
            

    def tournament_selection(self, tournament_size=5):
        #select best individual from tournament
        tournament = np.random.choice(self.population, size=tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)

    def one_point_crossover(self, parent1, parent2):
        #crossover two parents to create two children
        child1 = parent1.individualCopy()
        child2 = parent2.individualCopy()

        cross_point = np.random.randint(1, (self.compressed_shape[0]-1))
        child1.imgArray[cross_point:] = parent2.imgArray[cross_point:]
        child2.imgArray[cross_point:] = parent1.imgArray[cross_point:]
        
        return child1, child2
    
    def two_point_crossover(self, parent1, parent2):
        child1 = parent1.individualCopy()
        child2 = parent2.individualCopy()

        while True: #to make sure cross_point1 and cross_point2 are not the same
            cross_point1 =np.random.randint(1, (self.compressed_shape[0]-1)) #while loop will continue to run until both points are different
            cross_point2 =np.random.randint(1, (self.compressed_shape[0]-1))
            if cross_point1!=cross_point2:
                break

        cross_point1, cross_point2=sorted((cross_point1,cross_point2)) #so first point is smaller than second one
        child1.imgArray[cross_point1:cross_point2] = parent2.imgArray[cross_point1:cross_point2]
        child2.imgArray[cross_point1:cross_point2] = parent1.imgArray[cross_point1:cross_point2]
        
        return child1, child2

    
    def mutate(self, individual):
        #mutate individual by adding random noise
        mutated = individual.individualCopy()
        mask = np.random.random(mutated.imgArray.shape) < self.mutation_rate
        noise = np.random.randint(-2, 3, mutated.imgArray.shape)
        mutated.imgArray[mask] = np.clip(mutated.imgArray[mask] + noise[mask], 0, 255)
        return mutated

    def evolve(self, generations=500, baseline=None):
        #evolve population for generations
        #store fitness history for convergence analysis
        fitness_history = []
        
        if baseline is None:
            baseline = cv2.resize(self.original_image, 
                                (self.compressed_shape[1], self.compressed_shape[0]),
                                interpolation=cv2.INTER_CUBIC)
        
        self.initialize_population(baseline)

        for gen in range(generations):

            #evaluate fitness of each individual
            self.evaluate_fitness()
            # Sort population by fitness descending
            sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

            best1 = sorted_population[0].individualCopy() #elitism
            best2 = sorted_population[1].individualCopy()
            
            #save best fitness for convergence curve
            best_mse = -best1.fitness
            fitness_history.append(best_mse)
            print(f"Gen {gen}: Best MSE = {best_mse:.2f}")
            new_population = [best1,best2] #elitism. copies best 2 individuals to next population

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                if self.crossover=='one_point':
                    
                    if np.random.random() < self.crossover_rate:
                        child1, child2 = self.one_point_crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.individualCopy(), parent2.individualCopy()
                else: #if not one point crossover, it will do two point crossover
                    if np.random.random() < self.crossover_rate:
                        child1, child2 = self.two_point_crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.individualCopy(), parent2.individualCopy()
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                child1.individual_fitness(self.original_image, self.original_shape)
                child2.individual_fitness(self.original_image, self.original_shape)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.population_size]
        
        self.evaluate_fitness()
        best_i = max(self.population, key=lambda x: x.fitness)
        print("Best Individual MSE is", -best_i.fitness)
        return best_i, fitness_history
