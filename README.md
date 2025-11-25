# Sample Pseudocode: Evolutionary Approach
### Load original image
original = load_image("test.png") # 2000x1200x3
### Generate baseline
baseline = cv2.resize(original, (200, 120), cv2.INTER_CUBIC)
baseline_upscaled = cv2.resize(baseline, (2000, 1200))
baseline_mse = MSE(original, baseline_upscaled)
### Initialize population with variations of baseline
population = [add_noise(baseline) for _ in range(100)]
### Evolution
for generation in range(500):
### Evaluate fitness
10
for individual in population:
upscaled = cv2.resize(individual, (2000, 1200))
fitness = -MSE(original, upscaled)
### Selection, crossover, mutation
parents = tournament_selection(population)
offspring = crossover(parents)
offspring = mutate(offspring, rate=0.1)
population = offspring
print(f"Gen {generation}: Best MSE = {-max(fitness)}")
### Output best result
best = population[best_index]
save_image("compressed.png", best
