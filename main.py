import cv2
import numpy as np
from GA import GA

def mse(original_image, upscaled_image):
    mse = np.mean((original_image.astype(np.float32) - upscaled_image.astype(np.float32)) ** 2)
    return mse

def main():
    original = cv2.imread("original.jpg")

    baseline_bicubic = cv2.resize(original, (200,120), cv2.INTER_CUBIC)
    baseline_upscaled_bicubic = cv2.resize(baseline_bicubic, (original.shape[1], original.shape[0]),interpolation=cv2.INTER_CUBIC)
    
    baseline_bicubic_mse = mse(original, baseline_upscaled_bicubic)
    

    baseline_bilinear = cv2.resize(original, (200,120), cv2.INTER_LINEAR)
    baseline_upscaled_bilinear = cv2.resize(baseline_bilinear, (original.shape[1], original.shape[0]),interpolation=cv2.INTER_LINEAR)
    
    baseline_bilinear_mse = mse(original, baseline_upscaled_bilinear)

    print(f"Bicubic MSE={baseline_bicubic_mse:.2f}")
    print(f"Bilinear MSE={baseline_bilinear_mse:.2f}")

    cv2.imwrite("baseline_bicubic.png", baseline_bicubic)
    cv2.imwrite("baseline_bilinear.png", baseline_bilinear)

    ga = GA(
        original_image=original,
        compressed_shape=(120,200,3),
        population_size=100,
        crossover_rate=0.8,
        mutation_rate=0.01
    )

    best_individual = ga.evolve(generations=100, baseline=baseline_bicubic)
    compressed = best_individual.imgArray
    compressed_up = cv2.resize(compressed, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("compressed_ga.png", compressed)
    cv2.imwrite("compressed_ga_upscaled.png", compressed_up)

    # mse_ga = mse(original, compressed_up)
    # print(f"GA MSE {mse_ga:.2f}")

if __name__ == "__main__":
    main()
