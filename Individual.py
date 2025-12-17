import cv2
import numpy as np

class Individual:

    def __init__(self, shape=(120,200,3), imgArray=None):
        self.shape=shape #size of array (120,200,3)
        if imgArray is None: #if array is empty, create random array
            self.imgArray = np.random.randint(0, 256, shape, dtype=np.uint8)
        else:
            self.imgArray = imgArray.astype(np.uint8)
        self.fitness = -np.inf

    def individualCopy(self):
        i =Individual(self.shape, self.imgArray.copy())
        i.fitness=self.fitness
        return i

    def individual_fitness(self, original_image, original_shape):
        upscaled = cv2.resize(self.imgArray.astype(np.uint8),
                              (original_shape[1], original_shape[0]),
                              interpolation=cv2.INTER_CUBIC)
        fitness_mse = np.mean((original_image.astype(np.float32) - upscaled.astype(np.float32)) ** 2)
        self.fitness = -fitness_mse
