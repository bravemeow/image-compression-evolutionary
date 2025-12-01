import numpy as np
class Individual:

    def __init__(self, shape=(120,200,3), imgArray=None):
        self.shape=shape #size of array (120,200,3)
        if imgArray is None: #if array is empty, create random array
            self.imgArray = np.random.randint(0, 256, shape)
        else:
            self.imgArray=imgArray


    def individualCopy(self):
        return Individual(self.shape,self.imgArray.copy())
    
    # def fitness():
        
