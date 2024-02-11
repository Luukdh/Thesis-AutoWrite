import sys
import numpy as np
import random

class RandomTransformation():
    def __init__(self):
        pass

    def getRotationMatrix(self):
        # print("rotate")
        alpha = (random.random() * 2 - 1) * 2 * np.pi / 36
        return np.array([[np.cos(alpha), -np.sin(alpha)],
                             [np.sin(alpha), np.cos(alpha)]])
    
    def getHShearMatrix(self):
        # print("h_shear")
        return np.array([[1, 0],
                         [random.uniform(-0.2, 0.2), 1]])
    
    def getVShearMatrix(self):
        # print("v_shear")
        return np.array([[1, random.uniform(-0.2, 0.2)],
                         [0, 1]])
    
    def getHFlatMatrix(self):
        # print("h_flat")
        return np.array([[1, 0],
                         [0, 1 - random.uniform(-0.1, 0.1)]])
    
    def getVFlatMatrix(self):
        # print("v_flat")
        return np.array([[1 - random.uniform(-0.1, 0.1), 0],
                         [0, 1]])

    def __call__(self, sample):
        x, y = xy = sample[:-1]
        p = sample[-1]
        transform = ["none",
                     "reverse", 
                     "rotate", 
                     "h_shear",
                     "h_flat",
                     "v_flat",
                     "v_shear",
                     "translate",]
        
        random_transform = random.choice(transform)
        matrix = None
        if random_transform == "reverse":
            # print("reverse")
            return np.flip(sample, axis=1)
        elif random_transform == "translate":
            # print("translate")
            return sample + np.array([[random.uniform(-10, 10)], 
                                      [random.uniform(-10, 10)], 
                                      [0]])
        elif random_transform == "rotate":
            matrix = self.getRotationMatrix()
        elif random_transform == "h_shear":
            matrix = self.getHShearMatrix()
        elif random_transform == "h_flat":
            matrix = self.getHFlatMatrix()
        elif random_transform == "v_flat":
            matrix = self.getVFlatMatrix()
        elif random_transform == "v_shear":
            matrix = self.getVShearMatrix()
        else:
            # print("none")
            return np.stack([x, y, p], axis=0)

        x, y = xy = matrix @ xy
        return np.stack([x, y, p], axis=0)

