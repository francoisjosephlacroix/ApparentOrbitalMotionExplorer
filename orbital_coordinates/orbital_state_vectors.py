import numpy as np


class OrbitalStateVectors:
    def __init__(self, position: np.array, velocity: np.array):
        self.position: np.array = position
        self.velocity: np.array = velocity
