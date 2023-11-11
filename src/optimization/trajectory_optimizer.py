import numpy as np

from src.prediction.satellite import Satellite


class FitnessEvaluator:

    def __init__(self):
        pass

    def evaluate_fitness(self, ref_sat: Satellite, sat_2: Satellite) -> float:
        rx, ry, rz = ref_sat.get_trajectory_position_per_axis()
        rx2, ry2, rz2 = sat_2.get_trajectory_position_per_axis()

        distances = [self.absolute_distance(np.array([x, y, z]), np.array([x2, y2, z2])) for (x, y, z, x2, y2, z2) in
                     zip(rx, ry, rz, rx2, ry2, rz2)]

        return sum(distances)

    def absolute_distance(self, vec_ref: np.array, vec_2: np.array):
        dist = np.linalg.norm(vec_2 - vec_ref)
        return dist
