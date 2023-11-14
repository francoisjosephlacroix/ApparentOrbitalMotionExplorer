import math
from typing import Tuple

import numpy as np

from prediction.constants import GRAVITATIONAL_PARAMETER


class OrbitalStateVectors:
    def __init__(self, position: np.array, velocity: np.array):
        self.position: np.array = position
        self.velocity: np.array = velocity

    def get_position_magnitude(self) -> float:
        return np.linalg.norm(self.position)

    def get_velocity_magnitude(self) -> float:
        return np.linalg.norm(self.velocity)

    def eccentricity_vec(self) -> np.array:
        velocity_magnitude = self.get_velocity_magnitude()
        velocity_magnitude_square = (velocity_magnitude ** 2)
        position_magnitude = self.get_position_magnitude()
        position_component = (velocity_magnitude_square - GRAVITATIONAL_PARAMETER / position_magnitude) * self.position
        velocity_component = (np.dot(self.position, self.velocity) * self.velocity)
        eccentricity_vec = (1 / GRAVITATIONAL_PARAMETER) * (position_component - velocity_component)
        return eccentricity_vec

    def semi_major_axis(self) -> float:
        velocity_magnitude = self.get_velocity_magnitude()
        velocity_magnitude_square = (velocity_magnitude ** 2)
        position_magnitude = self.get_position_magnitude()

        mech_energy = (velocity_magnitude_square / 2) - (GRAVITATIONAL_PARAMETER / position_magnitude)
        semi_major_axis = -GRAVITATIONAL_PARAMETER / (2 * mech_energy)

        return semi_major_axis

    def inclination(self) -> float:
        angular_momentum = self.angular_momentum()
        angular_momentum_magnitude = np.linalg.norm(angular_momentum)

        inclination = math.acos(angular_momentum[2] / angular_momentum_magnitude)

        return inclination

    def angular_momentum(self) -> np.array:
        return np.cross(self.position, self.velocity)

    def right_ascension(self) -> float:
        ascending_node = self.ascending_node()
        ascending_node_magnitude = np.linalg.norm(ascending_node)
        right_ascention = math.acos(np.dot(np.array([1.0, 0.0, 0.0]), ascending_node) / ascending_node_magnitude)

        if ascending_node[1] >= 0.0:
            return right_ascention
        else:
            return 2 * math.pi - right_ascention

    def ascending_node(self) -> np.array:
        return np.cross(np.array([0.0, 0.0, 1.0]), self.angular_momentum())

    def argument_periapsis(self) -> float:
        eccentricity_vec = self.eccentricity_vec()
        eccentricity = np.linalg.norm(eccentricity_vec)
        ascending_node = self.ascending_node()
        ascending_node_magnitude = np.linalg.norm(ascending_node)

        argument_periapsis = math.acos(
            np.dot(ascending_node, eccentricity_vec) / (ascending_node_magnitude * eccentricity))

        if eccentricity_vec[2] >= 0:
            return argument_periapsis
        else:
            return 2 * math.pi - argument_periapsis

    def true_anomaly(self) -> float:
        eccentricity_vec = self.eccentricity_vec()
        eccentricity = np.linalg.norm(eccentricity_vec)
        true_anomaly = math.acos(
            np.dot(eccentricity_vec, self.position) / (eccentricity * self.get_position_magnitude()))

        if np.dot(self.position, self.velocity) >= 0:
            return true_anomaly
        else:
            return 2 * math.pi - true_anomaly

    def get_sat_reference_frame(self) -> Tuple[np.array, np.array, np.array]:

        i_vec = self.position / np.linalg.norm(self.position)
        pos_vel_cross = np.cross(self.position, self.velocity)
        k_vec = pos_vel_cross / np.linalg.norm(pos_vel_cross)
        j_vec = np.cross(k_vec, i_vec)

        return i_vec, j_vec, k_vec
