import math
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

from src.prediction.constants import GRAVITATIONAL_PARAMETER
from src.orbital_coordinates.orbital_state_vectors import OrbitalStateVectors
from src.orbital_coordinates.orbital_elements import OrbitalElements


class SatelliteStatus:
    def __init__(self, timestamp: datetime, orbital_elements: OrbitalElements):
        self.timestamp: datetime = timestamp
        self.orbital_elements: OrbitalElements = orbital_elements
        self.orbital_state_vectors = self.convert_orbital_elements_to_state_vectors()

    @staticmethod
    def from_state_vector(timestamp: datetime, orbital_state_vectors: OrbitalStateVectors):
        orbital_elements: OrbitalElements = SatelliteStatus.convert_state_vectors_to_orbital_elements(
            orbital_state_vectors)
        return SatelliteStatus(timestamp, orbital_elements)

    @staticmethod
    def convert_state_vectors_to_orbital_elements(orbital_state_vectors: OrbitalStateVectors) -> OrbitalElements:
        eccentricity_vec = orbital_state_vectors.eccentricity_vec()
        eccentricity = np.linalg.norm(eccentricity_vec)
        semi_major_axis = orbital_state_vectors.semi_major_axis()
        inclination = orbital_state_vectors.inclination()
        longitude_ascending_node = orbital_state_vectors.right_ascension()
        argument_periapsis = orbital_state_vectors.argument_periapsis()
        true_anomaly = orbital_state_vectors.true_anomaly()

        return OrbitalElements(eccentricity, semi_major_axis, inclination, longitude_ascending_node, argument_periapsis,
                               true_anomaly)

    def get_rotation_matrix_lvlh(self):
        i_vec, j_vec, k_vec = self.orbital_state_vectors.get_sat_reference_frame()

        return Rotation.from_matrix(np.array([i_vec, j_vec, k_vec]))

    def convert_classical_to_perifocal(self):
        # AyansolaOgundele_NonlinearDynamicsandControlofSpacecraftRelativeMotion.pdf
        # Pages 5-10 eq 1.4, 1.20
        radius = (self.orbital_elements.semi_major_axis * (1 - self.orbital_elements.eccentricity ** 2)) / (
                1 + self.orbital_elements.eccentricity * math.cos(self.orbital_elements.true_anomaly))

        rx = radius * math.cos(self.orbital_elements.true_anomaly)
        ry = radius * math.sin(self.orbital_elements.true_anomaly)
        rz = 0.0
        radius_vec = np.array([rx, ry, rz])

        gravitational_ratio = math.sqrt(GRAVITATIONAL_PARAMETER / self.orbital_elements.semi_latus_rectum)

        vx = gravitational_ratio * (-math.sin(self.orbital_elements.true_anomaly))
        vy = gravitational_ratio * (self.orbital_elements.eccentricity + math.cos(self.orbital_elements.true_anomaly))
        vz = 0.0
        velocity_vec = np.array([vx, vy, vz])

        return radius_vec, velocity_vec

    def rotate_state_vector(self, vector: np.array) -> np.array:
        rotation_matrix = Rotation.from_euler("ZXZ", [-self.orbital_elements.argument_periapsis,
                                                      -self.orbital_elements.inclination,
                                                      -self.orbital_elements.longitude_ascending_node])
        vector_rot = vector @ rotation_matrix.as_matrix()
        return vector_rot

    def convert_orbital_elements_to_state_vectors(self) -> OrbitalStateVectors:
        (radius_vec, velocity_vec) = self.convert_classical_to_perifocal()

        position_state_vector = self.rotate_state_vector(radius_vec)
        velocity_state_vector = self.rotate_state_vector(velocity_vec)

        return OrbitalStateVectors(position_state_vector, velocity_state_vector)

    def get_eccentricity(self):
        return self.orbital_elements.eccentricity

    def get_semi_major_axis(self):
        return self.orbital_elements.semi_major_axis

    def get_inclination(self):
        return self.orbital_elements.inclination

    def get_longitude_ascending_node(self):
        return self.orbital_elements.longitude_ascending_node

    def get_argument_periapsis(self):
        return self.orbital_elements.argument_periapsis

    def get_true_anomaly(self):
        return self.orbital_elements.true_anomaly

    def get_position(self):
        return self.orbital_state_vectors.position

    def get_velocity(self):
        return self.orbital_state_vectors.velocity
