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

    def get_rotation_matrix_lvlh(self):
        pos_vec = self.orbital_state_vectors.position
        vel_vec = self.orbital_state_vectors.velocity

        i_vec, j_vec, k_vec = self.get_sat_reference_frame(pos_vec, vel_vec)

        return Rotation.from_matrix(np.array([i_vec, j_vec, k_vec]))

    def get_sat_reference_frame(self, pos_vec, vel_vec):
        i_vec = pos_vec / np.linalg.norm(pos_vec)
        pos_vel_cross = np.cross(pos_vec, vel_vec)
        k_vec = pos_vel_cross / np.linalg.norm(pos_vel_cross)
        j_vec = np.cross(k_vec, i_vec)

        return i_vec, j_vec, k_vec

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

    def convert_state_vectors_to_orbital_elements(self) -> OrbitalElements:
        return OrbitalElements(0, 0, 0, 0, 0, 0)

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
