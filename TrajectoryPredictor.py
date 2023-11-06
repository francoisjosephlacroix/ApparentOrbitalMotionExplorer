from datetime import datetime, timedelta
from typing import List
import numpy as np
import scipy
import math
from scipy.spatial.transform import Rotation

GRAVITATIONAL_PARAMETER = 3.986E5  # (km^3/s^2)


# class Vector:
#
#     def __init__(self, values: np.array):
#         self.values: np.array = values

class PositionVector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class VelocityVector:
    def __init__(self, vx, vy, vz):
        self.vx = vx
        self.vy = vy
        self.vz = vz


class OrbitalStateVectors:
    def __init__(self, position: np.array, velocity: np.array):
        self.position: np.array = position
        self.velocity: np.array = velocity


class OrbitalElements:
    def __init__(self, eccentricity, semi_major_axis, inclination, longitude_ascending_node, argument_periapsis,
                 true_anomaly):
        self.eccentricity = eccentricity
        self.semi_major_axis = semi_major_axis  # in km
        self.inclination = inclination  # in rads
        self.longitude_ascending_node = longitude_ascending_node  # in rads
        self.argument_periapsis = argument_periapsis  # in rads
        self.true_anomaly = true_anomaly  # in rads
        self.semi_latus_rectum = self.semi_major_axis * (1 - self.eccentricity ** 2)

class SatelliteStatus:
    def __init__(self, timestamp: datetime, orbital_elements: OrbitalElements):
        self.timestamp: datetime = timestamp
        self.orbital_elements: OrbitalElements = orbital_elements
        self.orbital_state_vectors = self.convert_orbital_elements_to_state_vectors()

    def convert_classical_to_perifocal(self):
        # AyansolaOgundele_NonlinearDynamicsandControlofSpacecraftRelativeMotion.pdf
        # Pages 5-10 eq 1.4, 1.20
        radius = (self.orbital_elements.semi_major_axis * (1 - self.orbital_elements.eccentricity ** 2)) / (1 + self.orbital_elements.eccentricity * math.cos(self.orbital_elements.true_anomaly))

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
        rotation_matrix = Rotation.from_euler("ZXZ", [-self.orbital_elements.argument_periapsis, -self.orbital_elements.inclination, -self.orbital_elements.longitude_ascending_node])
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


class TrajectoryPredictor:

    def __init__(self, step, eccentric_anomaly_tolerance=10e-10, eccentric_anomaly_max_iter=100):
        self.step: timedelta = step  # One time increment, in seconds
        self.eccentric_anomaly_tolerance = eccentric_anomaly_tolerance
        self.eccentric_anomaly_max_iter = eccentric_anomaly_max_iter

    def convert_rad_angle_to_normal_range(self, angle):
        if angle >= 0 and angle < 2 * math.pi:
            return angle
        else:
            return (angle + 2 * math.pi) % (2 * math.pi)

    def choose_half_plane_cos(self, old_value, new_value):
        # Values in rad

        old_value = self.convert_rad_angle_to_normal_range(old_value)
        new_value = self.convert_rad_angle_to_normal_range(new_value)

        if old_value <= math.pi:
            if new_value <= math.pi:
                return new_value
            else:
                return self.convert_rad_angle_to_normal_range(math.pi * 2 - new_value)
        else:
            if new_value > math.pi:
                return new_value
            else:
                return self.convert_rad_angle_to_normal_range(math.pi * 2 - new_value)

    def choose_half_plane_sin(self, old_value, new_value):
        # Values in rad

        old_value = self.convert_rad_angle_to_normal_range(old_value)
        new_value = self.convert_rad_angle_to_normal_range(new_value)

        if old_value <= math.pi / 2 or old_value > 3 * math.pi / 2:
            if new_value <= math.pi / 2 or new_value > 3 * math.pi / 2:
                return new_value
            else:
                return self.convert_rad_angle_to_normal_range(math.pi - new_value)
        else:
            if new_value > math.pi / 2 and new_value <= 3 * math.pi / 2:
                return new_value
            else:
                return self.convert_rad_angle_to_normal_range(math.pi - new_value)

    def compute_mean_motion(self, semi_major_axis):
        return math.sqrt(GRAVITATIONAL_PARAMETER / (semi_major_axis ** 3))

    def compute_eccentric_anomaly(self, eccentricity, true_anomaly):
        cos_true_anomaly = math.cos(true_anomaly)
        eccentric_anomaly = math.acos((eccentricity + cos_true_anomaly) / (1 + eccentricity * cos_true_anomaly))

        return self.choose_half_plane_cos(true_anomaly, eccentric_anomaly)

    def compute_mean_anomaly(self, eccentric_anomaly, eccentricity):
        mean_anomaly = eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly)
        return self.choose_half_plane_cos(eccentric_anomaly, mean_anomaly)

    def compute_mean_anomaly_future(self, mean_anomaly, mean_motion):
        mean_anomaly_future = mean_anomaly + mean_motion * self.step.seconds.real
        return self.convert_rad_angle_to_normal_range(mean_anomaly_future)

    def compute_eccentric_anomaly_future(self, mean_anomaly_future, eccentricity):
        eccentric_anomaly_future = mean_anomaly_future
        nb_iter = 0

        while nb_iter < self.eccentric_anomaly_max_iter:

            new_eccentric_anomaly_future = mean_anomaly_future + eccentricity * math.sin(eccentric_anomaly_future)

            if abs(new_eccentric_anomaly_future - eccentric_anomaly_future) < self.eccentric_anomaly_tolerance:
                break
            else:
                eccentric_anomaly_future = new_eccentric_anomaly_future

        return new_eccentric_anomaly_future

    def compute_true_anomaly_future(self, eccentric_anomaly_future, eccentricity):
        true_anomaly_future = math.acos((math.cos(eccentric_anomaly_future) - eccentricity) / (
                    1 - eccentricity * math.cos(eccentric_anomaly_future)))
        return self.choose_half_plane_cos(eccentric_anomaly_future, true_anomaly_future)

    def next_step(self, satellite_status: SatelliteStatus) -> SatelliteStatus:
        # Find initial mean motion
        mean_motion_initial = self.compute_mean_motion(satellite_status.get_semi_major_axis())

        # Find E initial (eccentric anomaly)
        eccentric_anomaly_initial = self.compute_eccentric_anomaly(satellite_status.get_eccentricity(),
                                                                  satellite_status.get_true_anomaly())

        # Find M initial (mean anomaly)
        mean_anomaly_initial = self.compute_mean_anomaly(eccentric_anomaly_initial, satellite_status.get_eccentricity())

        # Move mean anomaly to the desired time

        mean_anomaly_future = self.compute_mean_anomaly_future(mean_anomaly_initial, mean_motion_initial)

        # Solve for E future (eccentric anomaly)

        eccentric_anomaly_future = self.compute_eccentric_anomaly_future(mean_anomaly_future,
                                                                         satellite_status.get_eccentricity())

        # Finv v future (true anomaly)

        true_anomaly_future = self.compute_true_anomaly_future(eccentric_anomaly_future,
                                                               satellite_status.get_eccentricity())

        new_orbital_elements = OrbitalElements(satellite_status.get_eccentricity(),
                                               satellite_status.get_semi_major_axis(),
                                               satellite_status.get_inclination(),
                                               satellite_status.get_longitude_ascending_node(),
                                               satellite_status.get_argument_periapsis(),
                                               true_anomaly_future)
        new_satellite_status = SatelliteStatus(satellite_status.timestamp + self.step, new_orbital_elements)

        return new_satellite_status

    def next_step_with_perturbations(self, satellite_status: SatelliteStatus) -> SatelliteStatus:
        # Find initial mean motion
        mean_motion_initial = self.compute_mean_motion(satellite_status.get_semi_major_axis())

        # Find E initial (eccentric anomaly)
        eccentric_anomaly_initial = self.compute_eccentric_anomaly(satellite_status.get_eccentricity(),
                                                                  satellite_status.get_true_anomaly())

        # Find M initial (mean anomaly)
        mean_anomaly_initial = self.compute_mean_anomaly(eccentric_anomaly_initial, satellite_status.get_eccentricity())

        # Find average mean motion
        # average_mean_motion = self.compute_average_mean_motion(mean_motion_initial, )

        # Move mean anomaly to the desired time

        # Solve for E future

        # Finv v future

        # check quadrant
        raise NotImplementedError("Not implemented yet, refer to page 284-285 for algorithm")
        pass

    def predict(self, limit=1, nb_steps=100):
        pass


class Satellite:

    def __init__(self, satellite_status: SatelliteStatus, trajectory_predictor: TrajectoryPredictor):
        self.satellite_status: SatelliteStatus = satellite_status
        self.trajectory: List[SatelliteStatus] = [self.satellite_status]
        self.trajectory_predictor: TrajectoryPredictor = trajectory_predictor

    def update_satellite_status(self, new_satellite_status: SatelliteStatus):
        self.satellite_status = new_satellite_status
        self.trajectory.append(self.satellite_status)

    def extend_trajectory(self, steps=1):
        for step in range(steps):
            new_satellite_status = self.trajectory_predictor.next_step(self.satellite_status)
            self.update_satellite_status(new_satellite_status)


if __name__ == "__main__":
    trajectory_predictor = TrajectoryPredictor(timedelta(seconds=5))

    orbital_elements = OrbitalElements(0.05, 7000, math.radians(50), 0, 0, math.radians(89.99999999))
    date = datetime(2023, 11, 4, 4, 44, 44)
    satellite_status = SatelliteStatus(date, orbital_elements)
    satellite = Satellite(satellite_status, trajectory_predictor)

    satellite.extend_trajectory()
    print("test")
