from datetime import timedelta
import math

from prediction.constants import GRAVITATIONAL_PARAMETER
from prediction.satellite_status import SatelliteStatus
from orbital_coordinates.orbital_elements import OrbitalElements


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
        # mean_motion_initial = self.compute_mean_motion(satellite_status.get_semi_major_axis())

        # Find E initial (eccentric anomaly)
        # eccentric_anomaly_initial = self.compute_eccentric_anomaly(satellite_status.get_eccentricity(),
        #                                                            satellite_status.get_true_anomaly())

        # Find M initial (mean anomaly)
        # mean_anomaly_initial = self.compute_mean_anomaly(eccentric_anomaly_initial,
        #                                                  satellite_status.get_eccentricity())

        # Find average mean motion
        # average_mean_motion = self.compute_average_mean_motion(mean_motion_initial, )

        # Move mean anomaly to the desired time

        # Solve for E future

        # Finv v future

        # check quadrant
        raise NotImplementedError("Not implemented yet, refer to page 284-285 for algorithm")

    def predict(self, limit=1, nb_steps=100):
        pass
