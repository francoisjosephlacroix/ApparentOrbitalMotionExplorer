import unittest
from datetime import timedelta, datetime
import math
import numpy as np

from src.prediction.trajectory_predictor import TrajectoryPredictor
from src.prediction.satellite_status import SatelliteStatus
from src.orbital_coordinates.orbital_elements import OrbitalElements


class TestTrajectoryPredictor(unittest.TestCase):

    def setUp(self):
        self.trajectory_predictor = TrajectoryPredictor(timedelta(hours=6))

        self.orbital_elements = OrbitalElements(0.05, 7000, math.radians(50), 0, 0, math.radians(270))
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        self.satellite_status = SatelliteStatus(self.date, self.orbital_elements)

    def test_convert_rad_angle_to_normal_range_pi(self):
        normal_angle = math.pi
        expected_angle = math.pi

        new_angle = self.trajectory_predictor.convert_rad_angle_to_normal_range(normal_angle)
        self.assertEqual(new_angle, expected_angle)

    def test_convert_rad_angle_to_normal_range_zero(self):
        normal_angle = 0
        expected_angle = 0

        new_angle = self.trajectory_predictor.convert_rad_angle_to_normal_range(normal_angle)
        self.assertEqual(new_angle, expected_angle)

    def test_convert_rad_angle_to_normal_range_2pi(self):
        normal_angle = 2 * math.pi
        expected_angle = 0

        new_angle = self.trajectory_predictor.convert_rad_angle_to_normal_range(normal_angle)
        self.assertEqual(new_angle, expected_angle)

    def test_convert_rad_angle_to_normal_range_negative(self):
        normal_angle = -math.pi
        expected_angle = math.pi

        new_angle = self.trajectory_predictor.convert_rad_angle_to_normal_range(normal_angle)
        self.assertEqual(new_angle, expected_angle)

    def test_convert_rad_angle_to_normal_range_large(self):
        normal_angle = 5 * math.pi
        expected_angle = math.pi

        new_angle = self.trajectory_predictor.convert_rad_angle_to_normal_range(normal_angle)
        self.assertEqual(new_angle, expected_angle)

    def test_choose_half_plane_cos_first_quadrant(self):
        old_angle = math.pi / 4
        new_angle = -math.pi / 4
        expected_angle = math.pi / 4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_second_quadrant(self):
        old_angle = 3 * math.pi / 4
        new_angle = -3 * math.pi / 4
        expected_angle = 3 * math.pi / 4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_third_quadrant(self):
        old_angle = -3 * math.pi / 4
        new_angle = 3 * math.pi / 4
        expected_angle = 2 * math.pi - 3 * math.pi / 4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_fourth_quadrant(self):
        old_angle = -math.pi / 4
        new_angle = math.pi / 4
        expected_angle = 2 * math.pi - math.pi / 4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_first_half(self):
        old_angle = math.pi / 4
        new_angle = 3 * math.pi / 4
        expected_angle = 3 * math.pi / 4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_second_half(self):
        old_angle = -math.pi / 4
        new_angle = -3 * math.pi / 4
        expected_angle = 2 * math.pi - 3 * math.pi / 4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_opposite_quarters(self):
        old_angle = -math.pi / 4
        new_angle = 3 * math.pi / 4
        expected_angle = 2 * math.pi - 3 * math.pi / 4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_sin_first_quadrant(self):
        old_angle = math.pi / 4
        new_angle = 3 * math.pi / 4
        expected_angle = math.pi / 4

        result_angle = self.trajectory_predictor.choose_half_plane_sin(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_sin_second_quadrant(self):
        old_angle = 3 * math.pi / 4
        new_angle = math.pi / 4
        expected_angle = 3 * math.pi / 4

        result_angle = self.trajectory_predictor.choose_half_plane_sin(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_sin_third_quadrant(self):
        old_angle = 5 * math.pi / 4
        new_angle = math.pi / 4
        expected_angle = math.pi - math.pi / 4

        result_angle = self.trajectory_predictor.choose_half_plane_sin(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_sin_furth_quadrant(self):
        old_angle = -math.pi / 4
        new_angle = 3 * math.pi / 4
        expected_angle = math.pi / 4

        result_angle = self.trajectory_predictor.choose_half_plane_sin(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_compute_mean_motion(self):
        expected_mean_motion = 0.001078
        mean_motion = self.trajectory_predictor.compute_mean_motion(self.satellite_status.get_semi_major_axis())

        self.assertAlmostEqual(expected_mean_motion, mean_motion)

    def test_compute_eccentric_anomaly(self):
        expected_eccentric_anomaly = 4.762
        actual_eccentric_anomaly = self.trajectory_predictor.compute_eccentric_anomaly(
            self.satellite_status.get_eccentricity(), self.satellite_status.get_true_anomaly())
        self.assertAlmostEqual(expected_eccentric_anomaly, actual_eccentric_anomaly, 3)

    def test_compute_mean_anomaly(self):
        expected_mean_anomaly = 4.812
        eccentric_anomaly = 4.762
        actual_mean_anomaly = self.trajectory_predictor.compute_mean_anomaly(eccentric_anomaly,
                                                                             self.satellite_status.get_eccentricity())
        self.assertAlmostEqual(expected_mean_anomaly, actual_mean_anomaly, 3)

    def test_compute_mean_anomaly_future(self):
        expected_mean_anomaly_future = 2.964
        mean_anomaly = 4.812
        mean_motion = 0.001078
        actual_mean_anomaly_future = self.trajectory_predictor.compute_mean_anomaly_future(mean_anomaly, mean_motion)

        self.assertAlmostEqual(expected_mean_anomaly_future, actual_mean_anomaly_future, 3)

    def test_compute_eccentric_anomaly_future(self):
        expected_eccentric_anomaly_future = 2.972
        mean_anomaly_future = 2.964
        actual_eccentric_anomaly_future = self.trajectory_predictor.compute_eccentric_anomaly_future(
            mean_anomaly_future, self.satellite_status.get_eccentricity())
        self.assertAlmostEqual(expected_eccentric_anomaly_future, actual_eccentric_anomaly_future, 3)

    def test_compute_true_anomaly_future(self):
        expected_true_anomaly_future = math.radians(170.75)
        eccentric_anomaly_future = 2.972
        actual_true_anomaly_future = self.trajectory_predictor.compute_true_anomaly_future(
            eccentric_anomaly_future,
            self.satellite_status.get_eccentricity())

        self.assertAlmostEqual(expected_true_anomaly_future, actual_true_anomaly_future, 3)

    def test_end_to_end(self):
        expected_true_anomaly_future = math.radians(170.75)
        new_satellite_statue = self.trajectory_predictor.next_step(self.satellite_status)

        self.assertAlmostEqual(expected_true_anomaly_future, new_satellite_statue.get_true_anomaly(), 2)

    def test_convert_classical_to_perifocal_perigee(self):
        self.orbital_elements = OrbitalElements(0.05, 7000, 0.0, 0.0, 0.0, 0.0)
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        self.satellite_status = SatelliteStatus(self.date, self.orbital_elements)

        radius_vec, velocity_vec = self.satellite_status.convert_classical_to_perifocal()

        self.assertEqual(radius_vec[0], 6650)
        self.assertEqual(radius_vec[1], 0)
        self.assertEqual(radius_vec[2], 0)

        self.assertEqual(velocity_vec[0], 0)
        self.assertAlmostEqual(velocity_vec[1], 7.933, 3)
        self.assertEqual(velocity_vec[2], 0)

    def test_convert_classical_to_perifocal_perigee_with_angles(self):
        self.orbital_elements = OrbitalElements(0.05, 7000, math.pi / 4, math.pi / 4, math.pi / 4, 0.0)
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        self.satellite_status = SatelliteStatus(self.date, self.orbital_elements)

        radius_vec, velocity_vec = self.satellite_status.convert_classical_to_perifocal()

        self.assertEqual(radius_vec[0], 6650)
        self.assertEqual(radius_vec[1], 0)
        self.assertEqual(radius_vec[2], 0)

        self.assertEqual(velocity_vec[0], 0)
        self.assertAlmostEqual(velocity_vec[1], 7.933, 3)
        self.assertEqual(velocity_vec[2], 0)

    def test_convert_classical_to_perifocal_apogee(self):
        self.orbital_elements = OrbitalElements(0.05, 7000, 0.0, 0.0, 0.0, math.pi)
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        self.satellite_status = SatelliteStatus(self.date, self.orbital_elements)

        radius_vec, velocity_vec = self.satellite_status.convert_classical_to_perifocal()

        self.assertEqual(radius_vec[0], -7350)
        self.assertAlmostEqual(radius_vec[1], 0, 10)
        self.assertAlmostEqual(radius_vec[2], 0, 10)

        self.assertAlmostEqual(velocity_vec[0], 0, 10)
        self.assertAlmostEqual(velocity_vec[1], -7.178, 3)
        self.assertAlmostEqual(velocity_vec[2], 0, 10)

    def test_rotate_state_vector(self):
        radius_vec = np.array([6650, 0, 0])

        self.orbital_elements = OrbitalElements(0.05, 7000, math.pi / 4, 0, 0, 0.0)
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        self.satellite_status = SatelliteStatus(self.date, self.orbital_elements)

        rotated_vec = self.satellite_status.rotate_state_vector(radius_vec)

        self.assertEqual(rotated_vec[0], radius_vec[0])
        self.assertEqual(rotated_vec[1], radius_vec[1])
        self.assertEqual(rotated_vec[2], radius_vec[2])

    def test_rotate_state_vector_half_rotation(self):
        radius_vec = np.array([6650, 0, 0])

        self.orbital_elements = OrbitalElements(0.05, 7000, math.pi / 4, math.pi, math.pi, 0.0)
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        self.satellite_status = SatelliteStatus(self.date, self.orbital_elements)

        rotated_vec = self.satellite_status.rotate_state_vector(radius_vec)

        self.assertEqual(rotated_vec[0], radius_vec[0])
        self.assertAlmostEqual(rotated_vec[1], radius_vec[1])
        self.assertAlmostEqual(rotated_vec[2], radius_vec[2])

    def test_convert_classical_to_perifocal(self):
        self.orbital_elements = OrbitalElements(0.05, 7000, 0.0, 0.0, 0.0, 0.0)
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        self.satellite_status = SatelliteStatus(self.date, self.orbital_elements)

        self.satellite_status.convert_classical_to_perifocal()

        self.assertEqual(self.satellite_status.orbital_state_vectors.position[0], 6650)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.position[1], 0)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.position[2], 0)

        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[0], 0)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[1], 7.933, 3)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[2], 0)

    def test_convert_classical_to_perifocal_flipped(self):
        self.orbital_elements = OrbitalElements(0.05, 7000, math.pi, 0.0, math.pi, 0.0)
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        self.satellite_status = SatelliteStatus(self.date, self.orbital_elements)

        self.satellite_status.convert_classical_to_perifocal()

        self.assertEqual(self.satellite_status.orbital_state_vectors.position[0], -6650)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.position[1], 0)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.position[2], 0)

        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[0], 0)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[1], 7.933, 3)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[2], 0)

    def test_convert_classical_to_perifocal_vertical(self):
        self.orbital_elements = OrbitalElements(0.05, 7000, math.pi / 2, 0.0, 0.0, 0.0)
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        self.satellite_status = SatelliteStatus(self.date, self.orbital_elements)

        self.satellite_status.convert_classical_to_perifocal()

        self.assertEqual(self.satellite_status.orbital_state_vectors.position[0], 6650)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.position[1], 0)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.position[2], 0)

        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[0], 0)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[1], 0)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[2], 7.933, 3)

    def test_convert_classical_to_perifocal_angles(self):
        self.orbital_elements = OrbitalElements(0.05, 7000, math.pi / 2, 0.0, math.pi / 2, 0.0)
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        self.satellite_status = SatelliteStatus(self.date, self.orbital_elements)

        self.satellite_status.convert_classical_to_perifocal()

        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.position[0], 0)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.position[1], 0)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.position[2], 6650)

        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[0], -7.933, 3)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[1], 0)
        self.assertAlmostEqual(self.satellite_status.orbital_state_vectors.velocity[2], 0)

    def test_lvlh_reference_frame(self):
        self.orbital_elements = OrbitalElements(0.05, 7000, 0.0, 0.0, 0.0, 0.0)
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        self.satellite_status = SatelliteStatus(self.date, self.orbital_elements)

        pos_vec = np.array([1, 0, 0])
        vel_vec = np.array([0, 1, 1])
        i_vec, j_vec, k_vec = self.satellite_status.get_sat_reference_frame(pos_vec, vel_vec)

        self.assertEqual(i_vec[0], 1)
        self.assertEqual(i_vec[1], 0)
        self.assertEqual(i_vec[2], 0)

        self.assertEqual(j_vec[0], 0)
        self.assertAlmostEqual(j_vec[1], math.sqrt(0.5))
        self.assertAlmostEqual(j_vec[2], math.sqrt(0.5))

        self.assertEqual(k_vec[0], 0)
        self.assertAlmostEqual(k_vec[1], -math.sqrt(0.5))
        self.assertAlmostEqual(k_vec[2], math.sqrt(0.5))
