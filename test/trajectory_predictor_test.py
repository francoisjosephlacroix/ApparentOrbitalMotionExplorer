import unittest
from datetime import timedelta, datetime
import math

from TrajectoryPredictor import TrajectoryPredictor, OrbitalElements, SatelliteStatus, Satellite


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
        normal_angle = 2*math.pi
        expected_angle = 0

        new_angle = self.trajectory_predictor.convert_rad_angle_to_normal_range(normal_angle)
        self.assertEqual(new_angle, expected_angle)

    def test_convert_rad_angle_to_normal_range_negative(self):
        normal_angle = -math.pi
        expected_angle = math.pi

        new_angle = self.trajectory_predictor.convert_rad_angle_to_normal_range(normal_angle)
        self.assertEqual(new_angle, expected_angle)

    def test_convert_rad_angle_to_normal_range_large(self):
        normal_angle = 5*math.pi
        expected_angle = math.pi

        new_angle = self.trajectory_predictor.convert_rad_angle_to_normal_range(normal_angle)
        self.assertEqual(new_angle, expected_angle)

    def test_choose_half_plane_cos_first_quadrant(self):
        old_angle = math.pi/4
        new_angle = -math.pi/4
        expected_angle = math.pi/4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_second_quadrant(self):
        old_angle = 3*math.pi/4
        new_angle = -3*math.pi/4
        expected_angle = 3*math.pi/4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_third_quadrant(self):
        old_angle = -3*math.pi/4
        new_angle = 3*math.pi/4
        expected_angle = 2*math.pi-3*math.pi/4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_fourth_quadrant(self):
        old_angle = -math.pi/4
        new_angle = math.pi/4
        expected_angle = 2*math.pi-math.pi/4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_first_half(self):
        old_angle = math.pi/4
        new_angle = 3*math.pi/4
        expected_angle = 3*math.pi/4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_second_half(self):
        old_angle = -math.pi/4
        new_angle = -3*math.pi/4
        expected_angle = 2*math.pi-3*math.pi/4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_cos_opposite_quarters(self):
        old_angle = -math.pi/4
        new_angle = 3*math.pi/4
        expected_angle = 2*math.pi-3*math.pi/4

        result_angle = self.trajectory_predictor.choose_half_plane_cos(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_sin_first_quadrant(self):
        old_angle = math.pi/4
        new_angle = 3*math.pi/4
        expected_angle = math.pi/4

        result_angle = self.trajectory_predictor.choose_half_plane_sin(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_sin_second_quadrant(self):
        old_angle = 3*math.pi/4
        new_angle = math.pi/4
        expected_angle = 3*math.pi/4

        result_angle = self.trajectory_predictor.choose_half_plane_sin(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_sin_third_quadrant(self):
        old_angle = 5*math.pi/4
        new_angle = math.pi/4
        expected_angle = math.pi - math.pi/4

        result_angle = self.trajectory_predictor.choose_half_plane_sin(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_choose_half_plane_sin_furth_quadrant(self):
        old_angle = -math.pi/4
        new_angle = 3*math.pi/4
        expected_angle = math.pi/4

        result_angle = self.trajectory_predictor.choose_half_plane_sin(old_angle, new_angle)
        self.assertEqual(result_angle, expected_angle)

    def test_compute_mean_motion(self):
        expected_mean_motion = 0.001078
        mean_motion = self.trajectory_predictor.compute_mean_motion(self.satellite_status.get_semi_major_axis())

        self.assertAlmostEqual(expected_mean_motion, mean_motion)

    def test_compute_eccentric_anomaly(self):
        expected_eccentric_anomaly = 4.762
        actual_eccentric_anomaly = self.trajectory_predictor.compute_eccentric_anomaly(self.satellite_status.get_eccentricity(), self.satellite_status.get_true_anomaly())
        self.assertAlmostEqual(expected_eccentric_anomaly, actual_eccentric_anomaly, 3)

    def test_compute_mean_anomaly(self):
        expected_mean_anomaly = 4.812
        eccentric_anomaly = 4.762
        actual_mean_anomaly = self.trajectory_predictor.compute_mean_anomaly(eccentric_anomaly, self.satellite_status.get_eccentricity())
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
        actual_eccentric_anomaly_future = self.trajectory_predictor.compute_eccentric_anomaly_future(mean_anomaly_future, self.satellite_status.get_eccentricity())
        self.assertAlmostEqual(expected_eccentric_anomaly_future, actual_eccentric_anomaly_future, 3)

    def test_compute_true_anomaly_future(self):
        expected_true_anomaly_future = math.radians(170.75)
        eccentric_anomaly_future = 2.972
        actual_true_anomaly_future = self.trajectory_predictor.compute_true_anomaly_future(eccentric_anomaly_future, self.satellite_status.get_eccentricity())
        self.assertAlmostEqual(expected_true_anomaly_future, actual_true_anomaly_future, 3)

    def test_end_to_end(self):
        expected_true_anomaly_future = math.radians(170.75)
        new_satellite_statue = self.trajectory_predictor.next_step(self.satellite_status)

        self.assertAlmostEqual(expected_true_anomaly_future, new_satellite_statue.get_true_anomaly(), 2)
