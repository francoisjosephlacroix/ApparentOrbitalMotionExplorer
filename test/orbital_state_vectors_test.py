import unittest
import numpy as np
import math

from orbital_coordinates.orbital_state_vectors import OrbitalStateVectors


class TestTrajectoryPredictor(unittest.TestCase):

    def setUp(self):
        pos_vec = np.array([8228.0, 389.0, 6888.0])
        vel_vec = np.array([-0.7, 6.6, -0.6])
        self.orbital_state_vectors = OrbitalStateVectors(pos_vec, vel_vec)

    def test_lvlh_reference_frame(self):
        pos_vec = np.array([1, 0, 0])
        vel_vec = np.array([0, 1, 1])
        orbital_state_vectors = OrbitalStateVectors(pos_vec, vel_vec)
        i_vec, j_vec, k_vec = orbital_state_vectors.get_sat_reference_frame()

        self.assertEqual(i_vec[0], 1)
        self.assertEqual(i_vec[1], 0)
        self.assertEqual(i_vec[2], 0)

        self.assertEqual(j_vec[0], 0)
        self.assertAlmostEqual(j_vec[1], math.sqrt(0.5))
        self.assertAlmostEqual(j_vec[2], math.sqrt(0.5))

        self.assertEqual(k_vec[0], 0)
        self.assertAlmostEqual(k_vec[1], -math.sqrt(0.5))
        self.assertAlmostEqual(k_vec[2], math.sqrt(0.5))

    def test_eccentricity_vec(self):
        ecc_vec = self.orbital_state_vectors.eccentricity_vec()
        ecc = np.linalg.norm(ecc_vec)

        self.assertAlmostEqual(ecc_vec[0], 0.1376, 4)
        self.assertAlmostEqual(ecc_vec[1], 0.1284, 4)
        self.assertAlmostEqual(ecc_vec[2], 0.1149, 4)
        self.assertAlmostEqual(ecc, 0.2205, 4)

    def test_semi_major_axis(self):
        semi_major_axis = self.orbital_state_vectors.semi_major_axis()

        self.assertAlmostEqual(semi_major_axis, 1.336E4, -1)

    def test_inclination(self):
        inclination = self.orbital_state_vectors.inclination()

        self.assertAlmostEqual(inclination, math.radians(39.94), 4)

    def test_angular_momentum(self):
        angular_momentum = self.orbital_state_vectors.angular_momentum()

        self.assertAlmostEqual(angular_momentum[0], -45694.2, 1)
        self.assertAlmostEqual(angular_momentum[1], 115.2, 1)
        self.assertAlmostEqual(angular_momentum[2], 54577.1, 1)

    def test_right_ascension(self):
        right_ascension = self.orbital_state_vectors.right_ascension()

        self.assertAlmostEqual(right_ascension, math.radians(269.9), 2)

    def test_ascending_node(self):
        ascending_node = self.orbital_state_vectors.ascending_node()

        self.assertAlmostEqual(ascending_node[0], -115.2, 1)
        self.assertAlmostEqual(ascending_node[1], -45694.2, 1)
        self.assertAlmostEqual(ascending_node[2], 0.0, 1)

    def test_argument_periapsis(self):
        argument_periapsis = self.orbital_state_vectors.argument_periapsis()

        self.assertAlmostEqual(argument_periapsis, math.radians(125.7), 3)

    def test_true_anomaly(self):
        true_anomaly = self.orbital_state_vectors.true_anomaly()

        self.assertAlmostEqual(true_anomaly, math.radians(326.5), 2)
