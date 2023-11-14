import unittest
from datetime import datetime

import numpy as np

from orbital_coordinates.orbital_state_vectors import OrbitalStateVectors
from prediction.satellite_status import SatelliteStatus


class TestSatelliteStatus(unittest.TestCase):

    def test_from_state_vector(self):
        pos_vec = np.array([8228.0, 389.0, 6888.0])
        vel_vec = np.array([-0.7, 6.6, -0.6])
        orbital_state_vectors = OrbitalStateVectors(pos_vec, vel_vec)
        date = datetime(2023, 11, 4, 4, 44, 44)

        satellite_status = SatelliteStatus.from_state_vector(date, orbital_state_vectors)

        self.assertAlmostEqual(pos_vec[0], satellite_status.orbital_state_vectors.position[0])
        self.assertAlmostEqual(pos_vec[1], satellite_status.orbital_state_vectors.position[1])
        self.assertAlmostEqual(pos_vec[2], satellite_status.orbital_state_vectors.position[2])

        self.assertAlmostEqual(vel_vec[0], satellite_status.orbital_state_vectors.velocity[0])
        self.assertAlmostEqual(vel_vec[1], satellite_status.orbital_state_vectors.velocity[1])
        self.assertAlmostEqual(vel_vec[2], satellite_status.orbital_state_vectors.velocity[2])

        self.assertEqual(np.linalg.norm(orbital_state_vectors.eccentricity_vec()),
                         satellite_status.orbital_elements.eccentricity)
        self.assertEqual(orbital_state_vectors.semi_major_axis(), satellite_status.orbital_elements.semi_major_axis)
        self.assertEqual(orbital_state_vectors.inclination(), satellite_status.orbital_elements.inclination)
        self.assertEqual(orbital_state_vectors.right_ascension(),
                         satellite_status.orbital_elements.longitude_ascending_node)
        self.assertEqual(orbital_state_vectors.argument_periapsis(),
                         satellite_status.orbital_elements.argument_periapsis)
        self.assertEqual(orbital_state_vectors.true_anomaly(), satellite_status.orbital_elements.true_anomaly)
