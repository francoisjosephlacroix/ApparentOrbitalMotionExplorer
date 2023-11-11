import unittest
from datetime import timedelta, datetime
import math
import numpy as np

from src.optimization.trajectory_optimizer import FitnessEvaluator
from src.orbital_coordinates.orbital_elements import OrbitalElements
from src.prediction.satellite import Satellite
from src.prediction.satellite_status import SatelliteStatus
from src.prediction.trajectory_predictor import TrajectoryPredictor


class TestFitnessEvaluator(unittest.TestCase):

    def setUp(self):
        self.steps = 10
        trajectory_predictor = TrajectoryPredictor(timedelta(seconds=5))

        orbital_elements = OrbitalElements(0.05, 7500, 0, 0, 0, 0)
        self.date = datetime(2023, 11, 4, 4, 44, 44)
        satellite_status = SatelliteStatus(self.date, orbital_elements)
        self.satellite = Satellite(satellite_status, trajectory_predictor)
        self.satellite.extend_trajectory(steps=self.steps)

        trajectory_predictor2 = TrajectoryPredictor(timedelta(seconds=5))
        # eccentricity, semi_major_axis, inclination, longitude_ascending_node, argument_periapsis, true_anomaly
        orbital_elements2 = OrbitalElements(0.05, 7500, 0, 0, 0, 0)
        satellite_status2 = SatelliteStatus(self.date, orbital_elements2)
        self.satellite2 = Satellite(satellite_status2, trajectory_predictor2)
        self.satellite2.extend_trajectory(steps=self.steps)

        self.fitness_evaluator = FitnessEvaluator()

    def test_absolute_distance_same_vector(self):
        vec_ref = np.array([1, 1, 1])
        vec_2 = np.array([1, 1, 1])
        expected_dist = 0
        dist = self.fitness_evaluator.absolute_distance(vec_ref, vec_2)
        self.assertEqual(dist, expected_dist)

    def test_absolute_distance(self):
        vec_ref = np.array([0, 0, 0])
        vec_2 = np.array([1, 1, 1])
        expected_dist = math.sqrt(3)
        dist = self.fitness_evaluator.absolute_distance(vec_ref, vec_2)
        self.assertEqual(dist, expected_dist)

    def test_evaluate_fitness(self):
        expected_fitness = 0
        fitness = self.fitness_evaluator.evaluate_fitness(self.satellite, self.satellite2)
        self.assertEqual(fitness, expected_fitness)

    def test_evaluate_fitness_unfit(self):
        expected_fitness = 1000

        trajectory_predictor2 = TrajectoryPredictor(timedelta(seconds=5))
        # eccentricity, semi_major_axis, inclination, longitude_ascending_node, argument_periapsis, true_anomaly
        orbital_elements2 = OrbitalElements(0.05, 7600, 0, 0, 0, 0)
        satellite_status2 = SatelliteStatus(self.date, orbital_elements2)
        unfit_satellite = Satellite(satellite_status2, trajectory_predictor2)
        unfit_satellite.extend_trajectory(steps=self.steps)

        fitness = self.fitness_evaluator.evaluate_fitness(self.satellite, unfit_satellite)
        self.assertGreater(fitness, expected_fitness)
