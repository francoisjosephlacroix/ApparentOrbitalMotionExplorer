import math
import pathlib
import time
from datetime import timedelta, datetime

from ConfigSpace import Configuration, ConfigurationSpace

from smac import HyperparameterOptimizationFacade, Scenario

from optimization.trajectory_optimizer import FitnessEvaluator
from orbital_coordinates.orbital_elements import OrbitalElements
from prediction.satellite import Satellite
from prediction.satellite_status import SatelliteStatus
from prediction.trajectory_predictor import TrajectoryPredictor


def train(config: Configuration, seed: int = 0) -> float:
    steps = 1000
    step_size = timedelta(seconds=10)
    trajectory_predictor = TrajectoryPredictor(step_size)

    orbital_elements = OrbitalElements.from_dict(ref_sat_config)
    date = datetime(2023, 11, 4, 4, 44, 44)
    satellite_status = SatelliteStatus(date, orbital_elements)
    satellite = Satellite(satellite_status, trajectory_predictor)
    satellite.extend_trajectory(steps=steps)

    trajectory_predictor2 = TrajectoryPredictor(step_size)
    # eccentricity, semi_major_axis, inclination, longitude_ascending_node, argument_periapsis, true_anomaly
    orbital_elements2 = OrbitalElements(config.get("eccentricity"),
                                        config.get('semi_major_axis'),
                                        config.get('inclination'),
                                        config.get('longitude_ascending_node'),
                                        config.get('argument_periapsis'),
                                        config.get('true_anomaly'))
    satellite_status2 = SatelliteStatus(date, orbital_elements2)
    satellite2 = Satellite(satellite_status2, trajectory_predictor2)
    satellite2.extend_trajectory(steps=steps)

    fitness_evaluator = FitnessEvaluator()

    fitness = fitness_evaluator.evaluate_fitness(satellite, satellite2)
    print(fitness)
    return fitness


ref_sat_config = {
    'eccentricity': 0.05,
    'semi_major_axis': 7500,
    'inclination': 0.0,
    'longitude_ascending_node': 0.0,
    'argument_periapsis': 0.0,
    'true_anomaly': 0.0,
}

configspace = ConfigurationSpace({"eccentricity": (0.0, 1.0),
                                  'semi_major_axis': (6800.0, 10000.0),
                                  'inclination': (0.0, math.pi),
                                  'longitude_ascending_node': (0.0, 2.0 * math.pi),
                                  'argument_periapsis': (0.0, 2.0 * math.pi),
                                  'true_anomaly': (0.0, 2.0 * math.pi)})

# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=True, n_trials=2000,
                    output_directory=pathlib.Path("../../smac3_output/"), name=str(time.time()))
# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train)

incumbent = smac.optimize()

print(incumbent)
