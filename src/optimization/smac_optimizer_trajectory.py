import json
import math
import os
import time
from datetime import timedelta, datetime
from pathlib import Path

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer, UniformFloatHyperparameter,
)
from smac import HyperparameterOptimizationFacade, Scenario

from optimization.trajectory_optimizer import FitnessEvaluator
from orbital_coordinates.orbital_elements import OrbitalElements
from prediction.constants import EARTH_RADIUS
from prediction.relative_motion import RelativeMotion
from prediction.satellite import Satellite
from prediction.satellite_status import SatelliteStatus
from prediction.trajectory import Trajectory
from prediction.trajectory_predictor import TrajectoryPredictor

INFINITE_COST = float('inf')

steps = 1000
step_size = timedelta(seconds=15)
ref_sat_config = {
    'eccentricity': 0.05,
    'semi_major_axis': 7500,
    'inclination': 0.0,
    'longitude_ascending_node': 0.0,
    'argument_periapsis': 0.0,
    'true_anomaly': 0.0,
}

sat_2_config = {
    'eccentricity': 0.06,
    'semi_major_axis': 7600,
    'inclination': math.radians(15),
    'longitude_ascending_node': math.radians(5),
    'argument_periapsis': math.radians(5),
    'true_anomaly': math.radians(15)
}
date = datetime(2023, 11, 4, 4, 44, 44)


trajectory_predictor = TrajectoryPredictor(step_size)

orbital_elements = OrbitalElements.from_dict(ref_sat_config)
satellite_status = SatelliteStatus(date, orbital_elements)
satellite = Satellite(satellite_status, trajectory_predictor)
satellite.extend_trajectory(steps=steps)

rx, ry, rz = satellite.get_trajectory_position_per_axis()

trajectory_predictor2 = TrajectoryPredictor(step_size)
# eccentricity, semi_major_axis, inclination, longitude_ascending_node, argument_periapsis, true_anomaly
orbital_elements2 = OrbitalElements.from_dict(sat_2_config)
satellite_status2 = SatelliteStatus(date, orbital_elements2)
satellite2 = Satellite(satellite_status2, trajectory_predictor2)
satellite2.extend_trajectory(steps=steps)

rx2, ry2, rz2 = satellite2.get_trajectory_position_per_axis()

relative_motion = RelativeMotion(satellite, satellite2)

delta_x, delta_y, delta_z = relative_motion.get_relative_motion_lvlh()

optimal_trajectory = Trajectory(delta_x, delta_y, delta_z)


def evaluate(config: Configuration, seed: int = 0) -> float:
    trajectory_predictor = TrajectoryPredictor(step_size)

    orbital_elements = OrbitalElements.from_dict(ref_sat_config)
    satellite_status = SatelliteStatus(date, orbital_elements)
    satellite = Satellite(satellite_status, trajectory_predictor)

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

    if orbital_elements2.perigee < (EARTH_RADIUS + 200):
        return INFINITE_COST

    satellite.extend_trajectory(steps=steps)
    satellite2.extend_trajectory(steps=steps)

    relative_motion = RelativeMotion(satellite, satellite2)
    delta_x, delta_y, delta_z = relative_motion.get_relative_motion_lvlh()

    candidate_trajectory = Trajectory(delta_x, delta_y, delta_z)

    fitness_evaluator = FitnessEvaluator()

    fitness = fitness_evaluator.evaluate_trajectory_fitness(optimal_trajectory, candidate_trajectory)
    print(fitness)
    return fitness


base_path = Path("../../smac3_output/")
run_name = str(time.time())
run_path = Path.joinpath(base_path, run_name)
os.mkdir(run_path)

run_metadata = {
    "steps": steps,
    "step_size": step_size.seconds,
    "ref_sat_config": ref_sat_config,
    "sat_2_config": sat_2_config,
    "date": date.isoformat()
}

json.dump(run_metadata, open(Path.joinpath(run_path, "metadata.json"), "w"))

configspace = ConfigurationSpace()

eccentricity = Float("eccentricity", (0.0, 1.0), default=0.05)
semi_major_axis = Float("semi_major_axis", (6800.0, 10000.0), default=7500)
inclination = Float("inclination", (0.0, math.pi), default=0.0)
longitude_ascending_node = Float("longitude_ascending_node", (0.0, 2.0 * math.pi), default=0.0)
argument_periapsis = Float("argument_periapsis", (0.0, 2.0 * math.pi), default=0.0)
true_anomaly = Float("true_anomaly", (0.0, 2.0 * math.pi), default=0.0)

configspace.add_hyperparameters([eccentricity, semi_major_axis, inclination, longitude_ascending_node, argument_periapsis, true_anomaly])

# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=True, n_trials=10000,
                    output_directory=base_path, name=run_name)
# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, evaluate)

incumbent = smac.optimize()

print(incumbent)
