import json
import math
import os
import time
from datetime import timedelta, datetime
from pathlib import Path

import ConfigSpace.conditions
import numpy as np

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer, UniformFloatHyperparameter,
)
from sklearn.gaussian_process.kernels import RBF
from smac import HyperparameterOptimizationFacade, Scenario, BlackBoxFacade
from smac.acquisition.function import EI
from smac.acquisition.maximizer import LocalSearch
from smac.initial_design import DefaultInitialDesign, LatinHypercubeInitialDesign, FactorialInitialDesign
from smac.model.gaussian_process import MCMCGaussianProcess, GaussianProcess
from smac.model.random_forest import RandomForest
from smac.random_design import CosineAnnealingRandomDesign

from optimization.trajectory_optimizer import FitnessEvaluator
from orbital_coordinates.orbital_elements import OrbitalElements
from orbital_coordinates.orbital_state_vectors import OrbitalStateVectors
from prediction.constants import EARTH_RADIUS
from prediction.relative_motion import RelativeMotion
from prediction.satellite import Satellite
from prediction.satellite_status import SatelliteStatus
from prediction.trajectory import Trajectory
from prediction.trajectory_predictor import TrajectoryPredictor

if __name__ == "__main__":

    # INFINITE_COST = float('inf')
    INFINITE_COST = 10E10

    steps = 8
    step_size = timedelta(minutes=15)
    ref_sat_config = {
        'eccentricity': 0.05,
        'semi_major_axis': 7500,
        'inclination': 0.0,
        'longitude_ascending_node': 0.0,
        'argument_periapsis': 0.0,
        'true_anomaly': 0.0,
    }

    date = datetime(2023, 11, 4, 4, 44, 44)

    trajectory_predictor = TrajectoryPredictor(step_size)

    orbital_elements = OrbitalElements.from_dict(ref_sat_config)
    satellite_status = SatelliteStatus(date, orbital_elements)
    satellite = Satellite(satellite_status, trajectory_predictor)
    satellite.extend_trajectory(steps=steps)

    rx, ry, rz = satellite.get_trajectory_position_per_axis()

    # Cube trajectory
    delta_x = np.array([1, -1, -1, 1, 1, -1, -1, 1])
    delta_y = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    delta_z = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    optimal_trajectory = Trajectory(delta_x, delta_y, delta_z)

    initial_position = np.array([rx[0] + delta_x[0],
                                 ry[0] + delta_y[0],
                                 rz[0] + delta_z[0]])

    def evaluate(config: Configuration, seed: int = 0) -> float:
        try:
            trajectory_predictor = TrajectoryPredictor(step_size)

            orbital_elements = OrbitalElements.from_dict(ref_sat_config)
            satellite_status = SatelliteStatus(date, orbital_elements)
            satellite = Satellite(satellite_status, trajectory_predictor)

            trajectory_predictor2 = TrajectoryPredictor(step_size)

            new_vel_vec = np.array([config.get("vx"), config.get("vy"), config.get("vz")])
            if np.linalg.norm(new_vel_vec) >= 11.2:
                return INFINITE_COST

            orbital_state_vectors = OrbitalStateVectors(initial_position, new_vel_vec)

            satellite_status2 = SatelliteStatus.from_state_vector(date, orbital_state_vectors)
            satellite2 = Satellite(satellite_status2, trajectory_predictor2)

            if satellite_status2.orbital_elements.perigee < (EARTH_RADIUS + 200):
                return INFINITE_COST

            satellite.extend_trajectory(steps=steps)
            satellite2.extend_trajectory(steps=steps)

            relative_motion = RelativeMotion(satellite, satellite2)
            delta_x, delta_y, delta_z = relative_motion.get_relative_motion_lvlh()

            candidate_trajectory = Trajectory(delta_x, delta_y, delta_z)

            fitness_evaluator = FitnessEvaluator()

            fitness = fitness_evaluator.evaluate_trajectory_fitness(optimal_trajectory, candidate_trajectory)
            print(fitness)
        except:
            return INFINITE_COST

        return fitness


    base_path = Path("../../smac3_output/")
    run_name = str(time.time())
    run_path = Path.joinpath(base_path, run_name)
    os.mkdir(run_path)

    run_metadata = {
        "steps": steps,
        "step_size": step_size.seconds,
        "ref_sat_config": ref_sat_config,
        "sat_2_initial_position": initial_position.tolist(),
        "optimal_trajectory": optimal_trajectory.to_list(),
        "date": date.isoformat()
    }

    json.dump(run_metadata, open(Path.joinpath(run_path, "metadata.json"), "w"))

    configspace = ConfigurationSpace()

    vx = Float("vx", (-11.2, 11.2), default=satellite.satellite_status.orbital_state_vectors.velocity[0])
    vy = Float("vy", (-11.2, 11.2), default=satellite.satellite_status.orbital_state_vectors.velocity[1])
    vz = Float("vz", (-11.2, 11.2), default=satellite.satellite_status.orbital_state_vectors.velocity[2])

    configspace.add_hyperparameters([vx, vy, vz])

    # Scenario object specifying the optimization environment
    scenario = Scenario(configspace, deterministic=True, n_trials=10000,
                        output_directory=base_path, name=run_name)

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, evaluate)
    incumbent = smac.optimize()

    print(incumbent)
