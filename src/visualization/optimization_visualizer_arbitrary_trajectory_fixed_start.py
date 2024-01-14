import json
import math
from datetime import timedelta, datetime
from pathlib import Path

import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

from orbital_coordinates.orbital_elements import OrbitalElements
from orbital_coordinates.orbital_state_vectors import OrbitalStateVectors
from prediction.constants import EARTH_RADIUS
from prediction.relative_motion import RelativeMotion
from prediction.satellite import Satellite
from prediction.satellite_status import SatelliteStatus
from prediction.trajectory_predictor import TrajectoryPredictor

show_orbits = True
show_relative_motion = True
show_relative_motion_lvlh = True
leaders_only = True
show_solution = True
show_top = 1
display_granulatity = 1000

linecolor = 'royalblue'
sat_color = "forestgreen"
sat_color_start = 'y'
sat_color_end = 'r'

base_path = Path("../../smac3_output/")
run_name = '1700197030.1055977'
run_path = Path.joinpath(base_path, run_name)
metadata_path = Path.joinpath(run_path, "metadata.json")
run_metadata = json.load(open(metadata_path, "r"))

step_size = timedelta(seconds=run_metadata.get("step_size"))
steps = run_metadata.get("steps")

if display_granulatity:
    new_step_size = steps * step_size.seconds / display_granulatity
    step_size = timedelta(seconds=new_step_size)
    steps = display_granulatity

ref_sat_config = run_metadata.get("ref_sat_config")
sat_2_config = run_metadata.get("sat_2_config")
date = datetime.strptime(run_metadata.get("date"), "%Y-%m-%dT%H:%M:%S")

initial_position = run_metadata.get("sat_2_initial_position")
optimal_trajectory = run_metadata.get("optimal_trajectory")

smac_history = Path.joinpath(run_path, "0", "runhistory.json")
run_history = json.load(open(smac_history, "r"))
configs = run_history.get("configs")
configs_to_plot = configs

def add_optimal_solution_to_lvlh_graph(graph):
    delta_x = optimal_trajectory[0]
    delta_y = optimal_trajectory[1]
    delta_z = optimal_trajectory[2]

    graph.plot3D(delta_x, delta_y, delta_z, "g")
    graph.scatter(delta_x[0], delta_y[0], delta_z[0], linewidths=3, marker="*", edgecolors=sat_color_start)
    graph.scatter(delta_x[-1], delta_y[-1], delta_z[-1], linewidths=3, marker="*", edgecolors=sat_color_end)

if leaders_only:
    smac_intensifier = Path.joinpath(run_path, "0", "intensifier.json")
    intensifier = json.load(open(smac_intensifier, "r"))
    smac_trajectory = intensifier.get("trajectory")
    smac_trajectory = smac_trajectory[-show_top:]
    leaders = {str(l.get('config_ids')[0]): configs.get(str(l.get('config_ids')[0])) for l in smac_trajectory}
    print(leaders)
    configs_to_plot = leaders


trajectory_predictor = TrajectoryPredictor(step_size)

orbital_elements = OrbitalElements.from_dict(ref_sat_config)
satellite_status = SatelliteStatus(date, orbital_elements)
satellite = Satellite(satellite_status, trajectory_predictor)
satellite.extend_trajectory(steps=steps)

rx, ry, rz = satellite.get_trajectory_position_per_axis()

sns.set_style("dark", {"grid.color": ".1", "grid.linestyle": ":"})
plt.figure(figsize=(12, 12))
ax_orbits = plt.axes(projection='3d')

sns.set_style("dark", {"grid.color": ".1", "grid.linestyle": ":"})
plt.figure(figsize=(12, 12))
ax_relative = plt.axes(projection='3d')

sns.set_style("dark", {"grid.color": ".1", "grid.linestyle": ":"})
plt.figure(figsize=(12, 12))
ax_lvlh = plt.axes(projection='3d')

ax_orbits.plot3D(rx, ry, rz)

for idx, config in configs_to_plot.items():

    trajectory_predictor2 = TrajectoryPredictor(step_size)
    # eccentricity, semi_major_axis, inclination, longitude_ascending_node, argument_periapsis, true_anomaly

    new_vel_vec = np.array([config.get("vx"), config.get("vy"), config.get("vz")])
    orbital_state_vectors_2 = OrbitalStateVectors(np.array(initial_position), new_vel_vec)

    satellite_status2 = SatelliteStatus.from_state_vector(date, orbital_state_vectors_2)

    satellite2 = Satellite(satellite_status2, trajectory_predictor2)
    satellite2.extend_trajectory(steps=steps)

    rx2, ry2, rz2 = satellite2.get_trajectory_position_per_axis()

    if show_orbits:
        ax_orbits.plot3D(rx2, ry2, rz2)

    if show_relative_motion:
        relative_motion = RelativeMotion(satellite, satellite2)

        delta_x, delta_y, delta_z = relative_motion.get_relative_motion()

        ax_relative.plot3D(delta_x, delta_y, delta_z, linecolor)

        ax_relative.scatter(0, 0, 0, linewidths=3, marker="*", edgecolors=sat_color)
        ax_relative.scatter(delta_x[0], delta_y[0], delta_z[0], linewidths=3, marker="*", edgecolors=sat_color_start)
        ax_relative.scatter(delta_x[-1], delta_y[-1], delta_z[-1], linewidths=3, marker="*", edgecolors=sat_color_end)

    if show_relative_motion_lvlh:
        relative_motion = RelativeMotion(satellite, satellite2)

        delta_x, delta_y, delta_z = relative_motion.get_relative_motion_lvlh()

        ax_lvlh.plot3D(delta_x, delta_y, delta_z, linecolor)

        ax_lvlh.scatter(0, 0, 0, linewidths=3, marker="*", edgecolors=sat_color)
        ax_lvlh.scatter(delta_x[0], delta_y[0], delta_z[0], linewidths=3, marker="*", edgecolors=sat_color_start)
        ax_lvlh.scatter(delta_x[-1], delta_y[-1], delta_z[-1], linewidths=3, marker="*", edgecolors=sat_color_end)

        if show_solution:
            add_optimal_solution_to_lvlh_graph(ax_lvlh)

u, v = np.mgrid[0:2 * np.pi:180j, 0:np.pi:90j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

SPHERE_RADIUS = EARTH_RADIUS
cm = sns.color_palette("blend:#b300ff,#7959e3", as_cmap=True)
ax_orbits.plot_surface(SPHERE_RADIUS * x, SPHERE_RADIUS * y, SPHERE_RADIUS * z, cmap=cm, alpha=0.2)

ax_relative.set_xlabel("X (Sun)")
ax_relative.set_ylabel("Y (Eccleptic)")
ax_relative.set_zlabel("Z (Pole)")

ax_lvlh.set_xlabel("X (Sat Radius)")
ax_lvlh.set_ylabel("Y (Sat Velocity)")
ax_lvlh.set_zlabel("Z (Cross RxV")

plt.show()
