import json
from datetime import timedelta, datetime

import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

from orbital_coordinates.orbital_elements import OrbitalElements
from prediction.relative_motion import RelativeMotion
from prediction.satellite import Satellite
from prediction.satellite_status import SatelliteStatus
from prediction.trajectory_predictor import TrajectoryPredictor

smac_run = """../../smac3_output/1699503694.4798608"""

smac_history = smac_run + "/0/runhistory.json"

run_history = json.load(open(smac_history, "r"))

print(run_history)

configs = run_history.get("configs")
configs_to_plot = configs
leaders_only = True

if leaders_only:
    smac_intensifier = smac_run + "\\0\\intensifier.json"
    intensifier = json.load(open(smac_intensifier, "r"))
    smac_trajectory = intensifier.get("trajectory")

    leaders = {str(l.get('config_ids')[0]): configs.get(str(l.get('config_ids')[0])) for l in smac_trajectory}
    configs_to_plot = leaders

show_orbits = True
show_relative_motion = True
show_relative_motion_lvlh = True

linecolor = 'royalblue'
sat_color = "forestgreen"
sat_color_start = 'y'
sat_color_end = 'r'

steps = 10000

trajectory_predictor = TrajectoryPredictor(timedelta(seconds=5))

orbital_elements = OrbitalElements(0.05, 7500, 0, 0, 0, 0)
date = datetime(2023, 11, 4, 4, 44, 44)
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

    trajectory_predictor2 = TrajectoryPredictor(timedelta(seconds=5))
    # eccentricity, semi_major_axis, inclination, longitude_ascending_node, argument_periapsis, true_anomaly
    orbital_elements2 = OrbitalElements.from_dict(config)
    satellite_status2 = SatelliteStatus(date, orbital_elements2)
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

u, v = np.mgrid[0:2 * np.pi:180j, 0:np.pi:90j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

SPHERE_RADIUS = 6371
# cm = sns.color_palette("light:#5A9", as_cmap=True)
# cm = sns.color_palette("dark:#5A9_r", as_cmap=True)
# cm = sns.color_palette("blend:#ffffff,#376efa,#266e2e,#acad8c,#266e2e,#124d18,#ffffff", as_cmap=True)
cm = sns.color_palette("blend:#b300ff,#7959e3", as_cmap=True)
ax_orbits.plot_surface(SPHERE_RADIUS * x, SPHERE_RADIUS * y, SPHERE_RADIUS * z, cmap=cm, alpha=0.2)

ax_relative.set_xlabel("X (Sun)")
ax_relative.set_ylabel("Y (Eccleptic)")
ax_relative.set_zlabel("Z (Pole)")

ax_lvlh.set_xlabel("X (Sat Radius)")
ax_lvlh.set_ylabel("Y (Sat Velocity)")
ax_lvlh.set_zlabel("Z (Cross RxV")

plt.show()
