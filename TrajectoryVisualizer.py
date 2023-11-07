import math
from datetime import timedelta, datetime
from typing import List

import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from TrajectoryPredictor import TrajectoryPredictor, OrbitalElements, SatelliteStatus, Satellite


class RelativeMotion:

    def __init__(self, ref_satellite: Satellite, child_satellite: Satellite):
        self.ref_satellite: Satellite = ref_satellite
        self.child_satellite: Satellite = child_satellite

    def get_relative_motion(self):
        rx, ry, rz = self.ref_satellite.get_trajectory_position_per_axis()
        rx2, ry2, rz2 = self.child_satellite.get_trajectory_position_per_axis()

        delta_x = self.compute_relative_distance(rx, rx2)
        delta_y = self.compute_relative_distance(ry, ry2)
        delta_z = self.compute_relative_distance(rz, rz2)

        return delta_x, delta_y, delta_z

    def compute_relative_distance(self, x, x2):
        return np.subtract(x2, x)

    def get_relative_motion_LVLH(self):
        delta_x, delta_y, delta_z = self.get_relative_motion()
        ref_frames: List[Rotation] = self.ref_satellite.get_reference_frames()

        lvlh_x, lvlh_y, lvlh_z = [], [], []

        for x, y, z, ref_frame in zip(delta_x, delta_y, delta_z, ref_frames):
            rotated_vec = ref_frame.apply(np.array([x, y, z]))
            lvlh_x.append(rotated_vec[0])
            lvlh_y.append(rotated_vec[1])
            lvlh_z.append(rotated_vec[2])

        return lvlh_x, lvlh_y, lvlh_z

def main():

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

    trajectory_predictor2 = TrajectoryPredictor(timedelta(seconds=5))
    # eccentricity, semi_major_axis, inclination, longitude_ascending_node, argument_periapsis, true_anomaly
    orbital_elements2 = OrbitalElements(0.06, 8000, math.radians(15), math.radians(5), math.radians(5), math.radians(15))
    satellite_status2 = SatelliteStatus(date, orbital_elements2)
    satellite2 = Satellite(satellite_status2, trajectory_predictor2)
    satellite2.extend_trajectory(steps=steps)

    rx2, ry2, rz2 = satellite2.get_trajectory_position_per_axis()

    if show_orbits:
        sns.set_style("dark", {"grid.color": ".1", "grid.linestyle": ":"})
        plt.figure(figsize=(12, 12))
        ax = plt.axes(projection='3d')
        ax.plot3D(rx, ry, rz)
        ax.plot3D(rx2, ry2, rz2)

        u, v = np.mgrid[0:2 * np.pi:180j, 0:np.pi:90j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        sphere_radius = 6371
        # cm = sns.color_palette("light:#5A9", as_cmap=True)
        # cm = sns.color_palette("dark:#5A9_r", as_cmap=True)
        # cm = sns.color_palette("blend:#ffffff,#376efa,#266e2e,#acad8c,#266e2e,#124d18,#ffffff", as_cmap=True)
        cm = sns.color_palette("blend:#b300ff,#7959e3", as_cmap=True)
        ax.plot_surface(sphere_radius*x, sphere_radius*y, sphere_radius*z, cmap=cm, alpha=0.2)

        total_max = np.array([max(a) for a in [rx, ry, rz, rx2, ry2, rz2]]).max()
        ax.set_zlim(-total_max, total_max)
        plt.xlim([-total_max, total_max])
        plt.ylim([-total_max, total_max])

        ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], length=total_max)

        # plt.tight_layout()
        plt.show()


    if show_relative_motion:

        relative_motion = RelativeMotion(satellite, satellite2)

        delta_x, delta_y, delta_z = relative_motion.get_relative_motion()
        total_max = max([np.absolute(delta_x).max(), np.absolute(delta_y).max(), np.absolute(delta_z).max()])

        sns.set_style("dark", {"grid.color": ".1", "grid.linestyle": ":"})
        plt.figure(figsize=(12, 12))
        ax = plt.axes(projection='3d')
        ax.plot3D(delta_x, delta_y, delta_z, linecolor)

        ax.scatter(0, 0, 0, linewidths=3, marker="*", edgecolors=sat_color)
        ax.scatter(delta_x[0], delta_y[0], delta_z[0], linewidths=3, marker="*", edgecolors=sat_color_start)
        ax.scatter(delta_x[-1], delta_y[-1], delta_z[-1], linewidths=3, marker="*", edgecolors=sat_color_end)

        ax.set_zlim(-total_max, total_max)
        plt.xlim([-total_max, total_max])
        plt.ylim([-total_max, total_max])

        ax.set_xlabel("X (Sun)")
        ax.set_ylabel("Y (Eccleptic)")
        ax.set_zlabel("Z (Pole)")
        plt.tight_layout()
        plt.show()

    if show_relative_motion_lvlh:


        relative_motion = RelativeMotion(satellite, satellite2)

        delta_x, delta_y, delta_z = relative_motion.get_relative_motion_LVLH()
        total_max = max([np.absolute(delta_x).max(), np.absolute(delta_y).max(), np.absolute(delta_z).max()])

        sns.set_style("dark", {"grid.color": ".1", "grid.linestyle": ":"})
        plt.figure(figsize=(12, 12))
        ax = plt.axes(projection='3d')

        ax.plot3D(delta_x, delta_y, delta_z, linecolor)

        ax.set_zlim(-total_max, total_max)
        plt.xlim([-total_max, total_max])
        plt.ylim([-total_max, total_max])

        ax.scatter(0, 0, 0, linewidths=3, marker="*", edgecolors=sat_color)
        ax.scatter(delta_x[0], delta_y[0], delta_z[0], linewidths=3, marker="*", edgecolors=sat_color_start)
        ax.scatter(delta_x[-1], delta_y[-1], delta_z[-1], linewidths=3, marker="*", edgecolors=sat_color_end)

        ax.set_xlabel("X (Sat Radius)")
        ax.set_ylabel("Y (Sat Velocity)")
        ax.set_zlabel("Z (Cross RxV")

        # plt.tight_layout()
        plt.show()










if __name__ == "__main__":
    main()



