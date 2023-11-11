from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from prediction.satellite import Satellite


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

    def get_relative_motion_lvlh(self):
        delta_x, delta_y, delta_z = self.get_relative_motion()
        ref_frames: List[Rotation] = self.ref_satellite.get_reference_frames()

        lvlh_x, lvlh_y, lvlh_z = [], [], []

        for x, y, z, ref_frame in zip(delta_x, delta_y, delta_z, ref_frames):
            rotated_vec = ref_frame.apply(np.array([x, y, z]))
            lvlh_x.append(rotated_vec[0])
            lvlh_y.append(rotated_vec[1])
            lvlh_z.append(rotated_vec[2])

        return lvlh_x, lvlh_y, lvlh_z
