from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from TrajectoryPredictor import TrajectoryPredictor
from SatelliteStatus import SatelliteStatus


class Satellite:

    def __init__(self, satellite_status: SatelliteStatus, trajectory_predictor: TrajectoryPredictor):
        self.satellite_status: SatelliteStatus = satellite_status
        self.trajectory: List[SatelliteStatus] = [self.satellite_status]
        self.trajectory_predictor: TrajectoryPredictor = trajectory_predictor

    def get_trajectory_position_per_axis(self):
        rx, ry, rz = [], [], []

        for sat_status in self.trajectory:
            pos = sat_status.orbital_state_vectors.position
            rx.append(pos[0])
            ry.append(pos[1])
            rz.append(pos[2])

        return np.array(rx), np.array(ry), np.array(rz)

    def get_reference_frames(self) -> List[Rotation]:
        ref_frames = []

        for sat_status in self.trajectory:
            ref_frame = sat_status.get_rotation_matrix_LVLH()
            ref_frames.append(ref_frame)

        return ref_frames

    def update_satellite_status(self, new_satellite_status: SatelliteStatus):
        self.satellite_status = new_satellite_status
        self.trajectory.append(self.satellite_status)

    def extend_trajectory(self, steps=1):
        for step in range(steps):
            new_satellite_status = self.trajectory_predictor.next_step(self.satellite_status)
            self.update_satellite_status(new_satellite_status)
