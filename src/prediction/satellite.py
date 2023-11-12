from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from prediction.satellite_status import SatelliteStatus
from prediction.trajectory import Trajectory
from prediction.trajectory_predictor import TrajectoryPredictor


class Satellite:

    def __init__(self, satellite_status: SatelliteStatus,
                 trajectory_predictor: TrajectoryPredictor):
        self.satellite_status: SatelliteStatus = satellite_status
        self.trajectory: List[SatelliteStatus] = [self.satellite_status]
        self.trajectory_predictor: TrajectoryPredictor = trajectory_predictor

    def get_trajectory_position_per_axis(self):
        return Trajectory.from_satellite_status_list(self.trajectory).get_coordinates()

    def get_reference_frames(self) -> List[Rotation]:
        ref_frames = []

        for sat_status in self.trajectory:
            ref_frame = sat_status.get_rotation_matrix_lvlh()
            ref_frames.append(ref_frame)

        return ref_frames

    def update_satellite_status(self, new_satellite_status: SatelliteStatus):
        self.satellite_status = new_satellite_status
        self.trajectory.append(self.satellite_status)

    def extend_trajectory(self, steps=1):
        for step in range(steps):
            new_satellite_status = self.trajectory_predictor.next_step(self.satellite_status)
            self.update_satellite_status(new_satellite_status)
