from typing import List, Tuple
import numpy as np

from prediction.satellite_status import SatelliteStatus


class Trajectory:

    def __init__(self, x: np.array, y: np.array, z: np.array):
        self.x: np.array = x
        self.y: np.array = y
        self.z: np.array = z

    def get_coordinates(self) -> Tuple[np.array, np.array, np.array]:
        return self.x, self.y, self.z

    def from_satellite_status_list(satellite_status_list: List[SatelliteStatus]):
        x, y, z = [], [], []

        for satellite_status in satellite_status_list:
            position = satellite_status.get_position()
            x.append(position[0])
            y.append(position[1])
            z.append(position[2])

        return Trajectory(np.array(x), np.array(y), np.array(z))

    def as_matrix(self) -> np.array:
        return np.array(self.x, self.y, self.z)

    def get_initial_position(self):
        return self.x[0], self.y[0], self.z[0]

    def to_list(self):
        return [self.x.tolist(),
                self.y.tolist(),
                self.z.tolist()]
