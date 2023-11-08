import math
from datetime import timedelta, datetime

from orbital_coordinates.orbital_elements import OrbitalElements
from prediction.satellite import Satellite
from prediction.satellite_status import SatelliteStatus
from prediction.trajectory_predictor import TrajectoryPredictor

if __name__ == "__main__":
    trajectory_predictor = TrajectoryPredictor(timedelta(seconds=5))
    orbital_elements = OrbitalElements(0.05, 7000, math.radians(50), 0, 0, math.radians(89.99999999))
    date = datetime(2023, 11, 4, 4, 44, 44)
    satellite_status = SatelliteStatus(date, orbital_elements)
    satellite = Satellite(satellite_status, trajectory_predictor)
    satellite.extend_trajectory()
