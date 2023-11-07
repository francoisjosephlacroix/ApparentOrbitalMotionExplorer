import math
from datetime import timedelta, datetime

from OrbitalCoordinates.OrbitalElements import OrbitalElements
from Prediction.Satellite import Satellite
from Prediction.SatelliteStatus import SatelliteStatus
from Prediction.TrajectoryPredictor import TrajectoryPredictor

if __name__ == "__main__":
    trajectory_predictor = TrajectoryPredictor(timedelta(seconds=5))
    orbital_elements = OrbitalElements(0.05, 7000, math.radians(50), 0, 0, math.radians(89.99999999))
    date = datetime(2023, 11, 4, 4, 44, 44)
    satellite_status = SatelliteStatus(date, orbital_elements)
    satellite = Satellite(satellite_status, trajectory_predictor)
    satellite.extend_trajectory()
