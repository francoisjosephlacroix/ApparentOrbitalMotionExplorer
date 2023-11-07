class OrbitalElements:
    def __init__(self, eccentricity, semi_major_axis, inclination, longitude_ascending_node, argument_periapsis,
                 true_anomaly):
        self.eccentricity = float(eccentricity)
        self.semi_major_axis = float(semi_major_axis)  # in km
        self.inclination = float(inclination)  # in rads
        self.longitude_ascending_node = float(longitude_ascending_node)  # in rads
        self.argument_periapsis = float(argument_periapsis)  # in rads
        self.true_anomaly = float(true_anomaly)  # in rads
        self.semi_latus_rectum = self.semi_major_axis * (1 - self.eccentricity ** 2)
