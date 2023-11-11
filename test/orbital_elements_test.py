import unittest

from orbital_coordinates.orbital_elements import OrbitalElements


class OrbitalElementsTest(unittest.TestCase):

    def test_from_dict(self):
        input_dict = {
            'argument_periapsis': 4.055597577266145,
            'eccentricity': 0.04546875245259896,
            'inclination': 0.00711747023195512,
            'longitude_ascending_node': 1.8001780849322229,
            'semi_major_axis': 7489.777383978899,
            'true_anomaly': 0.4620830455294944,
        }

        oe = OrbitalElements.from_dict(input_dict)

        self.assertEqual(input_dict.get("argument_periapsis"), oe.argument_periapsis)
        self.assertEqual(input_dict.get("eccentricity"), oe.eccentricity)
        self.assertEqual(input_dict.get("inclination"), oe.inclination)
        self.assertEqual(input_dict.get("longitude_ascending_node"), oe.longitude_ascending_node)
        self.assertEqual(input_dict.get("semi_major_axis"), oe.semi_major_axis)
        self.assertEqual(input_dict.get("true_anomaly"), oe.true_anomaly)
