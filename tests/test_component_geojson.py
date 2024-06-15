import unittest

from sigmf.component.geo_json import GeoJSONPoint


class CoverageCases(unittest.TestCase):
    """Default cases for GeoJSONPoint for codecov."""

    def setUp(self):
        # example given in SigMF spec.
        self.point ={"type":"Point", "coordinates": [-107.6183682, 34.0787916, 2120.0]}
        # w/out altitude
        self.point2 = {"type":"Point", "coordinates": [-107.6183682, 34.0787916]}

    def test_cov_wgs64_serialization(self):
        """Tests whether the wgs64 object serializes properly (codecov)."""
        point_dict = GeoJSONPoint(**self.point).model_dump(by_alias=True)
        self.assertEqual(self.point, point_dict)
        point_dict2 = GeoJSONPoint(**self.point2).model_dump(by_alias=True)
        self.assertEqual(self.point2, point_dict2)
        

class WarningCases(unittest.TestCase):
    """Cases that do not 'fail' but raise a warning to its behaviour."""
    
    def test_correct_point_type(self):
        """Tests whether warning is raised when point type differs."""
        point ={"type":"RandomType",
                 "coordinates": [-107.6183682, 34.0787916, 2120.0]}
        # when type != Point
        with self.assertWarns(Warning):
            GeoJSONPoint(**point)


class FailingCases(unittest.TestCase):
    """Cases where the validator should throw an exception."""

    def setUp(self):
        # two incorrect examples.
        self.point ={"type":"Point", "coordinates": [-107.6183682, 34.0787916, 2120.0, 0.5, 0.4]}
        self.point2 = {"type":"Point", "coordinates": [-107.6183682]}
        # no coordinates.
        self.point3 = {"type": "Point"}

    def test_error_raises_incorrect_length(self):
        """Checks that the correct number of coordinates is specified."""
        with self.assertRaises(ValueError):
            GeoJSONPoint(**self.point)
        with self.assertRaises(ValueError):
            GeoJSONPoint(**self.point2)
    
    def test_coordinates_key_present(self):
        with self.assertRaises(KeyError):
            # missing key.
            GeoJSONPoint(**self.point3)
