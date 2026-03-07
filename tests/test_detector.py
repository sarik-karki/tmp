import numpy as np
import pytest

from src.vehicleDetector import VehicleDet, _clamp_bbox_xyxy as clamp_vehicle
from src.plateDetector import PlateDet, _clamp_bbox_xyxy as clamp_plate


# ---------------------------------------------------------------------------
# _clamp_bbox_xyxy (shared logic in both detectors)
# ---------------------------------------------------------------------------

class TestClampBboxVehicle:

    def test_valid_bbox_unchanged(self):
        b = clamp_vehicle((10, 20, 100, 200), 640, 480)
        assert b == (10, 20, 100, 200)

    def test_clamp_negative_coords(self):
        b = clamp_vehicle((-10, -20, 100, 200), 640, 480)
        assert b == (0, 0, 100, 200)

    def test_clamp_exceeding_coords(self):
        b = clamp_vehicle((10, 20, 700, 500), 640, 480)
        assert b == (10, 20, 639, 479)

    def test_zero_width_returns_none(self):
        assert clamp_vehicle((50, 50, 50, 100), 640, 480) is None

    def test_zero_height_returns_none(self):
        assert clamp_vehicle((50, 50, 100, 50), 640, 480) is None

    def test_inverted_returns_none(self):
        assert clamp_vehicle((100, 100, 50, 50), 640, 480) is None

    def test_fully_outside_returns_none(self):
        # Both x coords clamp to w-1=639, resulting in x1==x2
        assert clamp_vehicle((700, 500, 800, 600), 640, 480) is None

    def test_boundary_box(self):
        b = clamp_vehicle((0, 0, 639, 479), 640, 480)
        assert b == (0, 0, 639, 479)

    def test_single_pixel(self):
        b = clamp_vehicle((5, 5, 6, 6), 640, 480)
        assert b == (5, 5, 6, 6)


class TestClampBboxPlate:

    def test_valid_bbox_unchanged(self):
        b = clamp_plate((10, 20, 100, 50), 200, 100)
        assert b == (10, 20, 100, 50)

    def test_clamp_negative(self):
        b = clamp_plate((-5, -5, 50, 50), 200, 100)
        assert b == (0, 0, 50, 50)

    def test_inverted_returns_none(self):
        assert clamp_plate((100, 100, 50, 50), 200, 200) is None


# ---------------------------------------------------------------------------
# VehicleDet / PlateDet dataclass
# ---------------------------------------------------------------------------

class TestDetDataclasses:

    def test_vehicle_det_immutable(self):
        d = VehicleDet(bbox=(1, 2, 3, 4), conf=0.9, cls=2)
        with pytest.raises(AttributeError):
            d.conf = 0.5

    def test_plate_det_immutable(self):
        d = PlateDet(bbox=(1, 2, 3, 4), conf=0.8, cls=0)
        with pytest.raises(AttributeError):
            d.cls = 1

    def test_vehicle_det_fields(self):
        d = VehicleDet(bbox=(10, 20, 30, 40), conf=0.95, cls=3)
        assert d.bbox == (10, 20, 30, 40)
        assert d.conf == 0.95
        assert d.cls == 3

    def test_plate_det_equality(self):
        d1 = PlateDet(bbox=(1, 2, 3, 4), conf=0.8, cls=0)
        d2 = PlateDet(bbox=(1, 2, 3, 4), conf=0.8, cls=0)
        assert d1 == d2
