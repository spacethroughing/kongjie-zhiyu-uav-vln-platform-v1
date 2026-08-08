import pytest

from harness.models import Polygon, SafetyEnvelope, SearchZone, Vec3
from harness.safety import SafetyViolation, approach_point, validate_position


ZONE = SearchZone(
    id="z", name="Zone", polygon=Polygon(points=[(-10, -10), (10, -10), (10, 10), (-10, 10)])
)
ENVELOPE = SafetyEnvelope(min_altitude_m=2, max_altitude_m=10)


def test_geofence_accepts_boundary_and_safe_altitude():
    validate_position(Vec3(x=10, y=0, z=-5), ZONE, ENVELOPE)


def test_geofence_rejects_outside_point():
    with pytest.raises(SafetyViolation):
        validate_position(Vec3(x=11, y=0, z=-5), ZONE, ENVELOPE)


def test_approach_keeps_standoff():
    point = approach_point(Vec3(x=0, y=0, z=-5), Vec3(x=10, y=0, z=0), 5, 3)
    assert point.x == pytest.approx(7)
    assert point.y == pytest.approx(0)
    assert point.z == pytest.approx(-5)

