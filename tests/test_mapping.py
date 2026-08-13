from harness.mapping import TopologicalMapper
from harness.models import Vec3


def test_mapper_builds_occupancy_topology_and_semantic_landmarks():
    mapper = TopologicalMapper(place_spacing_m=3.0)
    assert mapper.integrate_pose(Vec3(x=0, y=0, z=-5)) is True
    assert mapper.integrate_pose(Vec3(x=1, y=0, z=-5)) is False
    assert mapper.integrate_pose(Vec3(x=4, y=0, z=-5)) is True

    accepted = mapper.integrate_lidar(
        [
            Vec3(x=5, y=2, z=-5),
            Vec3(x=5.2, y=2.1, z=-5.2),
            Vec3(x=6, y=3, z=0),
            Vec3(x=100, y=100, z=-5),
        ],
        Vec3(x=4, y=0, z=-5),
    )
    assert accepted == 2
    mapper.integrate_lidar(
        [Vec3(x=5.1, y=2.0, z=-5), Vec3(x=5.3, y=2.2, z=-5.1)],
        Vec3(x=4, y=0, z=-5),
    )

    first = mapper.integrate_semantic(
        "橙色球体", Vec3(x=8, y=2, z=-3), 0.91, "frame-1"
    )
    second = mapper.integrate_semantic(
        "橙色圆球", Vec3(x=8.4, y=2.2, z=-3), 0.95, "frame-2"
    )
    assert first is not None and second is not None
    assert first["id"] == second["id"]
    assert second["observations"] == 2
    assert mapper.integrate_semantic(
        "天空", Vec3(x=8, y=2, z=-3), 0.99, "frame-3"
    ) is None
    cone = mapper.integrate_semantic(
        "蓝色圆锥体", Vec3(x=8.2, y=2.1, z=-3), 0.9, "frame-4"
    )
    assert cone is not None and cone["id"] != second["id"]

    snapshot = mapper.snapshot()
    assert snapshot["stats"]["occupancy_cells"] == 1
    assert snapshot["stats"]["occupancy_tracks"] == 1
    assert snapshot["stats"]["explored_cells"] > 0
    assert snapshot["explored"]
    assert snapshot["stats"]["place_nodes"] == 2
    assert snapshot["stats"]["semantic_objects"] == 1
    assert snapshot["stats"]["semantic_tracks"] == 2
    tentative = mapper.snapshot(include_tentative_semantics=True)
    semantic_nodes = [node for node in tentative["nodes"] if node["kind"] == "object"]
    assert len(semantic_nodes) == 2
    assert {node["confirmed"] for node in semantic_nodes} == {True, False}
    assert any(edge["kind"] == "traversal" for edge in snapshot["edges"])
    assert any(edge["kind"] == "observed" for edge in snapshot["edges"])
    assert mapper.is_explored(Vec3(x=4.5, y=0, z=-5)) is True
    assert mapper.is_explored(Vec3(x=20, y=20, z=-5)) is False


def test_lidar_rays_mark_free_space_as_explored_without_marking_off_ray_cells():
    mapper = TopologicalMapper(coverage_cell_size_m=2.0)
    mapper.integrate_lidar(
        [Vec3(x=10, y=0, z=-5)],
        Vec3(x=0, y=0, z=-5),
    )

    assert mapper.exploration_coverage(Vec3(x=5, y=0, z=-5), 2.5) >= 0.35
    assert mapper.is_explored(Vec3(x=5, y=0, z=-5), 2.5) is True
    assert mapper.is_explored(Vec3(x=5, y=8, z=-5), 2.5) is False


def test_polygon_coverage_ratio_uses_lidar_observed_cells():
    mapper = TopologicalMapper(coverage_cell_size_m=2.0)
    polygon = [(0, 0), (8, 0), (8, 8), (0, 8)]
    assert mapper.coverage_ratio_in_polygon(polygon) == 0
    mapper.integrate_lidar(
        [Vec3(x=7, y=1, z=-5), Vec3(x=7, y=7, z=-5)],
        Vec3(x=1, y=1, z=-5),
    )
    assert 0 < mapper.coverage_ratio_in_polygon(polygon) <= 1


def test_mapper_separates_actual_traversal_from_lidar_observation():
    mapper = TopologicalMapper(coverage_cell_size_m=2.0, place_spacing_m=4.0)
    for x in (0.0, 4.0, 8.0):
        mapper.integrate_pose(Vec3(x=x, y=0, z=-5))

    assert mapper.trajectory_revisit_ratio(
        Vec3(x=0, y=0, z=-5), Vec3(x=8, y=0, z=-5)
    ) == 1.0
    assert mapper.trajectory_revisit_ratio(
        Vec3(x=8, y=0, z=-5), Vec3(x=16, y=0, z=-5)
    ) == 0.0

    # A LiDAR ray marks visible free space but must not turn it into a flown
    # trajectory corridor.
    mapper.integrate_lidar(
        [Vec3(x=8, y=10, z=-5)],
        Vec3(x=8, y=0, z=-5),
    )
    assert mapper.segment_exploration_coverage(
        Vec3(x=8, y=0, z=-5), Vec3(x=8, y=8, z=-5)
    ) > 0.5
    assert mapper.trajectory_revisit_ratio(
        Vec3(x=8, y=0, z=-5), Vec3(x=8, y=8, z=-5)
    ) == 0.0
    snapshot = mapper.snapshot()
    assert snapshot["stats"]["traversed_cells"] == 5
    assert snapshot["traversed"]


def test_semantic_tracks_coalesce_after_filtered_positions_converge():
    mapper = TopologicalMapper()
    first = mapper.integrate_semantic("black cube", Vec3(x=0, y=0, z=-3), 0.8, "a")
    second = mapper.integrate_semantic("black cube", Vec3(x=4, y=0, z=-3), 0.8, "b")
    assert first and second and first["id"] != second["id"]

    mapper.integrate_semantic("black cube", Vec3(x=1.4, y=0, z=-3), 0.9, "c")
    mapper.integrate_semantic("black cube", Vec3(x=2.6, y=0, z=-3), 0.9, "d")

    snapshot = mapper.snapshot()
    objects = [node for node in snapshot["nodes"] if node["kind"] == "object"]
    assert len(objects) == 1
    assert objects[0]["observations"] == 4
