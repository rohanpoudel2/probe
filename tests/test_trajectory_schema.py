from data.trajectory_schema import (
    build_trajectory_prefix_views,
    parse_trajectory_prefix_stack_view,
    trajectory_prefix_stack_view,
    parse_trajectory_prefix_view,
)


def test_build_trajectory_prefix_views_respects_bounds_and_rounding() -> None:
    assert build_trajectory_prefix_views([10, 25, 100], token_count=10) == [
        ("trajectory_prefix_p10", 1),
        ("trajectory_prefix_p25", 3),
        ("trajectory_prefix_p100", 10),
    ]


def test_parse_trajectory_prefix_view() -> None:
    assert parse_trajectory_prefix_view("trajectory_prefix_p33") == 33
    assert parse_trajectory_prefix_view("answer") is None


def test_trajectory_prefix_stack_view_and_parse() -> None:
    assert trajectory_prefix_stack_view(25) == "trajectory_prefix_stack_p25"
    assert parse_trajectory_prefix_stack_view("trajectory_prefix_stack_p25") == 25
    assert parse_trajectory_prefix_stack_view("trajectory_prefix_p25") is None
