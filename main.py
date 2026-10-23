"""Plan collision-aware paths through ROS occupancy maps."""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


@dataclass(frozen=True)
class OccupancyMap:
    free: np.ndarray
    occupied: np.ndarray
    unknown: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    image_path: Path

    @property
    def height(self) -> int:
        return int(self.free.shape[0])

    @property
    def width(self) -> int:
        return int(self.free.shape[1])

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        column = math.floor((x - self.origin_x) / self.resolution)
        row_from_bottom = math.floor((y - self.origin_y) / self.resolution)
        row = self.height - 1 - row_from_bottom
        if not (0 <= column < self.width and 0 <= row < self.height):
            raise ValueError(f"World coordinate ({x:g}, {y:g}) is outside the occupancy map.")
        return row, column

    def cell_center(self, row: int, column: int) -> tuple[float, float]:
        x = self.origin_x + (column + 0.5) * self.resolution
        y = self.origin_y + (self.height - row - 0.5) * self.resolution
        return x, y


def finite_number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def load_occupancy_map(yaml_path: Path) -> OccupancyMap:
    if not yaml_path.is_file():
        raise ValueError(f"Map YAML file does not exist: {yaml_path}")
    try:
        metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse map YAML: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Map YAML must contain a mapping of map-server settings.")

    image_name = metadata.get("image")
    if not isinstance(image_name, str) or not image_name.strip():
        raise ValueError("Map YAML must provide an image path.")
    image_path = (yaml_path.parent / image_name).resolve()
    if not image_path.is_file():
        raise ValueError(f"Occupancy image does not exist: {image_path}")

    mode = metadata.get("mode", "trinary")
    if mode != "trinary":
        raise ValueError("Only ROS map mode 'trinary' is supported.")
    resolution = finite_number(metadata.get("resolution"), "Map resolution")
    if resolution <= 0:
        raise ValueError("Map resolution must be greater than zero.")
    origin = metadata.get("origin", [0.0, 0.0, 0.0])
    if not isinstance(origin, (list, tuple)) or len(origin) != 3:
        raise ValueError("Map origin must contain x, y, and yaw values.")
    origin_x = finite_number(origin[0], "Map origin x")
    origin_y = finite_number(origin[1], "Map origin y")
    origin_yaw = finite_number(origin[2], "Map origin yaw")
    if not math.isclose(origin_yaw, 0.0, abs_tol=1e-9):
        raise ValueError("Rotated map origins are not supported; use a map with origin yaw 0.")

    free_threshold = finite_number(metadata.get("free_thresh", 0.196), "Free threshold")
    occupied_threshold = finite_number(metadata.get("occupied_thresh", 0.65), "Occupied threshold")
    if not (0 <= free_threshold < occupied_threshold <= 1):
        raise ValueError("Map thresholds must satisfy 0 <= free_thresh < occupied_thresh <= 1.")
    negate = metadata.get("negate", 0)
    if negate not in (0, 1, False, True):
        raise ValueError("Map 'negate' must be 0 or 1.")

    try:
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float32)
    except OSError as exc:
        raise ValueError(f"Could not read occupancy image: {exc}") from exc
    if pixels.ndim != 2 or not pixels.size:
        raise ValueError("Occupancy image must be a non-empty grayscale image.")

    probability = pixels / 255.0 if bool(negate) else (255.0 - pixels) / 255.0
    free = probability <= free_threshold
    occupied = probability >= occupied_threshold
    unknown = ~(free | occupied)
    if not np.any(free):
        raise ValueError("Map contains no cells classified as free.")
    return OccupancyMap(free, occupied, unknown, resolution, origin_x, origin_y, image_path)


def inflate_obstacles(occupied: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    """Conservatively grow occupied cells by the robot's circular footprint."""
    inflated = occupied.copy()
    if radius_m == 0:
        return inflated
    height, width = occupied.shape
    max_offset = math.ceil(radius_m / resolution + 0.5)
    for dy in range(-min(max_offset, height - 1), min(max_offset, height - 1) + 1):
        dy_clearance = max(0.0, abs(dy) * resolution - resolution / 2)
        for dx in range(-min(max_offset, width - 1), min(max_offset, width - 1) + 1):
            dx_clearance = max(0.0, abs(dx) * resolution - resolution / 2)
            if math.hypot(dx_clearance, dy_clearance) > radius_m:
                continue
            source_top = max(0, -dy)
            source_bottom = min(height, height - dy)
            source_left = max(0, -dx)
            source_right = min(width, width - dx)
            target_top, target_bottom = source_top + dy, source_bottom + dy
            target_left, target_right = source_left + dx, source_right + dx
            inflated[target_top:target_bottom, target_left:target_right] |= occupied[
                source_top:source_bottom, source_left:source_right
            ]
    return inflated


def octile_distance(row: int, column: int, goal_row: int, goal_column: int) -> float:
    delta_row = abs(goal_row - row)
    delta_column = abs(goal_column - column)
    diagonal = min(delta_row, delta_column)
    straight = max(delta_row, delta_column) - diagonal
    return diagonal * math.sqrt(2.0) + straight


def astar(
    traversable: np.ndarray,
    unknown: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    unknown_cost: float,
) -> list[tuple[int, int]]:
    height, width = traversable.shape
    if not traversable[start]:
        raise ValueError("Start coordinate is occupied or unknown; choose a traversable cell.")
    if not traversable[goal]:
        raise ValueError("Goal coordinate is occupied or unknown; choose a traversable cell.")

    start_index = start[0] * width + start[1]
    goal_index = goal[0] * width + goal[1]
    cell_count = height * width
    scores = np.full(cell_count, np.inf, dtype=np.float64)
    parents = np.full(cell_count, -1, dtype=np.int64)
    closed = np.zeros(cell_count, dtype=np.bool_)
    scores[start_index] = 0.0
    queue = [(octile_distance(*start, *goal), 0.0, start_index)]
    neighbors = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    )

    while queue:
        _priority, current_score, current_index = heapq.heappop(queue)
        if closed[current_index] or current_score != scores[current_index]:
            continue
        if current_index == goal_index:
            path: list[tuple[int, int]] = []
            cursor = goal_index
            while cursor != -1:
                path.append(divmod(cursor, width))
                if cursor == start_index:
                    break
                cursor = int(parents[cursor])
            return path[::-1]

        closed[current_index] = True
        row, column = divmod(current_index, width)
        for delta_row, delta_column, step_cost in neighbors:
            next_row, next_column = row + delta_row, column + delta_column
            if not (0 <= next_row < height and 0 <= next_column < width):
                continue
            if not traversable[next_row, next_column]:
                continue
            if delta_row and delta_column and (
                not traversable[row + delta_row, column]
                or not traversable[row, column + delta_column]
            ):
                continue

            next_index = next_row * width + next_column
            if closed[next_index]:
                continue
            terrain_cost = unknown_cost if unknown[next_row, next_column] else 1.0
            candidate_score = current_score + step_cost * terrain_cost
            if candidate_score >= scores[next_index]:
                continue
            scores[next_index] = candidate_score
            parents[next_index] = current_index
            estimate = octile_distance(next_row, next_column, *goal)
            heapq.heappush(queue, (candidate_score + estimate, candidate_score, next_index))

    raise ValueError("No collision-free path connects the requested start and goal.")


def parse_xy(value: str) -> tuple[float, float]:
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Coordinates must be written as x,y in map-frame meters.") from exc
    if len(parts) != 2 or not all(math.isfinite(part) for part in parts):
        raise argparse.ArgumentTypeError("Coordinates must be written as finite x,y map-frame meters.")
    return parts[0], parts[1]


def parse_nonnegative(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a finite non-negative number.") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Value must be a finite non-negative number.")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a collision-aware A* route through a ROS trinary occupancy map."
    )
    parser.add_argument("--map", required=True, type=Path, help="ROS map-server YAML file.")
    parser.add_argument("--start", required=True, type=parse_xy, metavar="X,Y", help="Start in map-frame meters.")
    parser.add_argument("--goal", required=True, type=parse_xy, metavar="X,Y", help="Goal in map-frame meters.")
    parser.add_argument("--output", type=Path, default=Path("path.csv"), help="Waypoint CSV destination.")
    parser.add_argument(
        "--robot-radius", type=parse_nonnegative, default=0.0,
        help="Circular robot footprint radius in meters; occupied cells are inflated by this amount.",
    )
    parser.add_argument(
        "--allow-unknown", action="store_true",
        help="Permit unknown map cells, with the configured traversal penalty.",
    )
    parser.add_argument(
        "--unknown-cost", type=parse_nonnegative, default=3.0,
        help="Relative cost of unknown cells (at least 1; default: 3).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.unknown_cost < 1:
            raise ValueError("Unknown-cell cost must be at least 1.")
        occupancy_map = load_occupancy_map(args.map)
        output_path = args.output.resolve()
        if output_path in {args.map.resolve(), occupancy_map.image_path}:
            raise ValueError("Output path must not overwrite the map YAML or occupancy image.")

        blocked = inflate_obstacles(
            occupancy_map.occupied, args.robot_radius, occupancy_map.resolution
        )
        traversable = occupancy_map.free & ~blocked
        if args.allow_unknown:
            traversable |= occupancy_map.unknown & ~blocked
        start = occupancy_map.world_to_cell(*args.start)
        goal = occupancy_map.world_to_cell(*args.goal)
        path = astar(traversable, occupancy_map.unknown, start, goal, args.unknown_cost)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(("waypoint", "x_m", "y_m", "grid_column", "grid_row", "distance_from_start_m"))
            distance = 0.0
            previous: tuple[float, float] | None = None
            for index, (row, column) in enumerate(path):
                x, y = occupancy_map.cell_center(row, column)
                if previous is not None:
                    distance += math.hypot(x - previous[0], y - previous[1])
                writer.writerow((index, f"{x:.6f}", f"{y:.6f}", column, row, f"{distance:.6f}"))
                previous = (x, y)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Route: {len(path)} waypoints, {distance:.3f} m, saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
