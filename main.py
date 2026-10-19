import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Pose:
    x: float; y: float; theta: float

    def move(self, v: float, omega: float, dt: float = 0.1) -> "Pose":
        theta_new = self.theta + omega * dt
        x_new = self.x + v * np.cos(self.theta) * dt
        y_new = self.y + v * np.sin(self.theta) * dt
        return Pose(x_new, y_new, theta_new)

class OccupancyGrid:
    def __init__(self, width: int = 100, height: int = 100, resolution: float = 0.05):
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.resolution = resolution
        self.origin = Pose(-width * resolution / 2, -height * resolution / 2, 0)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.origin.x) / self.resolution)
        gy = int((y - self.origin.y) / self.resolution)
        return gx, gy

def astar(grid: np.ndarray, start: Tuple, goal: Tuple) -> Optional[List[Tuple]]:
    import heapq
    h, w = grid.shape
    def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(0 + heuristic(start, goal), 0, start, [])]
    visited = set()
    while open_set:
        f, g, node, path = heapq.heappop(open_set)
        if node in visited: continue
        visited.add(node)
        path = path + [node]
        if node == goal: return path
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = node[0]+dx, node[1]+dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] < 0.5 and (nx,ny) not in visited:
                ng = g + 1
                heapq.heappush(open_set, (ng + heuristic((nx,ny), goal), ng, (nx,ny), path))
    return None

if __name__ == "__main__":
    occ = OccupancyGrid(100, 100, 0.05)
    robot = Pose(0.0, 0.0, 0.0)
    np.random.seed(42)
    for step in range(50):
        robot = robot.move(v=0.1, omega=np.random.uniform(-0.2, 0.2))
    print(f"Grid shape: {occ.grid.shape}")
    print(f"Robot pose: x={robot.x:.2f} y={robot.y:.2f} theta={np.degrees(robot.theta):.1f} deg")
    path = astar(occ.grid, (50,50), (80,80))
    if path:
        print(f"Path found: {len(path)} waypoints")
    else:
        print("No path found (open grid, direct)")
