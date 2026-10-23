# ROS Occupancy-Map Global Planner

This repository turns a saved ROS map and two map-frame coordinates into a collision-aware grid route. It loads the map-server YAML and grayscale image, classifies free, occupied, and unknown cells, optionally accounts for a circular robot footprint, and uses A* to export world-coordinate waypoints as CSV.

The tool uses a map you provide. It does not generate example sensor readings or a fabricated environment.

## Install

```bash
python -m pip install -r requirements.txt
```

Export a trinary occupancy map from your ROS mapping workflow. Keep its YAML file and referenced image together, or use an image path relative to that YAML file. Then plan a route in the map frame:

```bash
python main.py --map maps/warehouse.yaml --start 1.0,1.0 --goal 7.5,4.0 --robot-radius 0.20 --output routes/warehouse.csv
```

Coordinates are `x,y` in meters in the map frame. `--robot-radius` grows occupied cells to provide footprint clearance. Unknown cells are blocked by default. To allow them with the default three-times traversal cost:

```bash
python main.py --map maps/warehouse.yaml --start 1.0,1.0 --goal 7.5,4.0 --allow-unknown --output routes/warehouse.csv
```

The CSV contains ordered waypoint centers (`x_m`, `y_m`), their grid indices, and cumulative route distance. No output file is written when the start/goal is invalid or no path can be found.

## Map support and scope

- Supports ROS map-server YAML with `mode: trinary` (or omitted mode), a grayscale map image, `resolution`, `origin`, `negate`, `free_thresh`, and `occupied_thresh`.
- Requires an unrotated map origin (`origin` yaw of zero). Map images use the ROS image convention with the first image row at the top.
- Uses eight-direction A* and prevents diagonal moves through obstacle corners. Unknown-cell traversal is optional and penalized.
- This is a global path planner for an existing map. It does not perform SLAM, localize a robot, avoid moving obstacles, create velocity commands, or connect to TurtleBot/Unitree hardware. A robot integration still needs localization, a local planner/controller, and runtime safety checks.

## License

MIT. See [LICENSE](LICENSE).
