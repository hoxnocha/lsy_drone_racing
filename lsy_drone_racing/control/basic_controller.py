"""Example controller implementation moved out of the base module.

This file contains the user's example controller `MyController`. It must subclass the
`Controller` class from `lsy_drone_racing.control.controller`. The project's loader
`utils.load_controller()` expects exactly one controller class per file and will
load classes that subclass that base class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from drone_models.core import load_params
from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MyController(Controller):
    """Concrete controller implementation for demo purposes."""

    def normalize(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def quat_to_rot_matrix(self, quaternion: Tuple[float, float, float, float]) -> npt.NDArray[np.float64]:
        x, y, z, w = quaternion
        norm = np.sqrt(x * x + y * y + z * z + w * w)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

        Rm = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

        return Rm

    def rotate_local_normal(self, q: Tuple[float, float, float, float], local_normal=(1, 0, 0)) -> npt.NDArray[np.float64]:
        Rm = self.quat_to_rot_matrix(q)
        n_world = Rm.dot(np.array(local_normal))
        return self.normalize(n_world)
    


    
    def distance_point_to_line(self, A, B, P):
        # calculate the distance of the line from A to B to the point P
        AB = A - B
        AP = P - A
        return np.linalg.norm(np.cross(AB, AP)) / np.linalg.norm(AB)
    
    def detour_point(self, A, B, P, clearance):
        AB = B - A
        AB_unit = AB / np.linalg.norm(AB)
        # Direction of Line to Obstacle
        AP = P - A
        # Component orthogonal to AB
        ortho = AP - np.dot(AP, AB_unit) * AB_unit
        ortho_unit = ortho / np.linalg.norm(ortho)
        return P + clearance * ortho_unit

    



    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]

        # Safety distance from gate and obstacle
        d_gate = 0.5
        d_obstacle = 0.5
        # waypoint list
        way_points = []

        ### Compute way points
        drone_position = obs["pos"]
        gates_positions = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
        obstacle_positions = obs["obstacles_pos"]

        way_points.append(drone_position)

        for gate_quad, gate_pos in zip(gates_quat, gates_positions):
            normal_vector = self.rotate_local_normal(gate_quad)
            entry_point = gate_pos - d_gate * normal_vector
            exit_point = gate_pos + d_gate * normal_vector

            ### Check for collisions with obstacles:
            for obst_pos in obstacle_positions:
                beginning_of_line = way_points[-1]
                end_of_line = entry_point
                d = self.distance_point_to_line(beginning_of_line, end_of_line, obst_pos)
                if d <= d_obstacle:
                    way_points.append(self.detour_point(beginning_of_line, end_of_line, obst_pos, 1.5*d_obstacle))
                    break

            way_points.append(entry_point)
            way_points.append(gate_pos)
            way_points.append(exit_point)

        del way_points[0]



        ### convert waypoint list to numpy array
        way_points = np.array(way_points)

        self._t_total = 15
        t = np.linspace(0, self._t_total, len(way_points))
        self._des_pos_spline = CubicSpline(t, way_points)

        self._tick = 0
        self._finished = False

        print("This is my code")

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        des_pos = self._des_pos_spline(t)
        action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._tick = 0

    def reset(self):
        pass

    def episode_reset(self):
        pass
