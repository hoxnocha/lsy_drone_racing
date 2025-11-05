from __future__ import annotations  

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import Controller 

from drone_models.core import load_params

if TYPE_CHECKING:
    from numpy.typing import NDArray
    

class DynamicTrajectoryController(Controller):
    

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        
        self._freq = config.env.freq
        self.t_total = 30 # time duration for the trajectory
        self._tick = 0
        self._finished = False
        
        
        
      
        self.gates_pos = obs['gates_pos']
        self.init_pos = obs['pos']
        self.gates_norm, self.gate_y_axes, self.gate_z_axes = \
            self._extract_gate_coordinate_frames(obs['gates_quat'])

        
        waypoints = self.calc_waypoints(self.init_pos, self.gates_pos, self.gates_norm)
        
        
        waypoints = self._add_detour_waypoints(
            waypoints,
            self.gates_pos,
            self.gates_norm,
            self.gate_y_axes,
            self.gate_z_axes,
            num_intermediate_points=5,
            angle_threshold=120.0,
            detour_distance=0.65
        )

       
        t, waypoints_avoided = self.avoid_collision(waypoints, obs['obstacles_pos'], 0.3)

        
        if len(t) < 2: 
            print("警告: avoid_collision 返回点少于2个。使用原始路径点。")
            self.trajectory = self.trajectory_generate(self.t_total, waypoints)
        else:
            self.trajectory = CubicSpline(t, waypoints_avoided)
            self.t_total = self.trajectory.x[-1] 
            
        
        if load_params:
            drone_params = load_params(config.sim.physics, config.sim.drone_model)
            self.drone_mass = drone_params["mass"]
            
    
    def calc_waypoints(
            self, drone_init_pos: NDArray[np.floating], gates_pos: NDArray[np.floating], gates_norm: NDArray[np.floating], distance: float = 0.5 , num_int_pnts: int = 5,
    ) -> NDArray[np.floating]:
        """计算门之间的插值路径点 (来自 level2_1.py)。"""
        num_gates = gates_pos.shape[0]
        # 在每个门的前后创建路径点
        wp = np.concatenate([gates_pos - distance * gates_norm + i/(num_int_pnts-1) * 2 * distance * gates_norm for i in range(num_int_pnts)], axis=1).reshape(num_gates, num_int_pnts, 3).reshape(-1,3)
        # 将无人机初始位置添加为第一个点
        wp = np.concatenate([np.array([drone_init_pos]), wp], axis=0)
        return wp
    
    def trajectory_generate(
        self, t_total: float, waypoints: NDArray[np.floating],
    ) -> CubicSpline:
        """从路径点生成三次样条轨迹 (来自 level2_1.py)。"""
        diffs = np.diff(waypoints, axis=0)
        segment_length = np.linalg.norm(diffs, axis=1)
        arc_cum_length = np.concatenate([[0], np.cumsum(segment_length)])
        t = arc_cum_length / arc_cum_length[-1] * t_total
        return CubicSpline(t, waypoints)
    
    def avoid_collision(
        self, waypoints: NDArray[np.floating], obstacles_pos: NDArray[np.floating], safe_dist: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """修改路径点以避免与障碍物碰撞 (来自 level2_1.py)。"""
        # 首先，生成一个初步的轨迹
        pre_trajectory = self.trajectory_generate(self.t_total, waypoints)
        
        num_steps = int(self._freq * self.t_total)
        if num_steps <= 0: # 防止 t_total 几乎为0时出错
            num_steps = 1
            
        t_axis = np.linspace(0, self.t_total, num_steps)
        
        wp = pre_trajectory(t_axis)

        for obst_idx, obst in enumerate(obstacles_pos):
            flag = False
            t_results = []
            wp_results = []
            
            for i in range(wp.shape[0]):
                point = wp[i]
                if np.linalg.norm(obst[:2] - point[:2]) < safe_dist and not flag: 
                    flag = True
                    in_idx = i
                elif np.linalg.norm(obst[:2] - point[:2]) >= safe_dist and flag:    
                    out_idx = i
                    flag = False
                    direction = wp[in_idx][:2] - obst[:2] + wp[out_idx][:2] - obst[:2]
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    new_point_xy = obst[:2] + direction * safe_dist
                    new_point_z = (wp[in_idx][2] + wp[out_idx][2])/2
                    new_point = np.concatenate([new_point_xy, [new_point_z]])
                    
                    t_results.append((t_axis[in_idx] + t_axis[out_idx])/2)
                    wp_results.append(new_point)
                elif np.linalg.norm(obst[:2] - point[:2]) >= safe_dist:   
                    t_results.append(t_axis[i])
                    wp_results.append(point)
            
            if flag:
                t_results.append(t_axis[-1])
                wp_results.append(wp[-1])

            t_axis = np.array(t_results)
            wp = np.array(wp_results)

        if len(t_axis) > 0:
            unique_indices = np.unique(t_axis, return_index=True)[1]
            t_axis = t_axis[unique_indices]
            wp = wp[unique_indices]

        if len(t_axis) < 2:
            print("Avoid_collision: 过滤后点不足，返回原始路径点。")
            t_axis_fallback = self.trajectory_generate(self.t_total, waypoints).x
            wp_fallback = waypoints
            return t_axis_fallback, wp_fallback

        return t_axis, wp
    
    def pos_change_detect(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        
        if not hasattr(self, 'last_gate_flags'):
            self.last_gate_flags = np.array(obs['gates_visited'], dtype=bool)
            self.last_obst_flags = np.array(obs['obstacles_visited'], dtype=bool)
            return False

        curr_gate_flags = np.array(obs['gates_visited'], dtype=bool)
        curr_obst_flags = np.array(obs['obstacles_visited'], dtype=bool)

        gate_triggered = np.any((~self.last_gate_flags) & curr_gate_flags)
        obst_triggered = np.any((~self.last_obst_flags) & curr_obst_flags)

        self.last_gate_flags = curr_gate_flags
        self.last_obst_flags = curr_obst_flags

        return gate_triggered or obst_triggered

    
    
    def _extract_gate_coordinate_frames(
        self, 
        gates_quaternions: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        
        rotations = R.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        
        normals = rotation_matrices[:, :, 0]  #normalvektor (x-axis)
        y_axes = rotation_matrices[:, :, 1]   
        z_axes = rotation_matrices[:, :, 2]   
        
        return normals, y_axes, z_axes

    def _add_detour_waypoints(
        self,
        waypoints: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
        num_intermediate_points: int = 5,
        angle_threshold: float = 120.0,
        detour_distance: float = 0.65
    ) -> NDArray[np.floating]:
        
        num_gates = gate_positions.shape[0]
        waypoints_list = list(waypoints)  
        
        inserted_count = 0
        
        for i in range(num_gates - 1):
            
            last_idx_gate_i = 1 + (i + 1) * num_intermediate_points - 1 + inserted_count
            first_idx_gate_i_plus_1 = 1 + (i + 1) * num_intermediate_points + inserted_count
            
            p1 = waypoints_list[last_idx_gate_i]
            p2 = waypoints_list[first_idx_gate_i_plus_1]
            
            v = p2 - p1
            v_norm = np.linalg.norm(v)
            
            if v_norm < 1e-6:
                continue
            
            normal_i = gate_normals[i]
            cos_angle = np.dot(v, normal_i) / v_norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.arccos(cos_angle) * 180 / np.pi
            
            if angle_deg > angle_threshold:
                gate_center = gate_positions[i]
                y_axis = gate_y_axes[i]
                z_axis = gate_z_axes[i]
                
                v_proj = v - np.dot(v, normal_i) * normal_i
                v_proj_norm = np.linalg.norm(v_proj)
                
                if v_proj_norm < 1e-6:
                    detour_direction_vector = y_axis
                else:
                    v_proj_y = np.dot(v_proj, y_axis)
                    v_proj_z = np.dot(v_proj, z_axis)
                    proj_angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi
                    
                    if -90 <= proj_angle_deg < 45:
                        detour_direction_vector = y_axis
                    elif 45 <= proj_angle_deg < 135:
                        detour_direction_vector = z_axis
                    else:
                        detour_direction_vector = -y_axis
                
                detour_waypoint = gate_center + detour_distance * detour_direction_vector
                
                insert_position = last_idx_gate_i + 1
                waypoints_list.insert(insert_position, detour_waypoint)
                inserted_count += 1
        
        return np.array(waypoints_list)

   


    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            A 13-element drone state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate].
        """
        
     
        if self.pos_change_detect(obs):
            print(f"T={self._tick / self._freq:.2f}: 探测到新物体！重新规划轨迹。")
            
          
            self.gates_pos = obs['gates_pos']
            self.gates_norm, self.gate_y_axes, self.gate_z_axes = \
                self._extract_gate_coordinate_frames(obs['gates_quat'])
            
            
            waypoints = self.calc_waypoints(self.init_pos, self.gates_pos, self.gates_norm)
            
            # 3. 添加绕行路径点 (来自 level2.py)
            waypoints = self._add_detour_waypoints(
                waypoints,
                self.gates_pos,
                self.gates_norm,
                self.gate_y_axes,
                self.gate_z_axes,
                num_intermediate_points=5,
                angle_threshold=120.0,
                detour_distance=0.65
            )
            
            # 4. 重新运行碰撞规避 (来自 level2_1.py)
            t, waypoints_avoided = self.avoid_collision(waypoints, obs['obstacles_pos'], 0.3)
            
            # 5. 用新数据生成新的样条 (来自 level2_1.py)
            if len(t) < 2:
                print("警告: 重规划时 avoid_collision 返回点少于2个。")
                self.trajectory = self.trajectory_generate(self.t_total, waypoints)
            else:
                self.trajectory = CubicSpline(t, waypoints_avoided)
            
            self.t_total = self.trajectory.x[-1]

        # ---== End of Re-planning Logic ==---

        tau = min(self._tick / self._freq, self.trajectory.x[-1])
        target_pos = self.trajectory(tau)

        if tau >= self.trajectory.x[-1]:  
            self._finished = True

        action = np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)
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
        """
        Callback function called once after the control step.
        Increment the tick counter.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        
        pass

    def episode_reset(self):
        """
        Reset the controller's internal state.
        """
        self._tick = 0
        self._finished = False
        
        if hasattr(self, 'last_gate_flags'):
            delattr(self, 'last_gate_flags')
        if hasattr(self, 'last_obst_flags'):
            delattr(self, 'last_obst_flags')