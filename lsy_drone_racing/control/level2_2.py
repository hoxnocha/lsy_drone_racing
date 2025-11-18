# minsnap

from __future__ import annotations 

from typing import TYPE_CHECKING, Tuple, Any

import numpy as np

import minsnap_trajectories as ms


from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import Controller 

from drone_models.core import load_params



if TYPE_CHECKING:
    from numpy.typing import NDArray



class DynamicTrajectoryController(Controller):
    
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        
        self._freq = config.env.freq
        self.t_total = 20
        self._tick = 0
        self._finished = False

      
        self.SPEED_FAST = 2.0  
        self.SPEED_SLOW = 1.5  
        self.SLOW_DOWN_DISTANCE_GATE = 1.0 
        self.SLOW_DOWN_DISTANCE_OBSTACLE = 1.0
       
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

        
        t, waypoints_avoided = self.avoid_collision(
            waypoints, 
            obs['obstacles_pos'], 
            0.3
        )
      
        if len(t) < 2: 
            print("aviod collision returned less than 2 points during initialization.")
            
            t_fallback = self.allocate_time(
                waypoints, 
                self.gates_pos, 
                obs['obstacles_pos']
            )
            self.trajectory = self.generate_trajectory(waypoints, t_fallback)
        else:   
            self.trajectory = self.generate_trajectory(waypoints_avoided, t)
            
        
        self.t_total = t[-1] if len(t) > 0 else 0
        
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
            
    
    def calc_waypoints(
            self, drone_init_pos: NDArray[np.floating], gates_pos: NDArray[np.floating], gates_norm: NDArray[np.floating], distance: float = 0.5 , num_int_pnts: int = 5,
    ) -> NDArray[np.floating]:
        
        num_gates = gates_pos.shape[0]
        wp = np.concatenate([gates_pos - distance * gates_norm + i/(num_int_pnts-1) * 2 * distance * gates_norm for i in range(num_int_pnts)], axis=1).reshape(num_gates, num_int_pnts, 3).reshape(-1,3)
        wp = np.concatenate([np.array([drone_init_pos]), wp], axis=0)
        return wp

    def allocate_time(
        self, 
        waypoints: NDArray[np.floating],
        gates_pos: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        according to the distance to gates and obstacles, allocate time for each segment.
        """
        times = [0.0]
        cumulative_time = 0.0
        
        
        if len(obstacles_pos) == 0:
            obstacles_pos_2d = np.empty((0, 2))
        else:
            obstacles_pos_2d = obstacles_pos[:, :2]

        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i+1]
            segment_midpoint = (p1 + p2) / 2
            segment_distance = np.linalg.norm(p2 - p1)

            if segment_distance < 1e-6:
                times.append(cumulative_time)
                continue

            is_slow_zone = False

            
            min_gate_dist = np.min(np.linalg.norm(segment_midpoint - gates_pos, axis=1))
            if min_gate_dist < self.SLOW_DOWN_DISTANCE_GATE:
                is_slow_zone = True

           
            if not is_slow_zone and obstacles_pos_2d.shape[0] > 0:
                min_obst_dist = np.min(np.linalg.norm(segment_midpoint[:2] - obstacles_pos_2d, axis=1))
                if min_obst_dist < self.SLOW_DOWN_DISTANCE_OBSTACLE:
                    is_slow_zone = True
            
            
            target_speed = self.SPEED_SLOW if is_slow_zone else self.SPEED_FAST
            segment_time = segment_distance / target_speed
            
            cumulative_time += segment_time
            times.append(cumulative_time)

        return np.array(times)

    def generate_trajectory(
        self, waypoints: NDArray[np.floating], times: NDArray[np.floating]
    ) -> Any: # <-- 移除了 Tuple 和 bool 返回类型
        """
        使用 minsnap_trajectories 生成轨迹。
        
        """
        
        refs = []
        
        # 起点：约束位置、速度、加速度
        refs.append(ms.Waypoint(
            time=times[0],
            position=waypoints[0],
            velocity=np.zeros(3),
            acceleration=np.zeros(3)
        ))
        
        # 中间点：只约束位置
        for i in range(1, len(waypoints) - 1):
            refs.append(ms.Waypoint(
                time=times[i],
                position=waypoints[i]
            ))
        
        # 终点：约束位置、速度、加速度
        refs.append(ms.Waypoint(
            time=times[-1],
            position=waypoints[-1],
            velocity=np.zeros(3),
            acceleration=np.zeros(3)
        ))
        
        # 2. 生成轨迹
        print(f"[MinSnap] 正在为 {len(refs)} 个路径点求解...")
        polys = ms.generate_trajectory(
            refs,
            degree=8,
            idx_minimized_orders=(3, 4), # 最小化 Jerk 和 Snap
            num_continuous_orders=3, # 约束 Jerk 连续
            algorithm="closed-form",
        )
        print("[MinSnap] ...求解完成。")
        
        
        return polys

    
    def avoid_collision(
        self, 
        waypoints: NDArray[np.floating], 
        obstacles_pos: NDArray[np.floating], # (修改) 传入障碍物位置
        safe_dist: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        
        
        t_axis_initial = self.allocate_time(
            waypoints, 
            self.gates_pos, 
            obstacles_pos
        )
        
        pre_trajectory = CubicSpline(t_axis_initial, waypoints)
        
        current_t_total = t_axis_initial[-1]
        num_steps = int(self._freq * current_t_total)
        if num_steps <= 0: 
            num_steps = 1
            
        t_sampled = np.linspace(0, current_t_total, num_steps)
        wp = pre_trajectory(t_sampled)

        for obst_idx, obst in enumerate(obstacles_pos):
            flag = False
            t_results = []
            wp_results = []
            
            for i in range(len(t_sampled)):
                point = wp[i]
                t_current = t_sampled[i]
                
                if np.linalg.norm(obst[:2] - point[:2]) < safe_dist and not flag: 
                    flag = True
                    in_idx = i
                elif np.linalg.norm(obst[:2] - point[:2]) >= safe_dist and flag:    
                    out_idx = i
                    flag = False
                    
                    in_time = t_sampled[in_idx]
                    out_time = t_sampled[out_idx]
                    in_point = wp[in_idx]
                    out_point = wp[out_idx]

                    direction = in_point[:2] - obst[:2] + out_point[:2] - obst[:2]
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    new_point_xy = obst[:2] + direction * safe_dist
                    new_point_z = (in_point[2] + out_point[2])/2
                    new_point = np.concatenate([new_point_xy, [new_point_z]])
                    
                    t_results.append((in_time + out_time) / 2)
                    wp_results.append(new_point)
                elif np.linalg.norm(obst[:2] - point[:2]) >= safe_dist:   
                    t_results.append(t_current)
                    wp_results.append(point)
            
            if flag:
                t_results.append(t_sampled[-1])
                wp_results.append(wp[-1])

            t_sampled = np.array(t_results)
            wp = np.array(wp_results)

        if len(t_sampled) > 0:
            unique_indices = np.unique(t_sampled, return_index=True)[1]
            t_axis = t_sampled[unique_indices]
            wp_final = wp[unique_indices]
        else:
            t_axis = t_sampled
            wp_final = wp

        if len(t_axis) < 2:
            print("Avoid_collision: returned less than 2 points, using fallback allocation.")
            t_axis_fallback = self.allocate_time(
                waypoints, 
                self.gates_pos, 
                obstacles_pos
            )
            wp_fallback = waypoints
            return t_axis_fallback, wp_fallback

        return t_axis, wp_final
    
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
        """
        extract the full local coordinate frames of each gate 
        """
        rotations = R.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        
        normals = rotation_matrices[:, :, 0]  # normalvektor (x-axis)
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
        angle_threshold: float = 110.0,
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
        
        # ---== 动态重规划逻辑 (已更新) ==---
        if self.pos_change_detect(obs):
            print(f"T={self._tick / self._freq:.2f}: 探测到新物体！重新规划轨迹。")
            
            # 1. 提取完整的门坐标系
            self.gates_pos = obs['gates_pos']
            self.gates_norm, self.gate_y_axes, self.gate_z_axes = \
                self._extract_gate_coordinate_frames(obs['gates_quat'])
            
            # 2. 计算初始路径点
            waypoints = self.calc_waypoints(self.init_pos, self.gates_pos, self.gates_norm)
            
            # 3. 添加绕行路径点
            waypoints = self._add_detour_waypoints(
                waypoints,
                self.gates_pos,
                self.gates_norm,
                self.gate_y_axes,
                self.gate_z_axes
            )
            
            t, waypoints_avoided = self.avoid_collision(
                waypoints, 
                obs['obstacles_pos'], 
                0.3
            )
            
            
            if len(t) < 2:
                print("警告: 重规划时 avoid_collision 返回点少于2个。")
                t_fallback = self.allocate_time(
                    waypoints, 
                    self.gates_pos, 
                    obs['obstacles_pos']
                )
                
                self.trajectory = self.generate_trajectory(waypoints, t_fallback)
            else:
                
                self.trajectory = self.generate_trajectory(waypoints_avoided, t)
            
            
            self.t_total = t[-1] if len(t) > 0 else 0

        
        tau = min(self._tick / self._freq, self.t_total)
        
        pva = ms.compute_trajectory_derivatives(self.trajectory, np.array([tau]), 1)
        target_pos = pva[0, 0, :]
        
        if tau >= self.t_total:  
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
        self._tick += 1
        return self._finished

    def episode_callback(self):
        pass

    def episode_reset(self):
        self._tick = 0
        self._finished = False
        
        if hasattr(self, 'last_gate_flags'):
            delattr(self, 'last_gate_flags')
        if hasattr(self, 'last_obst_flags'):
            delattr(self, 'last_obst_flags')