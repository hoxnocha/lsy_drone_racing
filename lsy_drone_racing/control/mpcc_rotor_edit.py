from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple
from enum import IntEnum

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat, dot, DM, norm_2, floor, if_else, substitute
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from drone_models.core import load_params
from drone_models.so_rpy_rotor import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ObstacleType(IntEnum):
    CYLINDER_2D = 0  # 无限高圆柱：只计算 XY 平面距离 (用于大障碍物、左右门柱)
    CAPSULE_3D  = 2  # 有限长线段/胶囊：计算点到线段距离 (用于上下门框)

class MathUtils:
    """包含四元数处理、向量计算等静态辅助函数"""

    @staticmethod
    def quat_to_axis(quat: NDArray[np.floating], axis_index: int = 1) -> NDArray[np.floating]:
        """将四元数转换为旋转矩阵并提取指定轴 (0:x, 1:y, 2:z)"""
        rot = R.from_quat(quat)
        mats = np.asarray(rot.as_matrix())
        if mats.ndim == 3:
            return mats[:, :, axis_index]
        if mats.ndim == 2:
            return mats[:, axis_index]
        return None

    @staticmethod
    def extract_gate_frames(gates_quaternions: NDArray[np.floating]) -> Tuple[NDArray, NDArray, NDArray]:
        """从门框四元数提取 Normal(x), Y-axis, Z-axis"""
        normals = MathUtils.quat_to_axis(gates_quaternions, axis_index=0)
        y_axes = MathUtils.quat_to_axis(gates_quaternions, axis_index=1)
        z_axes = MathUtils.quat_to_axis(gates_quaternions, axis_index=2)
        return normals, y_axes, z_axes

    @staticmethod
    def normalize_vec(vec: NDArray[np.floating]) -> NDArray[np.floating]:
        nrm = np.linalg.norm(vec)
        return vec if nrm < 1e-6 else vec / nrm


# ==============================================================================
# 2. 路径规划类 (RacingPathPlanner)
# ==============================================================================

class RacingPathPlanner:
    """
    负责路径生成、避障（含虚拟门框障碍物）、以及带约束的平滑处理。
    """

    def __init__(self, ctrl_freq: float):
        self.ctrl_freq = ctrl_freq

    def build_trajectory(
        self, 
        obs: dict, 
        current_pos: NDArray[np.floating], 
        planned_duration: float,
        gate_positions_memory: NDArray[np.floating]  # 使用记忆中的稳定门坐标
    ) -> Tuple[CubicSpline, float]:
        """
        构建包含过门、避障（真实障碍物+虚拟门框）和平滑逻辑的完整轨迹。
        """
        obstacle_positions = obs["obstacles_pos"]
        gate_quats = obs["gates_quat"]

        # 1. 提取门框坐标系
        gate_normals, gate_y, gate_z = MathUtils.extract_gate_frames(gate_quats)

        # 2. 构建基础 Waypoints
        base_waypoints = self._build_gate_waypoints(current_pos, gate_positions_memory, gate_normals)
        if base_waypoints.shape[0] > 1:
            base_waypoints[1:, 2] += 0.0 # 可选的高度修正

        # 3. 插入门框 Detours (几何倒圆角，保证穿门角度)
        with_gate_detours = self._insert_gate_detours(
            base_waypoints, gate_positions_memory, gate_normals, gate_y, gate_z
        )

        # 4. 生成虚拟门框障碍物 (门柱 + 横梁)
        virt_pos, virt_types, virt_vecs, virt_lens = self._get_virtual_gate_obstacles(
            gate_positions_memory, gate_quats, gate_width=0.7, gate_height=0.7
        )

        # 5. 合并真实障碍物和虚拟障碍物
        # 真实障碍物默认为 CYLINDER_2D，半径安全距离设为 0.35 (Inflation)
        if len(obstacle_positions) > 0:
            n_real = len(obstacle_positions)
            types_real = np.full(n_real, ObstacleType.CYLINDER_2D, dtype=int)
            vecs_real = np.zeros((n_real, 3))
            lens_real = np.zeros(n_real)
            margins_real = np.full(n_real, 0.35) 

            # 虚拟门框安全距离设小一点 (0.15)，允许紧贴穿过
            margins_virt = np.full(len(virt_pos), 0.15)

            all_pos = np.vstack([obstacle_positions, virt_pos])
            all_types = np.concatenate([types_real, virt_types])
            all_vecs = np.vstack([vecs_real, virt_vecs])
            all_lens = np.concatenate([lens_real, virt_lens])
            all_margins = np.concatenate([margins_real, margins_virt])
        else:
            all_pos = virt_pos
            all_types = virt_types
            all_vecs = virt_vecs
            all_lens = virt_lens
            all_margins = np.full(len(virt_pos), 0.15)

        # 6. 执行避障注入 (Push-out logic)
        t_axis, raw_points = self._inject_obstacle_detours(
            with_gate_detours, 
            all_pos, 
            all_margins,
            all_types,
            all_vecs,
            all_lens,
            planned_duration
        )

        # 7. 受约束的平滑处理 (Constrained Moving Average)
        # 这里只传入真实障碍物用于 Repulsion 检查，避免把门框当做排斥物导致无法穿门
        # (因为门框避障已经在第6步处理好了，且我们希望穿过门框中心)
        smoothed_points = self._apply_constrained_smoothing(
            raw_points, 
            gate_positions_memory, 
            obstacle_positions, # 仅传入真实障碍物进行二次安全检查
            safe_threshold=0.25, 
            iterations=2,
            alpha=0.3
        )

        # 8. 生成最终样条
        if len(t_axis) != len(smoothed_points):
            trajectory = CubicSpline(t_axis, raw_points)
        else:
            trajectory = CubicSpline(t_axis, smoothed_points)
        
        new_duration = float(trajectory.x[-1])
        return trajectory, new_duration

    def prepare_mpcc_trajectory(
        self, 
        base_trajectory: CubicSpline, 
        model_traj_length: float
    ) -> CubicSpline:
        """为 MPCC 准备轨迹：延长尾部并按弧长重参数化。"""
        extended = self.extend_spline_tail(base_trajectory, extend_length=model_traj_length)
        arc_traj = self.reparametrize_by_arclength(extended)
        return arc_traj

    
    def _get_virtual_gate_obstacles(
        self,
        gate_positions: NDArray[np.floating],
        gate_quats: NDArray[np.floating],
        gate_width: float = 0.7,   
        gate_height: float = 0.7
    ) -> tuple[NDArray[np.floating], NDArray[np.int_], NDArray[np.floating], NDArray[np.floating]]:
        """生成门柱(圆柱)和横梁(胶囊)的虚拟障碍物参数"""
        gate_y_axes = MathUtils.quat_to_axis(gate_quats, axis_index=1) # 横向
        gate_z_axes = MathUtils.quat_to_axis(gate_quats, axis_index=2) # 垂直

        obs_positions = []
        obs_types = []
        obs_vecs = []
        obs_lens = []

        half_w = gate_width / 2.0
        half_h = gate_height / 2.0

        for i in range(len(gate_positions)):
            c = gate_positions[i]
            y = gate_y_axes[i]
            z = gate_z_axes[i]

            # A. 左右门柱 (Side Posts) -> CYLINDER_2D
            for sign in [1.0, -1.0]:
                post_pos = c + sign * half_w * y
                obs_positions.append(post_pos)
                obs_types.append(ObstacleType.CYLINDER_2D)
                obs_vecs.append(np.zeros(3)) 
                obs_lens.append(0.0)         

            # B. 上下横梁 (Top & Bottom Bars) -> CAPSULE_3D
            for sign in [1.0, -1.0]:
                bar_pos = c + sign * half_h * z
                obs_positions.append(bar_pos)
                obs_types.append(ObstacleType.CAPSULE_3D)
                obs_vecs.append(y)      # 方向: 沿着门的横向延伸
                obs_lens.append(half_w) # 长度: 从中心向两侧延伸半宽

        return (
            np.array(obs_positions),
            np.array(obs_types, dtype=int),
            np.array(obs_vecs),
            np.array(obs_lens)
        )

    def _inject_obstacle_detours(
        self,
        base_waypoints: NDArray[np.floating],
        all_obstacles_pos: NDArray[np.floating],
        safe_dist_list: NDArray[np.floating],
        types_list: NDArray[np.int_],
        vecs_list: NDArray[np.floating],
        lens_list: NDArray[np.floating],
        planned_duration: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        基于几何投影的避障注入。
        如果路径穿过障碍物（圆柱或胶囊），则在进出点之间插入一个被“推离”的 Detour 点。
        """
        pre_spline = self.spline_through_points(planned_duration, base_waypoints)
        n_samples = max(1, int(self.ctrl_freq * planned_duration))

        t_axis = np.linspace(0.0, planned_duration, n_samples)
        wp_samples = pre_spline(t_axis)

        # 遍历每一个障碍物
        for obst_c, safe_dist, o_type, o_vec, o_len in zip(
            all_obstacles_pos, safe_dist_list, types_list, vecs_list, lens_list
        ):
            new_t = []
            new_pts = []
            
            inside_region = False
            idx_in = -1
            idx_out = -1

            for idx in range(wp_samples.shape[0]):
                pt = wp_samples[idx]
                
                # 计算距离
                if o_type == ObstacleType.CYLINDER_2D:
                    # 2D 欧氏距离
                    dist = np.linalg.norm(obst_c[:2] - pt[:2])
                elif o_type == ObstacleType.CAPSULE_3D:
                    # 点到线段距离
                    vec_cp = pt - obst_c
                    proj = np.dot(vec_cp, o_vec)
                    proj_clamped = np.clip(proj, -o_len, o_len)
                    closest_pt = obst_c + proj_clamped * o_vec
                    dist = np.linalg.norm(pt - closest_pt)
                else:
                    dist = np.linalg.norm(obst_c - pt)

                # 状态机：进入/保持/离开 危险区域
                if dist < safe_dist and not inside_region:
                    inside_region = True
                    idx_in = idx
                elif dist >= safe_dist and inside_region:
                    inside_region = False
                    idx_out = idx
                    
                    # --- 生成 Detour 点 ---
                    p_in = wp_samples[idx_in]
                    p_out = wp_samples[idx_out]
                    p_mid = 0.5 * (p_in + p_out) # 冲突段中点

                    # 计算推力向量 (Push Vector)
                    if o_type == ObstacleType.CYLINDER_2D:
                        push_vec = p_mid - obst_c
                        push_vec[2] = 0.0 # 柱子不推高度
                    elif o_type == ObstacleType.CAPSULE_3D:
                        vec_cp = p_mid - obst_c
                        proj = np.clip(np.dot(vec_cp, o_vec), -o_len, o_len)
                        closest_on_seg = obst_c + proj * o_vec
                        push_vec = p_mid - closest_on_seg 
                    else:
                        push_vec = p_mid - obst_c

                    norm_push = np.linalg.norm(push_vec)
                    if norm_push < 1e-6: push_dir = np.array([0,0,1.0])
                    else: push_dir = push_vec / norm_push

                    # 计算最终 Detour 坐标
                    if o_type == ObstacleType.CYLINDER_2D:
                        detour_xy = obst_c[:2] + push_dir[:2] * safe_dist
                        detour_z = 0.5 * (p_in[2] + p_out[2]) # 高度保持原有趋势
                        detour_pt = np.concatenate([detour_xy, [detour_z]])
                    else:
                        # 3D: 从最近点向外推 safe_dist
                        vec_cp = p_mid - obst_c
                        proj = np.clip(np.dot(vec_cp, o_vec), -o_len, o_len) if o_type == ObstacleType.CAPSULE_3D else 0
                        base_pt = obst_c + proj * o_vec
                        detour_pt = base_pt + push_dir * safe_dist

                    # 插入 Detour 点
                    mid_t = 0.5 * (t_axis[idx_in] + t_axis[idx_out])
                    new_t.append(mid_t)
                    new_pts.append(detour_pt)
                    
                    # 添加当前的 safe point
                    new_t.append(t_axis[idx])
                    new_pts.append(pt)
                    
                elif dist >= safe_dist:
                    new_t.append(t_axis[idx])
                    new_pts.append(pt)
            
            # 处理结尾还在里面的情况 (边界保护)
            if inside_region:
                new_t.append(t_axis[-1])
                new_pts.append(wp_samples[-1])

            t_axis = np.asarray(new_t)
            wp_samples = np.asarray(new_pts)

        if t_axis.size > 1:
            _, uniq = np.unique(t_axis, return_index=True)
            return t_axis[uniq], wp_samples[uniq]
        
        return np.array([]), np.array([])

    # --- 核心平滑逻辑 ---

    def _apply_constrained_smoothing(
        self, 
        points: NDArray, 
        fixed_gates: NDArray, 
        obstacles_pos: NDArray,
        safe_threshold: float,
        iterations: int = 5,
        alpha: float = 0.5
    ) -> NDArray:
        """
        带约束的滑动平均平滑算法。
        注意：这里的 obstacles_pos 仅包含真实环境障碍物，不含虚拟门框。
        """
        path = points.copy()
        n_points = len(path)
        if n_points < 3: return path

        # 1. 找到所有需要锁定的点（起点 + 门中心）
        fixed_indices = []
        fixed_coords = []
        
        fixed_indices.append(0)
        fixed_coords.append(path[0].copy())

        if len(fixed_gates) > 0:
            for gate_pos in fixed_gates:
                dists = np.linalg.norm(path - gate_pos, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < 0.05: 
                    fixed_indices.append(min_idx)
                    fixed_coords.append(gate_pos)

        # 2. 迭代平滑
        for _ in range(iterations):
            prev_pts = path[:-2]
            curr_pts = path[1:-1]
            next_pts = path[2:]
            
            # (A) Smoothing
            new_inner = (1 - alpha) * curr_pts + alpha * (prev_pts + next_pts) / 2.0
            
            # (B) Repulsion: 仅针对真实障碍物
            if len(obstacles_pos) > 0:
                diff = new_inner[:, None, :] - obstacles_pos[None, :, :] 
                dists = np.linalg.norm(diff, axis=2)
                min_dists = np.min(dists, axis=1)
                unsafe_mask = min_dists < safe_threshold
                
                if np.any(unsafe_mask):
                    nearest_obst_idx = np.argmin(dists, axis=1)
                    indices_unsafe = np.where(unsafe_mask)[0]
                    for i in indices_unsafe:
                        obs_idx = nearest_obst_idx[i]
                        obs_center = obstacles_pos[obs_idx]
                        p_curr = new_inner[i]
                        
                        push_vec = p_curr - obs_center
                        nrm = np.linalg.norm(push_vec)
                        if nrm < 1e-6: push_vec = np.array([1.0, 0.0, 0.0])
                        else: push_vec /= nrm
                        
                        new_inner[i] = obs_center + push_vec * safe_threshold

            path[1:-1] = new_inner
            
            # (C) Locking
            for idx, coord in zip(fixed_indices, fixed_coords):
                path[idx] = coord

        return path

    # --- 基础路径生成与辅助 ---

    def _build_gate_waypoints(self, start_pos, gates_positions, gates_normals, half_span=0.5, samples_per_gate=5):
        n_gates = gates_positions.shape[0]
        grid = []
        for idx in range(samples_per_gate):
            alpha = idx / (samples_per_gate - 1) if samples_per_gate > 1 else 0.0
            grid.append(gates_positions - half_span * gates_normals + 2.0 * half_span * alpha * gates_normals)
        stacked = np.stack(grid, axis=1).reshape(n_gates, samples_per_gate, 3).reshape(-1, 3)
        return np.vstack([start_pos[None, :], stacked])

    def _insert_gate_detours(self, waypoints, gate_positions, gate_normals, gate_y, gate_z):
        # 简单透传，因为几何避障和虚拟门框已经能处理大部分碰撞
        return np.asarray(waypoints)

    def spline_through_points(self, duration: float, waypoints: NDArray[np.floating]) -> CubicSpline:
        diffs = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        if cum_len[-1] < 1e-6:
             return CubicSpline([0.0, duration], np.vstack([waypoints[0], waypoints[-1]]))
        t_axis = cum_len / cum_len[-1] * duration
        return CubicSpline(t_axis, waypoints)

    def reparametrize_by_arclength(self, trajectory: CubicSpline, arc_step: float = 0.05) -> CubicSpline:
        total_param_range = trajectory.x[-1] - trajectory.x[0]
        for _ in range(3):
            n_segments = max(2, int(total_param_range / arc_step))
            t_samples = np.linspace(0.0, total_param_range, n_segments)
            pts = trajectory(t_samples)
            deltas = np.diff(pts, axis=0)
            seg_lengths = np.linalg.norm(deltas, axis=1)
            cum_arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total_param_range = float(cum_arc[-1])
            trajectory = CubicSpline(cum_arc, pts)
        return CubicSpline(cum_arc, pts)

    def extend_spline_tail(self, trajectory: CubicSpline, extend_length: float = 1.0) -> CubicSpline:
        base_knots = trajectory.x
        base_dt = min(base_knots[1] - base_knots[0], 0.2)
        p_end = trajectory(base_knots[-1])
        v_end = trajectory.derivative(1)(base_knots[-1])
        v_dir = MathUtils.normalize_vec(v_end)

        extra_knots = np.arange(base_knots[-1] + base_dt, base_knots[-1] + extend_length, base_dt)
        p_extend = np.array([p_end + v_dir * (s - base_knots[-1]) for s in extra_knots])

        theta_new = np.concatenate([base_knots, extra_knots])
        p_new = np.vstack([trajectory(base_knots), p_extend])
        return CubicSpline(theta_new, p_new, axis=0)


# ==============================================================================
# 3. MPCC 控制器类 (Controller)
# ==============================================================================

class MPCC(Controller):
    """
    Model Predictive Contouring Control for drone racing.
    集成特性:
    - 真实动力学 (so_rpy_rotor)
    - 虚拟门框避障 + 几何投影推离 (Geometric Push-out)
    - 路径平滑 (Moving Average Smoothing)
    - 门坐标记忆 (Gate Memory Update)
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._cfg = config
        self._step_count = 0
        self._ctrl_freq = config.env.freq
        
        self._dyn_params = load_params("so_rpy_rotor", config.sim.drone_model)
        mass_val = float(self._dyn_params["mass"])
        gravity_mag = -float(self._dyn_params["gravity_vec"][-1])
        self.hover_thrust = mass_val * gravity_mag

        self.tau_rpy_act = 0.05
        self.tau_yaw_act = 0.08
        self.tau_f_act = 0.10
        self.rate_limit_df = 10.0
        self.rate_limit_drpy = 10.0
        self.rate_limit_v_theta = 4.0

        self._initial_pos = obs["pos"]
        self._cached_obstacles = obs["obstacles_pos"]
        self._planned_duration = 30.0

        # --- Gate Memory 初始化 ---
        self.memory_gate_pos = np.array(obs["gates_pos"], dtype=float)
        self.memory_gate_visited = np.array(obs.get("gates_visited", []), dtype=bool)
        if len(self.memory_gate_visited) == 0:
             self.memory_gate_visited = np.zeros(len(self.memory_gate_pos), dtype=bool)

        self.planner = RacingPathPlanner(ctrl_freq=self._ctrl_freq)
        
        self._replan(obs)

        self.N = 35
        self.T_HORIZON = 0.7
        self.dt = self.T_HORIZON / self.N
        self.model_arc_length = 0.05
        self.model_traj_length = 12.0

        self.arc_trajectory = self.planner.prepare_mpcc_trajectory(
            self.trajectory, self.model_traj_length
        )

        self.acados_ocp_solver, self.ocp = self._build_ocp_and_solver(
            self.T_HORIZON, self.N
        )

        self.pos_bound = [np.array([-2.6, 2.6]), np.array([-2.0, 1.8]), np.array([-0.1, 2.0])]
        self.velocity_bound = [-1.0, 4.0]

        self.last_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_f_act = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self.last_rpy_act = np.zeros(3)
        self.finished = False

    def _update_gate_memory_logic(self, obs: dict) -> bool:
        curr_visited = np.array(obs.get("gates_visited", []), dtype=bool)
        curr_pos = obs["gates_pos"]
        changed = False
        
        for i in range(len(self.memory_gate_pos)):
            if i < len(curr_visited) and curr_visited[i]:
                if np.linalg.norm(self.memory_gate_pos[i] - curr_pos[i]) > 1e-3:
                    self.memory_gate_pos[i] = curr_pos[i]
                    changed = True
        
        self.memory_gate_visited = curr_visited
        return changed

    def _replan(self, obs: dict):
        """调用 Planner 重新生成名义轨迹，使用 Memory Gate Pos"""
        self._cached_obstacles = obs["obstacles_pos"]
        traj, duration = self.planner.build_trajectory(
            obs, 
            current_pos=self._initial_pos if self._step_count == 0 else obs["pos"], 
            planned_duration=self._planned_duration,
            gate_positions_memory=self.memory_gate_pos 
        )
        self.trajectory = traj
        self._planned_duration = duration

    # ------------------------------------------------------------------
    # OCP 构建与 Cost 定义
    # ------------------------------------------------------------------

    def _export_dynamics_model(self) -> AcadosModel:
        model_name = "lsy_example_mpc_real"
        params = self._dyn_params

        X_dot_phys, X_phys, U_phys, _ = symbolic_dynamics_euler(
            mass=params["mass"], gravity_vec=params["gravity_vec"],
            J=params["J"], J_inv=params["J_inv"],
            acc_coef=params["acc_coef"], cmd_f_coef=params["cmd_f_coef"],
            rpy_coef=params["rpy_coef"], rpy_rates_coef=params["rpy_rates_coef"],
            cmd_rpy_coef=params["cmd_rpy_coef"], thrust_time_coef=params["thrust_time_coef"],
        )
        
        self.nx_phys = X_phys.shape[0]
        self.px, self.py, self.pz = X_phys[0], X_phys[1], X_phys[2]
        self.roll, self.pitch, self.yaw = X_phys[3], X_phys[4], X_phys[5]

        self.r_cmd_state = MX.sym("r_cmd_state")
        self.p_cmd_state = MX.sym("p_cmd_state")
        self.y_cmd_state = MX.sym("y_cmd_state")
        self.f_cmd_state = MX.sym("f_cmd_state")
        
        self.r_act = MX.sym("r_act")
        self.p_act = MX.sym("p_act")
        self.y_act = MX.sym("y_act")
        self.f_act = MX.sym("f_act")
        
        self.theta = MX.sym("theta")

        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")
        self.v_theta_cmd = MX.sym("v_theta_cmd")

        states = vertcat(X_phys, self.r_cmd_state, self.p_cmd_state, self.y_cmd_state, self.f_cmd_state,
                         self.r_act, self.p_act, self.y_act, self.f_act, self.theta)
        inputs = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd, self.v_theta_cmd)

        base_idx = self.nx_phys
        self.idx_r_cmd_state, self.idx_p_cmd_state = base_idx + 0, base_idx + 1
        self.idx_y_cmd_state, self.idx_f_cmd_state = base_idx + 2, base_idx + 3
        self.idx_r_act, self.idx_p_act = base_idx + 4, base_idx + 5
        self.idx_y_act, self.idx_f_act = base_idx + 6, base_idx + 7
        self.idx_theta = base_idx + 8

        U_phys_full = vertcat(self.r_act, self.p_act, self.y_act, self.f_act)
        f_dyn_phys = substitute(X_dot_phys, U_phys, U_phys_full)

        f_dyn = vertcat(
            f_dyn_phys,
            self.dr_cmd, self.dp_cmd, self.dy_cmd, self.df_cmd,
            (self.r_cmd_state - self.r_act) / float(self.tau_rpy_act),
            (self.p_cmd_state - self.p_act) / float(self.tau_rpy_act),
            (self.y_cmd_state - self.y_act) / float(self.tau_yaw_act),
            (self.f_cmd_state - self.f_act) / float(self.tau_f_act),
            self.v_theta_cmd
        )

        n_samples = int(self.model_traj_length / self.model_arc_length)
        self.pd_list = MX.sym("pd_list", 3 * n_samples)
        self.tp_list = MX.sym("tp_list", 3 * n_samples)
        self.qc_gate = MX.sym("qc_gate", 1 * n_samples)
        self.qc_obst = MX.sym("qc_obst", 1 * n_samples)
        
        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f_dyn
        model.x = states
        model.u = inputs
        model.p = vertcat(self.pd_list, self.tp_list, self.qc_gate, self.qc_obst)
        return model

    def _piecewise_linear_interp(self, theta, theta_vec, flattened_points, dim: int = 3):
        M = len(theta_vec)
        idx_float = (theta - theta_vec[0]) / (theta_vec[-1] - theta_vec[0]) * (M - 1)
        idx_low = floor(idx_float)
        alpha = idx_float - idx_low
        idx_low = if_else(idx_low < 0, 0, idx_low)
        idx_high = if_else(idx_low + 1 >= M, M - 1, idx_low + 1)
        
        p_low = vertcat(*[flattened_points[dim * idx_low + i] for i in range(dim)])
        p_high = vertcat(*[flattened_points[dim * idx_high + i] for i in range(dim)])
        return (1.0 - alpha) * p_low + alpha * p_high

    def _stage_cost_expression(self):
        pos_vec = vertcat(self.px, self.py, self.pz)
        att_vec = vertcat(self.roll, self.pitch, self.yaw)
        ctrl_vec = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)

        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_length)
        
        pd = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)
        w_gate = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_gate, dim=1)
        w_obst = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_obst, dim=1)

        tp_unit = tp / (norm_2(tp) + 1e-6)
        e_err = pos_vec - pd
        e_lag = dot(tp_unit, e_err) * tp_unit
        e_cont = e_err - e_lag

        q_l_curr = self.q_l + self.q_l_gate_peak * w_gate + self.q_l_obst_peak * w_obst
        q_c_curr = self.q_c + self.q_c_gate_peak * w_gate + self.q_c_obst_peak * w_obst
        
        track_cost = q_l_curr * dot(e_lag, e_lag) + q_c_curr * dot(e_cont, e_cont) + att_vec.T @ self.Q_w @ att_vec
        smooth_cost = ctrl_vec.T @ self.R_df @ ctrl_vec
        progress_cost = -self.miu * self.v_theta_cmd + \
                        self.w_v_gate * w_gate * (self.v_theta_cmd**2) + \
                        self.w_v_obst * w_obst * (self.v_theta_cmd**2)

        return track_cost + smooth_cost + progress_cost

    def _build_ocp_and_solver(self, Tf: float, N_horizon: int) -> tuple[AcadosOcpSolver, AcadosOcp]:
        ocp = AcadosOcp()
        model = self._export_dynamics_model()
        ocp.model = model
        self.nx, self.nu = model.x.rows(), model.u.rows()
        
        ocp.solver_options.N_horizon = N_horizon
        ocp.cost.cost_type = "EXTERNAL"
        
        # 权重配置
        self.q_l = 522.327621281147
        self.q_c = 279.45878291502595
        self.Q_w = 1 * DM(np.eye(3))
        self.q_l_gate_peak = 520.2687042765319
        self.q_c_gate_peak = 764.3037075176835
        self.q_l_obst_peak = 207.83845749683678
        self.q_c_obst_peak = 110.51885732449591
        self.R_df = DM(np.diag([0.1, 0.5, 0.5, 0.5]))
        self.miu = 14.3377785384655
        self.w_v_gate = 2.7327203765511516
        self.w_v_obst = 2.460291111562401

        ocp.model.cost_expr_ext_cost = self._stage_cost_expression()

        # 约束
        thrust_min = float(self._dyn_params["thrust_min"]) * 4.0
        thrust_max = float(self._dyn_params["thrust_max"]) * 4.0
        
        ocp.constraints.lbx = np.array([thrust_min, thrust_min, -1.57, -1.57, -1.57])
        ocp.constraints.ubx = np.array([thrust_max, thrust_max, 1.57, 1.57, 1.57])
        ocp.constraints.idxbx = np.array([self.idx_f_act, self.idx_f_cmd, 
                                          self.idx_r_cmd_state, self.idx_p_cmd_state, self.idx_y_cmd_state])

        limit_arr = np.array([self.rate_limit_df] + [self.rate_limit_drpy]*3 + [self.rate_limit_v_theta])
        ocp.constraints.lbu = np.concatenate([-limit_arr[:-1], [0.0]])
        ocp.constraints.ubu = limit_arr
        ocp.constraints.idxbu = np.arange(5)

        ocp.constraints.x0 = np.zeros(self.nx)
        ocp.parameter_values = np.zeros(model.p.rows())

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tf = Tf

        solver = AcadosOcpSolver(ocp, json_file="mpcc_prescripted_real_dyn.json", verbose=False)
        return solver, ocp

    def _encode_traj_params(self, trajectory: CubicSpline) -> np.ndarray:
        theta_samples = np.arange(0.0, self.model_traj_length, self.model_arc_length)
        pd_vals = trajectory(theta_samples)
        tp_vals = trajectory.derivative(1)(theta_samples)

        qc_gate = np.zeros_like(theta_samples, dtype=float)
        qc_obst = np.zeros_like(theta_samples, dtype=float)

        if len(self.memory_gate_pos) > 0:
            for gc in self.memory_gate_pos:
                d = np.linalg.norm(pd_vals - gc, axis=-1)
                qc_gate = np.maximum(qc_gate, np.exp(-2.0 * d**2))
        
        if len(self._cached_obstacles) > 0:
            for oc in self._cached_obstacles:
                d = np.linalg.norm(pd_vals[:, :2] - oc[:2], axis=-1)
                qc_obst = np.maximum(qc_obst, 0.7 * np.exp(-1.0 * d**2))

        return np.concatenate([pd_vals.reshape(-1), tp_vals.reshape(-1), qc_gate, qc_obst])

    # ------------------------------------------------------------------
    # Control Loop
    # ------------------------------------------------------------------

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray[np.floating]:
        self._current_obs_pos = obs["pos"]

        # 1. 更新门坐标记忆
        gates_moved = self._update_gate_memory_logic(obs)

        # 2. 检测环境变化
        if self._detect_env_change(obs) or gates_moved:
            print(f"T={self._step_count/self._ctrl_freq:.2f}: Replanning (Env/Gate changed)...")
            self._replan(obs)
            self.arc_trajectory = self.planner.prepare_mpcc_trajectory(
                self.trajectory, self.model_traj_length
            )
            param_vec = self._encode_traj_params(self.arc_trajectory)
            for k in range(self.N + 1):
                self.acados_ocp_solver.set(k, "p", param_vec)

        # 3. 状态组装
        quat = obs["quat"]
        rpy = R.from_quat(quat).as_euler("xyz", degrees=False)
        drpy = ang_vel2rpy_rates(quat, obs["ang_vel"]) if "ang_vel" in obs else np.zeros(3)
        
        x_now = np.zeros(self.nx)
        x_now[0:3] = obs["pos"]
        x_now[3:6] = rpy
        x_now[6:9] = obs["vel"]
        x_now[9:12] = drpy
        x_now[self.idx_r_cmd_state:self.idx_f_cmd_state+1] = list(self.last_rpy_cmd) + [self.last_f_cmd]
        x_now[self.idx_r_act:self.idx_f_act+1] = list(self.last_rpy_act) + [self.last_f_act]
        x_now[self.idx_theta] = self.last_theta

        # 4. Solver Warm Start & Init
        if not hasattr(self, "_x_warm"):
            self._x_warm = [x_now.copy() for _ in range(self.N + 1)]
            self._u_warm = [np.zeros(self.nu) for _ in range(self.N)]
        else:
            self._x_warm = self._x_warm[1:] + [self._x_warm[-1]]
            self._u_warm = self._u_warm[1:] + [self._u_warm[-1]]

        for i in range(self.N):
            self.acados_ocp_solver.set(i, "x", self._x_warm[i])
            self.acados_ocp_solver.set(i, "u", self._u_warm[i])
        self.acados_ocp_solver.set(self.N, "x", self._x_warm[self.N])
        
        self.acados_ocp_solver.set(0, "lbx", x_now)
        self.acados_ocp_solver.set(0, "ubx", x_now)

        # 5. 终止检查
        if self.last_theta >= float(self.arc_trajectory.x[-1]) or \
           self._check_safety(obs["pos"], obs["vel"]):
            self.finished = True

        # 6. 求解
        status = self.acados_ocp_solver.solve()
        if status != 0:
            print(f"[MPCC] Solver failed with status {status}")

        # 7. 提取结果
        self._x_warm = [self.acados_ocp_solver.get(i, "x") for i in range(self.N + 1)]
        self._u_warm = [self.acados_ocp_solver.get(i, "u") for i in range(self.N)]
        
        x_next = self._x_warm[1]
        self.last_rpy_cmd = x_next[self.idx_r_cmd_state:self.idx_y_cmd_state+1]
        self.last_f_cmd = float(x_next[self.idx_f_cmd_state])
        self.last_rpy_act = x_next[self.idx_r_act:self.idx_y_act+1]
        self.last_f_act = float(x_next[self.idx_f_act])
        self.last_theta = float(x_next[self.idx_theta])

        self._step_count += 1
        return np.concatenate([self.last_rpy_cmd, [self.last_f_cmd]])

    # --- Helpers ---

    def _detect_env_change(self, obs: dict) -> bool:
        if not hasattr(self, "_last_flags"):
            self._last_flags = (np.array(obs.get("gates_visited", [])), np.array(obs.get("obstacles_visited", [])))
            return False
        curr = (np.array(obs.get("gates_visited", [])), np.array(obs.get("obstacles_visited", [])))
        changed = (curr[0].shape != self._last_flags[0].shape) or \
                  np.any((~self._last_flags[0]) & curr[0]) or \
                  np.any((~self._last_flags[1]) & curr[1])
        self._last_flags = curr
        return bool(changed)

    def _check_safety(self, pos, vel):
        out_pos = any(pos[i] < self.pos_bound[i][0] or pos[i] > self.pos_bound[i][1] for i in range(3))
        speed = np.linalg.norm(vel)
        out_vel = not (self.velocity_bound[0] < speed < self.velocity_bound[1])
        if out_pos or out_vel:
            print(f"[MPCC] Safety Triggered: Pos={out_pos}, Vel={out_vel}")
            return True
        return False

    def step_callback(self, *args, **kwargs) -> bool:
        return self.finished

    def episode_callback(self):
        print("[MPCC] Episode reset.")
        self._step_count = 0
        self.finished = False
        if hasattr(self, "_x_warm"): del self._x_warm
        if hasattr(self, "_u_warm"): del self._u_warm
        self.last_theta = 0.0
        self.last_f_cmd = self.hover_thrust
        self.last_f_act = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self.last_rpy_act = np.zeros(3)

    def get_debug_lines(self):
        lines = []
        if hasattr(self, "arc_trajectory"):
            path = self.arc_trajectory(self.arc_trajectory.x)
            lines.append((path, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0))
        if hasattr(self, "_x_warm"):
            pred = np.array([x[:3] for x in self._x_warm])
            lines.append((pred, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))
        return lines