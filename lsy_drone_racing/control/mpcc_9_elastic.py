from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat, dot, DM, norm_2, floor, if_else, substitute
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates

from lsy_drone_racing.control import Controller
from enum import IntEnum

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ==============================================================================
#  MODULE 1: Utilities & Enums
# ==============================================================================

class ObstacleType(IntEnum):
    CYLINDER_2D = 0  
    CAPSULE_3D  = 2  
    FLOOR_PLANE = 9  

class FrameUtils:
    @staticmethod
    def quat_to_axis(quat: NDArray[np.floating], axis_index: int = 1) -> NDArray[np.floating]:
        rot = R.from_quat(quat)
        mats = np.asarray(rot.as_matrix())
        if mats.ndim == 3: return mats[:, :, axis_index]
        return mats[:, axis_index]

# ==============================================================================
#  MODULE 2: Path Tools (Spline & Geometry)
# ==============================================================================

class PathTools:
    def spline_through_points(self, duration: float, waypoints: NDArray[np.floating]) -> CubicSpline:
        diffs = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        t_axis = cum_len / (cum_len[-1] + 1e-6) * duration
        return CubicSpline(t_axis, waypoints)

    def reparametrize_by_arclength(
        self, trajectory: CubicSpline, arc_step: float = 0.05, epsilon: float = 1e-5
    ) -> CubicSpline:

        total_param_range = trajectory.x[-1] - trajectory.x[0]

        for _ in range(99):
            n_segments = max(2, int(total_param_range / arc_step))
            t_samples = np.linspace(0.0, total_param_range, n_segments)
            pts = trajectory(t_samples)
            deltas = np.diff(pts, axis=0)
            seg_lengths = np.linalg.norm(deltas, axis=1)
            cum_arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total_param_range = float(cum_arc[-1])
            trajectory = CubicSpline(cum_arc, pts)
            if np.std(seg_lengths) <= epsilon:
                return CubicSpline(cum_arc, pts)

        return CubicSpline(cum_arc, pts)

    def extend_spline_tail(self, trajectory: CubicSpline, extend_length: float = 1.0) -> CubicSpline:
        base_knots = trajectory.x
        base_dt = min(base_knots[1] - base_knots[0], 0.2)
        p_end = trajectory(base_knots[-1])
        v_end = trajectory.derivative(1)(base_knots[-1])
        v_dir = v_end / (np.linalg.norm(v_end) + 1e-6)

        extra_knots = np.arange(
            base_knots[-1] + base_dt,
            base_knots[-1] + extend_length,
            base_dt,
        )
        p_extend = np.array(
            [p_end + v_dir * (s - base_knots[-1]) for s in extra_knots]
        )

        theta_new = np.concatenate([base_knots, extra_knots])
        p_new = np.vstack([trajectory(base_knots), p_extend])
        return CubicSpline(theta_new, p_new, axis=0)

# ==============================================================================
#  MODULE 3: Path Planner (Freq-based Sampling + Vectorized Optimization)
# ==============================================================================

class PathPlanner:
    def __init__(self, initial_pos, initial_gates_pos, ctrl_freq, planned_duration=30.0):
        self.initial_pos = initial_pos
        self.trajectory = None
        self.arc_trajectory = None
        
        # 保存频率和规划时长
        self.ctrl_freq = ctrl_freq
        self.planned_duration = planned_duration
        
        self.cached_gates_pos = initial_gates_pos
        self.cached_obstacles_pos = np.array([])
        
        self._last_gate_flags = None
        self._last_obst_flags = None
        
        self.tools = PathTools()
        
        # --- 优化参数 ---
        self.ITERATIONS = 50           
        self.LEARNING_RATE = 0.05      # 点多了，步长稍微减小以防震荡
        
        # --- 物理参数 ---
        self.SAFE_Z = 0.55             
        self.OBS_RADIUS = 0.60         # 基础半径
        
        # 强力参数
        self.K_ELASTIC = 0.2           # 张力
        self.K_REP_OBS = 2.5           # 障碍物斥力
        self.K_REP_FLOOR = 2.0         # 地面斥力
        self.GATE_ALIGN_DIST = 0.5     

    def check_env_update(self, obs) -> bool:
        curr_gates = np.array(obs.get("gates_visited", []), dtype=bool)
        curr_obst = np.array(obs.get("obstacles_visited", []), dtype=bool)
        if self._last_gate_flags is None:
            self._last_gate_flags = curr_gates; self._last_obst_flags = curr_obst; return True 
        changed = np.any((~self._last_gate_flags) & curr_gates) or np.any((~self._last_obst_flags) & curr_obst)
        self._last_gate_flags = curr_gates; self._last_obst_flags = curr_obst
        return bool(changed)

    def replan(self, obs, model_traj_length):
        gate_pos = obs["gates_pos"]
        gate_quat = obs["gates_quat"]
        obst_pos = obs["obstacles_pos"]
        
        self.cached_gates_pos = gate_pos
        self.cached_obstacles_pos = obst_pos

        # 1. 生成基于频率的高密度骨架
        sparse_wps, fixed_mask = self._generate_freq_based_skeleton(self.initial_pos, gate_pos, gate_quat)
        
        # 2. 收集障碍物
        obstacles = self._collect_obstacles(gate_pos, gate_quat, obst_pos)
        
        # 3. 向量化弹性带优化 (处理大量点)
        safe_wps = self._optimize_vectorized(sparse_wps, fixed_mask, obstacles)

        # 4. 后处理平滑
        smoothed_wps = self._apply_discrete_smoothing(safe_wps, fixed_mask, iterations=5)

        # 5. 关键点提取 (保留避障细节)
        #    注意：因为点非常密，这里提取时可以用更宽松的阈值
        final_wps = self._extract_keyframes(smoothed_wps, fixed_mask, sparse_wps)
        
        # 6. 生成样条
        self.trajectory = self.tools.spline_through_points(self.planned_duration, final_wps)
            
        # 7. 延长 & 重参数化
        extended = self.tools.extend_spline_tail(self.trajectory, model_traj_length)
        self.arc_trajectory = self.tools.reparametrize_by_arclength(extended)
        
        return self.arc_trajectory

    def _generate_freq_based_skeleton(self, start_pos, gate_pos, gate_quat):
        """根据控制频率生成高密度骨架"""
        all_wps = []
        is_fixed = []
        
        gate_normals = FrameUtils.quat_to_axis(gate_quat, 0)
        safe_gate_pos = gate_pos.copy()
        safe_gate_pos[:, 2] = np.maximum(safe_gate_pos[:, 2], 0.5)

        # 1. 计算总路径的大致几何长度
        #    Start -> Pre1 -> Center1 -> Post1 -> Pre2 ...
        #    为了简化，我们计算 Start -> Center1 -> Center2 ... 的长度作为估算
        waypoints_approx = np.vstack([start_pos, safe_gate_pos])
        diffs = np.diff(waypoints_approx, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        total_dist = np.sum(dists)
        
        # 2. 计算平均速度和线性密度
        #    假设在 planned_duration 内跑完 total_dist
        if total_dist < 1e-3: total_dist = 1.0
        avg_speed = total_dist / self.planned_duration
        if avg_speed < 0.1: avg_speed = 0.1
        
        #    采样密度 (points/meter) = freq / speed
        #    例如: 100Hz / 5m/s = 20 points/m (即每5cm一个点，这与mpcc_9一致)
        points_per_meter = self.ctrl_freq / avg_speed
        
        # 3. 生成点
        last_anchor = start_pos
        all_wps.append(start_pos); is_fixed.append(True)

        for i in range(len(safe_gate_pos)):
            center = safe_gate_pos[i]
            normal = gate_normals[i]
            normal = normal / (np.linalg.norm(normal) + 1e-6)
            
            p_pre = center - normal * self.GATE_ALIGN_DIST
            p_post = center + normal * self.GATE_ALIGN_DIST
            
            # --- 段落: Last -> Pre ---
            dist_seg = np.linalg.norm(p_pre - last_anchor)
            # 计算该段需要的点数
            n_points = int(dist_seg * points_per_meter)
            n_points = max(2, n_points) # 至少2个点
            
            segment_pts = np.linspace(last_anchor, p_pre, n_points + 1)[1:] # [1:] 避免重复上一个点
            
            # 最后一个点是 p_pre (Fixed)，前面的都是 Floating
            for k, pt in enumerate(segment_pts):
                all_wps.append(pt)
                if k == len(segment_pts) - 1:
                    is_fixed.append(True) # p_pre
                else:
                    is_fixed.append(False) # floating
            
            # --- 段落: Pre -> Center -> Post (Fixed Corridor) ---
            # 这些点距离很近，直接加入即可
            all_wps.append(center); is_fixed.append(True)
            all_wps.append(p_post); is_fixed.append(True)
            
            last_anchor = p_post
            
        return np.array(all_wps), np.array(is_fixed, dtype=bool)

    def _optimize_vectorized(self, wps, fixed_mask, obstacles):
        """向量化的弹性带优化 (比循环快 50-100倍)"""
        wps_optim = wps.copy()
        N = len(wps_optim)
        start_z = wps[0, 2]
        
        # 预计算一些索引，避免循环
        idx_curr = np.arange(1, N-1)
        idx_prev = idx_curr - 1
        idx_next = idx_curr + 1
        
        # 找出非固定点的索引
        non_fixed_indices = np.where(~fixed_mask)[0]
        # 去掉首尾 (首尾通常是fixed，但为了安全)
        non_fixed_indices = non_fixed_indices[(non_fixed_indices > 0) & (non_fixed_indices < N-1)]
        
        inflation = 0.25
        effective_radius = self.OBS_RADIUS + inflation

        for _ in range(self.ITERATIONS):
            # 初始化力矩阵
            forces = np.zeros_like(wps_optim)
            
            # --- 1. 弹性张力 (Vectorized) ---
            # F_elastic = K * (Prev + Next - 2*Curr)
            # 也可以理解为指向中点
            p_curr = wps_optim[idx_curr]
            p_prev = wps_optim[idx_prev]
            p_next = wps_optim[idx_next]
            
            elastic_force = self.K_ELASTIC * (0.5 * (p_prev + p_next) - p_curr)
            forces[idx_curr] += elastic_force
            
            # --- 2. 地面斥力 (Vectorized) ---
            # 计算每个点对应的 Safe Z (起飞坡道)
            # 简单起见，按索引比例估算距离 (假设点分布均匀)
            # 或者直接全段使用 SAFE_Z，只在前 5% 做坡道
            z_vals = wps_optim[:, 2]
            
            # 创建坡道 Z 阈值数组
            safe_z_targets = np.full(N, self.SAFE_Z)
            ramp_len = int(0.15 * N) # 前15%的点做坡道
            if ramp_len > 0:
                ramp = np.linspace(start_z, self.SAFE_Z, ramp_len)
                safe_z_targets[:ramp_len] = ramp
            
            # 地面逻辑：只对低于 safe_z 的点施加力
            floor_mask = z_vals < safe_z_targets
            forces[floor_mask, 2] += self.K_REP_FLOOR * (safe_z_targets[floor_mask] - z_vals[floor_mask])
            
            # --- 3. 障碍物斥力 (Broadcasting) ---
            # 这是一个 (N_points, 3) 的数组
            # 障碍物数量通常很少 (<20)，我们可以循环障碍物，对所有点进行向量化计算
            for obs in obstacles:
                o_pos = obs['pos']
                
                if obs['type'] == ObstacleType.CYLINDER_2D:
                    # XY 平面距离
                    diff = wps_optim[:, :2] - o_pos[:2]
                    dist = np.linalg.norm(diff, axis=1)
                    
                    # 找到冲突点
                    mask = dist < effective_radius
                    if not np.any(mask): continue
                    
                    # 计算力
                    # dir = diff / dist (在XY平面)
                    # force = K * (R - dist) * dir
                    d_mask = dist[mask]
                    # 避免除零
                    d_mask = np.maximum(d_mask, 1e-6)
                    
                    push_dir = np.zeros((np.sum(mask), 3))
                    push_dir[:, :2] = diff[mask] / d_mask[:, None]
                    
                    # 模拟距离 (Inflation)
                    sim_dist = np.maximum(0.0, d_mask - inflation)
                    mag = self.K_REP_OBS * (self.OBS_RADIUS - sim_dist)
                    
                    forces[mask] += mag[:, None] * push_dir
                    
                elif obs['type'] == ObstacleType.CAPSULE_3D:
                    # 点到线段距离 (比较复杂，需要逐点算? 不，也可以向量化)
                    # Vector P (Nx3), A (1x3 start), B (1x3 end)
                    # PA = P - A
                    # BA = B - A
                    # t = dot(PA, BA) / dot(BA, BA)
                    # clip t in [0, 1]
                    # Closest = A + t * BA
                    # Dist = norm(P - Closest)
                    
                    half_len = obs['half_len']
                    vec = obs['vec']
                    start_p = o_pos - vec * half_len
                    end_p = o_pos + vec * half_len
                    segment = end_p - start_p # BA
                    seg_len_sq = np.dot(segment, segment)
                    
                    point_vec = wps_optim - start_p # PA
                    
                    t = np.dot(point_vec, segment) / (seg_len_sq + 1e-9)
                    t = np.clip(t, 0.0, 1.0)
                    
                    closest = start_p + t[:, None] * segment
                    diff = wps_optim - closest
                    dist = np.linalg.norm(diff, axis=1)
                    
                    mask = dist < effective_radius
                    if not np.any(mask): continue
                    
                    d_mask = dist[mask]
                    d_mask = np.maximum(d_mask, 1e-6)
                    
                    push_dir = diff[mask] / d_mask[:, None]
                    sim_dist = np.maximum(0.0, d_mask - inflation)
                    mag = self.K_REP_OBS * (self.OBS_RADIUS - sim_dist)
                    
                    forces[mask] += mag[:, None] * push_dir

            # 应用力 (只针对非固定点)
            # 注意：forces 是全量的，我们需要 mask 掉固定点
            forces[fixed_mask] = 0
            
            wps_optim += self.LEARNING_RATE * forces
            
            # 硬约束 (不穿地)
            non_fixed_mask = ~fixed_mask
            wps_optim[non_fixed_mask, 2] = np.maximum(wps_optim[non_fixed_mask, 2], 0.05)
            
        return wps_optim

    def _extract_keyframes(self, safe_wps, fixed_mask, original_wps):
        final_wps = []
        final_wps.append(safe_wps[0]) 
        last_kept_pos = safe_wps[0]
        
        displacements = np.linalg.norm(safe_wps - original_wps, axis=1)
        MOVED_THRESHOLD = 0.02 

        # 动态样条间距：直线段 1.0m 插一个点
        SPLINE_DIST = 1.0

        for i in range(1, len(safe_wps)):
            curr_pos = safe_wps[i]
            is_fixed = fixed_mask[i]
            is_last = (i == len(safe_wps) - 1)
            is_moved = displacements[i] > MOVED_THRESHOLD
            
            dist = np.linalg.norm(curr_pos - last_kept_pos)
            keep_this = False
            
            # 始终保留前几个点 (防止起飞抖动)
            if i < 10: keep_this = True
            elif is_fixed: keep_this = True
            elif is_last: keep_this = True
            
            elif is_moved:
                # 避障区：高密度 (每0.15m)
                if dist >= 0.15: keep_this = True
            
            else:
                # 直线区：低密度
                if dist >= SPLINE_DIST: keep_this = True
            
            if keep_this:
                final_wps.append(curr_pos)
                last_kept_pos = curr_pos
                
        return np.array(final_wps)

    def _apply_discrete_smoothing(self, wps, fixed_mask, iterations=1):
        smoothed = wps.copy()
        n = len(smoothed)
        # 向量化平滑： P_i = 0.25*P_i-1 + 0.5*P_i + 0.25*P_i+1
        # 可以用卷积或切片实现，比循环快
        for _ in range(iterations):
            # 不包括首尾
            p_prev = smoothed[:-2]
            p_curr = smoothed[1:-1]
            p_next = smoothed[2:]
            
            p_new = 0.25 * p_prev + 0.5 * p_curr + 0.25 * p_next
            
            # 只更新非固定点
            # mask 切片对应 1:-1 的部分
            mask_inner = ~fixed_mask[1:-1]
            smoothed[1:-1][mask_inner] = p_new[mask_inner]
            
        return smoothed

    def _collect_obstacles(self, gate_pos, gate_quat, real_obst_pos):
        obs_list = []
        for pos in real_obst_pos:
            obs_list.append({'type': ObstacleType.CYLINDER_2D, 'pos': np.array(pos)})
        y_axes = FrameUtils.quat_to_axis(gate_quat, 1)
        w = 0.7; half_w = w/2.0
        for i, c in enumerate(gate_pos):
            y = y_axes[i]
            for s in [1, -1]:
                obs_list.append({'type': ObstacleType.CYLINDER_2D, 'pos': c + s * half_w * y})
            for s in [1, -1]:
                obs_list.append({'type': ObstacleType.CAPSULE_3D, 'pos': c, 'vec': y, 'half_len': half_w})
        return obs_list
# ==============================================================================
#  MODULE 4: MPCC Controller (Strictly aligned with mpcc_9.py)
# ==============================================================================

class MPCC(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self._cfg = config
        
        self._dyn_params = load_params("so_rpy_rotor_drag", config.sim.drone_model)
        # 初始化推力为悬停推力，与 mpcc_9.py 保持一致
        self.hover_thrust = float(self._dyn_params["mass"]) * -float(self._dyn_params["gravity_vec"][-1])
        
        # --- 1. 初始化 Planner 并生成第一条路径 ---
        self.planner = PathPlanner(
            initial_pos=obs["pos"], 
            initial_gates_pos=obs["gates_pos"],
            ctrl_freq=config.env.freq,    # [新增] 传入频率
            planned_duration=30.0         # 保持与 mpcc_9 一致
        )
        
        self.N = 35 
        self.T_HORIZON = 0.7
        self.model_arc_length = 0.05
        self.model_traj_length = 12.0
        
        # 初次规划：确保 self.planner.arc_trajectory 存在
        self.planner.check_env_update(obs)
        self.planner.replan( obs, self.model_traj_length)
        
        # --- 2. 构建 Solver (传入初始轨迹以计算参数) ---
        # 关键修正：必须在创建 solver 前设置好 parameter_values
        self.acados_ocp_solver, self.ocp = self._build_ocp_and_solver(
            self.T_HORIZON, self.N, self.planner.arc_trajectory
        )
        
        # 安全边界 (来自 mpcc_9.py)
        self.pos_bound = [
            np.array([-2.6, 2.6]),
            np.array([-2.0, 1.8]),
            np.array([-0.1, 2.0]),
        ]
        self.velocity_bound = [-1.0, 4.0]

        self.last_theta = 0.0
        self.last_f_cmd = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self.finished = False
        self._x_warm = None

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        self._current_obs_pos = obs["pos"]
        
        # 1. 检查环境变化与重规划
        if self.planner.check_env_update(obs):
            # print(f"T={self._step_count / self._ctrl_freq:.2f}: Replanning...")
            self.planner.replan( obs, self.model_traj_length)
            
            # 更新参数：新的轨迹 -> 新的参数
            param_vec = self._encode_traj_params(self.planner.arc_trajectory)
            for k in range(self.N + 1): 
                self.acados_ocp_solver.set(k, "p", param_vec)

        # 2. 状态获取
        quat = obs["quat"]
        r_obj = R.from_quat(quat)
        roll_pitch_yaw = r_obj.as_euler("xyz", degrees=False)
        drpy = ang_vel2rpy_rates(quat, obs["ang_vel"]) if "ang_vel" in obs else np.zeros(3)
        
        # X_phys: [pos(3), rpy(3), vel(3), drpy(3)]
        X_phys_now = np.concatenate((obs["pos"], roll_pitch_yaw, obs["vel"], drpy))
        
        # Full State: [X_phys, r_cmd, p_cmd, y_cmd, f_cmd, theta]
        x_now = np.concatenate((
            X_phys_now,
            self.last_rpy_cmd, 
            [self.last_f_cmd], 
            [self.last_theta]
        ))

        # 3. Warm Start (完全复刻 mpcc_9.py)
        if self._x_warm is None:
            self._x_warm = [x_now.copy() for _ in range(self.N + 1)]
            self._u_warm = [np.zeros(5) for _ in range(self.N)]
        else:
            self._x_warm = self._x_warm[1:] + [self._x_warm[-1]]
            self._u_warm = self._u_warm[1:] + [self._u_warm[-1]]

        for i in range(self.N):
            self.acados_ocp_solver.set(i, "x", self._x_warm[i])
            self.acados_ocp_solver.set(i, "u", self._u_warm[i])
        self.acados_ocp_solver.set(self.N, "x", self._x_warm[self.N])
        
        # 初始状态约束 (强制 x0 = x_now)
        self.acados_ocp_solver.set(0, "lbx", x_now)
        self.acados_ocp_solver.set(0, "ubx", x_now)

        # 4. 安全与结束检查
        if self.last_theta >= self.planner.arc_trajectory.x[-1]: 
            self.finished = True
            print("[MPCC] Stop: finished path.")
        
        if self._pos_outside_limits(obs["pos"]):
            self.finished = True
            print("[MPCC] Stop: position out of safe bounds.")
            
        if self._speed_outside_limits(obs["vel"]):
            self.finished = True
            print("[MPCC] Stop: velocity out of safe range.")
        
        # 5. 求解
        status = self.acados_ocp_solver.solve()
        if status != 0:
            print("[MPCC] acados solver returned non-zero status:", status)
        
        # 6. 获取结果
        self._x_warm = [self.acados_ocp_solver.get(i, "x") for i in range(self.N + 1)]
        self._u_warm = [self.acados_ocp_solver.get(i, "u") for i in range(self.N)]
        x_next = self.acados_ocp_solver.get(1, "x")
        
        self.last_rpy_cmd = x_next[12:15]
        self.last_f_cmd = x_next[15]
        self.last_theta = x_next[16]
        
        self._step_count += 1
        return np.array([*self.last_rpy_cmd, self.last_f_cmd], dtype=float)

    def _build_ocp_and_solver(
        self, Tf: float, N_horizon: int, initial_trajectory, verbose: bool = False
    ) -> tuple[AcadosOcpSolver, AcadosOcp]:
        """
        按照 mpcc_9.py 的配置构建 OCP，并使用 initial_trajectory 初始化参数。
        """
        ocp = AcadosOcp()
        model = self._export_dynamics_model()
        ocp.model = model

        self.nx = model.x.rows()
        self.nu = model.u.rows()
        ocp.solver_options.N_horizon = N_horizon
        ocp.cost.cost_type = "EXTERNAL"

        # --- 权重设置 (mpcc_9.py) ---
        self.q_l = 200
        self.q_c = 100
        self.Q_w = 1 * DM(np.eye(3))

        self.q_l_gate_peak = 640
        self.q_c_gate_peak = 800
        self.q_l_obst_peak = 100
        self.q_c_obst_peak = 50

        # mpcc_9.py 原始阻尼 (响应快)
        self.R_df = DM(np.diag([0.1, 0.5, 0.5, 0.5])) 

        self.miu = 8.0
        self.w_v_gate = 4.0
        self.w_v_obst = 1.0

        ocp.model.cost_expr_ext_cost = self._stage_cost_expression()

        # --- 约束设置 (mpcc_9.py) ---
        thrust_min = float(self._dyn_params["thrust_min"]) * 4.0
        thrust_max = float(self._dyn_params["thrust_max"]) * 4.0

        # 状态约束: 仅约束 f_cmd(15) 和 rpy_cmd(12,13,14)
        ocp.constraints.lbx = np.array([thrust_min, -1.57, -1.57, -1.57]) 
        ocp.constraints.ubx = np.array([thrust_max, 1.57, 1.57, 1.57])   
        ocp.constraints.idxbx = np.array([15, 12, 13, 14])                  
        
        # 输入约束
        ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, 0.0])
        ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0, 4.0])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])
        ocp.constraints.x0 = np.zeros(self.nx)
        
        # --- 关键修复：初始化 Parameter Values ---
        # 如果这里不填入基于轨迹的有效值，solver 初始迭代会因为误差过大而崩满
        if initial_trajectory is not None:
            param_vec = self._encode_traj_params(initial_trajectory)
            ocp.parameter_values = param_vec
        else:
            # 备用（不应发生）
            ocp.parameter_values = np.zeros(3 * int(12.0/0.05) * 2 + 2 * int(12.0/0.05))
        
        # Solver Options (mpcc_9.py)
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tol = 1e-5
        ocp.solver_options.qp_solver_cond_N = N_horizon
        # mpcc_9.py 使用 Warm Start
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.qp_solver_iter_max = 20
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.tf = Tf
        
        solver = AcadosOcpSolver(ocp, json_file="mpcc_prescripted_real_dyn.json", verbose=verbose)
        return solver, ocp

    # --- 辅助函数 ---
    def _pos_outside_limits(self, pos: NDArray[np.floating]) -> bool:
        if self.pos_bound is None: return False
        for i_dim in range(3):
            low, high = self.pos_bound[i_dim]
            if pos[i_dim] < low or pos[i_dim] > high: return True
        return False

    def _speed_outside_limits(self, vel: NDArray[np.floating]) -> bool:
        if self.velocity_bound is None: return False
        speed = np.linalg.norm(vel)
        return not (self.velocity_bound[0] < speed < self.velocity_bound[1])

    def _encode_traj_params(self, trajectory):
        """将样条编码为 OCP 参数"""
        theta = np.arange(0.0, self.model_traj_length, self.model_arc_length)
        pd = trajectory(theta)
        tp = trajectory(theta, 1)
        qc_gate = np.zeros_like(theta)
        qc_obst = np.zeros_like(theta)
        
        # 使用 Elastic Planner 的缓存数据
        if hasattr(self.planner, "cached_gates_pos"):
            for gc in self.planner.cached_gates_pos:
                qc_gate = np.maximum(qc_gate, np.exp(-2.0 * np.linalg.norm(pd - gc, axis=1)**2))
        if hasattr(self.planner, "cached_obstacles_pos"):
            for oc in self.planner.cached_obstacles_pos:
                qc_obst = np.maximum(qc_obst, 1.0 * np.exp(-0.5 * np.linalg.norm(pd[:,:2] - oc[:2], axis=1)**2))
            
        return np.concatenate([pd.ravel(), tp.ravel(), qc_gate, qc_obst])

    def _export_dynamics_model(self) -> AcadosModel:
        """
        使用 drone_models.so_rpy.symbolic_dynamics_euler 的真实动力学：

        X_phys: [px,py,pz, roll,pitch,yaw, vx,vy,vz, dr,dp,dy]   (12)
        U_phys: [r_cmd, p_cmd, y_cmd, f_cmd]                     (4)

        在外面再加 4 个“命令状态” + 1 个 theta:
        X = [X_phys, r_cmd_state, p_cmd_state, y_cmd_state, f_cmd_state, theta] (17)
        U = [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta_cmd]

        其中真实动力学的控制输入 U_phys = [r_cmd_state, p_cmd_state, y_cmd_state, f_cmd_state]。
        """

        model_name = "lsy_example_mpc_real"

        params = self._dyn_params

        # 真实动力学（仅 12 状态 + 4 控制）
        X_dot_phys, X_phys, U_phys, _ = symbolic_dynamics_euler(
            mass=params["mass"],
            gravity_vec=params["gravity_vec"],
            J=params["J"],
            J_inv=params["J_inv"],
            acc_coef=params["acc_coef"],
            cmd_f_coef=params["cmd_f_coef"],
            rpy_coef=params["rpy_coef"],
            rpy_rates_coef=params["rpy_rates_coef"],
            cmd_rpy_coef=params["cmd_rpy_coef"],
        )

        # 物理状态别名
        self.px = X_phys[0]
        self.py = X_phys[1]
        self.pz = X_phys[2]
        self.roll = X_phys[3]
        self.pitch = X_phys[4]
        self.yaw = X_phys[5]
        self.vx = X_phys[6]
        self.vy = X_phys[7]
        self.vz = X_phys[8]
        self.dr = X_phys[9]
        self.dp = X_phys[10]
        self.dy = X_phys[11]

        # 命令状态（将作为真实动力学的输入）
        self.r_cmd_state = MX.sym("r_cmd_state")
        self.p_cmd_state = MX.sym("p_cmd_state")
        self.y_cmd_state = MX.sym("y_cmd_state")
        self.f_cmd_state = MX.sym("f_cmd_state")

        # 路径进度 theta
        self.theta = MX.sym("theta")

     
        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")
        self.v_theta_cmd = MX.sym("v_theta_cmd")

        states = vertcat(
            X_phys,
            self.r_cmd_state,
            self.p_cmd_state,
            self.y_cmd_state,
            self.f_cmd_state,
            self.theta,
        )
        inputs = vertcat(
            self.df_cmd,
            self.dr_cmd,
            self.dp_cmd,
            self.dy_cmd,
            self.v_theta_cmd,
        )

        # 真实动力学的控制输入由命令状态给出
        U_phys_full = vertcat(
            self.r_cmd_state,
            self.p_cmd_state,
            self.y_cmd_state,
            self.f_cmd_state,
        )

        # 用 casadi.substitute 把原本的 U_phys 换成 U_phys_full
        f_dyn_phys = substitute(X_dot_phys, U_phys, U_phys_full)

        # 命令状态一阶积分
        r_cmd_dot = self.dr_cmd
        p_cmd_dot = self.dp_cmd
        y_cmd_dot = self.dy_cmd
        f_cmd_dot = self.df_cmd

        theta_dot = self.v_theta_cmd

        f_dyn = vertcat(
            f_dyn_phys,
            r_cmd_dot,
            p_cmd_dot,
            y_cmd_dot,
            f_cmd_dot,
            theta_dot,
        )

        # 轨迹参数
        n_samples = int(self.model_traj_length / self.model_arc_length)
        self.pd_list = MX.sym("pd_list", 3 * n_samples)
        self.tp_list = MX.sym("tp_list", 3 * n_samples)

        # 拆成 gate / obstacle 两类“权重”曲线
        self.qc_gate = MX.sym("qc_gate", 1 * n_samples)
        self.qc_obst = MX.sym("qc_obst", 1 * n_samples)

        params_sym = vertcat(self.pd_list, self.tp_list, self.qc_gate, self.qc_obst)

        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f_dyn
        model.x = states
        model.u = inputs
        model.p = params_sym
        return model

    def _piecewise_linear_interp(self, theta, theta_vec, flattened_points, dim: int = 3):
        M = len(theta_vec); idx_float = (theta - theta_vec[0]) / (theta_vec[-1] - theta_vec[0]) * (M - 1)
        idx_low = floor(idx_float); idx_high = idx_low + 1; alpha = idx_float - idx_low
        idx_low = if_else(idx_low < 0, 0, idx_low); idx_high = if_else(idx_high >= M, M - 1, idx_high)
        p_low = vertcat(*[flattened_points[dim * idx_low + i] for i in range(dim)])
        p_high = vertcat(*[flattened_points[dim * idx_high + i] for i in range(dim)])
        return (1.0 - alpha) * p_low + alpha * p_high

    def _stage_cost_expression(self):
        position_vec = vertcat(self.px, self.py, self.pz); att_vec = vertcat(self.roll, self.pitch, self.yaw)
        ctrl_vec = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)
        
        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_length)
        pd_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)
        qc_gate_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_gate, dim=1)
        qc_obst_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_obst, dim=1)
        
        tp_unit = tp_theta / (norm_2(tp_theta) + 1e-6)
        e_theta = position_vec - pd_theta
        e_lag = dot(tp_unit, e_theta) * tp_unit
        e_contour = e_theta - e_lag
        
        track_cost = ((self.q_l + self.q_l_gate_peak * qc_gate_theta + self.q_l_obst_peak * qc_obst_theta) * dot(e_lag, e_lag) +
                      (self.q_c + self.q_c_gate_peak * qc_gate_theta + self.q_c_obst_peak * qc_obst_theta) * dot(e_contour, e_contour) +
                      att_vec.T @ self.Q_w @ att_vec)
        smooth_cost = ctrl_vec.T @ self.R_df @ ctrl_vec
        speed_cost = (- self.miu * self.v_theta_cmd + self.w_v_gate * qc_gate_theta * (self.v_theta_cmd ** 2) +
                      self.w_v_obst * qc_obst_theta * (self.v_theta_cmd ** 2))
        return track_cost + smooth_cost + speed_cost

    def step_callback(self, action, obs, reward, terminated, truncated, info): 
        return self.finished
    
    def episode_callback(self):
        print("[MPCC] Episode reset.")
        self._step_count = 0; self.finished = False; self._x_warm = None
        self.last_theta = 0.0; self.last_f_cmd = self.hover_thrust; self.last_rpy_cmd = np.zeros(3)
        self.planner._last_gate_flags = None

    def get_debug_lines(self):
        debug_lines = []
        if self.planner.arc_trajectory is not None:
            try:
                t_end = self.planner.arc_trajectory.x[-1]
                if t_end > 0.1:
                    pts = self.planner.arc_trajectory(np.linspace(0, t_end, 100))
                    debug_lines.append((pts, np.array([0.0, 1.0, 0.0, 1.0]), 2.0, 2.0))
            except: pass
        if self._x_warm is not None:
            try:
                pts = np.asarray([x[:3] for x in self._x_warm])
                debug_lines.append((pts, np.array([1.0, 0.0, 0.0, 1.0]), 3.0, 3.0))
            except: pass
        return debug_lines


