# trajectory aims to next gate
"""
try_mpc9_deque_path.py — Attitude NMPC with soft obstacle avoidance, 
and a PRE-COMPUTED global reference path, tracked with a 
SMOOTHLY RECEDING (deque) polyline.

[MODIFIED: 
 6. [MODIFIED v8.0 - User Req]
    - [NEW] Merged path planning from 'level2_1.py' with MPC from 'mpc_14.py'.
    - [NEW] _build_global_path now uses 'level2_1.py' logic (detours, collision 
      avoidance) to generate a smart CubicSpline trajectory.
    - [NEW] compute_control now samples *time* (not distance) along the spline.
    - [NEW] Re-planning is now handled by 'level2_1.py's 'pos_change_detect'.
    - [REMOVED] All deque, pre/post, climb, and dynamic v_des logic.
 7. [MODIFIED v6.8 - User Req] Tuned MPC to fix droop:
    - Increased Q-weight for Z-position in create_acados_model (2500.0).
 8. [FIX v6.5] Corrected typo 'pos_f'/'self_traj_hist' in compute_control.
 9. [KEPT] MPC EXTERNAL_COST for gate/obstacle avoidance.
]
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat, sqrt, log, exp
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline # [NEW v8.0]

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates

from lsy_drone_racing.control import Controller
from collections import deque # [KEPT] for _traj_hist visualization

if TYPE_CHECKING:
    from numpy.typing import NDArray


# -----------------------------
# ACADOS model & OCP definition
# -----------------------------
# [KEPT from mpc_14.py] MPC/ACADOS 定义保持不变
def create_acados_model(parameters: dict) -> AcadosModel:
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    model = AcadosModel()
    model.name = "att_mpc_lvl2_softobs_band"
    model.f_expl_expr = X_dot
    model.x = X
    model.u = U

    n_obs = 4
    obs_dim = n_obs * 3
    gate_dim = 11
    nx = X.rows()
    nu = U.rows()
    p_dim = obs_dim + gate_dim + nx + nu
    p = MX.sym("p", p_dim)
    model.p = p

    x_pos, y_pos, z_pos = X[0], X[1], X[2]
    h_list = []
    for i in range(n_obs):
        ox = p[3*i + 0]
        oy = p[3*i + 1]
        d2 = (x_pos - ox)**2 + (y_pos - oy)**2
        h_list.append(d2)
    model.con_h_expr = vertcat(*h_list)

    off = obs_dim
    gx = p[off+0]; gy = p[off+1]; gz = p[off+2]
    tx = p[off+3]; ty = p[off+4]
    nxg = p[off+5]; nyg = p[off+6]
    band_lo    = p[off+7]
    band_hi    = p[off+8]
    band_alpha = p[off+9]
    act_s      = p[off+10]

    xref_off = obs_dim + gate_dim
    x_ref = p[xref_off : xref_off + nx]
    u_ref = p[xref_off + nx : xref_off + nx + nu]

    dx = x_pos - gx
    dy = y_pos - gy
    dz = z_pos - gz
    s   = dx*nxg + dy*nyg
    lat = dx*tx  + dy*ty
    abslat = sqrt(lat*lat + 1e-9)
    absdz  = sqrt(dz*dz   + 1e-9)
    rho = sqrt(s*s + lat*lat + 1e-9)
    alpha_s = 1.0 / (1.0 + (rho / act_s) ** 2)

    def bump(r, lo, hi, k):
        sp1 = log(1 + exp(k*(r - lo))) / k
        sp2 = log(1 + exp(k*(hi - r))) / k
        return sp1 * sp2

    pen_lat = bump(abslat, band_lo, band_hi, band_alpha)
    pen_z   = bump(absdz , band_lo, band_hi, band_alpha)

    Q_diag = [1000, 1000, 2500, 5, 5, 5, 1, 1, 1, 5, 5, 5]
    R_diag = [0.5, 0.5, 0.5, 10.0]
    
    dx_vec = X - x_ref
    du_vec = U - u_ref
    track_cost = 0
    for i in range(nx): track_cost += Q_diag[i] * dx_vec[i]*dx_vec[i]
    for i in range(nu): track_cost += R_diag[i] * du_vec[i]*du_vec[i]

    w_lat, w_z = 200.0, 500
    ext_cost = track_cost + alpha_s * (w_lat * pen_lat + w_z * pen_z)
    
    model.cost_expr_ext_cost   = ext_cost
    model.cost_expr_ext_cost_0 = ext_cost

    term_cost = 0
    for i in range(nx): term_cost += dx_vec[i]*dx_vec[i]
    model.cost_expr_ext_cost_e = term_cost

    return model

def create_ocp_solver(Tf: float, N: int, parameters: dict, verbose: bool = False):
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    ocp.dims.N = N
    n_obs = 4
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([ 0.5,  0.5,  0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5], dtype=int)

    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4.0])
    ocp.constraints.ubu = np.array([ 0.5,  0.5,  0.5, parameters["thrust_max"] * 4.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3], dtype=int)
    ocp.constraints.x0 = np.zeros(nx)

    r_safe = 0.25
    ocp.dims.nh  = n_obs
    ocp.dims.nsh = n_obs
    ocp.constraints.lh = np.ones(n_obs) * (r_safe ** 2)
    ocp.constraints.uh = np.ones(n_obs) * 1e9
    ocp.constraints.idxsh = np.arange(n_obs, dtype=int)
    slack_w_lin, slack_w_quad = 5e2, 8e3
    ocp.cost.zl = slack_w_lin  * np.ones(n_obs)
    ocp.cost.zu = slack_w_lin  * np.ones(n_obs)
    ocp.cost.Zl = slack_w_quad * np.ones(n_obs)
    ocp.cost.Zu = slack_w_quad * np.ones(n_obs)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = Tf

    p_dim = n_obs*3 + 11 + nx + nu
    ocp.parameter_values = np.zeros(p_dim)

    json_name = "att_mpc_lvl2_softobs_band"
    solver = AcadosOcpSolver(
        ocp,
        json_file=f"c_generated_code/{json_name}.json",
        verbose=verbose,
        build=True,
        generate=True,
    )
    return solver, ocp



# -----------------------------
# Controller
# -----------------------------
class AttitudeMPC(Controller):
    """
    Attitude NMPC v8.0 (Hybrid):
    - Path Planning: Uses 'level2_1.py' logic (detours, collision avoidance)
      to generate a smart, time-based CubicSpline.
    - Path Following: MPC samples time along the spline for its horizon.
    - Control: Uses 'mpc_14.py' logic (EXTERNAL_COST, soft constraints)
      for robust execution and avoidance.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._N = 85
        self._dt = 1.0 / float(config.env.freq) # 50Hz -> 0.02s
        self._T_HORIZON = self._N * self._dt # 85 * 0.02 = 1.7s

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )

        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()

        self._finished = False
        self._tick = 0
        self.t_total_est = 30.0 # [NEW v8.0] 初始轨迹时间估计

        # Visualization caches
        self._traj_hist = deque(maxlen=4000)
        self._last_plan = None
        # [REMOVED v8.0] _last_polyline (deque logic)

        # --- [NEW v8.0] 动态地图/重规划状态 ---
        self._initial_obs = obs.copy()
        self.init_pos = obs['pos']
        self._known_gates_pos = np.asarray(obs["gates_pos"], float).copy()
        self._known_obstacles_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3).copy()
        
        # [NEW v8.0] 使用 level2_1.py 的重规划检测器
        self.last_gate_flags = None
        self.last_obst_flags = None
        self._replan_requested = False 
        
        # [REMOVED v8.0] d_pre/d_post lists

        # --- [NEW v8.0] 全局轨迹 (CubicSpline) ---
        self.trajectory: CubicSpline | None = None
        self.t_total: float = self.t_total_est

        # [NEW v8.0] 首次规划
        self.episode_callback() # 调用一次以初始化 flags 和 trajectory
        self._tick = 0 # 重置 episode_callback 增加的 tick
        self._finished = False

        # [REMOVED v8.0] Deque setup
        # --- [MODIFIED END] ---


    # ---------- helpers ----------
    
    def _hover_thrust(self) -> float:
        """计算悬停推力 (标量)。"""
        return float(self.drone_params["mass"]) * (-float(self.drone_params["gravity_vec"][-1]))

    # [NEW v8.0] --- 移植自 level2_1.py 的辅助函数 ---

    def _extract_gate_coordinate_frames(
        self, 
        gates_quaternions: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """从四元数中提取门的法向、Y轴和Z轴 (来自 level2_1.py)。"""
        # [FIX] 确保 quat 是 (N, 4)
        if gates_quaternions is None or gates_quaternions.size == 0:
            print("[AttitudeMPC] WARN: No quaternions found, using default [1,0,0] normal.")
            num_gates = len(self._known_gates_pos)
            normals = np.tile(np.array([1.0, 0.0, 0.0]), (num_gates, 1))
            y_axes =  np.tile(np.array([0.0, 1.0, 0.0]), (num_gates, 1))
            z_axes =  np.tile(np.array([0.0, 0.0, 1.0]), (num_gates, 1))
            return normals, y_axes, z_axes
            
        rotations = R.from_quat(gates_quaternions.reshape(-1, 4))
        rotation_matrices = rotations.as_matrix()
        
        normals = rotation_matrices[:, :, 0]  # 法向 (x-axis)
        y_axes = rotation_matrices[:, :, 1]   
        z_axes = rotation_matrices[:, :, 2]   
        
        return normals, y_axes, z_axes

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
        """为急转弯添加绕行点 (来自 level2_1.py)。"""
        num_gates = gate_positions.shape[0]
        waypoints_list = list(waypoints)  
        
        inserted_count = 0
        
        for i in range(num_gates - 1):
            
            last_idx_gate_i = 1 + (i + 1) * num_intermediate_points - 1 + inserted_count
            first_idx_gate_i_plus_1 = 1 + (i + 1) * num_intermediate_points + inserted_count
            
            # 检查索引是否越界
            if last_idx_gate_i >= len(waypoints_list) or first_idx_gate_i_plus_1 >= len(waypoints_list):
                break

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

    def trajectory_generate(
        self, t_total: float, waypoints: NDArray[np.floating],
    ) -> CubicSpline:
        """从路径点生成三次样条轨迹 (来自 level2_1.py)。"""
        diffs = np.diff(waypoints, axis=0)
        segment_length = np.linalg.norm(diffs, axis=1)
        arc_cum_length = np.concatenate([[0], np.cumsum(segment_length)])
        t = arc_cum_length / (arc_cum_length[-1] + 1e-9) * t_total
        # 确保 t 是单调递增的
        if not np.all(np.diff(t) > 0):
            print("[AttitudeMPC] trajectory_generate: 修复非单调的 t。")
            t = np.linspace(0, t_total, len(t))
        return CubicSpline(t, waypoints)
    
    def avoid_collision(
        self, t_total_est: float, waypoints: NDArray[np.floating], obstacles_pos: NDArray[np.floating], safe_dist: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """修改路径点以避免与障碍物碰撞 (来自 level2_1.py, 已修改为无状态)。"""
        # 首先，生成一个初步的轨迹
        pre_trajectory = self.trajectory_generate(t_total_est, waypoints)
        
        num_steps = int(self._N * 2) # 使用 MPC 视界长度的两倍进行采样
        t_axis = np.linspace(0, t_total_est, num_steps)
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
            
            if flag: # 如果在循环结束时仍然在区域内
                t_results.append(t_axis[-1])
                wp_results.append(wp[-1])

            t_axis = np.array(t_results)
            wp = np.array(wp_results)

        if len(t_axis) > 0:
            unique_indices = np.unique(t_axis, return_index=True)[1]
            t_axis = t_axis[unique_indices]
            wp = wp[unique_indices]

        if len(t_axis) < 2:
            print("[AttitudeMPC] avoid_collision: 过滤后点不足，返回原始路径点。")
            t_axis_fallback = self.trajectory_generate(t_total_est, waypoints).x
            wp_fallback = waypoints
            return t_axis_fallback, wp_fallback

        return t_axis, wp

    def pos_change_detect(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        """检测门或障碍物是否从 '未访问' 变为 '已访问' (来自 level2_1.py)。"""
        if self.last_gate_flags is None or self.last_obst_flags is None:
            # 这种情况不应该发生，因为 episode_callback 会初始化
            print("[AttitudeMPC] WARN: pos_change_detect called before flags were initialized.")
            return False

        curr_gate_flags = np.array(obs['gates_visited'], dtype=bool)
        curr_obst_flags = np.array(obs['obstacles_visited'], dtype=bool)

        gate_triggered = np.any((~self.last_gate_flags) & curr_gate_flags)
        obst_triggered = np.any((~self.last_obst_flags) & curr_obst_flags)

        self.last_gate_flags = curr_gate_flags
        self.last_obst_flags = curr_obst_flags

        return gate_triggered or obst_triggered

    # --- 移植自 mpc_14.py (v7.8) 的辅助函数 ---
    
    # [KEPT v7.8] 这个辅助函数被 _set_acados_parameters 需要
    def _gate_xy_dir(self, i: int, cur_pos: np.ndarray,
                    gates_pos: np.ndarray, gates_quat: np.ndarray | None,
                    use_cur_pos: np.ndarray | None = None) -> np.ndarray:
        """
        返回门 i 的 XY 法向 n_xy（指向赛道前进方向）。
        [V7.8] 使用 v_prog (来自 cur_pos/last_gate) 来翻转 quat (如果 quat 是反向的)
        """
        eps = 1e-12
        gi = np.asarray(gates_pos[i], float).reshape(3)

        # 1. 计算赛道前进向量 (v_prog)
        if use_cur_pos is not None and i == 0:
            v_prog = gi - np.asarray(use_cur_pos, float).reshape(3)
        else:
            ref_pos = cur_pos # 默认
            if i > 0:
                 ref_pos = gates_pos[i-1]
            v_prog = gi - np.asarray(ref_pos, float).reshape(3)
        
        v_prog_xy = v_prog[:2]
        if np.linalg.norm(v_prog_xy) < eps:
            v_prog_xy = np.array([1.0, 0.0])
        v_prog_xy /= (np.linalg.norm(v_prog_xy) + eps)

        # 2. 尝试从 Quaternion 获取方向
        if gates_quat is not None and i < len(gates_quat):
            q = np.asarray(gates_quat[i], float).reshape(4)
            fwd1 = np.array([1.0, 0.0])
            try: 
                fwd1_raw = R.from_quat(q).apply([1.0, 0.0, 0.0])[:2]
                if np.linalg.norm(fwd1_raw) > eps: fwd1 = fwd1_raw / np.linalg.norm(fwd1_raw)
            except Exception: pass 
            fwd2 = np.array([1.0, 0.0])
            try: 
                q_wxyz = np.array([q[3], q[0], q[1], q[2]]) 
                fwd2_raw = R.from_quat(q_wxyz).apply([1.0, 0.0, 0.0])[:2]
                if np.linalg.norm(fwd2_raw) > eps: fwd2 = fwd2_raw / np.linalg.norm(fwd2_raw)
            except Exception: pass 

            n_xy = fwd1 if abs(float(np.dot(fwd1, v_prog_xy))) >= abs(float(np.dot(fwd2, v_prog_xy))) else fwd2
            
            if float(np.dot(n_xy, v_prog_xy)) < 0.0:
                n_xy = -n_xy # 翻转它！
            
            return n_xy

        # 3. [FALLBACK] 如果没有 quat，使用赛道流向
        return v_prog_xy.copy()

    
    # [NEW v8.0] 全局路径生成器（现在使用 level2_1 逻辑）
    def _build_global_path(self):
        """
        [MODIFIED v8.0] 
        - 使用 level2_1.py 的智能路径规划逻辑。
        - 结果是一个 (M, 3) 的航点数组，存储在 self._global_waypoints。
        - 最终生成一个 CubicSpline 存储在 self.trajectory。
        """
        print("[AttitudeMPC] Re-building global path (v8.0 - level2_1 logic)...")
        
        # 1. 获取当前已知的世界状态
        gates_pos = self._known_gates_pos
        obstacles_pos = self._known_obstacles_pos
        init_pos = self.init_pos # 始终从原始起点开始
        
        # 假定 Quat 是固定的（level2.toml 中 randomize=false）
        gates_quat = np.asarray(self._initial_obs.get("gates_quat", None), float)
        gates_norm, y_axes, z_axes = self._extract_gate_coordinate_frames(gates_quat)

        # 2. 生成基础路径点 (来自 calc_waypoints)
        waypoints = self.calc_waypoints(init_pos, gates_pos, gates_norm)
        
        # 3. 为急转弯添加绕行点 (来自 _add_detour_waypoints)
        waypoints = self._add_detour_waypoints(
            waypoints, gates_pos, gates_norm, y_axes, z_axes,
            num_intermediate_points=5,
            angle_threshold=120.0,
            detour_distance=0.65
        )
        
        # 4. 运行碰撞规避 (来自 avoid_collision)
        # 估计一个总时间（仅用于 avoid_collision 采样）
        t_total_est = self.t_total_est
        t_axis, waypoints_avoided = self.avoid_collision(
            t_total_est, waypoints, obstacles_pos, 0.30
        )
        
        if len(t_axis) < 2:
            print("[AttitudeMPC] _build_global_path: 避障失败，使用绕行路径。")
            final_waypoints = waypoints
            self.trajectory = self.trajectory_generate(t_total_est, final_waypoints)
        else:
            final_waypoints = waypoints_avoided
            # [FIX] 使用避障后的 t_axis 创建样条
            self.trajectory = CubicSpline(t_axis, final_waypoints)

        # 5. 保存最终结果
        self.t_total = self.trajectory.x[-1] # 更新总时间
        self._global_waypoints = final_waypoints # 保存用于可视化的航点
        
        print(f"[AttitudeMPC] Global spline path built (T={self.t_total:.2f}s) with {len(self._global_waypoints)} points.")

    
    # [REMOVED v8.0] _reset_deque
    # [REMOVED v8.0] _build_local_ref
    
    # ---------- debug visualization (matplotlib) ----------
    def debug_plot(self, obs: dict):
        import matplotlib.pyplot as plt
        cur_pos = np.asarray(obs["pos"], float)
        gates_pos = self._known_gates_pos
        obstacles_pos = self._known_obstacles_pos
        cur_quat = np.asarray(obs["quat"], float)
        
        gate_idx = int(np.asarray(obs["target_gate"]).item())
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # [NEW v8.0] 绘制样条曲线
        if self.trajectory is not None:
             t_plot = np.linspace(self.trajectory.x[0], self.trajectory.x[-1], 200)
             path_plot = self.trajectory(t_plot)
             ax.plot(path_plot[:, 0], path_plot[:, 1], 
                     '--', color='gray', label="Global Spline Path")

        # [REMOVED v8.0] pos_ref (MPC ref path)
        ax.scatter(cur_pos[0], cur_pos[1], c='blue', marker='o', s=80, label="drone")
        ax.scatter(gates_pos[:, 0], gates_pos[:, 1], c='green', marker='s', s=40, label="gates")
        for i, gp in enumerate(gates_pos):
            ax.text(gp[0], gp[1], f"G{i}", color='green', fontsize=8)
        if obstacles_pos.size > 0:
            ax.scatter(obstacles_pos[:, 0], obstacles_pos[:, 1], c='red', marker='x', s=80, label="obstacles")
            r_safe_solver = 0.33 # (这匹配 create_ocp_solver)
            for (ox, oy, _oz) in obstacles_pos:
                circ1 = plt.Circle((ox, oy), r_safe_solver, fill=False, linestyle='--', color='red', alpha=0.5, label="MPC r_safe (0.33)")
                ax.add_patch(circ1)
        
        # [REMOVED v8.0] _last_polyline
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True); ax.legend()
        ax.set_title(f"Spline Follower (t={self._tick/self._dt:.2f}s)")
        plt.show()
    
    
    # [KEPT v7.8] 辅助函数，用于设置 acados 参数 p
    def _set_acados_parameters(self, cur_pos, pos_ref, vel_ref, yaw_ref):
        """
        为 EXTERNAL cost 设置 acados 参数向量 'p'。
        'p' 包含：[obs_pos, gate_geom, x_ref, u_ref]
        """
        
        # ---------- 1. 障碍物参数 (p[0:12]) ----------
        n_obs = 4 
        cur_pos_xy = cur_pos[:2]
        all_obs_pos = self._known_obstacles_pos
        selected_obs_pos = np.zeros((n_obs, 3))
        
        if all_obs_pos.shape[0] > 0:
            distances_xy = np.linalg.norm(all_obs_pos[:, :2] - cur_pos_xy, axis=1)
            closest_indices = np.argsort(distances_xy)
            num_to_take = min(all_obs_pos.shape[0], n_obs)
            indices_to_take = closest_indices[:num_to_take]
            selected_obs_pos[:num_to_take] = all_obs_pos[indices_to_take]
            if num_to_take < n_obs:
                selected_obs_pos[num_to_take:] = np.array([1e6, 1e6, 1e6])
        else:
             selected_obs_pos[:] = np.array([1e6, 1e6, 1e6])
        obs_flat = selected_obs_pos.flatten()

        # ---------- 2. 禁行带参数 (p[12:23]) ----------
        band_lo, band_hi = 0.30, 0.80
        band_alpha, act_s = 20.0, 0.50

        gates_pos = self._known_gates_pos
        gates_quat = np.asarray(self._initial_obs.get("gates_quat", None), float)

        dists = np.linalg.norm(gates_pos[:, :2] - cur_pos_xy[None, :], axis=1)
        best_i = int(np.argmin(dists))

        n_xy = self._gate_xy_dir(best_i, cur_pos, gates_pos, gates_quat, use_cur_pos=cur_pos)
        t_xy = np.array([-n_xy[1], n_xy[0]])
        g_band = gates_pos[best_i]
        gx, gy, gz = float(g_band[0]), float(g_band[1]), float(g_band[2])

        # ---------- 3. 组装并设置 p 向量 (p[0:end]) ----------
        nx, nu = self._nx, self._nu
        p_len = n_obs*3 + 11 + nx + nu # 总长度

        for j in range(self._N + 1):
            # 3.1 获取 x_ref, u_ref
            jj = min(j, self._N)
            xr = np.zeros(nx)
            xr[0:3] = pos_ref[jj]
            xr[5]   = yaw_ref[jj]
            xr[6:9] = vel_ref[jj]
            ur = np.zeros(nu)
            ur[3] = self._hover_thrust()

            # 3.2 组装完整的 p 向量
            p_full = np.zeros(p_len)
            off = 0
            p_full[off: off + n_obs*3] = obs_flat; off += n_obs*3 
            p_full[off + 0: off + 3] = [gx, gy, gz]
            p_full[off + 3: off + 5] = t_xy
            p_full[off + 5: off + 7] = n_xy
            p_full[off + 7] = band_lo
            p_full[off + 8] = band_hi
            p_full[off + 9] = band_alpha
            p_full[off +10] = act_s
            off += 11
            p_full[off: off + nx] = xr; off += nx
            p_full[off: off + nu] = ur; off += nu

            # 3.3 设置参数
            self._solver.set(j, "p", p_full)


    # ---------- main MPC step ----------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        
        cur_pos = np.asarray(obs["pos"], float)
        
        # --- [NEW v8.0] 1. 检查传感器数据并请求重规划 ---
        if self.pos_change_detect(obs):
            print("[AttitudeMPC] Sensor change detected!")
            self._replan_requested = True
            
            # 手动更新地图
            for i in range(len(self._known_gates_pos)):
                if obs["gates_visited"][i]:
                    self._known_gates_pos[i] = obs["gates_pos"][i]
            for i in range(len(self._known_obstacles_pos)):
                if obs["obstacles_visited"][i]:
                    self._known_obstacles_pos[i] = obs["obstacles_pos"][i]

        # --- [NEW v8.0] 2. 如果需要，执行重规划 ---
        if self._replan_requested:
            print("[AttitudeMPC] Replanning global path with new sensor data.")
            self._build_global_path() # 重建 self.trajectory
            self._replan_requested = False
            # [FIX] 重规划后重置时间
            self._tick = 0 
            print(f"[AttitudeMPC] New path T_total = {self.t_total:.2f}s")
        
        # --- 3. 设置初始状态 ---
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        # --- [REMOVED v8.0] Deque Update Logic ---

        # --- 5. [NEW v8.0] 构建局部参考轨迹 (从样条采样) ---
        
        # 5.1 计算当前时间和视界时间
        tau = min(self._tick * self._dt, self.t_total)
        
        # 创建时间向量 [tau, tau+dt, tau+2*dt, ..., tau+N*dt]
        t_horizon = np.linspace(
            tau, 
            min(tau + self._T_HORIZON, self.t_total), 
            self._N + 1
        )
        
        # 5.2 从样条采样 位置, 速度, 加速度
        pos_ref = self.trajectory(t_horizon)
        vel_ref = self.trajectory(t_horizon, 1) # 1阶导数
        
        # 5.3 计算 Yaw
        # [FIX] 使用 1 阶导数（速度）来计算 yaw
        dpos = vel_ref
        yaw_ref = np.arctan2(dpos[:, 1], dpos[:, 0])

        # 鲁棒 Yaw 计算
        unstable_yaw = np.linalg.norm(dpos[:, :2], axis=1) < 0.05
        if np.any(unstable_yaw):
            cur_yaw = R.from_quat(obs["quat"]).as_euler("xyz")[2]
            last_stable_yaw = cur_yaw
            for k in range(self._N + 1):
                if unstable_yaw[k]:
                    yaw_ref[k] = last_stable_yaw
                else:
                    last_stable_yaw = yaw_ref[k]
        yaw_ref = np.nan_to_num(yaw_ref, nan=0.0)
        
        # 缓存可视化
        self._last_plan = pos_ref.copy()
        self._traj_hist.append(cur_pos.reshape(3))

        # --- 6. [KEPT v7.8] 设置 Acados 参数 p ---
        self._set_acados_parameters(cur_pos, pos_ref, vel_ref, yaw_ref)

        # --- 7. 求解 MPC ---
        self._solver.solve()
        
        # --- 8. 提取结果 ---
        pred = np.zeros((self._N + 1, 3))
        for j in range(self._N + 1):
            xj = self._solver.get(j, "x")
            pred[j] = xj[0:3]
        self._last_plan = pred # 用实际预测覆盖参考
        
        u0 = self._solver.get(0, "u")
        return u0
    
    # ---------- callbacks ----------
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
        [MODIFIED v8.0]
        - 增加 tick。
        - 检查是否到达轨迹末端。
        """
        self._tick += 1
        if not self._finished:
             # 检查是否完成
             current_time = self._tick * self._dt
             if current_time >= self.t_total:
                 print(f"[AttitudeMPC] Trajectory finished at t={current_time:.2f}s")
                 self._finished = True
                 
        return self._finished

    def episode_callback(self):
        """
        [MODIFIED v8.0]
        - 重置所有状态和 flags。
        - 重建全局轨迹 (Spline)。
        """
        # cleanup between episodes
        self._traj_hist.clear()
        self._last_plan = None
        
        print("[AttitudeMPC] Episode reset.")
        self._tick = 0
        self._finished = False
        
        num_gates = len(self._initial_obs["gates_pos"])
        num_obstacles = len(self._initial_obs["obstacles_pos"])
        
        # [NEW v8.0] 初始化 (或重置) level2_1 的重规划检测器
        self.last_gate_flags = np.zeros(num_gates, dtype=bool)
        self.last_obst_flags = np.zeros(num_obstacles, dtype=bool)
        
        self._known_gates_pos = np.asarray(self._initial_obs["gates_pos"], float).copy()
        self._known_obstacles_pos = np.asarray(self._initial_obs["obstacles_pos"], float).reshape(-1, 3).copy()
        
        self._replan_requested = False
        self.t_total = self.t_total_est # 恢复估计的总时间
        
        # 用标称路径重建
        self._build_global_path()
        # [REMOVED v8.0] _reset_deque()


    def get_debug_lines(self):
        """
        Returns a list of tuples: (points(N,3), rgba(4,), min_size, max_size).
        - History (blue)
        - Current plan (red)
        - [REMOVED] Current polyline (yellow)
        - [NEW] Global path (waypoints)
        """
        out = []
        if len(self._traj_hist) >= 2:
            traj = np.asarray(self._traj_hist, float)
            out.append((traj, np.array([0.1, 0.3, 1.0, 0.9]), 2.5, 2.5))  # 蓝：历史
        if self._last_plan is not None and self._last_plan.shape[0] >= 2:
            out.append((self._last_plan, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))  # 红：预测
        
        # [MODIFIED v8.0] 显示生成的航点，而不是 polyline
        if getattr(self, "_global_waypoints", None) is not None and self._global_waypoints.shape[0] >= 2:
            out.append((self._global_waypoints, np.array([1.0, 0.9, 0.1, 0.95]), 3.0, 3.0))  # 黄：全局航点
        
        return out