from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat, dot, DM, norm_2, floor, if_else, substitute
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates

from lsy_drone_racing.control import Controller

import os
import sys


if TYPE_CHECKING:
    from numpy.typing import NDArray

from enum import IntEnum

class ObstacleType(IntEnum):
    CYLINDER_2D = 0  # 无限高圆柱：只计算 XY 平面距离 (用于大障碍物、左右门柱)
    CAPSULE_3D  = 2  # 有限长线段/胶囊：计算点到线段距离 (用于上下门框)

class GCOPTER_Lite:
    """
    极速版 GCOPTER (Minimum Jerk)。
    
    速度优化策略：
    1. 移除 scipy.optimize 的时间优化循环。
    2. 使用 '梯形速度剖面' (Trapezoidal Velocity Profile) 快速估算每段物理可行的时间。
    3. 仅保留一次闭式线性求解 (Closed-form Linear Solve)。
    
    耗时估计: < 3ms (Python)
    """
    def __init__(self, waypoints, avg_speed=4.0):
        self.waypoints = np.array(waypoints)
        self.n_seg = len(waypoints) - 1
        self.dim = waypoints.shape[1]
        
        # 1. 快速启发式时间分配
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        
        # 简单物理估算: t = dist / speed
        self.T = dists / avg_speed 
        
        # 强制每段至少 0.1s (防止两点过近导致求解爆炸)
        self.T = np.maximum(self.T, 0.1) 
        
        # 2. 闭式求解多项式系数
        self.coeffs = self._solve_min_jerk_fast(self.waypoints, self.T)
        self.ts_cumulative = np.concatenate(([0], np.cumsum(self.T)))
        self.x = self.ts_cumulative

    def _solve_min_jerk_fast(self, Q, T):
        """
        求解 Minimum Jerk 的闭式解 (无迭代)
        """
        n = self.n_seg
        dim = self.dim
        coeffs = np.zeros((n, dim, 6)) # 6 coefficients for 5th order

        # 为每个维度分别求解
        for d in range(dim):
            # --- 极速近似法 ---
            qs = Q[:, d]
            vs = np.zeros(n + 1)
            as_ = np.zeros(n + 1)
            
            # 速度启发式：前后两段平均速度 (Finite Difference)
            for i in range(1, n):
                v_prev = (qs[i] - qs[i-1]) / T[i-1]
                v_next = (qs[i+1] - qs[i]) / T[i]
                vs[i] = 0.5 * (v_prev + v_next)
                
            # 加速度启发式
            for i in range(1, n):
                a_prev = (vs[i] - vs[i-1]) / T[i-1]
                a_next = (vs[i+1] - vs[i]) / T[i]
                as_[i] = 0.5 * (a_prev + a_next)
                
            # --- 分段计算系数 (Closed-form mapping) ---
            for i in range(n):
                p0, p1 = qs[i], qs[i+1]
                v0, v1 = vs[i], vs[i+1]
                a0, a1 = as_[i], as_[i+1]
                t = T[i]
                
                t2, t3, t4, t5 = t*t, t*t*t, t*t*t*t, t*t*t*t*t
                
                delta_p = p1 - (p0 + v0*t + 0.5*a0*t2)
                delta_v = v1 - (v0 + a0*t)
                delta_a = a1 - a0
                
                # 逆矩阵系数预计算 (Inverse of [t^3, t^4, t^5; ...])
                k_c3 = (20*delta_p - 8*delta_v*t + delta_a*t2) / (2*t3)
                k_c4 = (-30*delta_p + 14*delta_v*t - 2*delta_a*t2) / (2*t4)
                k_c5 = (12*delta_p - 6*delta_v*t + delta_a*t2) / (2*t5)
                
                coeffs[i, d, :] = [p0, v0, 0.5*a0, k_c3, k_c4, k_c5]
                
        return coeffs

    def __call__(self, t_in, derivative=0):
        t_in = np.atleast_1d(t_in)
        res = np.zeros((len(t_in), self.dim))
        
        # 限制时间范围
        total_T = self.ts_cumulative[-1]
        t_in = np.clip(t_in, 0, total_T - 1e-6)

        # 向量化查找索引
        indices = np.searchsorted(self.ts_cumulative, t_in, side='right') - 1
        indices = np.clip(indices, 0, self.n_seg - 1)
        
        # 提取局部时间 dt
        t_start = self.ts_cumulative[indices]
        dt = t_in - t_start
        
        # 提取系数: (N_samples, Dim, 6)
        c = self.coeffs[indices] 
        
        dt2 = dt**2; dt3 = dt**3; dt4 = dt**4; dt5 = dt**5
        
        for d in range(self.dim):
            cd = c[:, d, :] # (N_samples, 6)
            if derivative == 0: # Pos
                res[:, d] = cd[:,0] + cd[:,1]*dt + cd[:,2]*dt2 + cd[:,3]*dt3 + cd[:,4]*dt4 + cd[:,5]*dt5
            elif derivative == 1: # Vel
                res[:, d] = cd[:,1] + 2*cd[:,2]*dt + 3*cd[:,3]*dt2 + 4*cd[:,4]*dt3 + 5*cd[:,5]*dt4
            elif derivative == 2: # Acc
                res[:, d] = 2*cd[:,2] + 6*cd[:,3]*dt + 12*cd[:,4]*dt2 + 20*cd[:,5]*dt3
                
        if len(res) == 1: return res[0]
        return res
    
    def derivative(self, n=1):
        return lambda t: self.__call__(t, derivative=n)


class FrameUtils:

    @staticmethod
    def quat_to_axis(quat: NDArray[np.floating], axis_index: int = 1) -> NDArray[np.floating]:
        rot = R.from_quat(quat)
        mats = np.asarray(rot.as_matrix())
        if mats.ndim == 3:
            return mats[:, :, axis_index]
        if mats.ndim == 2:
            return mats[:, axis_index]
        return None

    @staticmethod
    def z_axis_to_quat(target_vec: np.ndarray) -> NDArray[np.floating]:
        v = target_vec / (np.linalg.norm(target_vec) + 1e-9)
        z_axis = np.array([0.0, 0.0, 1.0])
        if np.allclose(v, z_axis):
            return np.array([0.0, 0.0, 0.0, 1.0])
        if np.allclose(v, -z_axis):
            return R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_quat()
        rot_axis = np.cross(z_axis, v)
        rot_axis /= np.linalg.norm(rot_axis) + 1e-9
        angle = np.arccos(np.clip(np.dot(z_axis, v), -1.0, 1.0))
        return R.from_rotvec(angle * rot_axis).as_quat()



class PathTools:
    def curvature_from_spline(
        self, spline: CubicSpline, t_vals: np.ndarray, eps: np.ndarray = 1e-8, positive: bool = True
    ) -> np.ndarray:
        v = spline(t_vals, 1)
        a = spline(t_vals, 2)
        cross_term = np.cross(v, a)
        num = np.linalg.norm(cross_term, axis=1)
        den = np.linalg.norm(v, axis=1) ** 3 + eps
        kappa = num / den
        return np.abs(kappa) if positive else kappa

    def turning_radius_from_spline(
        self, spline: CubicSpline, t_vals: np.ndarray, eps: np.ndarray = 1e-8, positive: bool = True
    ) -> np.ndarray:
        v = spline(t_vals, 1)
        a = spline(t_vals, 2)
        cross_term = np.cross(v, a)
        num = np.linalg.norm(v, axis=1) ** 3
        den = np.linalg.norm(cross_term, axis=1) + eps
        radius = num / den
        return np.abs(radius) if positive else radius

    def build_gate_waypoints(
        self,
        start_pos: NDArray[np.floating],
        gates_positions: NDArray[np.floating],
        gates_normals: NDArray[np.floating],
        half_span: float = 0.5,
        samples_per_gate: int = 5,
    ) -> NDArray[np.floating]:
        n_gates = gates_positions.shape[0]
        grid = []
        for idx in range(samples_per_gate):
            alpha = idx / (samples_per_gate - 1) if samples_per_gate > 1 else 0.0
            grid.append(gates_positions - half_span * gates_normals + 2.0 * half_span * alpha * gates_normals)
        stacked = np.stack(grid, axis=1).reshape(n_gates, samples_per_gate, 3).reshape(-1, 3)
        return np.vstack([start_pos[None, :], stacked])

    def spline_through_points(self, duration: float, waypoints: NDArray[np.floating]):
        # 调用我们新建的 GCOPTER_Lite
        # avg_speed: 6.0 意味着非常激进的竞速风格
        traj = GCOPTER_Lite(waypoints, avg_speed= 5.0)
        
        # 更新总规划时间 (这是 GCOPTER 算出来的物理最优时间)
        self._planned_duration = float(traj.ts_cumulative[-1])
        
        return traj

    def reparametrize_by_arclength(
        self, trajectory: CubicSpline, arc_step: float = 0.05, epsilon: float = 1e-5
    ) -> CubicSpline:
        
        # [Fix] 兼容 GCOPTER_Lite 和 CubicSpline
        if hasattr(trajectory, 'x'):
            total_param_range = trajectory.x[-1] - trajectory.x[0]
        elif hasattr(trajectory, 'ts_cumulative'):
            total_param_range = trajectory.ts_cumulative[-1]
        else:
            total_param_range = 10.0 # fallback

        for _ in range(99):
            n_segments = max(2, int(total_param_range / arc_step))
            t_samples = np.linspace(0.0, total_param_range, n_segments)
            pts = trajectory(t_samples)
            deltas = np.diff(pts, axis=0)
            seg_lengths = np.linalg.norm(deltas, axis=1)
            cum_arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            
            # [Fix] 关键修改：去除重复的弧长点 (距离为0的点)
            # CubicSpline 要求 x 严格单调递增
            valid_mask = np.concatenate(([True], np.diff(cum_arc) > 1e-6))
            cum_arc = cum_arc[valid_mask]
            pts = pts[valid_mask]

            if len(cum_arc) < 2:
                return trajectory
            
            total_param_range = float(cum_arc[-1])
            trajectory = CubicSpline(cum_arc, pts)
            
            # 在去重后的序列上计算间隔方差
            if np.std(np.diff(cum_arc)) <= epsilon:
                return trajectory

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

 

    def closest_point_on_path(
        self,
        trajectory: CubicSpline,
        pos: NDArray[np.floating],
        total_length: float | None = None,
        sample_interval: float = 0.05,
    ):
        if total_length is None:
            total_length = float(trajectory.x[-1])
        t_samples = np.arange(0.0, total_length, sample_interval)
        if t_samples.size == 0:
            return 0.0, trajectory(0.0)
        points = trajectory(t_samples)
        dists = np.linalg.norm(points - pos, axis=1)
        idx_min = int(np.argmin(dists))
        return idx_min * sample_interval, points[idx_min]

    def gate_points_on_path(
        self,
        trajectory: CubicSpline,
        gates_positions: NDArray[np.floating],
        total_length: float | None = None,
        sample_interval: float = 0.05,
    ):
        if total_length is None:
            total_length = float(trajectory.x[-1])

        theta_list = []
        gate_interp = []
        for center in gates_positions:
            theta_val, wp = self.closest_point_on_path(trajectory, center, total_length, sample_interval)
            theta_list.append(theta_val)
            gate_interp.append(wp)
        return np.asarray(theta_list), np.asarray(gate_interp)




class MPCC(Controller):
    """Model Predictive Contouring Control for drone racing."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self._cfg = config

        self._dyn_params = load_params("so_rpy_rotor_drag", config.sim.drone_model)
        mass_val = float(self._dyn_params["mass"])
        gravity_mag = -float(self._dyn_params["gravity_vec"][-1])

        self.hover_thrust = mass_val * gravity_mag

        self._initial_pos = obs["pos"]
        self._cached_gate_centers = obs["gates_pos"]
        self._planned_duration = 30.0

        self._path_utils = PathTools()

        
        self._rebuild_nominal_path_gate(obs)

        
        self.N = 35
        self.T_HORIZON = 0.7
        self.dt = self.T_HORIZON / self.N
        self.model_arc_length = 0.05
        self.model_traj_length = 12.0

        self.arc_trajectory = self._path_utils.reparametrize_by_arclength(
            self._path_utils.extend_spline_tail(self.trajectory, extend_length=self.model_traj_length)
        )

        self.acados_ocp_solver, self.ocp = self._build_ocp_and_solver(
            self.T_HORIZON, self.N, self.arc_trajectory
        )

        self.pos_bound = [
            np.array([-2.6, 2.6]),
            np.array([-2.0, 1.8]),
            np.array([-0.1, 2.0]),
        ]
        self.velocity_bound = [-1.0, 4.0]

        self.last_theta = 0.0
        self.last_v_theta = 0.0

        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self.finished = False

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

    # ------------------------------------------------------------------
    # MPCC cost（门 / 障碍物减速 + 贴轨权重分开）
    # ------------------------------------------------------------------

    def _piecewise_linear_interp(self, theta, theta_vec, flattened_points, dim: int = 3):
        M = len(theta_vec)
        idx_float = (theta - theta_vec[0]) / (theta_vec[-1] - theta_vec[0]) * (M - 1)

        idx_low = floor(idx_float)
        idx_high = idx_low + 1
        alpha = idx_float - idx_low

        idx_low = if_else(idx_low < 0, 0, idx_low)
        idx_high = if_else(idx_high >= M, M - 1, idx_high)

        p_low = vertcat(*[flattened_points[dim * idx_low + i] for i in range(dim)])
        p_high = vertcat(*[flattened_points[dim * idx_high + i] for i in range(dim)])

        return (1.0 - alpha) * p_low + alpha * p_high

    def _encode_traj_params(self, trajectory: CubicSpline) -> np.ndarray:
        """
        生成：
        - pd_vals: 参考轨迹点
        - tp_vals: 切向速度
        - qc_gate: 靠近门的权重（强）
        - qc_obst: 靠近障碍物的权重（弱）
        """
        theta_samples = np.arange(0.0, self.model_traj_length, self.model_arc_length)

        pd_vals = trajectory(theta_samples)               # (M, 3)
        tp_vals = trajectory.derivative(1)(theta_samples)

        qc_gate = np.zeros_like(theta_samples, dtype=float)
        qc_obst = np.zeros_like(theta_samples, dtype=float)

        # —— 门：距离门越近，权重越大 —— 
        if hasattr(self, "_cached_gate_centers"):
            for gate_center in self._cached_gate_centers:
                d_gate = np.linalg.norm(pd_vals - gate_center, axis=-1)
                # 衰减比较快，主要在门附近起作用
                qc_gate = np.maximum(qc_gate, np.exp(-2.0 * d_gate**2))

        # —— 障碍物：只看 XY 距离，作用范围稍大，强度略小 —— 
        if hasattr(self, "_cached_obstacles"):
            for obst_center in self._cached_obstacles:
                d_obs_xy = np.linalg.norm(pd_vals[:, :2] - obst_center[:2], axis=-1)
                qc_obst = np.maximum(qc_obst, 0.7 * np.exp(-1.0 * d_obs_xy**2))

        return np.concatenate(
            [
                pd_vals.reshape(-1),
                tp_vals.reshape(-1),
                qc_gate,
                qc_obst,
            ]
        )

    def _stage_cost_expression(self):
        """
        MPCC stage cost：
        - e_lag, e_contour
        - 姿态 roll/pitch/yaw 正则
        - 控制平滑：df_cmd, dr_cmd, dp_cmd, dy_cmd
        - 进度 v_theta_cmd + 靠近门 / 障碍物时减速
        """
        position_vec = vertcat(self.px, self.py, self.pz)
        att_vec = vertcat(self.roll, self.pitch, self.yaw)
        ctrl_vec = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)

        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_length)

        pd_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)

        # 门 / 障碍物权重
        qc_gate_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_gate, dim=1)
        qc_obst_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_obst, dim=1)

        tp_unit = tp_theta / (norm_2(tp_theta) + 1e-6)
        e_theta = position_vec - pd_theta
        e_lag = dot(tp_unit, e_theta) * tp_unit
        e_contour = e_theta - e_lag

        # 轨迹跟踪：门权重大于障碍物
        track_cost = (
            (self.q_l
             + self.q_l_gate_peak * qc_gate_theta
             + self.q_l_obst_peak * qc_obst_theta) * dot(e_lag, e_lag)
            + (self.q_c
               + self.q_c_gate_peak * qc_gate_theta
               + self.q_c_obst_peak * qc_obst_theta) * dot(e_contour, e_contour)
            + att_vec.T @ self.Q_w @ att_vec
        )

        smooth_cost = ctrl_vec.T @ self.R_df @ ctrl_vec

        # 速度：基础项鼓励前进，靠近门 / 障碍物时给 v_theta_cmd^2 加惩罚（门更强）
        speed_cost = (
            - self.miu * self.v_theta_cmd
            + self.w_v_gate * qc_gate_theta * (self.v_theta_cmd ** 2)
            + self.w_v_obst * qc_obst_theta * (self.v_theta_cmd ** 2)
        )

        return track_cost + smooth_cost + speed_cost

    def _build_ocp_and_solver(
        self, Tf: float, N_horizon: int, trajectory: CubicSpline, verbose: bool = False
    ) -> tuple[AcadosOcpSolver, AcadosOcp]:
        ocp = AcadosOcp()
        model = self._export_dynamics_model()
        ocp.model = model

        self.nx = model.x.rows()
        self.nu = model.u.rows()
        ocp.solver_options.N_horizon = N_horizon

        ocp.cost.cost_type = "EXTERNAL"

        # --------- 权重设置（可以再微调） ----------
        self.q_l = 200
        self.q_c = 100
        self.Q_w = 1 * DM(np.eye(3))

        # 门附近：贴轨更硬
        self.q_l_gate_peak = 640
        self.q_c_gate_peak = 800

        # 障碍物附近：贴轨也加强，但稍微弱一点
        self.q_l_obst_peak = 100
        self.q_c_obst_peak = 400

        self.R_df = DM(np.diag([0.1, 0.5, 0.5, 0.5]))

        # 进度项基础奖励
        self.miu = 8.0
        # 门减速强一点
        self.w_v_gate = 4.0
        # 障碍物减速弱一点
        self.w_v_obst = 1.0

        ocp.model.cost_expr_ext_cost = self._stage_cost_expression()

        # --- 状态约束：命令状态 ---
        thrust_min = float(self._dyn_params["thrust_min"]) * 4.0
        thrust_max = float(self._dyn_params["thrust_max"]) * 4.0

        # 状态顺序: [X_phys(12), r_cmd_state(12), p_cmd_state(13),
        #            y_cmd_state(14), f_cmd_state(15), theta(16)]
        ocp.constraints.lbx = np.array([thrust_min, -1.57, -1.57, -1.57])
        ocp.constraints.ubx = np.array([thrust_max, 1.57, 1.57, 1.57])
        ocp.constraints.idxbx = np.array([15, 12, 13, 14])

        # 输入约束
        ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, 0.0])
        ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0, 4.0])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

        ocp.constraints.x0 = np.zeros(self.nx)

        param_vec = self._encode_traj_params(self.arc_trajectory)
        ocp.parameter_values = param_vec

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tol = 1e-5
        ocp.solver_options.qp_solver_cond_N = N_horizon
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.qp_solver_iter_max = 20
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.tf = Tf

        solver = AcadosOcpSolver(ocp, json_file="mpcc_prescripted_real_dyn.json", verbose=verbose)
        return solver, ocp

    # ------------- trajectory planning & obstacle handling -------------

    def _rebuild_nominal_path_gate(self, obs: dict[str, NDArray[np.floating]]):
        print(f"T={self._step_count / self._ctrl_freq:.2f}: (Re)building path with VIRTUAL GATES...")

        gate_positions = obs["gates_pos"]
        obstacle_positions = obs["obstacles_pos"] 
        gate_quats = obs["gates_quat"]

        self._cached_gate_centers = gate_positions
        self._cached_obstacles = obstacle_positions

        gate_normals, gate_y, gate_z = self._extract_gate_frames(gate_quats)

        # 1. 基础路径 (过门心)
        base_waypoints = self._path_utils.build_gate_waypoints(
            self._initial_pos, gate_positions, gate_normals
        )
        # (可选) 抬高一点高度防止碰地
        if base_waypoints.shape[0] > 1:
            base_waypoints[1:, 2] += 0.0

        # 2. 几何倒圆角 (Detours)
        with_gate_detours = self._insert_gate_detours(
            base_waypoints, gate_positions, gate_normals, gate_y, gate_z,
        )
        
        # 3. 生成虚拟门框障碍物
        virt_pos, virt_types, virt_vecs, virt_lens = self._get_virtual_gate_obstacles(
            gate_positions, gate_quats, gate_width=0.7, gate_height=0.7
        )

        # 4. 合并真实障碍物和虚拟障碍物
        if len(obstacle_positions) > 0:
            n_real = len(obstacle_positions)
            # 真实障碍物默认为圆柱，半径0.3
            pos_real = obstacle_positions
            types_real = np.full(n_real, ObstacleType.CYLINDER_2D, dtype=int)
            vecs_real = np.zeros((n_real, 3))
            lens_real = np.zeros(n_real)
            margins_real = np.full(n_real, 0.3) # 真实障碍物安全距离

            # 虚拟门框安全距离 (可以设小一点，例如0.15，因为我们要紧贴穿过)
            margins_virt = np.full(len(virt_pos), 0.25)

            all_pos = np.vstack([pos_real, virt_pos])
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

        # 5. 调用新的避障函数 (传入列表参数)
        t_axis, collision_free_wps = self._inject_obstacle_detours(
            with_gate_detours, 
            all_pos, 
            all_margins,
            all_types,
            all_vecs,
            all_lens
        )


        if len(t_axis) < 2:
            print("[MPCC] Warning: fallback to raw path.")
            self.trajectory = self._path_utils.spline_through_points(self._planned_duration, with_gate_detours)
        else:
            self.trajectory = CubicSpline(t_axis, collision_free_wps)
            self._planned_duration = float(self.trajectory.x[-1])

    def _rebuild_nominal_path_obstacle(self, obs: dict[str, NDArray[np.floating]]):
        print(f"T={self._step_count / self._ctrl_freq:.2f}: (Re)building nominal path (obstacle)...")

        gate_positions = obs["gates_pos"]
        obstacle_positions = obs["obstacles_pos"]
        gate_quats = obs["gates_quat"]
        start_pos = obs["pos"]

        self._cached_gate_centers = gate_positions
        self._cached_obstacles = obstacle_positions

        gate_normals, gate_y, gate_z = self._extract_gate_frames(gate_quats)

        base_waypoints = self._path_utils.build_gate_waypoints(
            start_pos, gate_positions, gate_normals
        )

        altitude_offset = 0.0
        if base_waypoints.shape[0] > 1:
            base_waypoints[1:, 2] += altitude_offset

        with_gate_detours = self._insert_gate_detours(
            base_waypoints,
            gate_positions,
            gate_normals,
            gate_y,
            gate_z,
        )

        t_axis, collision_free_wps = self._inject_obstacle_detours(
            with_gate_detours, obstacle_positions, safe_dist=0.15
        )

        if len(t_axis) < 2:
            print("[MPCC] Warning: obstacle-avoid path fallback (too few points).")
            self.trajectory = self._path_utils.spline_through_points(self._planned_duration, with_gate_detours)
        else:
            self.trajectory = CubicSpline(t_axis, collision_free_wps)
            self._planned_duration = float(self.trajectory.x[-1])
            
    def _get_virtual_gate_obstacles(
        self,
        gate_positions: NDArray[np.floating],
        gate_quats: NDArray[np.floating],
        gate_width: float = 0.7,   
        gate_height: float = 0.7
    ) -> tuple[NDArray[np.floating], NDArray[np.int_], NDArray[np.floating], NDArray[np.floating]]:
        
        # 使用 mpcc_9 中已有的 FrameUtils
        gate_y_axes = FrameUtils.quat_to_axis(gate_quats, axis_index=1) # 横向
        gate_z_axes = FrameUtils.quat_to_axis(gate_quats, axis_index=2) # 垂直

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

            # --- A. 左右门柱 (Side Posts) ---
            # 防止从侧面或上方撞柱
            for sign in [1.0, -1.0]:
                post_pos = c + sign * half_w * y
                obs_positions.append(post_pos)
                obs_types.append(ObstacleType.CYLINDER_2D)
                obs_vecs.append(np.zeros(3)) 
                obs_lens.append(0.0)         

            # --- B. 上下横梁 (Top & Bottom Bars) ---
            # 保护门框边缘，允许从中间穿过
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
        obstacles_pos: NDArray[np.floating],
        safe_dist_list: NDArray[np.floating], # 注意这里改为接收列表
        types_list: NDArray[np.int_],         # 新增：类型
        vecs_list: NDArray[np.floating],      # 新增：向量
        lens_list: NDArray[np.floating],      # 新增：长度
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        
        pre_spline = self._path_utils.spline_through_points(self._planned_duration, base_waypoints)
        n_samples = int(self._ctrl_freq * self._planned_duration)
        if n_samples <= 0: n_samples = 1

        t_axis = np.linspace(0.0, self._planned_duration, n_samples)
        wp_samples = pre_spline(t_axis)

        # 遍历所有障碍物（混合了真实障碍物和虚拟门框）
        for obst_c, safe_dist, o_type, o_vec, o_len in zip(
            obstacles_pos, safe_dist_list, types_list, vecs_list, lens_list
        ):
            inside_region = False
            new_t = []
            new_pts = []
            
            idx_in = -1
            idx_out = -1

            
            for idx in range(wp_samples.shape[0]):
                pt = wp_samples[idx]
                
               
                if o_type == ObstacleType.CYLINDER_2D:
                    dist = np.linalg.norm(obst_c[:2] - pt[:2])
                elif o_type == ObstacleType.CAPSULE_3D:
                    vec_cp = pt - obst_c
                    proj = np.dot(vec_cp, o_vec)
                    proj_clamped = np.clip(proj, -o_len, o_len)
                    closest_pt = obst_c + proj_clamped * o_vec
                    dist = np.linalg.norm(pt - closest_pt)
                else:
                    dist = np.linalg.norm(obst_c - pt)


                if dist < safe_dist and not inside_region:
                    inside_region = True
                    idx_in = idx
                elif dist >= safe_dist and inside_region:
                    inside_region = False
                    idx_out = idx
                    
                    # 3. 处理冲突段：推离 (Push-out)
                    p_in = wp_samples[idx_in]
                    p_out = wp_samples[idx_out]
                    p_mid = 0.5 * (p_in + p_out) # 简单取中点，也可优化

                    # 计算推力方向
                    if o_type == ObstacleType.CYLINDER_2D:
                        push_vec = p_mid - obst_c
                        push_vec[2] = 0.0 # 柱子不推高度
                    elif o_type == ObstacleType.CAPSULE_3D:
                        vec_cp = p_mid - obst_c
                        proj = np.clip(np.dot(vec_cp, o_vec), -o_len, o_len)
                        closest_on_seg = obst_c + proj * o_vec
                        push_vec = p_mid - closest_on_seg # 从横梁最近点推开
                    else:
                        push_vec = p_mid - obst_c

                    norm_push = np.linalg.norm(push_vec)
                    if norm_push < 1e-6: push_dir = np.array([0,0,1.0]) # 防止重合
                    else: push_dir = push_vec / norm_push

                    # 生成 Detour 点
                    if o_type == ObstacleType.CYLINDER_2D:
                        detour_xy = obst_c[:2] + push_dir[:2] * safe_dist
                        # 高度保持原路径趋势
                        detour_z = 0.5 * (p_in[2] + p_out[2])
                        detour_pt = np.concatenate([detour_xy, [detour_z]])
                    else:
                        # 3D 物体：从最近点向外推
                        # 重新计算基准点
                        vec_cp = p_mid - obst_c
                        proj = np.clip(np.dot(vec_cp, o_vec), -o_len, o_len) if o_type == ObstacleType.CAPSULE_3D else 0
                        base_pt = obst_c + proj * o_vec
                        detour_pt = base_pt + push_dir * safe_dist

                    # 插入 Detour 点 (取中间时刻)
                    mid_t = 0.5 * (t_axis[idx_in] + t_axis[idx_out])
                    new_t.append(mid_t)
                    new_pts.append(detour_pt)
                    
                elif dist >= safe_dist:
                    new_t.append(t_axis[idx])
                    new_pts.append(pt)
            
            # 处理结尾还在里面的情况
            if inside_region:
                new_t.append(t_axis[-1])
                new_pts.append(wp_samples[-1])

            t_axis = np.asarray(new_t)
            wp_samples = np.asarray(new_pts)

        if t_axis.size > 1:
            _, uniq = np.unique(t_axis, return_index=True)
            return t_axis[uniq], wp_samples[uniq]
        
        # Fallback
        return np.array([]), np.array([])
  

    def _detect_event_change_gate(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        if not hasattr(self, "_last_gate_flags"):
            self._last_gate_flags = np.array(obs.get("gates_visited", []), dtype=bool)
            self._last_obst_flags = np.array(obs.get("obstacles_visited", []), dtype=bool)
            return False

        curr_gates = np.array(obs.get("gates_visited", []), dtype=bool)
        curr_obst = np.array(obs.get("obstacles_visited", []), dtype=bool)

        if curr_gates.shape != self._last_gate_flags.shape:
            self._last_gate_flags = curr_gates
            return False

        gate_trigger = np.any((~self._last_gate_flags) & curr_gates)
        obst_trigger = np.any((~self._last_obst_flags) & curr_obst)

        self._last_gate_flags = curr_gates
        self._last_obst_flags = curr_obst

        return bool(gate_trigger or obst_trigger)

    def _detect_event_change_obstacle(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        if not hasattr(self, "_last_gate_flags"):
            self._last_gate_flags = np.array(obs.get("gates_visited", []), dtype=bool)
            self._last_obst_flags = np.array(obs.get("obstacles_visited", []), dtype=bool)
            return False

        curr_gates = np.array(obs.get("gates_visited", []), dtype=bool)
        curr_obst = np.array(obs.get("obstacles_visited", []), dtype=bool)

        if curr_obst.shape != self._last_obst_flags.shape:
            self._last_obst_flags = curr_obst
            return False

        gate_trigger = np.any((~self._last_gate_flags) & curr_gates)
        obst_trigger = np.any((~self._last_obst_flags) & curr_obst)

        self._last_gate_flags = curr_gates
        self._last_obst_flags = curr_obst

        return bool(gate_trigger or obst_trigger)

    def _extract_gate_frames(
        self, gates_quaternions: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        normals = FrameUtils.quat_to_axis(gates_quaternions, axis_index=0)
        y_axes = FrameUtils.quat_to_axis(gates_quaternions, axis_index=1)
        z_axes = FrameUtils.quat_to_axis(gates_quaternions, axis_index=2)
        return normals, y_axes, z_axes

    def _insert_gate_detours(
        self,
        waypoints: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
        num_intermediate_points: int = 5,
        angle_threshold: float = 120.0,
        detour_distance: float = 0.65,
    ) -> NDArray[np.floating]:

        n_gates = gate_positions.shape[0]
        wp_list = list(waypoints)
        extra_inserted = 0

        for gate_idx in range(n_gates - 1):
            last_idx_curr_gate = 1 + (gate_idx + 1) * num_intermediate_points - 1 + extra_inserted
            first_idx_next_gate = 1 + (gate_idx + 1) * num_intermediate_points + extra_inserted

            if last_idx_curr_gate >= len(wp_list) or first_idx_next_gate >= len(wp_list):
                break

            p_curr = wp_list[last_idx_curr_gate]
            p_next = wp_list[first_idx_next_gate]
            delta_vec = p_next - p_curr
            delta_norm = np.linalg.norm(delta_vec)
            if delta_norm < 1e-6:
                continue

            normal_i = gate_normals[gate_idx]
            cos_ang = np.dot(delta_vec, normal_i) / delta_norm
            cos_ang = np.clip(cos_ang, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_ang))

            if angle_deg > angle_threshold:
                gate_center = gate_positions[gate_idx]
                y_axis = gate_y_axes[gate_idx]
                z_axis = gate_z_axes[gate_idx]

                tangential = delta_vec - np.dot(delta_vec, normal_i) * normal_i
                tangential_norm = np.linalg.norm(tangential)

                if tangential_norm < 1e-6:
                    detour_dir = y_axis
                else:
                    tangential /= tangential_norm
                    proj_y = np.dot(tangential, y_axis)
                    proj_z = np.dot(tangential, z_axis)
                    proj_angle = np.degrees(np.arctan2(proj_z, proj_y))

                    if -90.0 <= proj_angle < 45.0:
                        detour_dir = y_axis
                    elif 45.0 <= proj_angle < 135.0:
                        detour_dir = z_axis
                    else:
                        detour_dir = -y_axis

                detour_wp = gate_center + detour_distance * detour_dir
                insert_idx = last_idx_curr_gate + 1
                wp_list.insert(insert_idx, detour_wp)
                extra_inserted += 1

        return np.asarray(wp_list)

    # ------------------- safety check helpers -------------------

    def _pos_outside_limits(self, pos: NDArray[np.floating]) -> bool:
        if self.pos_bound is None:
            return False
        for i_dim in range(3):
            low, high = self.pos_bound[i_dim]
            if pos[i_dim] < low or pos[i_dim] > high:
                return True
        return False

    def _speed_outside_limits(self, vel: NDArray[np.floating]) -> bool:
        if self.velocity_bound is None:
            return False
        speed = np.linalg.norm(vel)
        return not (self.velocity_bound[0] < speed < self.velocity_bound[1])

  

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:

        self._current_obs_pos = obs["pos"]

        # 事件触发重规划（保持你现在的逻辑）
        if self._detect_event_change_gate(obs):
            print(f"T={self._step_count / self._ctrl_freq:.2f}: MPCC detected gate/env change, replanning...")
            self._rebuild_nominal_path_gate(obs)
            self.arc_trajectory = self._path_utils.reparametrize_by_arclength(
                self._path_utils.extend_spline_tail(
                    self.trajectory, extend_length=self.model_traj_length
                )
            )
            param_vec = self._encode_traj_params(self.arc_trajectory)
            for k in range(self.N + 1):
                self.acados_ocp_solver.set(k, "p", param_vec)

        if self._detect_event_change_obstacle(obs):
            print(f"T={self._step_count / self._ctrl_freq:.2f}: MPCC detected obstacle/env change, replanning...")
            self._rebuild_nominal_path_obstacle(obs)
            self.arc_trajectory = self._path_utils.reparametrize_by_arclength(
                self._path_utils.extend_spline_tail(
                    self.trajectory, extend_length=self.model_traj_length
                )
            )
            param_vec = self._encode_traj_params(self.arc_trajectory)
            for k in range(self.N + 1):
                self.acados_ocp_solver.set(k, "p", param_vec)

        quat = obs["quat"]
        r_obj = R.from_quat(quat)
        roll_pitch_yaw = r_obj.as_euler("xyz", degrees=False)

        if "ang_vel" in obs:
            drpy = ang_vel2rpy_rates(quat, obs["ang_vel"])
        else:
            drpy = np.zeros(3, dtype=float)

        # X_phys: [pos(3), rpy(3), vel(3), drpy(3)]
        X_phys_now = np.concatenate(
            (obs["pos"], roll_pitch_yaw, obs["vel"], drpy)
        )

        # 全状态: [X_phys(12), r_cmd_state, p_cmd_state, y_cmd_state, f_cmd_state, theta]
        x_now = np.concatenate(
            (
                X_phys_now,
                self.last_rpy_cmd,           # 命令状态 warm start
                np.array([self.last_f_cmd]),
                np.array([self.last_theta]),
            )
        )

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

        if self.last_theta >= float(self.arc_trajectory.x[-1]):
            self.finished = True
            print("[MPCC] Stop: finished path.")
        if self._pos_outside_limits(obs["pos"]):
            self.finished = True
            print("[MPCC] Stop: position out of safe bounds.")
        if self._speed_outside_limits(obs["vel"]):
            self.finished = True
            print("[MPCC] Stop: velocity out of safe range.")

        status = self.acados_ocp_solver.solve()
        if status != 0:
            print("[MPCC] acados solver returned non-zero status:", status)

        self._x_warm = [self.acados_ocp_solver.get(i, "x") for i in range(self.N + 1)]
        self._u_warm = [self.acados_ocp_solver.get(i, "u") for i in range(self.N)]

        x_next = self.acados_ocp_solver.get(1, "x")

        # 取出命令状态：索引 [12:15] = r,p,y cmd, [15] = f_cmd_state, [16] = theta
        self.last_rpy_cmd = np.array(x_next[12:15]).copy()
        self.last_f_cmd = float(x_next[15])
        self.last_f_collective = self.last_f_cmd
        self.last_theta = float(x_next[16])

        cmd = np.array(
            [
                self.last_rpy_cmd[0],
                self.last_rpy_cmd[1],
                self.last_rpy_cmd[2],
                self.last_f_cmd,
            ],
            dtype=float,
        )

        print(
            f"cmd: roll={cmd[0]:.3f}, pitch={cmd[1]:.3f}, yaw={cmd[2]:.3f}, thrust={cmd[3]:.3f}"
        )

        self._step_count += 1
        return cmd

    # --------------------- 回调 & debug ---------------------

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        return self.finished

    def episode_callback(self):
        print("[MPCC] Episode reset.")
        self._step_count = 0
        self.finished = False

        for attr in ["_last_gate_flags", "_last_obst_flags", "_x_warm", "_u_warm", "_current_obs_pos"]:
            if hasattr(self, attr):
                delattr(self, attr)

        self.last_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)

    def get_debug_lines(self):
        debug_lines = []

        if hasattr(self, "arc_trajectory"):
            try:
                full_path = self.arc_trajectory(self.arc_trajectory.x)
                debug_lines.append(
                    (full_path, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0)
                )
            except Exception:
                pass

        if hasattr(self, "_x_warm"):
            pred_states = np.asarray([x_state[:3] for x_state in self._x_warm])
            debug_lines.append(
                (pred_states, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0)
            )

        if (
            hasattr(self, "last_theta")
            and hasattr(self, "arc_trajectory")
            and hasattr(self, "_current_obs_pos")
        ):
            try:
                target_on_path = self.arc_trajectory(self.last_theta)
                segment = np.stack([self._current_obs_pos, target_on_path])
                debug_lines.append(
                    (segment, np.array([0.0, 0.0, 1.0, 1.0]), 1.0, 1.0)
                )
            except Exception:
                pass

        return debug_lines
