from __future__ import annotations

from typing import TYPE_CHECKING, List

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

 
# ----------------------------- 工具类 -----------------------------
# so_rpy_rotor & 把 mpcc_9 加上与 mpcc_4 同等复杂度的执行器滞后 + 速率限制


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


class VectorMath:

    @staticmethod
    def normalize(vec: NDArray[np.floating]) -> NDArray[np.floating]:
        nrm = np.linalg.norm(vec)
        return vec if nrm < 1e-6 else vec / nrm

    @staticmethod
    def bounded_dot(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
        return float(np.clip(np.dot(a, b), -1.0, 1.0))


class RootSolver:

    @staticmethod
    def cubic_real(a: np.floating, b: np.floating, c: np.floating, d: np.floating) -> List[np.float64]:
        roots = np.roots(np.array([a, b, c, d], dtype=np.float64))
        return [r.real for r in roots if np.isclose(r.imag, 0.0)]

    @staticmethod
    def quartic_real(
        a: np.floating, b: np.floating, c: np.floating, d: np.floating, e: np.floating
    ) -> List[np.float64]:
        roots = np.roots(np.array([a, b, c, d, e], dtype=np.float64))
        return [r.real for r in roots if np.isclose(r.imag, 0.0)]


class CompositeSpline:

    trajectory_1: CubicSpline
    trajectory_2: CubicSpline
    offset: np.floating
    x: NDArray[np.floating]

    def __init__(self, first: CubicSpline, second: CubicSpline, offset: np.floating):
        self.trajectory_1 = first
        self.trajectory_2 = second
        self.offset = offset
        self.x = np.concatenate([first.x, second.x + offset])

    def __call__(self, t):
        if np.isscalar(t):
            return self.trajectory_1(t) if t < self.offset else self.trajectory_2(t - self.offset)
        return np.array([self(t_i) for t_i in t])

    def derivative(self, order: int):
        return CompositeSpline(
            self.trajectory_1.derivative(order),
            self.trajectory_2.derivative(order),
            self.offset,
        )


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

    def preprocess_two_stage_trajectory(self, t: np.ndarray, pos: np.ndarray) -> CompositeSpline:
        idx_peak = 20 + int(np.argmax(np.asarray(pos)[20:, 1]))
        t = np.asarray(t)

        t_first, p_first = t[: idx_peak + 1], pos[: idx_peak + 1]
        t_second, p_second = t[idx_peak:] - t[idx_peak], pos[idx_peak:]

        spline_1 = CubicSpline(t_first, p_first)
        spline_2 = CubicSpline(t_second, p_second)

        arc_spline_1 = self.reparametrize_by_arclength(spline_1)
        arc_spline_2 = self.reparametrize_by_arclength(spline_2)

        arc_spline_1_cut = CubicSpline(arc_spline_1.x[:-1], arc_spline_1(arc_spline_1.x[:-1]))
        return CompositeSpline(arc_spline_1_cut, arc_spline_2, arc_spline_1.x[-1])

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


class VolumeInterp:
    """Trilinear interpolation over a regular 3D grid."""

    @staticmethod
    def trilinear(grid: np.ndarray, float_idx: np.ndarray) -> float:
        x_f, y_f, z_f = float_idx
        x0, y0, z0 = np.floor([x_f, y_f, z_f]).astype(int)
        dx, dy, dz = x_f - x0, y_f - y0, z_f - z0

        def safe_get(ix: int, iy: int, iz: int) -> float:
            if 0 <= ix < grid.shape[0] and 0 <= iy < grid.shape[1] and 0 <= iz < grid.shape[2]:
                return float(grid[ix, iy, iz])
            return 0.0

        c000 = safe_get(x0, y0, z0)
        c001 = safe_get(x0, y0, z0 + 1)
        c010 = safe_get(x0, y0 + 1, z0)
        c011 = safe_get(x0, y0 + 1, z0 + 1)
        c100 = safe_get(x0 + 1, y0, z0)
        c101 = safe_get(x0 + 1, y0, z0 + 1)
        c110 = safe_get(x0 + 1, y0 + 1, z0)
        c111 = safe_get(x0 + 1, y0 + 1, z0 + 1)

        c00 = c000 * (1 - dx) + c100 * dx
        c01 = c001 * (1 - dx) + c101 * dx
        c10 = c010 * (1 - dx) + c110 * dx
        c11 = c011 * (1 - dx) + c111 * dx

        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy
        return float(c0 * (1 - dz) + c1 * dz)


# ----------------------------- MPCC 控制器（真实动力学 + 原 cost） -----------------------------


class MPCC(Controller):
    """Model Predictive Contouring Control for drone racing."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self._cfg = config

        self._dyn_params = load_params("so_rpy_rotor", config.sim.drone_model)
        mass_val = float(self._dyn_params["mass"])
        gravity_mag = -float(self._dyn_params["gravity_vec"][-1])

        self.hover_thrust = mass_val * gravity_mag
        # --- 执行器/内环动态（用于“命令 -> 实际作用量”的一阶滞后）---
        # tau 越小响应越快（更偏 racing），越大越“肉”但更接近有滞后的真实系统
        self.tau_rpy_act = 0.05   # roll/pitch 命令到实际的时间常数 [s]
        self.tau_yaw_act = 0.08   # yaw 命令到实际的时间常数 [s]
        self.tau_f_act = 0.10     # thrust 命令到实际的时间常数 [s]

        # --- 速率限制（对输入 u=[df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta_cmd] 的 bounds）---
        self.rate_limit_df = 10.0        # thrust 命令变化率上限
        self.rate_limit_drpy = 10.0      # r/p/y 命令变化率上限
        self.rate_limit_v_theta = 4.0    # 路径进度速度上限


        self._initial_pos = obs["pos"]
        self._cached_gate_centers = obs["gates_pos"]
        self._planned_duration = 30.0

        self._path_utils = PathTools()

        # 初始名义轨迹
        self._rebuild_nominal_path_gate(obs)

        # MPC 配置
        self.N = 35       #55
        self.T_HORIZON = 0.7  #0.7
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
        self.last_f_act = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self.last_rpy_act = np.zeros(3)
        self.finished = False

    # ------------------------------------------------------------------
    # 使用真实动力学 symbolic_dynamics_euler + 命令积分器
    # ------------------------------------------------------------------
    def _export_dynamics_model(self) -> AcadosModel:
        """
        使用 drone_models.so_rpy_rotor.symbolic_dynamics_euler 的真实动力学。

        X_phys: so_rpy_rotor 返回的物理状态（含桨速），长度 self.nx_phys
        U_phys: 物理控制输入 [r_cmd, p_cmd, y_cmd, f_cmd]

        在外面再加 4 个“命令状态” + 1 个 theta:
        X = [X_phys, r_cmd_state, p_cmd_state, y_cmd_state, f_cmd_state, r_act, p_act, y_act, f_act, theta]
        U = [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta_cmd]
        """

        model_name = "lsy_example_mpc_real"

        params = self._dyn_params

        # 真实动力学
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
            thrust_time_coef=params["thrust_time_coef"],
        )

        # 物理状态维度（so_rpy_rotor 里通常 > 12）
        self.nx_phys = X_phys.shape[0]

        # 物理状态别名（只用到前 12 个姿态 + 速度）
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

        # 执行器/内环输出状态（真实动力学使用的“实际作用量”）
        self.r_act = MX.sym("r_act")
        self.p_act = MX.sym("p_act")
        self.y_act = MX.sym("y_act")
        self.f_act = MX.sym("f_act")

        # 路径进度 theta
        self.theta = MX.sym("theta")

        # 控制量：和原 cost 完全相同
        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")
        self.v_theta_cmd = MX.sym("v_theta_cmd")

        # 拼状态向量
        states = vertcat(
            X_phys,
            self.r_cmd_state,
            self.p_cmd_state,
            self.y_cmd_state,
            self.f_cmd_state,
            self.r_act,
            self.p_act,
            self.y_act,
            self.f_act,
            self.theta,
        )
        inputs = vertcat(
            self.df_cmd,
            self.dr_cmd,
            self.dp_cmd,
            self.dy_cmd,
            self.v_theta_cmd,
        )

        # 记录命令状态 / 执行器状态 / theta 的索引（方便后面读写）
        self.idx_r_cmd_state = int(self.nx_phys + 0)
        self.idx_p_cmd_state = int(self.nx_phys + 1)
        self.idx_y_cmd_state = int(self.nx_phys + 2)
        self.idx_f_cmd_state = int(self.nx_phys + 3)

        self.idx_r_act = int(self.nx_phys + 4)
        self.idx_p_act = int(self.nx_phys + 5)
        self.idx_y_act = int(self.nx_phys + 6)
        self.idx_f_act = int(self.nx_phys + 7)

        self.idx_theta = int(self.nx_phys + 8)

        # 真实动力学的控制输入由“执行器/内环输出状态”给出
        U_phys_full = vertcat(
            self.r_act,
            self.p_act,
            self.y_act,
            self.f_act,
        )

        # 用 casadi.substitute 把原本的 U_phys 换成 U_phys_full
        f_dyn_phys = substitute(X_dot_phys, U_phys, U_phys_full)

        # 命令状态一阶积分
        r_cmd_dot = self.dr_cmd
        p_cmd_dot = self.dp_cmd
        y_cmd_dot = self.dy_cmd
        f_cmd_dot = self.df_cmd


        # 执行器一阶滞后：act 跟随 cmd_state
        tau_rpy_act = float(self.tau_rpy_act)
        tau_yaw_act = float(self.tau_yaw_act)
        tau_f_act = float(self.tau_f_act)

        r_act_dot = (self.r_cmd_state - self.r_act) / tau_rpy_act
        p_act_dot = (self.p_cmd_state - self.p_act) / tau_rpy_act
        y_act_dot = (self.y_cmd_state - self.y_act) / tau_yaw_act
        f_act_dot = (self.f_cmd_state - self.f_act) / tau_f_act
        theta_dot = self.v_theta_cmd

        f_dyn = vertcat(
            f_dyn_phys,
            r_cmd_dot,
            p_cmd_dot,
            y_cmd_dot,
            f_cmd_dot,
            r_act_dot,
            p_act_dot,
            y_act_dot,
            f_act_dot,
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

        qc_gate_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_gate, dim=1)
        qc_obst_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_obst, dim=1)

        tp_unit = tp_theta / (norm_2(tp_theta) + 1e-6)
        e_theta = position_vec - pd_theta
        e_lag = dot(tp_unit, e_theta) * tp_unit
        e_contour = e_theta - e_lag

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
        self.q_l = 522.327621281147#200
        self.q_c = 279.45878291502595 #100
        self.Q_w = 1 * DM(np.eye(3))

        # 门附近：贴轨更硬
        self.q_l_gate_peak = 520.2687042765319 #640
        self.q_c_gate_peak = 764.3037075176835 #800

        # 障碍物附近：贴轨也加强，但稍微弱一点
        self.q_l_obst_peak = 207.83845749683678 #100
        self.q_c_obst_peak = 110.51885732449591 #50

        self.R_df = DM(np.diag([0.1, 0.5, 0.5, 0.5]))

        # 进度项基础奖励
        self.miu = 14.3377785384655 #8.0
        # 门减速强一点
        self.w_v_gate = 2.7327203765511516 #2.5
        # 障碍物减速弱一点
        self.w_v_obst = 2.460291111562401 #0.5

        ocp.model.cost_expr_ext_cost = self._stage_cost_expression()

        # --- 状态约束：同时约束 thrust 的“命令”和“实际”(一阶滞后输出) + r/p/y 命令 ---
        thrust_min = float(self._dyn_params["thrust_min"]) * 4.0
        thrust_max = float(self._dyn_params["thrust_max"]) * 4.0

        idx_r = self.idx_r_cmd_state
        idx_p = self.idx_p_cmd_state
        idx_y = self.idx_y_cmd_state
        idx_f_cmd = self.idx_f_cmd_state
        idx_f_act = self.idx_f_act

        ocp.constraints.lbx = np.array([thrust_min, thrust_min, -1.57, -1.57, -1.57])
        ocp.constraints.ubx = np.array([thrust_max, thrust_max, 1.57, 1.57, 1.57])
        ocp.constraints.idxbx = np.array([idx_f_act, idx_f_cmd, idx_r, idx_p, idx_y])

        # 输入约束
        ocp.constraints.lbu = np.array([-self.rate_limit_df, -self.rate_limit_drpy, -self.rate_limit_drpy, -self.rate_limit_drpy, 0.0])
        ocp.constraints.ubu = np.array([self.rate_limit_df, self.rate_limit_drpy, self.rate_limit_drpy, self.rate_limit_drpy, self.rate_limit_v_theta])
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
        print(f"T={self._step_count / self._ctrl_freq:.2f}: (Re)building nominal path (gate)...")

        gate_positions = obs["gates_pos"]
        obstacle_positions = obs["obstacles_pos"]
        gate_quats = obs["gates_quat"]
        start_pos = obs["pos"]

        self._cached_gate_centers = gate_positions
        self._cached_obstacles = obstacle_positions

        gate_normals, gate_y, gate_z = self._extract_gate_frames(gate_quats)

        base_waypoints = self._path_utils.build_gate_waypoints(
            self._initial_pos, gate_positions, gate_normals
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
            with_gate_detours, obstacle_positions, safe_dist=0.2
        )

        if len(t_axis) < 2:
            print("[MPCC] Warning: obstacle-avoid path fallback (too few points).")
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
            with_gate_detours, obstacle_positions, safe_dist=0.2
        )

        if len(t_axis) < 2:
            print("[MPCC] Warning: obstacle-avoid path fallback (too few points).")
            self.trajectory = self._path_utils.spline_through_points(self._planned_duration, with_gate_detours)
        else:
            self.trajectory = CubicSpline(t_axis, collision_free_wps)
            self._planned_duration = float(self.trajectory.x[-1])

    def _inject_obstacle_detours(
        self,
        base_waypoints: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
        safe_dist: float,
        arc_n: int = 5,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        避障时用圆弧 detour 替换离障碍物太近的一小段路径，使轨迹更平滑。
        同时避开“包含门附近”的路径段，尽量不破坏穿门轨迹。
        """

        # 先对原始 waypoint 拟合一条样条，方便均匀采样
        pre_spline = self._path_utils.spline_through_points(self._planned_duration, base_waypoints)

        n_samples = int(self._ctrl_freq * self._planned_duration)
        if n_samples <= 0:
            n_samples = 1

        t_axis = np.linspace(0.0, self._planned_duration, n_samples)
        wp_samples = pre_spline(t_axis)   # (N, 3)

        gate_margin = 3  # Gate 附近若干采样点范围内不做圆弧替换

        for obst in obstacles_pos:
            # 每次根据当前 wp_samples 重新估计门在路径上的索引
            gate_idx = np.array([], dtype=int)
            if hasattr(self, "_cached_gate_centers"):
                gates = np.asarray(self._cached_gate_centers)
                if gates.size > 0:
                    idx_list = []
                    for g in gates:
                        d_g = np.linalg.norm(wp_samples - g, axis=1)
                        idx_list.append(int(np.argmin(d_g)))
                    gate_idx = np.asarray(idx_list, dtype=int)

            # 计算每个采样点到障碍物的 XY 距离
            d_xy = np.linalg.norm(wp_samples[:, :2] - obst[:2], axis=1)
            inside = d_xy < safe_dist

            if not np.any(inside):
                # 这个障碍物对当前轨迹没有约束
                continue

            inside_idx = np.where(inside)[0]
            start_idx = int(inside_idx[0]) - 1
            end_idx = int(inside_idx[-1]) + 1

            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, len(t_axis) - 1)

            if end_idx <= start_idx + 1:
                # 有效区间太短，跳过
                continue

            # 如果这一段路径包含 gate 附近，就跳过这次圆弧替换，避免破坏穿门段
            if gate_idx.size > 0:
                if np.any((gate_idx >= start_idx - gate_margin) & (gate_idx <= end_idx + gate_margin)):
                    continue

            # 起止点
            p_start = wp_samples[start_idx]
            p_end = wp_samples[end_idx]

            # 起止向量（XY 平面），单位化
            v_start = p_start[:2] - obst[:2]
            v_end = p_end[:2] - obst[:2]
            nrm_s = np.linalg.norm(v_start)
            nrm_e = np.linalg.norm(v_end)
            if nrm_s < 1e-6 or nrm_e < 1e-6:
                # 非法几何情况，跳过这次绕障
                continue

            v_start /= nrm_s
            v_end /= nrm_e

            theta_start = np.arctan2(v_start[1], v_start[0])
            theta_end = np.arctan2(v_end[1], v_end[0])

            # 选择“较短”的圆弧方向
            d_theta = theta_end - theta_start
            if d_theta > np.pi:
                d_theta -= 2.0 * np.pi
            elif d_theta < -np.pi:
                d_theta += 2.0 * np.pi

            # 在起止角度之间线性插值 arc_n+2 个角度，剔除两端（起点终点已经保留）
            theta_list = np.linspace(theta_start, theta_start + d_theta, arc_n + 2)[1:-1]

            # 对应的 t 也线性插值
            t_start = t_axis[start_idx]
            t_end = t_axis[end_idx]
            t_list = np.linspace(t_start, t_end, arc_n + 2)[1:-1]

            detour_points = []
            for i, th in enumerate(theta_list):
                # 圆弧上的 XY
                dir_xy = np.array([np.cos(th), np.sin(th)], dtype=float)
                p_xy = obst[:2] + dir_xy * safe_dist

                # Z 用起止点线性插值
                alpha = (i + 1) / (arc_n + 1)
                z = (1.0 - alpha) * p_start[2] + alpha * p_end[2]

                detour_points.append(np.array([p_xy[0], p_xy[1], z], dtype=float))

            # 重新拼接整条路径：前段 + 圆弧 + 后段
            new_t_vals: List[float] = []
            new_points: List[np.ndarray] = []

            # 1) 起点到 start_idx
            for i in range(0, start_idx + 1):
                new_t_vals.append(t_axis[i])
                new_points.append(wp_samples[i])

            # 2) 圆弧 detour
            for t_i, p_i in zip(t_list, detour_points):
                new_t_vals.append(float(t_i))
                new_points.append(p_i)

            # 3) end_idx 到结尾
            for i in range(end_idx, len(t_axis)):
                new_t_vals.append(t_axis[i])
                new_points.append(wp_samples[i])

            # 更新 t_axis / wp_samples，供下一个障碍物使用
            t_axis = np.asarray(new_t_vals)
            wp_samples = np.asarray(new_points)

        # 去重保证严格增的参数
        if t_axis.size > 0:
            _, idx_unique = np.unique(t_axis, return_index=True)
            t_axis = t_axis[idx_unique]
            wp_samples = wp_samples[idx_unique]

        if t_axis.size < 2:
            print("[MPCC] Avoid_collision: too few points, reverting to original waypoints.")
            fallback_t = self._path_utils.spline_through_points(self._planned_duration, base_waypoints).x
            return fallback_t, base_waypoints

        return t_axis, wp_samples

    # 事件检测（仍然用 visited 信号）

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

    # --------------------- 核心控制 compute_control ---------------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:

        self._current_obs_pos = obs["pos"]

        # 事件触发重规划
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

        # 构造完整物理状态（nx_phys 维），前 12 维填 pos+rpy+vel+drpy，其余 rotor 等填 0
        X_phys_now_full = np.zeros(self.nx_phys, dtype=float)
        X_phys_now_full[0:3] = obs["pos"]
        X_phys_now_full[3:6] = roll_pitch_yaw
        X_phys_now_full[6:9] = obs["vel"]
        X_phys_now_full[9:12] = drpy
        # X_phys_now_full[12:] 保持 0（rotor 状态由 OCP 自己滚动）

        # 全状态: [X_phys, r_cmd_state, p_cmd_state, y_cmd_state, f_cmd_state, r_act, p_act, y_act, f_act, theta]
        x_now = np.zeros(self.nx, dtype=float)
        x_now[0:self.nx_phys] = X_phys_now_full
        x_now[self.idx_r_cmd_state] = self.last_rpy_cmd[0]
        x_now[self.idx_p_cmd_state] = self.last_rpy_cmd[1]
        x_now[self.idx_y_cmd_state] = self.last_rpy_cmd[2]
        x_now[self.idx_f_cmd_state] = self.last_f_cmd
        x_now[self.idx_r_act] = self.last_rpy_act[0]
        x_now[self.idx_p_act] = self.last_rpy_act[1]
        x_now[self.idx_y_act] = self.last_rpy_act[2]
        x_now[self.idx_f_act] = self.last_f_act
        x_now[self.idx_theta] = self.last_theta

        # warm start
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

        # 取出命令状态：用动态索引而不是写死 12/13/14/15/16
        self.last_rpy_cmd = np.array(
            [
                x_next[self.idx_r_cmd_state],
                x_next[self.idx_p_cmd_state],
                x_next[self.idx_y_cmd_state],
            ],
            dtype=float,
        )
        self.last_f_cmd = float(x_next[self.idx_f_cmd_state])
        # self.last_f_collective 由 f_act（执行器输出）更新
        self.last_rpy_act = np.array(
            [
                x_next[self.idx_r_act],
                x_next[self.idx_p_act],
                x_next[self.idx_y_act],
            ],
            dtype=float,
        )
        self.last_f_act = float(x_next[self.idx_f_act])
        self.last_f_collective = self.last_f_act
        self.last_theta = float(x_next[self.idx_theta])

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
        self.last_f_act = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self.last_rpy_act = np.zeros(3)

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