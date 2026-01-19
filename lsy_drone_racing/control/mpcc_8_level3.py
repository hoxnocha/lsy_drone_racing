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


# ----------------------------- Gate frame obstacle types -----------------------------


class ObstacleType(IntEnum):
    """Obstacle geometric models for path-planning avoidance (not MPC constraints)."""

    CYLINDER_2D = 0  # infinite cylinder: distance in XY plane only (for posts / 2D obstacles)
    CAPSULE_3D = 2   # finite capsule segment: point-to-segment distance (for gate bars)


# ----------------------------- Utilities -----------------------------


class FrameUtils:
    @staticmethod
    def quat_to_axis(quat: NDArray[np.floating], axis_index: int = 1) -> NDArray[np.floating]:
        rot = R.from_quat(quat)
        mats = np.asarray(rot.as_matrix())
        if mats.ndim == 3:
            return mats[:, :, axis_index]
        if mats.ndim == 2:
            return mats[:, axis_index]
        raise ValueError("quat_to_axis: unexpected quaternion shape")

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


class CompositeSpline:
    """Concatenate two splines with a time/arc-length offset (for two-stage tracks)."""

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
        p_extend = np.array([p_end + v_dir * (s - base_knots[-1]) for s in extra_knots])
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


# ----------------------------- MPCC Controller (real dynamics + gate frame planning) -----------------------------


class MPCC(Controller):
    """Model Predictive Contouring Control for drone racing (real dynamics + actuator lag + gate-frame-aware planning)."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self._cfg = config

        self._dyn_params = load_params("so_rpy_rotor", config.sim.drone_model)
        mass_val = float(self._dyn_params["mass"])
        gravity_mag = -float(self._dyn_params["gravity_vec"][-1])
        self.hover_thrust = mass_val * gravity_mag

        # --- actuator / inner-loop lag model ---
        self.tau_rpy_act = 0.05
        self.tau_yaw_act = 0.08
        self.tau_f_act = 0.10

        # --- input rate limits for u=[df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta_cmd] ---
        self.rate_limit_df = 10.0
        self.rate_limit_drpy = 10.0
        self.rate_limit_v_theta = 4.0

        # --- gate-frame geometry (adjust to your environment if different) ---
        self.gate_width = 0.7
        self.gate_height = 0.7
        self.gate_frame_margin = 0.18  # clearance for gate posts/bars
        self.obstacle_margin = 0.20    # clearance for regular obstacles

        self._initial_pos = obs["pos"]
        self._cached_gate_centers = obs["gates_pos"]
        self._planned_duration = 30.0

        self._path_utils = PathTools()

        # nominal trajectory
        self._rebuild_nominal_path_gate(obs)

        # MPC settings
        self.N = 35
        self.T_HORIZON = 0.7
        self.dt = self.T_HORIZON / self.N
        self.model_arc_length = 0.05
        self.model_traj_length = 20.0

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

        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_f_act = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self.last_rpy_act = np.zeros(3)
        self.finished = False

    # ------------------------------------------------------------------
    # Dynamics model export: so_rpy_rotor + command integrators + actuator lag + theta
    # ------------------------------------------------------------------

    def _export_dynamics_model(self) -> AcadosModel:
        model_name = "lsy_mpcc_real_dyn_gateframe"

        params = self._dyn_params

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

        self.nx_phys = X_phys.shape[0]

        # aliases for cost
        self.px = X_phys[0]
        self.py = X_phys[1]
        self.pz = X_phys[2]
        self.roll = X_phys[3]
        self.pitch = X_phys[4]
        self.yaw = X_phys[5]

        # command states
        self.r_cmd_state = MX.sym("r_cmd_state")
        self.p_cmd_state = MX.sym("p_cmd_state")
        self.y_cmd_state = MX.sym("y_cmd_state")
        self.f_cmd_state = MX.sym("f_cmd_state")

        # actuator output states
        self.r_act = MX.sym("r_act")
        self.p_act = MX.sym("p_act")
        self.y_act = MX.sym("y_act")
        self.f_act = MX.sym("f_act")

        # path progress
        self.theta = MX.sym("theta")

        # inputs (rates)
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

        # indices (for packing/unpacking)
        self.idx_r_cmd_state = int(self.nx_phys + 0)
        self.idx_p_cmd_state = int(self.nx_phys + 1)
        self.idx_y_cmd_state = int(self.nx_phys + 2)
        self.idx_f_cmd_state = int(self.nx_phys + 3)

        self.idx_r_act = int(self.nx_phys + 4)
        self.idx_p_act = int(self.nx_phys + 5)
        self.idx_y_act = int(self.nx_phys + 6)
        self.idx_f_act = int(self.nx_phys + 7)

        self.idx_theta = int(self.nx_phys + 8)

        # real dynamics take actuator outputs as physical inputs
        U_phys_full = vertcat(self.r_act, self.p_act, self.y_act, self.f_act)
        f_dyn_phys = substitute(X_dot_phys, U_phys, U_phys_full)

        # command integrators
        r_cmd_dot = self.dr_cmd
        p_cmd_dot = self.dp_cmd
        y_cmd_dot = self.dy_cmd
        f_cmd_dot = self.df_cmd

        # actuator lag
        r_act_dot = (self.r_cmd_state - self.r_act) / float(self.tau_rpy_act)
        p_act_dot = (self.p_cmd_state - self.p_act) / float(self.tau_rpy_act)
        y_act_dot = (self.y_cmd_state - self.y_act) / float(self.tau_yaw_act)
        f_act_dot = (self.f_cmd_state - self.f_act) / float(self.tau_f_act)

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

        # trajectory params (discrete samples along theta)
        n_samples = int(self.model_traj_length / self.model_arc_length)
        self.pd_list = MX.sym("pd_list", 3 * n_samples)
        self.tp_list = MX.sym("tp_list", 3 * n_samples)
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
    # Cost helpers
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
        theta_samples = np.arange(0.0, self.model_traj_length, self.model_arc_length)

        pd_vals = trajectory(theta_samples)
        tp_vals = trajectory.derivative(1)(theta_samples)

        qc_gate = np.zeros_like(theta_samples, dtype=float)
        qc_obst = np.zeros_like(theta_samples, dtype=float)

        if hasattr(self, "_cached_gate_centers"):
            for gate_center in self._cached_gate_centers:
                d_gate = np.linalg.norm(pd_vals - gate_center, axis=-1)
                qc_gate = np.maximum(qc_gate, np.exp(-2.0 * d_gate**2))

        if hasattr(self, "_cached_obstacles"):
            for obst_center in self._cached_obstacles:
                d_obs_xy = np.linalg.norm(pd_vals[:, :2] - obst_center[:2], axis=-1)
                qc_obst = np.maximum(qc_obst, 0.7 * np.exp(-1.0 * d_obs_xy**2))

        return np.concatenate([pd_vals.reshape(-1), tp_vals.reshape(-1), qc_gate, qc_obst])

    def _stage_cost_expression(self):
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
            (self.q_l + self.q_l_gate_peak * qc_gate_theta + self.q_l_obst_peak * qc_obst_theta) * dot(e_lag, e_lag)
            + (self.q_c + self.q_c_gate_peak * qc_gate_theta + self.q_c_obst_peak * qc_obst_theta)
            * dot(e_contour, e_contour)
            + att_vec.T @ self.Q_w @ att_vec
        )

        smooth_cost = ctrl_vec.T @ self.R_df @ ctrl_vec

        speed_cost = (
            -self.miu * self.v_theta_cmd
            + self.w_v_gate * qc_gate_theta * (self.v_theta_cmd**2)
            + self.w_v_obst * qc_obst_theta * (self.v_theta_cmd**2)
        )

        return track_cost + smooth_cost + speed_cost

    # ------------------------------------------------------------------
    # OCP build
    # ------------------------------------------------------------------

    def _build_ocp_and_solver(
        self, Tf: float, N_horizon: int, trajectory: CubicSpline, verbose: bool = False
    ) -> Tuple[AcadosOcpSolver, AcadosOcp]:
        ocp = AcadosOcp()
        model = self._export_dynamics_model()
        ocp.model = model

        self.nx = model.x.rows()
        self.nu = model.u.rows()
        ocp.solver_options.N_horizon = N_horizon

        ocp.cost.cost_type = "EXTERNAL"

        # weights
        self.q_l = 200
        self.q_c = 100
        self.Q_w = 1 * DM(np.eye(3))

        self.q_l_gate_peak = 640
        self.q_c_gate_peak = 800

        self.q_l_obst_peak = 100
        self.q_c_obst_peak = 50

        self.R_df = DM(np.diag([0.1, 0.5, 0.5, 0.5]))

        self.miu = 8.0
        self.w_v_gate = 2.5
        self.w_v_obst = 1.5

        ocp.model.cost_expr_ext_cost = self._stage_cost_expression()

        # ----- state bounds (path constraints) -----
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

        # ----- input bounds (rates + v_theta) -----
        ocp.constraints.lbu = np.array(
            [-self.rate_limit_df, -self.rate_limit_drpy, -self.rate_limit_drpy, -self.rate_limit_drpy, 0.0]
        )
        ocp.constraints.ubu = np.array(
            [self.rate_limit_df, self.rate_limit_drpy, self.rate_limit_drpy, self.rate_limit_drpy, self.rate_limit_v_theta]
        )
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

        # ----- IMPORTANT: initial state constraints (full x0) -----
        # This avoids dimension mismatch and properly pins the MPC initial condition online.
        ocp.constraints.idxbx_0 = np.arange(self.nx, dtype=int)
        ocp.constraints.lbx_0 = np.zeros(self.nx)
        ocp.constraints.ubx_0 = np.zeros(self.nx)

        # parameters
        param_vec = self._encode_traj_params(self.arc_trajectory)
        ocp.parameter_values = param_vec

        # solver opts
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

        solver = AcadosOcpSolver(ocp, json_file="mpcc_real_dyn_gateframe.json", verbose=verbose)
        return solver, ocp

    # ------------------------------------------------------------------
    # Gate-frame-aware path planning (virtual obstacles + avoidance)
    # ------------------------------------------------------------------

    def _get_virtual_gate_obstacles(
        self,
        gate_positions: NDArray[np.floating],
        gate_quats: NDArray[np.floating],
        gate_width: float,
        gate_height: float,
    ) -> Tuple[NDArray[np.floating], NDArray[np.int_], NDArray[np.floating], NDArray[np.floating]]:
        """
        Model gate frame as virtual obstacles:
          - left/right posts: CYLINDER_2D at center +/- y*(w/2)
          - top/bottom bars: CAPSULE_3D centered at center +/- z*(h/2), oriented along y, half-length w/2
        """
        gate_y_axes = FrameUtils.quat_to_axis(gate_quats, axis_index=1)
        gate_z_axes = FrameUtils.quat_to_axis(gate_quats, axis_index=2)

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

            # posts (2D cylinders)
            for sign in (1.0, -1.0):
                post_pos = c + sign * half_w * y
                obs_positions.append(post_pos)
                obs_types.append(ObstacleType.CYLINDER_2D)
                obs_vecs.append(np.zeros(3))
                obs_lens.append(0.0)

            # bars (3D capsules)
            y_u = y / (np.linalg.norm(y) + 1e-9)
            for sign in (1.0, -1.0):
                bar_center = c + sign * half_h * z
                obs_positions.append(bar_center)
                obs_types.append(ObstacleType.CAPSULE_3D)
                obs_vecs.append(y_u)
                obs_lens.append(half_w)

        return (
            np.asarray(obs_positions, dtype=float),
            np.asarray(obs_types, dtype=int),
            np.asarray(obs_vecs, dtype=float),
            np.asarray(obs_lens, dtype=float),
        )

    def _apply_obstacle_avoidance(
        self,
        base_waypoints: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
        safe_dist_list: NDArray[np.floating],
        types_list: NDArray[np.int_],
        vecs_list: NDArray[np.floating],
        lens_list: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Sample an initial spline and insert detour points whenever the sampled path enters/leaves an unsafe region.
        Supports:
          - CYLINDER_2D: XY distance to obstacle center
          - CAPSULE_3D: point-to-segment distance (segment centered at obstacle_pos, direction vec, half-length len)
        """
        pre_spline = self._path_utils.spline_through_points(self._planned_duration, base_waypoints)

        n_samples = int(self._ctrl_freq * self._planned_duration)
        n_samples = max(n_samples, 2)

        t_axis = np.linspace(0.0, self._planned_duration, n_samples)
        wp_samples = pre_spline(t_axis)

        for obst_c, safe_dist, o_type, o_vec, o_len in zip(
            obstacles_pos, safe_dist_list, types_list, vecs_list, lens_list
        ):
            inside_region = False
            new_t: list[float] = []
            new_pts: list[np.ndarray] = []

            def dist_to_obstacle(pt: np.ndarray) -> float:
                if int(o_type) == int(ObstacleType.CYLINDER_2D):
                    return float(np.linalg.norm(obst_c[:2] - pt[:2]))
                if int(o_type) == int(ObstacleType.CAPSULE_3D):
                    vec_cp = pt - obst_c
                    proj = float(np.dot(vec_cp, o_vec))
                    proj_clamped = float(np.clip(proj, -o_len, o_len))
                    closest_pt = obst_c + proj_clamped * o_vec
                    return float(np.linalg.norm(pt - closest_pt))
                return float(np.linalg.norm(obst_c - pt))

            for idx in range(wp_samples.shape[0]):
                pt = wp_samples[idx]
                dist = dist_to_obstacle(pt)

                if dist < safe_dist and not inside_region:
                    inside_region = True
                    idx_in = idx

                elif dist >= safe_dist and inside_region:
                    inside_region = False
                    idx_out = idx

                    p_in = wp_samples[idx_in]
                    p_out = wp_samples[idx_out]
                    p_mid = 0.5 * (p_in + p_out)

                    if int(o_type) == int(ObstacleType.CYLINDER_2D):
                        base_pt = obst_c.copy()
                        push_vec = p_mid - obst_c
                        push_vec[2] = 0.0
                    elif int(o_type) == int(ObstacleType.CAPSULE_3D):
                        vec_cp = p_mid - obst_c
                        proj = float(np.clip(np.dot(vec_cp, o_vec), -o_len, o_len))
                        base_pt = obst_c + proj * o_vec
                        push_vec = p_mid - base_pt
                    else:
                        base_pt = obst_c.copy()
                        push_vec = p_mid - obst_c

                    push_dir = push_vec / (np.linalg.norm(push_vec) + 1e-9)

                    if int(o_type) == int(ObstacleType.CYLINDER_2D):
                        detour_xy = base_pt[:2] + push_dir[:2] * safe_dist
                        detour_z = 0.5 * (p_in[2] + p_out[2])
                        detour_pt = np.array([detour_xy[0], detour_xy[1], detour_z], dtype=float)
                    else:
                        detour_pt = (base_pt + push_dir * safe_dist).astype(float)

                    new_t.append(0.5 * (t_axis[idx_in] + t_axis[idx_out]))
                    new_pts.append(detour_pt)

                if dist >= safe_dist:
                    new_t.append(float(t_axis[idx]))
                    new_pts.append(pt)

            if inside_region:
                new_t.append(float(t_axis[-1]))
                new_pts.append(wp_samples[-1])

            t_axis = np.asarray(new_t, dtype=float)
            wp_samples = np.asarray(new_pts, dtype=float)

        if t_axis.size > 0:
            _, uniq = np.unique(t_axis, return_index=True)
            return t_axis[uniq], wp_samples[uniq]

        return np.array([]), np.array([])

    # ------------------------------------------------------------------
    # Path rebuild (gate / obstacle events)
    # ------------------------------------------------------------------

    def _rebuild_nominal_path_gate(self, obs: dict[str, NDArray[np.floating]]):
        print(f"T={self._step_count / self._ctrl_freq:.2f}: (Re)building nominal path (gate + gate frame)...")

        gate_positions = obs["gates_pos"]
        obstacle_positions = obs["obstacles_pos"]
        gate_quats = obs["gates_quat"]

        self._cached_gate_centers = gate_positions
        self._cached_obstacles = obstacle_positions

        gate_normals, gate_y, gate_z = self._extract_gate_frames(gate_quats)

        base_waypoints = self._path_utils.build_gate_waypoints(self._initial_pos, gate_positions, gate_normals)

        with_gate_detours = self._insert_gate_detours(
            base_waypoints, gate_positions, gate_normals, gate_y, gate_z
        )

        # --- virtual obstacles: gate frame ---
        virt_pos, virt_types, virt_vecs, virt_lens = self._get_virtual_gate_obstacles(
            gate_positions, gate_quats, self.gate_width, self.gate_height
        )
        virt_margins = np.full(len(virt_pos), self.gate_frame_margin, dtype=float)

        # --- real obstacles (assumed cylinder in XY) ---
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            n_real = len(obstacle_positions)
            real_types = np.full(n_real, int(ObstacleType.CYLINDER_2D), dtype=int)
            real_vecs = np.zeros((n_real, 3), dtype=float)
            real_lens = np.zeros(n_real, dtype=float)
            real_margins = np.full(n_real, self.obstacle_margin, dtype=float)

            all_obstacles = np.vstack([obstacle_positions, virt_pos])
            all_types = np.concatenate([real_types, virt_types])
            all_vecs = np.vstack([real_vecs, virt_vecs])
            all_lens = np.concatenate([real_lens, virt_lens])
            all_margins = np.concatenate([real_margins, virt_margins])
        else:
            all_obstacles = virt_pos
            all_types = virt_types
            all_vecs = virt_vecs
            all_lens = virt_lens
            all_margins = virt_margins

        t_axis, collision_free_wps = self._apply_obstacle_avoidance(
            with_gate_detours,
            all_obstacles,
            safe_dist_list=all_margins,
            types_list=all_types,
            vecs_list=all_vecs,
            lens_list=all_lens,
        )

        if len(t_axis) < 2:
            print("[MPCC] Warning: avoid path fallback (too few points).")
            self.trajectory = self._path_utils.spline_through_points(self._planned_duration, with_gate_detours)
        else:
            self.trajectory = CubicSpline(t_axis, collision_free_wps)
            self._planned_duration = float(self.trajectory.x[-1])

    def _rebuild_nominal_path_obstacle(self, obs: dict[str, NDArray[np.floating]]):
        print(f"T={self._step_count / self._ctrl_freq:.2f}: (Re)building nominal path (obstacle + gate frame)...")

        gate_positions = obs["gates_pos"]
        obstacle_positions = obs["obstacles_pos"]
        gate_quats = obs["gates_quat"]
        start_pos = obs["pos"]

        self._cached_gate_centers = gate_positions
        self._cached_obstacles = obstacle_positions

        gate_normals, gate_y, gate_z = self._extract_gate_frames(gate_quats)

        base_waypoints = self._path_utils.build_gate_waypoints(start_pos, gate_positions, gate_normals)

        with_gate_detours = self._insert_gate_detours(
            base_waypoints, gate_positions, gate_normals, gate_y, gate_z
        )

        virt_pos, virt_types, virt_vecs, virt_lens = self._get_virtual_gate_obstacles(
            gate_positions, gate_quats, self.gate_width, self.gate_height
        )
        virt_margins = np.full(len(virt_pos), self.gate_frame_margin, dtype=float)

        if obstacle_positions is not None and len(obstacle_positions) > 0:
            n_real = len(obstacle_positions)
            real_types = np.full(n_real, int(ObstacleType.CYLINDER_2D), dtype=int)
            real_vecs = np.zeros((n_real, 3), dtype=float)
            real_lens = np.zeros(n_real, dtype=float)
            real_margins = np.full(n_real, self.obstacle_margin, dtype=float)

            all_obstacles = np.vstack([obstacle_positions, virt_pos])
            all_types = np.concatenate([real_types, virt_types])
            all_vecs = np.vstack([real_vecs, virt_vecs])
            all_lens = np.concatenate([real_lens, virt_lens])
            all_margins = np.concatenate([real_margins, virt_margins])
        else:
            all_obstacles = virt_pos
            all_types = virt_types
            all_vecs = virt_vecs
            all_lens = virt_lens
            all_margins = virt_margins

        t_axis, collision_free_wps = self._apply_obstacle_avoidance(
            with_gate_detours,
            all_obstacles,
            safe_dist_list=all_margins,
            types_list=all_types,
            vecs_list=all_vecs,
            lens_list=all_lens,
        )

        if len(t_axis) < 2:
            print("[MPCC] Warning: avoid path fallback (too few points).")
            self.trajectory = self._path_utils.spline_through_points(self._planned_duration, with_gate_detours)
        else:
            self.trajectory = CubicSpline(t_axis, collision_free_wps)
            self._planned_duration = float(self.trajectory.x[-1])

    # ------------------------------------------------------------------
    # Event detection (visited flags)
    # ------------------------------------------------------------------

    def _detect_event_change(self, obs: dict) -> bool:
        if not hasattr(self, "_last_gate_flags"):
            self._last_gate_flags = np.array(obs.get("gates_visited", []), dtype=bool)
            self._last_obst_flags = np.array(obs.get("obstacles_visited", []), dtype=bool)
            return False

        curr_gates = np.array(obs.get("gates_visited", []), dtype=bool)
        curr_obst = np.array(obs.get("obstacles_visited", []), dtype=bool)

        gate_trigger = curr_gates.shape == self._last_gate_flags.shape and np.any((~self._last_gate_flags) & curr_gates)
        obst_trigger = curr_obst.shape == self._last_obst_flags.shape and np.any((~self._last_obst_flags) & curr_obst)

        self._last_gate_flags = curr_gates
        self._last_obst_flags = curr_obst

        return bool(gate_trigger or obst_trigger)

    # ------------------------------------------------------------------
    # Gate frame extraction and detours (unchanged from your version)
    # ------------------------------------------------------------------

    def _extract_gate_frames(
        self, gates_quaternions: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
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

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def _pos_outside_limits(self, pos: NDArray[np.floating]) -> bool:
        for i_dim in range(3):
            low, high = self.pos_bound[i_dim]
            if pos[i_dim] < low or pos[i_dim] > high:
                return True
        return False

    def _speed_outside_limits(self, vel: NDArray[np.floating]) -> bool:
        speed = np.linalg.norm(vel)
        return not (self.velocity_bound[0] < speed < self.velocity_bound[1])

    # ------------------------------------------------------------------
    # Core control loop
    # ------------------------------------------------------------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        self._current_obs_pos = obs["pos"]

        # event-trigger replanning
        if self._detect_event_change(obs):
            print(f"T={self._step_count / self._ctrl_freq:.2f}: MPCC detected env change, replanning...")
            # prefer gate-based rebuild unless you want different policies
            self._rebuild_nominal_path_gate(obs)
            self.arc_trajectory = self._path_utils.reparametrize_by_arclength(
                self._path_utils.extend_spline_tail(self.trajectory, extend_length=self.model_traj_length)
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

        # physical state (nx_phys): fill first 12, remaining rotor states default to 0
        X_phys_now_full = np.zeros(self.nx_phys, dtype=float)
        X_phys_now_full[0:3] = obs["pos"]
        X_phys_now_full[3:6] = roll_pitch_yaw
        X_phys_now_full[6:9] = obs["vel"]
        X_phys_now_full[9:12] = drpy

        # full state
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

        # warm start shift
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

        # ---- pin initial state (full x0) using idxbx_0 ----
        self.acados_ocp_solver.set(0, "lbx", x_now)
        self.acados_ocp_solver.set(0, "ubx", x_now)

        # termination checks
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

        # update command / actuator / theta
        self.last_rpy_cmd = np.array(
            [x_next[self.idx_r_cmd_state], x_next[self.idx_p_cmd_state], x_next[self.idx_y_cmd_state]], dtype=float
        )
        self.last_f_cmd = float(x_next[self.idx_f_cmd_state])

        self.last_rpy_act = np.array(
            [x_next[self.idx_r_act], x_next[self.idx_p_act], x_next[self.idx_y_act]], dtype=float
        )
        self.last_f_act = float(x_next[self.idx_f_act])
        self.last_f_collective = self.last_f_act

        self.last_theta = float(x_next[self.idx_theta])

        cmd = np.array([self.last_rpy_cmd[0], self.last_rpy_cmd[1], self.last_rpy_cmd[2], self.last_f_cmd], dtype=float)

        print(f"cmd: roll={cmd[0]:.3f}, pitch={cmd[1]:.3f}, yaw={cmd[2]:.3f}, thrust={cmd[3]:.3f}")

        self._step_count += 1
        return cmd

    # ------------------------------------------------------------------
    # Callbacks & debug
    # ------------------------------------------------------------------

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
                debug_lines.append((full_path, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0))
            except Exception:
                pass

        if hasattr(self, "_x_warm"):
            pred_states = np.asarray([x_state[:3] for x_state in self._x_warm])
            debug_lines.append((pred_states, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))

        if hasattr(self, "last_theta") and hasattr(self, "arc_trajectory") and hasattr(self, "_current_obs_pos"):
            try:
                target_on_path = self.arc_trajectory(self.last_theta)
                segment = np.stack([self._current_obs_pos, target_on_path])
                debug_lines.append((segment, np.array([0.0, 0.0, 1.0, 1.0]), 1.0, 1.0))
            except Exception:
                pass

        return debug_lines