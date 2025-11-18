from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat, dot, DM, norm_2, floor, if_else
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from drone_models.core import load_params
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray



class FrameUtils:

    @staticmethod
    def quat_to_axis(quat: NDArray[np.floating], axis_index: int = 1) -> NDArray[np.floating]:
        """
        Convert quaternion(s) to the specified body axis expressed in world frame.

        Args:
            quat: shape (N,4) or (4,)
            axis_index: 0=x, 1=y, 2=z

        Returns:
            axis vectors with same leading shape as quat[...,0]
        """
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
        t_current = trajectory.x

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
        """
        Extend an existing trajectory along its terminal tangent direction.
        """
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
        """
        Cut original trajectory at a y-maximum (after first 20 samples) and
        reparametrize both parts by arc length, then combine.
        """
        idx_peak = 20 + int(np.argmax(np.asarray(pos)[20:, 1]))
        t = np.asarray(t)

        t_first, p_first = t[: idx_peak + 1], pos[: idx_peak + 1]
        t_second, p_second = t[idx_peak:] - t[idx_peak], pos[idx_peak:]

        spline_1 = CubicSpline(t_first, p_first)
        spline_2 = CubicSpline(t_second, p_second)

        arc_spline_1 = self.reparametrize_by_arclength(spline_1)
        arc_spline_2 = self.reparametrize_by_arclength(spline_2)

        # drop last knot of stage 1 to avoid duplicate time
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




class MPCC(Controller):
    """Model Predictive Contouring Control for drone racing."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self._cfg = config

        self._dyn_params = load_params("so_rpy", config.sim.drone_model)
        mass_val = float(self._dyn_params["mass"])
        gravity_mag = -float(self._dyn_params["gravity_vec"][-1])

        self.hover_thrust = mass_val * gravity_mag

        self._initial_pos = obs["pos"]
        self._cached_gate_centers = obs["gates_pos"]
        self._planned_duration = 30.0

        self._path_utils = PathTools()

        self._rebuild_nominal_path(obs)

        self.N = 50
        self.T_HORIZON = 0.6
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
        Continuous-time quadrotor dynamics in the same structure as the original version.
        """

        model_name = "lsy_example_mpc"

        mass_val = float(self._dyn_params["mass"])
        GRAVITY = -float(self._dyn_params["gravity_vec"][-1])

        params_pitch_rate = [-6.003842038081178, 6.213752925707588]
        params_roll_rate = [-3.960889336015948, 4.078293254657104]
        params_yaw_rate = [-0.005347588299390372, 0.0]

        self.px = MX.sym("px")
        self.py = MX.sym("py")
        self.pz = MX.sym("pz")
        self.vx = MX.sym("vx")
        self.vy = MX.sym("vy")
        self.vz = MX.sym("vz")
        self.roll = MX.sym("r")
        self.pitch = MX.sym("p")
        self.yaw = MX.sym("y")
        self.f_collective = MX.sym("f_collective")
        self.f_collective_cmd = MX.sym("f_collective_cmd")
        self.r_cmd = MX.sym("r_cmd")
        self.p_cmd = MX.sym("p_cmd")
        self.y_cmd = MX.sym("y_cmd")
        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")
        self.theta = MX.sym("theta")
        self.v_theta_cmd = MX.sym("v_theta_cmd")

        states = vertcat(
            self.px,
            self.py,
            self.pz,
            self.vx,
            self.vy,
            self.vz,
            self.roll,
            self.pitch,
            self.yaw,
            self.f_collective,
            self.f_collective_cmd,
            self.r_cmd,
            self.p_cmd,
            self.y_cmd,
            self.theta,
        )
        inputs = vertcat(
            self.df_cmd,
            self.dr_cmd,
            self.dp_cmd,
            self.dy_cmd,
            self.v_theta_cmd,
        )

        thrust = self.f_collective
        inv_mass = 1.0 / mass_val

        ax = inv_mass * thrust * (
            cos(self.roll) * sin(self.pitch) * cos(self.yaw)
            + sin(self.roll) * sin(self.yaw)
        )
        ay = inv_mass * thrust * (
            cos(self.roll) * sin(self.pitch) * sin(self.yaw)
            - sin(self.roll) * cos(self.yaw)
        )
        az = inv_mass * thrust * cos(self.roll) * cos(self.pitch) - GRAVITY

        f_dyn = vertcat(
            self.vx,
            self.vy,
            self.vz,
            ax,
            ay,
            az,
            params_roll_rate[0] * self.roll + params_roll_rate[1] * self.r_cmd,
            params_pitch_rate[0] * self.pitch + params_pitch_rate[1] * self.p_cmd,
            params_yaw_rate[0] * self.yaw + params_yaw_rate[1] * self.y_cmd,
            10.0 * (self.f_collective_cmd - self.f_collective),
            self.df_cmd,
            self.dr_cmd,
            self.dp_cmd,
            self.dy_cmd,
            self.v_theta_cmd,
        )

        self.pd_list = MX.sym("pd_list", 3 * int(self.model_traj_length / self.model_arc_length))
        self.tp_list = MX.sym("tp_list", 3 * int(self.model_traj_length / self.model_arc_length))
        self.qc_dyn = MX.sym("qc_dyn", 1 * int(self.model_traj_length / self.model_arc_length))
        params = vertcat(self.pd_list, self.tp_list, self.qc_dyn)

        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f_dyn
        model.x = states
        model.u = inputs
        model.p = params
        return model

    def _piecewise_linear_interp(self, theta, theta_vec, flattened_points, dim: int = 3):
        """
        CasADi-friendly linear interpolation along 1D parameter theta.
        """
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
        Build parameter vector [pd_list, tp_list, qc_dyn_list] for the MPCC cost.
        """
        theta_samples = np.arange(0.0, self.model_traj_length, self.model_arc_length)

        pd_vals = trajectory(theta_samples)
        tp_vals = trajectory.derivative(1)(theta_samples)
        qc_dyn = np.zeros_like(theta_samples)

        if hasattr(self, "_cached_gate_centers"):
            for gate_center in self._cached_gate_centers:
                d_gate = np.linalg.norm(pd_vals - gate_center, axis=-1)
                qc_gate = np.exp(-5.0 * d_gate**2)
                qc_dyn = np.maximum(qc_dyn, qc_gate)

        if hasattr(self, "_cached_obstacles"):
            for obst_center in self._cached_obstacles:
                d_obs_xy = np.linalg.norm(pd_vals[:, :2] - obst_center[:2], axis=-1)
                qc_obs = np.exp(-3.0 * d_obs_xy**2)
                qc_dyn = np.maximum(qc_dyn, qc_obs)

        return np.concatenate([pd_vals.reshape(-1), tp_vals.reshape(-1), qc_dyn])

    def _stage_cost_expression(self):
        """
        MPCC stage cost: path tracking (contouring + lag), attitude regularization,
        control smoothness and progress / speed shaping.
        """
        position_vec = vertcat(self.px, self.py, self.pz)
        att_vec = vertcat(self.roll, self.pitch, self.yaw)
        ctrl_vec = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)

        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_length)

        pd_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)
        qc_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_dyn, dim=1)

        tp_unit = tp_theta / (norm_2(tp_theta) + 1e-6)
        e_theta = position_vec - pd_theta
        e_lag = dot(tp_unit, e_theta) * tp_unit
        e_contour = e_theta - e_lag

        track_cost = (
            (self.q_l + self.q_l_peak * qc_theta) * dot(e_lag, e_lag)
            + (self.q_c + self.q_c_peak * qc_theta) * dot(e_contour, e_contour)
            + att_vec.T @ self.Q_w @ att_vec
        )

        smooth_cost = ctrl_vec.T @ self.R_df @ ctrl_vec

        speed_cost = -self.miu * self.v_theta_cmd + self.w_v_gate * qc_theta * (self.v_theta_cmd**2)

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

        self.q_l = 160
        self.q_l_peak = 640
        self.q_c = 80
        self.q_c_peak = 800
        self.Q_w = 1 * DM(np.eye(3))

        self.R_df = DM(np.diag([0.1, 0.5, 0.5, 0.5]))
        self.miu = 8.0
        self.w_v_gate = 2.0

        ocp.model.cost_expr_ext_cost = self._stage_cost_expression()

        # state bounds
        thrust_min = float(self._dyn_params["thrust_min"]) * 4.0
        thrust_max = float(self._dyn_params["thrust_max"]) * 4.0

        # [px,py,pz,vx,vy,vz,roll,pitch,yaw,f,f_cmd,r_cmd,p_cmd,y_cmd,theta]
        ocp.constraints.lbx = np.array([thrust_min, thrust_min, -1.57, -1.57, -1.57])
        ocp.constraints.ubx = np.array([thrust_max, thrust_max, 1.57, 1.57, 1.57])
        ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

        # input bounds
        ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, 0.0])
        ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0, 4.0])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

        # initial state (dummy, overwritten online)
        ocp.constraints.x0 = np.zeros(self.nx)

        # parameters: initial trajectory
        param_vec = self._encode_traj_params(self.arc_trajectory)
        ocp.parameter_values = param_vec

        # solver options
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

        solver = AcadosOcpSolver(ocp, json_file="mpcc_prescripted.json", verbose=verbose)
        return solver, ocp

    # ------------- trajectory planning & obstacle handling -------------

    def _rebuild_nominal_path(self, obs: dict[str, NDArray[np.floating]]):
        """
        Plan / replan nominal geometric path from gates + obstacles (purely in geometry level).
        """
        print(f"T={self._step_count / self._ctrl_freq:.2f}: (Re)building nominal path...")

        gate_positions = obs["gates_pos"]
        obstacle_positions = obs["obstacles_pos"]
        gate_quats = obs["gates_quat"]

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
            with_gate_detours, obstacle_positions, safe_dist=0.3
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
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Given a set of waypoints, generate a smooth path that stays at least
        `safe_dist` away (in XY) from each obstacle by inserting additional points.
        """
        pre_spline = self._path_utils.spline_through_points(self._planned_duration, base_waypoints)

        n_samples = int(self._ctrl_freq * self._planned_duration)
        if n_samples <= 0:
            n_samples = 1

        t_axis = np.linspace(0.0, self._planned_duration, n_samples)
        wp_samples = pre_spline(t_axis)

        for obst in obstacles_pos:
            inside_region = False
            new_t_vals: List[float] = []
            new_points: List[np.ndarray] = []

            for idx in range(wp_samples.shape[0]):
                pt = wp_samples[idx]
                dist_xy = np.linalg.norm(obst[:2] - pt[:2])

                if dist_xy < safe_dist and not inside_region:
                    inside_region = True
                    start_idx = idx
                elif dist_xy >= safe_dist and inside_region:
                    end_idx = idx
                    inside_region = False

                    direction_xy = (
                        wp_samples[start_idx][:2] - obst[:2]
                        + wp_samples[end_idx][:2] - obst[:2]
                    )
                    direction_xy /= np.linalg.norm(direction_xy) + 1e-6

                    detour_xy = obst[:2] + direction_xy * safe_dist
                    detour_z = 0.5 * (wp_samples[start_idx][2] + wp_samples[end_idx][2])
                    detour_point = np.concatenate([detour_xy, [detour_z]])

                    new_t_vals.append(0.5 * (t_axis[start_idx] + t_axis[end_idx]))
                    new_points.append(detour_point)
                elif dist_xy >= safe_dist:
                    new_t_vals.append(t_axis[idx])
                    new_points.append(pt)

            if inside_region:
                new_t_vals.append(t_axis[-1])
                new_points.append(wp_samples[-1])

            t_axis = np.asarray(new_t_vals)
            wp_samples = np.asarray(new_points)

        if t_axis.size > 0:
            _, idx_unique = np.unique(t_axis, return_index=True)
            t_axis = t_axis[idx_unique]
            wp_samples = wp_samples[idx_unique]

        if t_axis.size < 2:
            print("[MPCC] Avoid_collision: too few points, reverting to original waypoints.")
            fallback_t = self._path_utils.spline_through_points(self._planned_duration, base_waypoints).x
            return fallback_t, base_waypoints

        return t_axis, wp_samples

    def _detect_event_change(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        """
        Detect changes in 'visited' flags for gates and obstacles, to trigger replanning.
        """
        if not hasattr(self, "_last_gate_flags"):
            self._last_gate_flags = np.array(obs.get("gates_visited", []), dtype=bool)
            self._last_obst_flags = np.array(obs.get("obstacles_visited", []), dtype=bool)
            return False

        curr_gates = np.array(obs.get("gates_visited", []), dtype=bool)
        curr_obst = np.array(obs.get("obstacles_visited", []), dtype=bool)

        if curr_gates.shape != self._last_gate_flags.shape:
            self._last_gate_flags = curr_gates
            return False
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
                # need detour
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

        if self._detect_event_change(obs):
            print(f"T={self._step_count / self._ctrl_freq:.2f}: MPCC detected environment change, replanning...")
            self._rebuild_nominal_path(obs)
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

        x_now = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                roll_pitch_yaw,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
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

        self.last_f_collective = float(x_next[9])
        self.last_f_cmd = float(x_next[10])
        self.last_rpy_cmd = np.array(x_next[11:14]).copy()
        self.last_theta = float(x_next[14])

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
        """
        Return line segments for visualization, same semantics as原实现：
        - full arclength path
        - predicted trajectory from warm start
        - line from current drone pos to reference on path at last_theta
        """
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
