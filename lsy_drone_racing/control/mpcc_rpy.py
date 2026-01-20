from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, List, Optional
from enum import IntEnum

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat, dot, DM, norm_2, floor, if_else, substitute
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ==============================================================================
# 1. Enums & Helpers
# ==============================================================================

class ObstacleType(IntEnum):
    CYLINDER_2D = 0
    CAPSULE_3D = 2


class FrameUtils:
    """Utilities for coordinate frame extraction."""
    @staticmethod
    def quat_to_axis(quat: NDArray[np.floating], axis_index: int = 1) -> NDArray[np.floating]:
        rot = R.from_quat(quat)
        mats = np.asarray(rot.as_matrix())
        if mats.ndim == 3: return mats[:, :, axis_index]
        if mats.ndim == 2: return mats[:, axis_index]
        return None

    @staticmethod
    def extract_gate_frames(gates_quaternions: NDArray[np.floating]) -> Tuple[NDArray, NDArray, NDArray]:
        normals = FrameUtils.quat_to_axis(gates_quaternions, axis_index=0)
        y_axes = FrameUtils.quat_to_axis(gates_quaternions, axis_index=1)
        z_axes = FrameUtils.quat_to_axis(gates_quaternions, axis_index=2)
        return normals, y_axes, z_axes


# ==============================================================================
# 2. Path Planner 
# ==============================================================================

class RacingPathPlanner:
    """Handles trajectory generation, spline fitting, and detours."""

    def __init__(self, ctrl_freq: float):
        self.ctrl_freq = ctrl_freq

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
        extra_knots = np.arange(base_knots[-1] + base_dt, base_knots[-1] + extend_length, base_dt)
        p_extend = np.array([p_end + v_dir * (s - base_knots[-1]) for s in extra_knots])
        theta_new = np.concatenate([base_knots, extra_knots])
        p_new = np.vstack([trajectory(base_knots), p_extend])
        return CubicSpline(theta_new, p_new, axis=0)

    def build_gate_waypoints(
        self, start_pos: NDArray[np.floating], gates_pos: NDArray[np.floating], gates_normal: NDArray[np.floating], 
        half_span: float = 0.5, samples_per_gate: int = 5
    ) -> NDArray[np.floating]:
        n_gates = gates_pos.shape[0]
        grid = []
        for idx in range(samples_per_gate):
            alpha = idx / (samples_per_gate - 1) if samples_per_gate > 1 else 0.0
            grid.append(gates_pos - half_span * gates_normal + 2.0 * half_span * alpha * gates_normal)
        stacked = np.stack(grid, axis=1).reshape(n_gates, samples_per_gate, 3).reshape(-1, 3)
        return np.vstack([start_pos[None, :], stacked])

    def insert_gate_detours(
        self, waypoints: NDArray, gate_pos: NDArray, gate_normal: NDArray, 
        gate_y: NDArray, gate_z: NDArray, num_inter: int = 5, angle_thr: float = 120.0, dist: float = 0.65
    ) -> NDArray:
        wp_list = list(waypoints)
        extra = 0
        for i in range(gate_pos.shape[0] - 1):
            curr_idx = 1 + (i + 1) * num_inter - 1 + extra
            next_idx = 1 + (i + 1) * num_inter + extra
            if curr_idx >= len(wp_list) or next_idx >= len(wp_list): break
            
            vec = wp_list[next_idx] - wp_list[curr_idx]
            if np.linalg.norm(vec) < 1e-6: continue
            
            cos_ang = np.clip(np.dot(vec, gate_normal[i]) / np.linalg.norm(vec), -1.0, 1.0)
            if np.degrees(np.arccos(cos_ang)) > angle_thr:
                tangent = vec - np.dot(vec, gate_normal[i]) * gate_normal[i]
                norm_t = np.linalg.norm(tangent)
                detour_dir = gate_y[i]
                if norm_t > 1e-6:
                    tangent /= norm_t
                    proj_ang = np.degrees(np.arctan2(np.dot(tangent, gate_z[i]), np.dot(tangent, gate_y[i])))
                    if -90 <= proj_ang < 45: detour_dir = gate_y[i]
                    elif 45 <= proj_ang < 135: detour_dir = gate_z[i]
                    else: detour_dir = -gate_y[i]
                
                wp_list.insert(curr_idx + 1, gate_pos[i] + dist * detour_dir)
                extra += 1
        return np.asarray(wp_list)

    def inject_obstacle_detours(
        self, waypoints: NDArray, obs_pos: NDArray, duration: float, 
        gate_pos: NDArray, safe_dist: float = 0.2, arc_n: int = 5
    ) -> Tuple[NDArray, NDArray]:
        """
        Replaces path segments that are too close to obstacles with circular arcs.
        Strictly ported from original mpcc_rpy logic.
        """
        # 1. Initial spline for sampling
        pre_spline = self.spline_through_points(duration, waypoints)
        n_samples = max(1, int(self.ctrl_freq * duration))
        t_axis = np.linspace(0.0, duration, n_samples)
        wp_samples = pre_spline(t_axis)
        
        gate_margin = 3

        for obst in obs_pos:
            # Re-calculate gate indices on current path (Logic from original)
            gate_idx = np.array([], dtype=int)
            if len(gate_pos) > 0:
                idx_list = []
                for g in gate_pos:
                    d_g = np.linalg.norm(wp_samples - g, axis=1)
                    idx_list.append(int(np.argmin(d_g)))
                gate_idx = np.asarray(idx_list, dtype=int)

            # Check collision
            d_xy = np.linalg.norm(wp_samples[:, :2] - obst[:2], axis=1)
            inside = d_xy < safe_dist

            if not np.any(inside): continue

            inside_idx = np.where(inside)[0]
            start_idx = max(0, int(inside_idx[0]) - 1)
            end_idx = min(len(t_axis) - 1, int(inside_idx[-1]) + 1)

            if end_idx <= start_idx + 1: continue

            # Don't modify if near a gate (Critical for passing gates)
            if gate_idx.size > 0:
                if np.any((gate_idx >= start_idx - gate_margin) & (gate_idx <= end_idx + gate_margin)):
                    continue

            # Geometry: Circular Detour
            p_start = wp_samples[start_idx]
            p_end = wp_samples[end_idx]
            v_start = p_start[:2] - obst[:2]
            v_end = p_end[:2] - obst[:2]
            n_s, n_e = np.linalg.norm(v_start), np.linalg.norm(v_end)
            if n_s < 1e-6 or n_e < 1e-6: continue
            
            v_start /= n_s
            v_end /= n_e
            
            th_s = np.arctan2(v_start[1], v_start[0])
            th_e = np.arctan2(v_end[1], v_end[0])
            d_th = th_e - th_s
            if d_th > np.pi: d_th -= 2*np.pi
            elif d_th < -np.pi: d_th += 2*np.pi
            
            # Interpolate Arc
            theta_list = np.linspace(th_s, th_s + d_th, arc_n + 2)[1:-1]
            t_list = np.linspace(t_axis[start_idx], t_axis[end_idx], arc_n + 2)[1:-1]
            
            detour_pts = []
            for i, th in enumerate(theta_list):
                xy = obst[:2] + np.array([np.cos(th), np.sin(th)]) * safe_dist
                alpha = (i + 1) / (arc_n + 1)
                z = (1.0 - alpha) * p_start[2] + alpha * p_end[2]
                detour_pts.append(np.array([xy[0], xy[1], z]))
            
            # Stitch
            new_t = list(t_axis[:start_idx+1])
            new_p = list(wp_samples[:start_idx+1])
            for t_i, p_i in zip(t_list, detour_pts):
                new_t.append(float(t_i))
                new_p.append(p_i)
            for i in range(end_idx, len(t_axis)):
                new_t.append(t_axis[i])
                new_p.append(wp_samples[i])
                
            t_axis = np.asarray(new_t)
            wp_samples = np.asarray(new_p)

        # Cleanup
        if t_axis.size > 0:
            _, uniq = np.unique(t_axis, return_index=True)
            t_axis = t_axis[uniq]
            wp_samples = wp_samples[uniq]
            
        if t_axis.size < 2:
            return self.spline_through_points(duration, waypoints).x, waypoints
            
        return t_axis, wp_samples

    def build_complete_trajectory(
        self, start_pos: NDArray, obs: dict, planned_duration: float
    ) -> Tuple[CubicSpline, float]:
        gate_pos = obs["gates_pos"]
        gate_quats = obs["gates_quat"]
        gate_normals, gate_y, gate_z = FrameUtils.extract_gate_frames(gate_quats)
        
        base_wps = self.build_gate_waypoints(start_pos, gate_pos, gate_normals)
        if base_wps.shape[0] > 1: base_wps[1:, 2] += 0.0
        
        wps_detour = self.insert_gate_detours(base_wps, gate_pos, gate_normals, gate_y, gate_z)
        
        t_axis, wps_final = self.inject_obstacle_detours(
            wps_detour, obs["obstacles_pos"], planned_duration, gate_pos
        )
        
        if len(t_axis) < 2:
            traj = self.spline_through_points(planned_duration, wps_detour)
        else:
            traj = CubicSpline(t_axis, wps_final)
            
        return traj, float(traj.x[-1])


# ==============================================================================
# 3. MPCC Controller (RPY Model)
# ==============================================================================

class MPCC(Controller):
    """
    MPCC using `so_rpy` model (17 states). 
    Strict preservation of algorithm, solver, and weights from mpcc_rpy.py.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self._cfg = config

        # Dynamics (so_rpy_rotor_drag)
        self._dyn_params = load_params("so_rpy_rotor_drag", config.sim.drone_model)
        mass = float(self._dyn_params["mass"])
        g_mag = -float(self._dyn_params["gravity_vec"][-1])
        self.hover_thrust = mass * g_mag

        # Initial State
        self._initial_pos = obs["pos"]
        self._cached_gate_centers = obs["gates_pos"]
        self._cached_obstacles = obs["obstacles_pos"]
        self._planned_duration = 30.0

        # Planner
        self.planner = RacingPathPlanner(self._ctrl_freq)
        self._rebuild_nominal_path_gate(obs)

        # MPC Config
        self.N = 35
        self.T_HORIZON = 0.7
        self.dt = self.T_HORIZON / self.N
        self.model_arc_length = 0.05
        self.model_traj_length = 12.0

        # Solver Setup
        self.arc_trajectory = self.planner.reparametrize_by_arclength(
            self.planner.extend_spline_tail(self.trajectory, extend_length=self.model_traj_length)
        )
        self.acados_ocp_solver, self.ocp = self._build_ocp_and_solver(
            self.T_HORIZON, self.N, self.arc_trajectory
        )

        # Bounds & Limits
        self.pos_bound = [np.array([-2.6, 2.6]), np.array([-2.0, 1.8]), np.array([-0.1, 2.0])]
        self.velocity_bound = [-1.0, 4.0]
        
        # Rate Limits
        self.rate_limit_df = 10.0
        self.rate_limit_drpy = 10.0
        self.rate_limit_v_theta = 4.0

        # Actuator constants
        self.tau_rpy_act = 0.05
        self.tau_yaw_act = 0.08
        self.tau_f_act = 0.10

        # Runtime vars
        self.last_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_f_act = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self.finished = False

    def _export_dynamics_model(self) -> AcadosModel:
        """
        Exports `so_rpy` dynamics (17 states).
        X = [X_phys(12), r_cmd, p_cmd, y_cmd, f_cmd, theta]
        U = [df, dr, dp, dy, v_theta]
        """
        model_name = "lsy_example_mpc_real"
        params = self._dyn_params

        # Physical Dynamics (12 states)
        X_dot_phys, X_phys, U_phys, _ = symbolic_dynamics_euler(
            mass=params["mass"], gravity_vec=params["gravity_vec"], J=params["J"], J_inv=params["J_inv"],
            acc_coef=params["acc_coef"], cmd_f_coef=params["cmd_f_coef"], rpy_coef=params["rpy_coef"],
            rpy_rates_coef=params["rpy_rates_coef"], cmd_rpy_coef=params["cmd_rpy_coef"]
        )
        self.nx_phys = X_phys.shape[0]
        
        # Aliases
        self.px, self.py, self.pz = X_phys[0], X_phys[1], X_phys[2]
        self.roll, self.pitch, self.yaw = X_phys[3], X_phys[4], X_phys[5]

        # Extended States
        self.r_cmd_state = MX.sym("r_cmd_state")
        self.p_cmd_state = MX.sym("p_cmd_state")
        self.y_cmd_state = MX.sym("y_cmd_state")
        self.f_cmd_state = MX.sym("f_cmd_state")
        self.theta = MX.sym("theta")

        # Inputs
        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")
        self.v_theta_cmd = MX.sym("v_theta_cmd")

        states = vertcat(X_phys, self.r_cmd_state, self.p_cmd_state, self.y_cmd_state, self.f_cmd_state, self.theta)
        inputs = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd, self.v_theta_cmd)

        # Indices
        self.idx_r_cmd = int(self.nx_phys + 0)
        self.idx_p_cmd = int(self.nx_phys + 1)
        self.idx_y_cmd = int(self.nx_phys + 2)
        self.idx_f_cmd = int(self.nx_phys + 3)
        self.idx_theta = int(self.nx_phys + 4)

        # Link physical inputs to command states
        U_phys_full = vertcat(self.r_cmd_state, self.p_cmd_state, self.y_cmd_state, self.f_cmd_state)
        f_dyn_phys = substitute(X_dot_phys, U_phys, U_phys_full)

        f_dyn = vertcat(
            f_dyn_phys,
            self.dr_cmd, self.dp_cmd, self.dy_cmd, self.df_cmd, # Command integration
            self.v_theta_cmd
        )

        # Params
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
        idx_high = idx_low + 1
        alpha = idx_float - idx_low
        idx_low = if_else(idx_low < 0, 0, idx_low)
        idx_high = if_else(idx_high >= M, M - 1, idx_high)
        p_low = vertcat(*[flattened_points[dim * idx_low + i] for i in range(dim)])
        p_high = vertcat(*[flattened_points[dim * idx_high + i] for i in range(dim)])
        return (1.0 - alpha) * p_low + alpha * p_high

    def _stage_cost_expression(self):
        pos_vec = vertcat(self.px, self.py, self.pz)
        att_vec = vertcat(self.roll, self.pitch, self.yaw)
        ctrl_vec = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)

        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_length)
        pd_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)
        qc_gate = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_gate, dim=1)
        qc_obst = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_obst, dim=1)

        tp_unit = tp_theta / (norm_2(tp_theta) + 1e-6)
        e_theta = pos_vec - pd_theta
        e_lag = dot(tp_unit, e_theta) * tp_unit
        e_contour = e_theta - e_lag

        track_cost = (
            (self.q_l + self.q_l_gate_peak * qc_gate + self.q_l_obst_peak * qc_obst) * dot(e_lag, e_lag)
            + (self.q_c + self.q_c_gate_peak * qc_gate + self.q_c_obst_peak * qc_obst) * dot(e_contour, e_contour)
            + att_vec.T @ self.Q_w @ att_vec
        )
        smooth_cost = ctrl_vec.T @ self.R_df @ ctrl_vec
        speed_cost = (
            - self.miu * self.v_theta_cmd
            + self.w_v_gate * qc_gate * (self.v_theta_cmd ** 2)
            + self.w_v_obst * qc_obst * (self.v_theta_cmd ** 2)
        )
        return track_cost + smooth_cost + speed_cost

    def _build_ocp_and_solver(self, Tf: float, N_horizon: int, trajectory: CubicSpline, verbose: bool = False) -> Tuple[AcadosOcpSolver, AcadosOcp]:
        ocp = AcadosOcp()
        model = self._export_dynamics_model()
        ocp.model = model
        self.nx = model.x.rows()
        self.nu = model.u.rows()
        ocp.solver_options.N_horizon = N_horizon
        ocp.cost.cost_type = "EXTERNAL"

        self.q_l = 200
        self.q_c = 100
        self.Q_w = 1 * DM(np.eye(3))
        self.q_l_gate_peak = 640
        self.q_c_gate_peak = 800
        self.q_l_obst_peak = 100
        self.q_c_obst_peak = 50
        self.R_df = DM(np.diag([0.1, 0.5, 0.5, 0.5]))
        self.miu = 8.0
        self.w_v_gate = 4.0
        self.w_v_obst = 1.0

        ocp.model.cost_expr_ext_cost = self._stage_cost_expression()

        # Constraints
        t_min = float(self._dyn_params["thrust_min"]) * 4.0
        t_max = float(self._dyn_params["thrust_max"]) * 4.0
        
        # State Bounds: [f_cmd, r_cmd, p_cmd, y_cmd]
        ocp.constraints.lbx = np.array([t_min, -1.57, -1.57, -1.57])
        ocp.constraints.ubx = np.array([t_max, 1.57, 1.57, 1.57])
        ocp.constraints.idxbx = np.array([self.idx_f_cmd, self.idx_r_cmd, self.idx_p_cmd, self.idx_y_cmd])

        # Input Bounds
        ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, 0.0])
        ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0, 4.0])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

        ocp.constraints.x0 = np.zeros(self.nx)
        ocp.parameter_values = self._encode_traj_params(self.arc_trajectory)

        # Solver Options
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

    def _encode_traj_params(self, trajectory: CubicSpline) -> np.ndarray:
        theta = np.arange(0.0, self.model_traj_length, self.model_arc_length)
        pd = trajectory(theta)
        tp = trajectory.derivative(1)(theta)
        qc_gate = np.zeros_like(theta)
        qc_obst = np.zeros_like(theta)

        if hasattr(self, "_cached_gate_centers"):
            for g in self._cached_gate_centers:
                d = np.linalg.norm(pd - g, axis=-1)
                qc_gate = np.maximum(qc_gate, np.exp(-2.0 * d**2))
        
        if hasattr(self, "_cached_obstacles"):
            for o in self._cached_obstacles:
                d = np.linalg.norm(pd[:, :2] - o[:2], axis=-1)
                qc_obst = np.maximum(qc_obst, 0.7 * np.exp(-1.0 * d**2))

        return np.concatenate([pd.reshape(-1), tp.reshape(-1), qc_gate, qc_obst])

    # Rebuild & Detect Logic
    def _rebuild_nominal_path_gate(self, obs: dict):
        print(f"T={self._step_count/self._ctrl_freq:.2f}: (Re)building path (gate)...")
        self._cached_gate_centers = obs["gates_pos"]
        self._cached_obstacles = obs["obstacles_pos"]
        # Strict global replanning from start
        self.trajectory, duration = self.planner.build_complete_trajectory(
            self._initial_pos, obs, self._planned_duration
        )
        self._planned_duration = duration

    def _rebuild_nominal_path_obstacle(self, obs: dict):
        print(f"T={self._step_count/self._ctrl_freq:.2f}: (Re)building path (obst)...")
        self._cached_gate_centers = obs["gates_pos"]
        self._cached_obstacles = obs["obstacles_pos"]
        # Strict global replanning from start
        self.trajectory, duration = self.planner.build_complete_trajectory(
            self._initial_pos, obs, self._planned_duration
        )
        self._planned_duration = duration

    def _detect_event_change_gate(self, obs: dict) -> bool:
        if not hasattr(self, "_last_gate_flags"):
            self._last_gate_flags = np.array(obs.get("gates_visited", []), dtype=bool)
            self._last_obst_flags = np.array(obs.get("obstacles_visited", []), dtype=bool)
            return False
        curr = np.array(obs.get("gates_visited", []), dtype=bool)
        if curr.shape != self._last_gate_flags.shape:
            self._last_gate_flags = curr
            return False
        trig = np.any((~self._last_gate_flags) & curr)
        self._last_gate_flags = curr
        return bool(trig)

    def _detect_event_change_obstacle(self, obs: dict) -> bool:
        if not hasattr(self, "_last_obst_flags"): return False
        curr = np.array(obs.get("obstacles_visited", []), dtype=bool)
        if curr.shape != self._last_obst_flags.shape:
            self._last_obst_flags = curr
            return False
        trig = np.any((~self._last_obst_flags) & curr)
        self._last_obst_flags = curr
        return bool(trig)

    # Control Loop
    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray:
        self._current_obs_pos = obs["pos"]
        replanned = False

        if self._detect_event_change_gate(obs):
            self._rebuild_nominal_path_gate(obs)
            replanned = True
        elif self._detect_event_change_obstacle(obs):
            self._rebuild_nominal_path_obstacle(obs)
            replanned = True

        if replanned:
            self.arc_trajectory = self.planner.reparametrize_by_arclength(
                self.planner.extend_spline_tail(self.trajectory, self.model_traj_length)
            )
            param_vec = self._encode_traj_params(self.arc_trajectory)
            for k in range(self.N + 1): self.acados_ocp_solver.set(k, "p", param_vec)

        # State Construction (17 states)
        quat = obs["quat"]
        rpy = R.from_quat(quat).as_euler("xyz", degrees=False)
        drpy = ang_vel2rpy_rates(quat, obs["ang_vel"]) if "ang_vel" in obs else np.zeros(3)
        
        # 1. Physics (12)
        X_phys_now = np.concatenate([obs["pos"], rpy, obs["vel"], drpy])
        
        # 2. Cmd (4) + Theta (1)
        x_now = np.concatenate([
            X_phys_now,
            self.last_rpy_cmd,
            [self.last_f_cmd],
            [self.last_theta]
        ])

        # Warm Start
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

        # Termination
        if self.last_theta >= float(self.arc_trajectory.x[-1]): self.finished = True
        if self._pos_outside_limits(obs["pos"]): self.finished = True
        if self._speed_outside_limits(obs["vel"]): self.finished = True

        status = self.acados_ocp_solver.solve()
        if status != 0: print("acados status:", status)

        self._x_warm = [self.acados_ocp_solver.get(i, "x") for i in range(self.N + 1)]
        self._u_warm = [self.acados_ocp_solver.get(i, "u") for i in range(self.N)]
        
        x_next = self.acados_ocp_solver.get(1, "x")
        
        # Update Internal State (indices based on 17-state vector)
        # r_cmd=12, p_cmd=13, y_cmd=14, f_cmd=15, theta=16
        self.last_rpy_cmd = x_next[12:15]
        self.last_f_cmd = x_next[15]
        self.last_theta = x_next[16]
        
        # Note: rpy.py controller assumes cmd is instantaneous or integrated externally, 
        # so last_act is just copy of last_cmd for output purposes
        self.last_f_act = self.last_f_cmd 
        self.last_f_collective = self.last_f_cmd

        cmd = np.array([self.last_rpy_cmd[0], self.last_rpy_cmd[1], self.last_rpy_cmd[2], self.last_f_cmd])
        self._step_count += 1
        return cmd

    def _pos_outside_limits(self, pos):
        if self.pos_bound is None: return False
        for i in range(3):
            if pos[i] < self.pos_bound[i][0] or pos[i] > self.pos_bound[i][1]: return True
        return False

    def _speed_outside_limits(self, vel):
        if self.velocity_bound is None: return False
        s = np.linalg.norm(vel)
        return not (self.velocity_bound[0] < s < self.velocity_bound[1])

    def step_callback(self, *args, **kwargs) -> bool: return self.finished

    def episode_callback(self):
        print("[MPCC] Episode reset.")
        self._step_count = 0
        self.finished = False
        for attr in ["_last_gate_flags", "_last_obst_flags", "_x_warm", "_u_warm", "_current_obs_pos"]:
            if hasattr(self, attr): delattr(self, attr)
        self.last_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_f_act = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)

    def get_debug_lines(self):
        lines = []
        if hasattr(self, "arc_trajectory"):
            try:
                lines.append((self.arc_trajectory(self.arc_trajectory.x), np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0))
            except: pass
        if hasattr(self, "_x_warm"):
            lines.append((np.asarray([x[:3] for x in self._x_warm]), np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))
        return lines