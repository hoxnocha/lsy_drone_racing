from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import DM, MX, dot, floor, if_else, norm_2, substitute, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from drone_models.core import load_params
from drone_models.so_rpy_rotor import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Helper Class
class FrameUtils:
    """extraction coordinate frame  from quaternions."""
    @staticmethod
    def quat_to_axis(quat: NDArray[np.floating], axis_index: int = 1) -> NDArray[np.floating]:
        """Extract a specific column from the rotation matrix derived from quaternion."""
        rot = R.from_quat(quat)
        mats = np.asarray(rot.as_matrix())
        if mats.ndim == 3:
            return mats[:, :, axis_index]
        if mats.ndim == 2:
            return mats[:, axis_index]
        return None

    @staticmethod
    def extract_gate_frames(gates_quaternions: NDArray[np.floating]) -> Tuple[NDArray, NDArray, NDArray]:
        """Extract Normal(x), Y-axis, Z-axis from gate quaternions."""
        normals = FrameUtils.quat_to_axis(gates_quaternions, axis_index=0)
        y_axes = FrameUtils.quat_to_axis(gates_quaternions, axis_index=1)
        z_axes = FrameUtils.quat_to_axis(gates_quaternions, axis_index=2)
        return normals, y_axes, z_axes


#  Path Planner Class
class RacingPathPlanner:
    """
    Handles all trajectory generation, spline fitting, re-parameterization,
    and obstacle/gate detours.
    """

    def __init__(self, ctrl_freq: float):
        self.ctrl_freq = ctrl_freq

    def build_gate_waypoints(
        self,
        start_pos: NDArray[np.floating],
        gates_positions: NDArray[np.floating],
        gates_normals: NDArray[np.floating],
        half_span: float = 0.5,
        samples_per_gate: int = 5,
    ) -> NDArray[np.floating]:
        """Generate guide waypoints passing through gates."""
        n_gates = gates_positions.shape[0]
        grid = []
        for idx in range(samples_per_gate):
            alpha = idx / (samples_per_gate - 1) if samples_per_gate > 1 else 0.0
            grid.append(gates_positions - half_span * gates_normals + 2.0 * half_span * alpha * gates_normals)
        stacked = np.stack(grid, axis=1).reshape(n_gates, samples_per_gate, 3).reshape(-1, 3)
        return np.vstack([start_pos[None, :], stacked])

    def spline_through_points(self, duration: float, waypoints: NDArray[np.floating]) -> CubicSpline:
        """Fit a time-parameterized CubicSpline through waypoints."""
        diffs = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        t_axis = cum_len / (cum_len[-1] + 1e-6) * duration
        return CubicSpline(t_axis, waypoints)

    def reparametrize_by_arclength(
        self, trajectory: CubicSpline, arc_step: float = 0.05, epsilon: float = 1e-5
    ) -> CubicSpline:
        """
        Reparameterize spline by arc length.
        """
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
        """Extend the end of the spline linearly."""
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

    def insert_gate_detours(
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
        """
        Inserts detour waypoints to ensure proper gate entry angles.
        """
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
            if delta_norm < 1e-6: continue

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
                # [FIX] Insert at the correct index relative to current list expansion
                insert_idx = last_idx_curr_gate + 1
                wp_list.insert(insert_idx, detour_wp)
                extra_inserted += 1

        return np.asarray(wp_list)

    def _find_collision_indices(
        self,
        trajectory_points: NDArray[np.floating],
        obstacle_pos: NDArray[np.floating],
        safe_dist: float
    ) -> Tuple[int, int, bool]:
        """
        Identifies the start and end indices of the trajectory segment that violates 
        the safety distance.
        """
        d_xy = np.linalg.norm(trajectory_points[:, :2] - obstacle_pos[:2], axis=1)
        inside_mask = d_xy < safe_dist

        if not np.any(inside_mask):
            return -1, -1, False

        inside_indices = np.where(inside_mask)[0]
        start_idx = max(int(inside_indices[0]) - 1, 0)
        end_idx = min(int(inside_indices[-1]) + 1, len(trajectory_points) - 1)

        # Check if the segment is valid (has length)
        if end_idx <= start_idx + 1:
            return -1, -1, False
            
        return start_idx, end_idx, True

    def _generate_arc_detour(
        self,
        p_start: NDArray[np.floating],
        p_end: NDArray[np.floating],
        center: NDArray[np.floating],
        radius: float,
        n_points: int
    ) -> NDArray[np.floating]:
        """
        Generates a 3D arc of points around a center between start and end points.
        """
        # 1. Calculate vectors from center to start/end
        v_start = p_start[:2] - center[:2]
        v_end = p_end[:2] - center[:2]
        
        # Normalize
        nrm_s = np.linalg.norm(v_start) + 1e-9
        nrm_e = np.linalg.norm(v_end) + 1e-9
        v_start /= nrm_s
        v_end /= nrm_e

        # 2. Calculate angles
        theta_start = np.arctan2(v_start[1], v_start[0])
        theta_end = np.arctan2(v_end[1], v_end[0])

        # 3. Handle angle wrapping (ensure shortest path around circle)
        d_theta = theta_end - theta_start
        if d_theta > np.pi:
            d_theta -= 2.0 * np.pi
        elif d_theta < -np.pi:
            d_theta += 2.0 * np.pi

        # 4. Generate points
        theta_list = np.linspace(theta_start, theta_start + d_theta, n_points + 2)[1:-1]
        detour_points = []
        
        for i, th in enumerate(theta_list):
            # Interpolate Z (height) linearly
            alpha = (i + 1) / (n_points + 1)
            z = (1.0 - alpha) * p_start[2] + alpha * p_end[2]
            
            # XY position on circle
            p_x = center[0] + radius * np.cos(th)
            p_y = center[1] + radius * np.sin(th)
            detour_points.append(np.array([p_x, p_y, z]))

        return np.array(detour_points)

    def inject_obstacle_detours(
        self,
        base_waypoints: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
        planned_duration: float,
        gate_positions: NDArray[np.floating],
        safe_dist: float,
        arc_n: int = 5,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Injects circular detours around obstacles where the path violates safety margins.
        """
        # 1. Initial Sampling
        pre_spline = self.spline_through_points(planned_duration, base_waypoints)
        n_samples = max(int(self.ctrl_freq * planned_duration), 2)
        
        t_axis = np.linspace(0.0, planned_duration, n_samples)
        wp_samples = pre_spline(t_axis)

        gate_margin = 3 # indices margin to avoid modifying path inside a gate

        # 2. Iterate over obstacles
        for obst in obstacles_pos:
            
            # Check collision
            start_idx, end_idx, collision = self._find_collision_indices(wp_samples, obst, safe_dist)
            if not collision:
                continue

            # Check if this collision is too close to a gate (don't modify gate entry)
            # (Simplified check: look for nearest gate index)
            is_near_gate = False
            if len(gate_positions) > 0:
                # Find indices of points closest to gates
                gate_indices = [np.argmin(np.linalg.norm(wp_samples - g, axis=1)) for g in gate_positions]
                gate_indices = np.array(gate_indices)
                
                # Check overlap
                if np.any((gate_indices >= start_idx - gate_margin) & (gate_indices <= end_idx + gate_margin)):
                    is_near_gate = True
            
            if is_near_gate:
                continue

            # 3. Generate Detour
            p_start = wp_samples[start_idx]
            p_end = wp_samples[end_idx]
            
            t_start = t_axis[start_idx]
            t_end = t_axis[end_idx]
            t_detour = np.linspace(t_start, t_end, arc_n + 2)[1:-1]
            
            detour_pts = self._generate_arc_detour(p_start, p_end, obst, safe_dist, arc_n)

            # 4. Splice Arrays
            # Construct new lists: [Head] + [Detour] + [Tail]
            new_t = np.concatenate([t_axis[:start_idx+1], t_detour, t_axis[end_idx:]])
            new_wp = np.concatenate([wp_samples[:start_idx+1], detour_pts, wp_samples[end_idx:]])

            t_axis = new_t
            wp_samples = new_wp

        # 5. Finalize
        if t_axis.size > 0:
            _, idx_unique = np.unique(t_axis, return_index=True)
            t_axis = t_axis[idx_unique]
            wp_samples = wp_samples[idx_unique]

        if t_axis.size < 2:
            print("[Planner] Warning: Avoidance path invalid, reverting to nominal.")
            fallback_t = self.spline_through_points(planned_duration, base_waypoints).x
            return fallback_t, base_waypoints

        return t_axis, wp_samples

    def build_complete_trajectory(
        self,
        start_pos: NDArray[np.floating],
        obs: dict,
        planned_duration: float
    ) -> Tuple[CubicSpline, float]:
        """
        Orchestrates the full path generation process:
        Gate WPs -> Gate Detours -> Obstacle Detours -> Spline.
        """
        gate_positions = obs["gates_pos"]
        obstacle_positions = obs["obstacles_pos"]
        gate_quats = obs["gates_quat"]

        gate_normals, gate_y, gate_z = FrameUtils.extract_gate_frames(gate_quats)

        base_waypoints = self.build_gate_waypoints(
            start_pos, gate_positions, gate_normals
        )

        if base_waypoints.shape[0] > 1:
            base_waypoints[1:, 2] += 0.0

        with_gate_detours = self.insert_gate_detours(
            base_waypoints, gate_positions, gate_normals, gate_y, gate_z,
        )

        t_axis, collision_free_wps = self.inject_obstacle_detours(
            with_gate_detours, obstacle_positions, planned_duration, gate_positions, safe_dist=0.2
        )

        if len(t_axis) < 2:
            print("[Planner] Warning: obstacle-avoid path fallback.")
            trajectory = self.spline_through_points(planned_duration, with_gate_detours)
        else:
            trajectory = CubicSpline(t_axis, collision_free_wps)
        
        return trajectory, float(trajectory.x[-1])



# MPCC Controller Class
class MPCC(Controller):
    """
    Model Predictive Contouring Control for drone racing.
    Logic strictly mirrors mpcc_rotor.py.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self._cfg = config

        # Dynamics model
        self._dyn_params = load_params("so_rpy_rotor", config.sim.drone_model)
        mass_val = float(self._dyn_params["mass"])
        gravity_mag = -float(self._dyn_params["gravity_vec"][-1])
        self.hover_thrust = mass_val * gravity_mag
        
        # Actuator lag constants
        self.tau_rpy_act = 0.05
        self.tau_yaw_act = 0.08
        self.tau_f_act = 0.10

        # Input rate limits
        self.rate_limit_df = 10.0
        self.rate_limit_drpy = 10.0
        self.rate_limit_v_theta = 4.0

        # Path Planning State
        self._initial_pos = obs["pos"]
        self._cached_gate_centers = obs["gates_pos"]
        self._cached_obstacles = obs["obstacles_pos"]
        self._planned_duration = 30.0
        
        self.planner = RacingPathPlanner(ctrl_freq=self._ctrl_freq)

        # Build initial path (Logic: Gate mode uses initial_pos)
        self._rebuild_nominal_path_gate(obs)

        # MPC Configuration 
        self.N = 35
        self.T_HORIZON = 0.7
        self.dt = self.T_HORIZON / self.N
        self.model_arc_length = 0.05
        self.model_traj_length = 12.0

        # Prepare spline for solver
        self.arc_trajectory = self.planner.reparametrize_by_arclength(
            self.planner.extend_spline_tail(self.trajectory, extend_length=self.model_traj_length)
        )

        # Build Solver
        self.acados_ocp_solver, self.ocp = self._build_ocp_and_solver(
            self.T_HORIZON, self.N, self.arc_trajectory
        )

        # Safety Bounds
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

    # Dynamics & Solver Setup
    def _export_dynamics_model(self) -> AcadosModel:
        """Exports the symbolic dynamics model. Strictly matches original."""
        model_name = "lsy_example_mpc_real"
        params = self._dyn_params

        # Physical Dynamics
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

        # State Mapping
        self.px, self.py, self.pz = X_phys[0], X_phys[1], X_phys[2]
        self.roll, self.pitch, self.yaw = X_phys[3], X_phys[4], X_phys[5]
        
        # Symbolic States
        self.r_cmd_state = MX.sym("r_cmd_state")
        self.p_cmd_state = MX.sym("p_cmd_state")
        self.y_cmd_state = MX.sym("y_cmd_state")
        self.f_cmd_state = MX.sym("f_cmd_state")
        
        self.r_act = MX.sym("r_act")
        self.p_act = MX.sym("p_act")
        self.y_act = MX.sym("y_act")
        self.f_act = MX.sym("f_act")
        
        self.theta = MX.sym("theta")

        # Control Inputs
        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")
        self.v_theta_cmd = MX.sym("v_theta_cmd")

        # Vectors
        states = vertcat(
            X_phys,
            self.r_cmd_state, self.p_cmd_state, self.y_cmd_state, self.f_cmd_state,
            self.r_act, self.p_act, self.y_act, self.f_act,
            self.theta,
        )
        inputs = vertcat(
            self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd, self.v_theta_cmd,
        )

        # Indices
        self.idx_r_cmd_state = int(self.nx_phys + 0)
        self.idx_p_cmd_state = int(self.nx_phys + 1)
        self.idx_y_cmd_state = int(self.nx_phys + 2)
        self.idx_f_cmd_state = int(self.nx_phys + 3)

        self.idx_r_act = int(self.nx_phys + 4)
        self.idx_p_act = int(self.nx_phys + 5)
        self.idx_y_act = int(self.nx_phys + 6)
        self.idx_f_act = int(self.nx_phys + 7)
        self.idx_theta = int(self.nx_phys + 8)

        # Dynamics equation linking
        U_phys_full = vertcat(self.r_act, self.p_act, self.y_act, self.f_act)
        f_dyn_phys = substitute(X_dot_phys, U_phys, U_phys_full)

        tau_rpy = float(self.tau_rpy_act)
        tau_yaw = float(self.tau_yaw_act)
        tau_f = float(self.tau_f_act)

        f_dyn = vertcat(
            f_dyn_phys,
            self.dr_cmd, self.dp_cmd, self.dy_cmd, self.df_cmd,  # Command states integration
            (self.r_cmd_state - self.r_act) / tau_rpy,
            (self.p_cmd_state - self.p_act) / tau_rpy,
            (self.y_cmd_state - self.y_act) / tau_yaw,
            (self.f_cmd_state - self.f_act) / tau_f,
            self.v_theta_cmd  # Theta integration
        )

        # Cost parameters
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
        """CasADi symbolic linear interpolation."""
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
        """Constructs the symbolic cost function."""
        pos_vec = vertcat(self.px, self.py, self.pz)
        att_vec = vertcat(self.roll, self.pitch, self.yaw)
        ctrl_vec = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)

        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_length)

        # Interpolate
        pd_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)
        qc_gate_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_gate, dim=1)
        qc_obst_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_obst, dim=1)

        tp_unit = tp_theta / (norm_2(tp_theta) + 1e-6)
        e_theta = pos_vec - pd_theta
        e_lag = dot(tp_unit, e_theta) * tp_unit
        e_contour = e_theta - e_lag

        # Weights
        q_l_term = (self.q_l + self.q_l_gate_peak * qc_gate_theta + self.q_l_obst_peak * qc_obst_theta)
        q_c_term = (self.q_c + self.q_c_gate_peak * qc_gate_theta + self.q_c_obst_peak * qc_obst_theta)
        
        track_cost = (
            q_l_term * dot(e_lag, e_lag)
            + q_c_term * dot(e_contour, e_contour)
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
        """Configures and builds the ACADOS solver."""
        ocp = AcadosOcp()
        model = self._export_dynamics_model()
        ocp.model = model

        self.nx = model.x.rows()
        self.nu = model.u.rows()
        ocp.solver_options.N_horizon = N_horizon
        ocp.cost.cost_type = "EXTERNAL"

        # TUNING WEIGHTS(from auto tune)
        self.q_l = 522.327621281147
        self.q_c = 279.45878291502595
        self.Q_w = 1 * DM(np.eye(3))

        self.q_l_gate_peak = 520.2687042765319
        self.q_c_gate_peak = 764.3037075176835#820

        self.q_l_obst_peak = 207.83845749683678
        self.q_c_obst_peak = 110.51885732449591 #130

        self.R_df = DM(np.diag([0.1, 0.5, 0.5, 0.5]))

        self.miu = 14.3377785384655 #13.8
        self.w_v_gate = 2.7327203765511516
        self.w_v_obst = 2.460291111562401

        ocp.model.cost_expr_ext_cost = self._stage_cost_expression()

        # Constraints
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

        ocp.constraints.lbu = np.array([-self.rate_limit_df, -self.rate_limit_drpy, -self.rate_limit_drpy, -self.rate_limit_drpy, 0.0])
        ocp.constraints.ubu = np.array([self.rate_limit_df, self.rate_limit_drpy, self.rate_limit_drpy, self.rate_limit_drpy, self.rate_limit_v_theta])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])
        
        # Initial Parameters
        ocp.constraints.x0 = np.zeros(self.nx)
        param_vec = self._encode_traj_params(self.arc_trajectory)
        ocp.parameter_values = param_vec

        # Solver configuration
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

    # Path Planning & Parameter Encoding
    def _encode_traj_params(self, trajectory: CubicSpline) -> np.ndarray:
        """Samples the trajectory and generates weights."""
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

        return np.concatenate([
            pd_vals.reshape(-1), tp_vals.reshape(-1), qc_gate, qc_obst,
        ])

    def _rebuild_nominal_path_gate(self, obs: dict[str, NDArray[np.floating]]):
        """Replanning triggered by Gate: Uses INITIAL position (Global Consistency)."""
        print(f"T={self._step_count / self._ctrl_freq:.2f}: (Re)building nominal path (gate)...")
        # Update Cache
        self._cached_gate_centers = obs["gates_pos"]
        self._cached_obstacles = obs["obstacles_pos"]
        
        # build full trajectory
        self.trajectory, duration = self.planner.build_complete_trajectory(
            self._initial_pos,  # Strict: Use Initial Pos
            obs=obs,
            planned_duration=self._planned_duration
        )
        self._planned_duration = duration

    def _rebuild_nominal_path_obstacle(self, obs: dict[str, NDArray[np.floating]]):
        """Replanning triggered by obstacles: Uses INITIAL position (Global Consistency)."""
        
        print(f"T={self._step_count / self._ctrl_freq:.2f}: (Re)building nominal path (obstacle -> FORCE GLOBAL)...")
        self._cached_gate_centers = obs["gates_pos"]
        self._cached_obstacles = obs["obstacles_pos"]

        self.trajectory, duration = self.planner.build_complete_trajectory(
            self._initial_pos,  
            obs=obs,
            planned_duration=self._planned_duration
        )
        self._planned_duration = duration
    # --------------------------------------------------------------------------
    # Detection & Control Loop
    # --------------------------------------------------------------------------

    def _detect_event_change_gate(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        if not hasattr(self, "_last_gate_flags"):
            self._last_gate_flags = np.array(obs.get("gates_visited", []), dtype=bool)
            self._last_obst_flags = np.array(obs.get("obstacles_visited", []), dtype=bool)
            return False

        curr_gates = np.array(obs.get("gates_visited", []), dtype=bool)
        if curr_gates.shape != self._last_gate_flags.shape:
            self._last_gate_flags = curr_gates
            return False

        trigger = np.any((~self._last_gate_flags) & curr_gates)
        self._last_gate_flags = curr_gates
        return bool(trigger)

    def _detect_event_change_obstacle(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        if not hasattr(self, "_last_obst_flags"):
            return False 

        curr_obst = np.array(obs.get("obstacles_visited", []), dtype=bool)
        if curr_obst.shape != self._last_obst_flags.shape:
            self._last_obst_flags = curr_obst
            return False

        trigger = np.any((~self._last_obst_flags) & curr_obst)
        self._last_obst_flags = curr_obst
        return bool(trigger)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:

        self._current_obs_pos = obs["pos"]
        replanned = False

        # Event Detection & Replanning(get actual gates or obstacles position)
        if self._detect_event_change_gate(obs):
            print(f"T={self._step_count / self._ctrl_freq:.2f}: MPCC detected gate/env change, replanning...")
            self._rebuild_nominal_path_gate(obs)
            replanned = True
            
        elif self._detect_event_change_obstacle(obs):
            print(f"T={self._step_count / self._ctrl_freq:.2f}: MPCC detected obstacle/env change, replanning...")
            self._rebuild_nominal_path_obstacle(obs)
            replanned = True

        # Update Solver Params
        if replanned:
            # re-generate Spline
            self.arc_trajectory = self.planner.reparametrize_by_arclength(
                self.planner.extend_spline_tail(self.trajectory, extend_length=self.model_traj_length)
            )
            # recalculate (pd, tp, qc_gate, qc_obst)
            param_vec = self._encode_traj_params(self.arc_trajectory)
            for k in range(self.N + 1):
                self.acados_ocp_solver.set(k, "p", param_vec)

        # ----------------------------------------------------------------------
        # State Construction
        # ----------------------------------------------------------------------
        quat = obs["quat"]
        rpy = R.from_quat(quat).as_euler("xyz", degrees=False)
        
        # Euler Rates
        if "ang_vel" in obs:
            drpy = ang_vel2rpy_rates(quat, obs["ang_vel"])
        else:
            drpy = np.zeros(3, dtype=float)

        # physical states 12 dimensions(pos, rpy, vel, drpy)
        X_phys_now_full = np.zeros(self.nx_phys, dtype=float)
        X_phys_now_full[0:3] = obs["pos"]
        X_phys_now_full[3:6] = rpy
        X_phys_now_full[6:9] = obs["vel"]
        X_phys_now_full[9:12] = drpy
        # other dimensions of X_phys remain 0, let solver handles them

        # [X_phys, r_cmd, p_cmd, y_cmd, f_cmd, r_act, p_act, y_act, f_act, theta]
        x_now = np.zeros(self.nx, dtype=float)
        x_now[0:self.nx_phys] = X_phys_now_full
        
        # command states(last command)
        x_now[self.idx_r_cmd_state] = self.last_rpy_cmd[0]
        x_now[self.idx_p_cmd_state] = self.last_rpy_cmd[1]
        x_now[self.idx_y_cmd_state] = self.last_rpy_cmd[2]
        x_now[self.idx_f_cmd_state] = self.last_f_cmd
        
        #  (last input)
        x_now[self.idx_r_act] = self.last_rpy_act[0]
        x_now[self.idx_p_act] = self.last_rpy_act[1]
        x_now[self.idx_y_act] = self.last_rpy_act[2]
        x_now[self.idx_f_act] = self.last_f_act
        
    
        x_now[self.idx_theta] = self.last_theta

        # ----------------------------------------------------------------------
        # Solver Setup
        # ----------------------------------------------------------------------
        
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

        # x0
        self.acados_ocp_solver.set(0, "lbx", x_now)
        self.acados_ocp_solver.set(0, "ubx", x_now)

        # ----------------------------------------------------------------------
        # Termination Checks
        # ----------------------------------------------------------------------
        if self.last_theta >= float(self.arc_trajectory.x[-1]):
            self.finished = True
            print("[MPCC] Stop: finished path.")
        
        if self._pos_outside_limits(obs["pos"]):
            self.finished = True
            print("[MPCC] Stop: position out of safe bounds.")
            
        if self._speed_outside_limits(obs["vel"]):
            self.finished = True
            print("[MPCC] Stop: velocity out of safe range.")

    
        # Solve & Output
        status = self.acados_ocp_solver.solve()
        
        if status != 0:
            print("[MPCC] acados solver returned non-zero status:", status)
            # return np.array([0, 0, 0, self.hover_thrust])

        self._x_warm = [self.acados_ocp_solver.get(i, "x") for i in range(self.N + 1)]
        self._u_warm = [self.acados_ocp_solver.get(i, "u") for i in range(self.N)]

        x_next = self.acados_ocp_solver.get(1, "x")

        self.last_rpy_cmd = np.array([
            x_next[self.idx_r_cmd_state], 
            x_next[self.idx_p_cmd_state], 
            x_next[self.idx_y_cmd_state]
        ])
        self.last_f_cmd = float(x_next[self.idx_f_cmd_state])
        
        self.last_rpy_act = np.array([
            x_next[self.idx_r_act], 
            x_next[self.idx_p_act], 
            x_next[self.idx_y_act]
        ])
        self.last_f_act = float(x_next[self.idx_f_act])
        self.last_f_collective = self.last_f_act
        self.last_theta = float(x_next[self.idx_theta])

        # command [roll, pitch, yaw, thrust]
        cmd = np.array([
            self.last_rpy_cmd[0], 
            self.last_rpy_cmd[1], 
            self.last_rpy_cmd[2], 
            self.last_f_cmd
        ], dtype=float)

        self._step_count += 1
        return cmd

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

    # Callbacks & Debug
    def step_callback(self, *args, **kwargs) -> bool:
        return self.finished

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
        self.last_rpy_act = np.zeros(3)

    def get_debug_lines(self):
        debug_lines = []
        if hasattr(self, "arc_trajectory"):
            try:
                full_path = self.arc_trajectory(self.arc_trajectory.x)
                debug_lines.append((full_path, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0))
            except Exception: pass
        if hasattr(self, "_x_warm"):
            pred_states = np.asarray([x_state[:3] for x_state in self._x_warm])
            debug_lines.append((pred_states, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))
        if hasattr(self, "last_theta") and hasattr(self, "arc_trajectory") and hasattr(self, "_current_obs_pos"):
            try:
                target_on_path = self.arc_trajectory(self.last_theta)
                segment = np.stack([self._current_obs_pos, target_on_path])
                debug_lines.append((segment, np.array([0.0, 0.0, 1.0, 1.0]), 1.0, 1.0))
            except Exception: pass
        return debug_lines