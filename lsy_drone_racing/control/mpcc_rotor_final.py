from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

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


class MathUtils:
    """Static utility functions for quaternion processing, vector calculations"""

    @staticmethod
    def quat_to_axis(quat: NDArray[np.floating], axis_index: int = 1) -> NDArray[np.floating]:
        """Convert quaternion to rotation matrix and extract specific axis (0:x, 1:y, 2:z)."""
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
        normals = MathUtils.quat_to_axis(gates_quaternions, axis_index=0)
        y_axes = MathUtils.quat_to_axis(gates_quaternions, axis_index=1)
        z_axes = MathUtils.quat_to_axis(gates_quaternions, axis_index=2)
        return normals, y_axes, z_axes

    @staticmethod
    def normalize_vec(vec: NDArray[np.floating]) -> NDArray[np.floating]:
        nrm = np.linalg.norm(vec)
        return vec if nrm < 1e-6 else vec / nrm


class RacingPathPlanner:
    """
    Path planner responsible for generation, re-parameterization, 
    and corrections of trajectory for gates and obstacles.
    """

    def __init__(self, ctrl_freq: float):
        self.ctrl_freq = ctrl_freq

    def build_trajectory(
        self, 
        obs: dict, 
        current_pos: NDArray[np.floating], 
        planned_duration: float
    ) -> Tuple[CubicSpline, float]:
        """
        Build complete trajectory including gate passing and obstacle avoidance.
        Returns: (arclength_spline, total_duration)
        """
        gate_positions = obs["gates_pos"]
        obstacle_positions = obs["obstacles_pos"]
        gate_quats = obs["gates_quat"]

        # Extract gate frame coordinates
        gate_normals, gate_y, gate_z = MathUtils.extract_gate_frames(gate_quats)

        # Build base gate waypoints
        base_waypoints = self._build_gate_waypoints(current_pos, gate_positions, gate_normals)
        
        # Simple altitude compensation 
        if base_waypoints.shape[0] > 1:
            base_waypoints[1:, 2] += 0.0

        # detours (ensure gate passing angle)
        with_gate_detours = self._insert_gate_detours(
            base_waypoints, gate_positions, gate_normals, gate_y, gate_z
        )

        # obstacle avoidance
        # Note: Pass cached_gate_centers to avoid obstacle modification near gates
        t_axis, collision_free_wps = self._inject_obstacle_detours(
            with_gate_detours, obstacle_positions, 
            planned_duration, gate_positions, safe_dist=0.2
        )

        # 5. Generate spline
        if len(t_axis) < 2:
            print("[Planner] Warning: obstacle-avoid path fallback.")
            trajectory = self.spline_through_points(planned_duration, with_gate_detours)
        else:
            trajectory = CubicSpline(t_axis, collision_free_wps)
        
        new_duration = float(trajectory.x[-1])
        return trajectory, new_duration

    def prepare_mpcc_trajectory(
        self, 
        base_trajectory: CubicSpline, 
        model_traj_length: float
    ) -> CubicSpline:
        """
        Prepare trajectory for MPCC:
        1. Extend tail (prevent out of range during prediction)
        2. Reparameterize by arc length
        """
        extended = self.extend_spline_tail(base_trajectory, extend_length=model_traj_length)
        arc_traj = self.reparametrize_by_arclength(extended)
        return arc_traj

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

        # 99 iterations to ensure convergence
        for _ in range(99):
            n_segments = max(2, int(total_param_range / arc_step))
            t_samples = np.linspace(0.0, total_param_range, n_segments)
            pts = trajectory(t_samples)
            
            deltas = np.diff(pts, axis=0)
            seg_lengths = np.linalg.norm(deltas, axis=1)
            cum_arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            
            total_param_range = float(cum_arc[-1])
            trajectory = CubicSpline(cum_arc, pts)
            
            # Convergence check: ensures points are evenly spaced
            if np.std(seg_lengths) <= epsilon:
                return CubicSpline(cum_arc, pts)

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

    def _build_gate_waypoints(
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
            # Simple strategy: Generate auxiliary points before and after the gate along the normal vector
            grid.append(gates_positions - half_span * gates_normals + 2.0 * half_span * alpha * gates_normals)
        stacked = np.stack(grid, axis=1).reshape(n_gates, samples_per_gate, 3).reshape(-1, 3)
        return np.vstack([start_pos[None, :], stacked])

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
        """
        If the angle is too large when entering the gate, 
        add extra guide points to adjust attitude.
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
            angle_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

            if angle_deg > angle_threshold:
                gate_center = gate_positions[gate_idx]
                tangential = delta_vec - np.dot(delta_vec, normal_i) * normal_i
                tangential_norm = np.linalg.norm(tangential)

                if tangential_norm < 1e-6:
                    detour_dir = gate_y_axes[gate_idx]
                else:
                    tangential /= tangential_norm
                    proj_y = np.dot(tangential, gate_y_axes[gate_idx])
                    proj_z = np.dot(tangential, gate_z_axes[gate_idx])
                    proj_angle = np.degrees(np.arctan2(proj_z, proj_y))

                    if -90.0 <= proj_angle < 45.0:
                        detour_dir = gate_y_axes[gate_idx]
                    elif 45.0 <= proj_angle < 135.0:
                        detour_dir = gate_z_axes[gate_idx]
                    else:
                        detour_dir = -gate_y_axes[gate_idx]

                detour_wp = gate_center + detour_distance * detour_dir
                wp_list.insert(last_idx_curr_gate + 1, detour_wp)
                extra_inserted += 1

        return np.asarray(wp_list)

    def _inject_obstacle_detours(
        self,
        base_waypoints: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
        planned_duration: float,
        gate_positions: NDArray[np.floating],
        safe_dist: float,
        arc_n: int = 5,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        
        # Initial spline for sampling
        pre_spline = self.spline_through_points(planned_duration, base_waypoints)
        
        n_samples = int(self.ctrl_freq * planned_duration)
        if n_samples <= 0: n_samples = 1
            
        t_axis = np.linspace(0.0, planned_duration, n_samples)
        wp_samples = pre_spline(t_axis)
        gate_margin = 3 

        for obst in obstacles_pos:
            gate_idx = np.array([], dtype=int)
            if len(gate_positions) > 0:
                idx_list = []
                for g in gate_positions:
                    # Find closest sample point to each gate center
                    d_g = np.linalg.norm(wp_samples - g, axis=1)
                    idx_list.append(int(np.argmin(d_g)))
                gate_idx = np.array(idx_list, dtype=int)
 
            # Check distance to obstacle (XY plane)
            d_xy = np.linalg.norm(wp_samples[:, :2] - obst[:2], axis=1)
            inside = d_xy < safe_dist

            if not np.any(inside):
                continue

            inside_idx = np.where(inside)[0]
            start_idx = int(inside_idx[0]) - 1
            end_idx = int(inside_idx[-1]) + 1

            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, len(t_axis) - 1)

            if end_idx <= start_idx + 1:
                continue

            # If the detour overlaps with a gate area, SKIP avoidance to prevent hitting the gate frame
            if gate_idx.size > 0:
                # Check if any gate index falls within [start - margin, end + margin]
                is_near_gate = np.any((gate_idx >= start_idx - gate_margin) & 
                                      (gate_idx <= end_idx + gate_margin))
                if is_near_gate:
                    continue
            
            p_start = wp_samples[start_idx]
            p_end = wp_samples[end_idx]
            
            # Geometry: Calculate Arc
            v_start = p_start[:2] - obst[:2]
            v_end = p_end[:2] - obst[:2]
            nrm_s = np.linalg.norm(v_start)
            nrm_e = np.linalg.norm(v_end)
            
            if nrm_s < 1e-6 or nrm_e < 1e-6:
                continue
                
            v_start /= nrm_s
            v_end /= nrm_e
            
            theta_start = np.arctan2(v_start[1], v_start[0])
            theta_end = np.arctan2(v_end[1], v_end[0])
            d_theta = theta_end - theta_start
            
            # Shortest path logic
            if d_theta > np.pi: d_theta -= 2.0 * np.pi
            elif d_theta < -np.pi: d_theta += 2.0 * np.pi

            # Interpolate arc points
            theta_list = np.linspace(theta_start, theta_start + d_theta, arc_n + 2)[1:-1]
            t_list = np.linspace(t_axis[start_idx], t_axis[end_idx], arc_n + 2)[1:-1]

            detour_points = []
            for i, th in enumerate(theta_list):
                dir_xy = np.array([np.cos(th), np.sin(th)])
                p_xy = obst[:2] + dir_xy * safe_dist
                
                # Linear interpolation for Z
                alpha = (i + 1) / (arc_n + 1)
                z = (1.0 - alpha) * p_start[2] + alpha * p_end[2]
                detour_points.append(np.array([p_xy[0], p_xy[1], z]))

            # Splicing: Construct new arrays
            # 1. Before obstacle
            new_t_vals = list(t_axis[:start_idx+1])
            new_points = list(wp_samples[:start_idx+1])
            
            # 2. The Arc
            for t_i, p_i in zip(t_list, detour_points):
                new_t_vals.append(float(t_i))
                new_points.append(p_i)
                
            # 3. After obstacle
            new_t_vals.extend(t_axis[end_idx:])
            new_points.extend(wp_samples[end_idx:])
            
            # Update main arrays for next iteration
            t_axis = np.array(new_t_vals)
            wp_samples = np.array(new_points)

        # Final cleanup: Remove duplicates and ensure strictly increasing time
        if t_axis.size > 0:
            _, idx_unique = np.unique(t_axis, return_index=True)
            t_axis = t_axis[idx_unique]
            wp_samples = wp_samples[idx_unique]
            
        # Fallback check
        if len(t_axis) < 2:
            print("[Planner] Warning: obstacle-avoid path fallback.")
            fallback_spline = self.spline_through_points(planned_duration, base_waypoints)
            return fallback_spline.x, base_waypoints

        return t_axis, wp_samples


class MPCC(Controller):
    """
    Model Predictive Contouring Control for drone racing.
    Integrates so_rpy_rotor model
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
        self._cached_gate_centers = obs["gates_pos"]
        self._cached_obstacles = obs["obstacles_pos"]
        self._planned_duration = 30.0

        self.planner = RacingPathPlanner(ctrl_freq=self._ctrl_freq)
        
        self._replan(obs)

        # MPC setup 
        self.N = 35
        self.T_HORIZON = 0.7
        self.dt = self.T_HORIZON / self.N
        self.model_arc_length = 0.05
        self.model_traj_length = 12.0

        # Generate arc-length parameterized trajectory for OCP
        self.arc_trajectory = self.planner.prepare_mpcc_trajectory(
            self.trajectory, self.model_traj_length
        )

        # Build solver
        self.acados_ocp_solver, self.ocp = self._build_ocp_and_solver(
            self.T_HORIZON, self.N
        )

        param_vec = self._encode_traj_params(self.arc_trajectory)
        for k in range(self.N + 1):
            self.acados_ocp_solver.set(k, "p", param_vec)
        # Safety bounds
        self.pos_bound = [np.array([-2.6, 2.6]), np.array([-2.0, 1.8]), np.array([-0.1, 2.0])]
        self.velocity_bound = [-1.0, 4.0]

        # State recording
        self.last_theta = 0.0
        self.last_v_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_f_act = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self.last_rpy_act = np.zeros(3)
        self.finished = False

    def _replan(self, obs: dict):
        """Call Planner to regenerate nominal trajectory"""
        # Update cached positions for Cost Function usage
        self._cached_gate_centers = obs["gates_pos"]
        self._cached_obstacles = obs["obstacles_pos"]
        
        traj, duration = self.planner.build_trajectory(
            obs, 
            current_pos=self._initial_pos if self._step_count == 0 else obs["pos"], 
            planned_duration=self._planned_duration
        )
        self.trajectory = traj
        self._planned_duration = duration

    def _export_dynamics_model(self) -> AcadosModel:
        """
        Define symbolic model containing real dynamics + actuator lag + virtual state theta.
        """
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

        # 5. Link dynamics equations
        # Replace physical model input U_phys with actuator state U_phys_full
        U_phys_full = vertcat(self.r_act, self.p_act, self.y_act, self.f_act)
        f_dyn_phys = substitute(X_dot_phys, U_phys, U_phys_full)

        f_dyn = vertcat(
            f_dyn_phys,
            self.dr_cmd, self.dp_cmd, self.dy_cmd, self.df_cmd, # cmd dot
            (self.r_cmd_state - self.r_act) / float(self.tau_rpy_act),
            (self.p_cmd_state - self.p_act) / float(self.tau_rpy_act),
            (self.y_cmd_state - self.y_act) / float(self.tau_yaw_act),
            (self.f_cmd_state - self.f_act) / float(self.tau_f_act),
            self.v_theta_cmd
        )

        # 6. Parameters (Spline trajectory params + weight maps)
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
        """Linear interpolation for CasADi symbolic functions"""
        M = len(theta_vec)
        idx_float = (theta - theta_vec[0]) / (theta_vec[-1] - theta_vec[0]) * (M - 1)
        idx_low = floor(idx_float)
        alpha = idx_float - idx_low
        idx_low = if_else(idx_low < 0, 0, idx_low)
        idx_high = if_else(idx_low + 1 >= M, M - 1, idx_low + 1)
        
        # Simplified reshape logic for code brevity, principle remains the same
        p_low = vertcat(*[flattened_points[dim * idx_low + i] for i in range(dim)])
        p_high = vertcat(*[flattened_points[dim * idx_high + i] for i in range(dim)])
        return (1.0 - alpha) * p_low + alpha * p_high

    def _stage_cost_expression(self):
        """Build MPCC Stage Cost"""
        pos_vec = vertcat(self.px, self.py, self.pz)
        att_vec = vertcat(self.roll, self.pitch, self.yaw)
        ctrl_vec = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)

        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_length)
        
        # Interpolate to get reference point and weights for current theta
        pd = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)
        w_gate = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_gate, dim=1)
        w_obst = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_obst, dim=1)

        tp_unit = tp / (norm_2(tp) + 1e-6)
        e_err = pos_vec - pd
        e_lag = dot(tp_unit, e_err) * tp_unit
        e_cont = e_err - e_lag

        # Dynamic weights
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
        
        
        # --- Weight configuration ---
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

        # --- Constraints ---
        thrust_min = float(self._dyn_params["thrust_min"]) * 4.0
        thrust_max = float(self._dyn_params["thrust_max"]) * 4.0
        
        # State constraints: Actuator & Cmd Limits
        idx_r = self.idx_r_cmd_state
        idx_p = self.idx_p_cmd_state
        idx_y = self.idx_y_cmd_state
        idx_f_cmd = self.idx_f_cmd_state
        idx_f_act = self.idx_f_act
        ocp.constraints.lbx = np.array([thrust_min, thrust_min, -1.57, -1.57, -1.57])
        ocp.constraints.ubx = np.array([thrust_max, thrust_max, 1.57, 1.57, 1.57])
        ocp.constraints.idxbx = np.array([idx_f_act, idx_f_cmd, idx_r, idx_p, idx_y])
        # Input constraints: Rate Limits
        ocp.constraints.lbu = np.array([-self.rate_limit_df, -self.rate_limit_drpy, -self.rate_limit_drpy, -self.rate_limit_drpy, 0.0])
        ocp.constraints.ubu = np.array([self.rate_limit_df, self.rate_limit_drpy, self.rate_limit_drpy, self.rate_limit_drpy, self.rate_limit_v_theta])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])
        ocp.solver_options.tol = 1e-5
        ocp.solver_options.qp_solver_cond_N = N_horizon
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.qp_solver_iter_max = 20
        ocp.solver_options.nlp_solver_max_iter = 50

        ocp.constraints.x0 = np.zeros(self.nx)


        
        # Fill initial parameters (prevent errors, will be overwritten later)
        ocp.parameter_values = np.zeros(model.p.rows())

        # Solver Options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        
        ocp.solver_options.tf = Tf

        solver = AcadosOcpSolver(ocp, json_file="mpcc_prescripted_real_dyn.json", verbose=False)
        return solver, ocp

    def _encode_traj_params(self, trajectory: CubicSpline) -> np.ndarray:
        """Generate parameter vector required by Acados (pos, tangent, weights)"""
        theta_samples = np.arange(0.0, self.model_traj_length, self.model_arc_length)
        pd_vals = trajectory(theta_samples)
        tp_vals = trajectory.derivative(1)(theta_samples)

        qc_gate = np.zeros_like(theta_samples, dtype=float)
        qc_obst = np.zeros_like(theta_samples, dtype=float)

        # Simple weight map generation: closer to gate/obstacle -> higher weight
        if len(self._cached_gate_centers) > 0:
            for gc in self._cached_gate_centers:
                d = np.linalg.norm(pd_vals - gc, axis=-1)
                qc_gate = np.maximum(qc_gate, np.exp(-2.0 * d**2))
        
        if len(self._cached_obstacles) > 0:
            for oc in self._cached_obstacles:
                d = np.linalg.norm(pd_vals[:, :2] - oc[:2], axis=-1)
                qc_obst = np.maximum(qc_obst, 0.7 * np.exp(-1.0 * d**2))

        return np.concatenate([pd_vals.reshape(-1), tp_vals.reshape(-1), qc_gate, qc_obst])


    # Control Loop
    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray[np.floating]:
        self._current_obs_pos = obs["pos"]

        # 1. Detect environment change and replan
        if self._detect_env_change(obs):
            print(f"T={self._step_count/self._ctrl_freq:.2f}: Env changed, replanning...")
            self._replan(obs)
            self.arc_trajectory = self.planner.prepare_mpcc_trajectory(
                self.trajectory, self.model_traj_length
            )
            # Update Solver parameters
            param_vec = self._encode_traj_params(self.arc_trajectory)
            for k in range(self.N + 1):
                self.acados_ocp_solver.set(k, "p", param_vec)

        # 2. State estimation 
        quat = obs["quat"]
        rpy = R.from_quat(quat).as_euler("xyz", degrees=False)
        drpy = ang_vel2rpy_rates(quat, obs["ang_vel"]) if "ang_vel" in obs else np.zeros(3)
        
        x_now = np.zeros(self.nx)
        # Physical part
        x_now[0:3] = obs["pos"]
        x_now[3:6] = rpy
        x_now[6:9] = obs["vel"]
        x_now[9:12] = drpy
        # Extended part (Cmd, Act, Theta)
        x_now[self.idx_r_cmd_state:self.idx_f_cmd_state+1] = list(self.last_rpy_cmd) + [self.last_f_cmd]
        x_now[self.idx_r_act:self.idx_f_act+1] = list(self.last_rpy_act) + [self.last_f_act]
        x_now[self.idx_theta] = self.last_theta

        # 3. Solver setup (Warm Start & Init)
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

        # 4. Termination check
        if self.last_theta >= float(self.arc_trajectory.x[-1]) or \
           self._check_safety(obs["pos"], obs["vel"]):
            self.finished = True

        # 5. Solve
        status = self.acados_ocp_solver.solve()
        if status != 0:
            print(f"[MPCC] Solver failed with status {status}")

        # 6. Extract results
        self._x_warm = [self.acados_ocp_solver.get(i, "x") for i in range(self.N + 1)]
        self._u_warm = [self.acados_ocp_solver.get(i, "u") for i in range(self.N)]
        
        x_next = self._x_warm[1]
        self.last_rpy_cmd = x_next[self.idx_r_cmd_state:self.idx_y_cmd_state+1]
        self.last_f_cmd = float(x_next[self.idx_f_cmd_state])
        self.last_rpy_act = x_next[self.idx_r_act:self.idx_y_act+1]
        self.last_f_act = float(x_next[self.idx_f_act])
        self.last_theta = float(x_next[self.idx_theta])

        self._step_count += 1
        
        cmd = np.concatenate([self.last_rpy_cmd, [self.last_f_cmd]])
        return cmd


    def _detect_env_change(self, obs: dict) -> bool:
        """Simple event detection: Gate or obstacle state change"""
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
        # Draw Trajectory
        if hasattr(self, "arc_trajectory"):
            path = self.arc_trajectory(self.arc_trajectory.x)
            lines.append((path, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0))
        # Draw Prediction
        if hasattr(self, "_x_warm"):
            pred = np.array([x[:3] for x in self._x_warm])
            lines.append((pred, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))
        return lines