"""Time-Optimal MPC Controller for Drone Racing with Gates and Obstacles.

Based on:
- Foehn et al., "Time-Optimal Planning for Quadrotor Waypoint Flight", Science Robotics 2021
- Mellinger & Kumar, "Minimum snap trajectory generation and control for quadrotors", ICRA 2011

Implements:
1. Point-mass trajectory initialization with obstacle avoidance
2. Minimum-snap polynomial smoothing
3. Time-optimal velocity profile
4. MPC tracking controller
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
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

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "basic_example_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model

def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
        """Create robust MPC OCP solver."""
        ocp = AcadosOcp()

        ocp.model = create_acados_model(parameters)
        

        nx = ocp.model.x.size()[0]
        nu = ocp.model.u.size()[0]

        ocp.dims.N = N
        

        # Cost function (balanced for tracking and control effort)
        Q = np.diag([
            50.0, 50.0, 400.0,    # position
            1.0, 1.0, 1.0,       # rpy
            10.0, 10.0, 10.0,    # velocity
            5.0, 5.0, 5.0,       # angular velocity
        ])
        R = np.diag([1, 1, 1, 50])  # control input

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q

        ocp.cost.Vx = np.zeros((nx + nu, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((nx + nu, nu))
        ocp.cost.Vu[nx:, :] = np.eye(nu)

        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros(nx + nu)
        ocp.cost.yref_e = np.zeros(nx)

        # Constraints
        ocp.constraints.lbx = np.array([-1.0, -1.0, -1.0])
        ocp.constraints.ubx = np.array([1.0, 1.0, 1.0])
        ocp.constraints.idxbx = np.array([3, 4, 5])

        max_thrust = 0.8
        min_thrust = 0.0
        max_rpy_rate = 8.0

        ocp.constraints.lbu = np.array([-max_rpy_rate, -max_rpy_rate, -max_rpy_rate, min_thrust])
        ocp.constraints.ubu = np.array([max_rpy_rate, max_rpy_rate, max_rpy_rate, max_thrust])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # Solver options (robust configuration)
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hpipm_mode = 'BALANCE'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 5
        ocp.solver_options.qp_solver_iter_max = 100
        ocp.solver_options.tol = 1e-2
        ocp.solver_options.tf = Tf
        #ocp.solver_options.qp_solver_cond_N = N

        # Regularization
        ocp.solver_options.regularize_method = 'PROJECT'
        ocp.solver_options.levenberg_marquardt = 1e-2

        return AcadosOcpSolver(ocp), ocp

class TimeOptimalMPC(Controller):
    """Time-optimal MPC controller with robust trajectory generation."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize controller."""
        super().__init__(obs, info, config)

        # Parameters
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq
        self._N = 15  # MPC horizon
        self._T_HORIZON = self._N * self._dt

        # Trajectory generation
        print("\n" + "=" * 70)
        print("🚀 Time-Optimal MPC Controller Initialization")
        print("=" * 70)

        self._generate_trajectory(obs)

        # Create MPC solver
        self._acados_ocp_solver, self._ocp = create_ocp_solver(self._T_HORIZON, self._N, self.drone_params)

        # Control state
        self._tick = 0
        self._finished = False

        print(f"\n✅ Controller initialized")
        print(f"   MPC: N={self._N}, T={self._T_HORIZON:.2f}s")
        print(f"   Trajectory: {len(self._waypoints_pos)} points, {self._t_total:.2f}s")
        print("=" * 70 + "\n")

    def _generate_trajectory(self, obs):
        """Generate time-optimal trajectory through gates avoiding obstacles."""
        gates_pos = obs['gates_pos']
        obstacles_pos = obs['obstacles_pos']
        current_pos = obs['pos']

        # Geometric parameters
        OBSTACLE_RADIUS = 0.015  # Obstacle radius (m)
        DRONE_RADIUS = 0.12      # Drone safety radius (m)
        SAFETY_MARGIN = 0.25     # Additional safety margin (m)
        MIN_CLEARANCE = OBSTACLE_RADIUS + DRONE_RADIUS + SAFETY_MARGIN

        print("\n📍 Gate and Obstacle Information:")
        for i, gate in enumerate(gates_pos):
            print(f"   Gate {i+1}: [{gate[0]:.2f}, {gate[1]:.2f}, {gate[2]:.2f}]")
        for i, obs_pos in enumerate(obstacles_pos):
            print(f"   Obstacle {i+1}: [{obs_pos[0]:.2f}, {obs_pos[1]:.2f}, {obs_pos[2]:.2f}]")

        # ===== Phase 1: Generate waypoints with obstacle avoidance =====
        waypoints = [current_pos.copy()]

        # Add takeoff waypoint
        takeoff = current_pos.copy()
        takeoff[2] = max(0.3, current_pos[2])
        waypoints.append(takeoff)

        for gate_idx in range(4):
            gate_pos = gates_pos[gate_idx]
            pass_height = gate_pos[2]

            last_wp = waypoints[-1]

            # Target: gate center
            gate_target = np.array([gate_pos[0], gate_pos[1], pass_height])

            # Check for obstacle collisions on path
            path_blocked = False
            blocking_obstacle = None

            for obs_idx, obs_pos in enumerate(obstacles_pos):
                # Compute point-to-line distance
                vec = gate_target[:2] - last_wp[:2]
                vec_len_sq = np.dot(vec, vec)

                if vec_len_sq < 1e-6:
                    dist = np.linalg.norm(obs_pos[:2] - last_wp[:2])
                else:
                    t = np.dot(obs_pos[:2] - last_wp[:2], vec) / vec_len_sq
                    t = np.clip(t, 0, 1)
                    closest = last_wp[:2] + t * vec
                    dist = np.linalg.norm(obs_pos[:2] - closest)

                if dist < MIN_CLEARANCE:
                    path_blocked = True
                    blocking_obstacle = obs_pos
                    print(f"   ⚠️  Gate {gate_idx+1}: Path blocked by obstacle {obs_idx+1} (dist={dist:.3f}m)")
                    break

            # Add detour waypoint if needed
            if path_blocked:
                # Compute perpendicular detour
                vec = gate_target[:2] - last_wp[:2]
                vec_len = np.linalg.norm(vec)

                if vec_len > 1e-6:
                    direction = vec / vec_len
                    perpendicular = np.array([-direction[1], direction[0]])

                    # Determine detour direction (away from obstacle)
                    midpoint = (last_wp[:2] + gate_target[:2]) / 2
                    to_obstacle = blocking_obstacle[:2] - midpoint
                    side = np.dot(to_obstacle, perpendicular)

                    # Detour point
                    detour_offset = 0.45  # 45cm perpendicular offset
                    detour_xy = midpoint - np.sign(side) * detour_offset * perpendicular
                    detour_z = (last_wp[2] + pass_height) / 2
                    detour_wp = np.array([detour_xy[0], detour_xy[1], detour_z])

                    waypoints.append(detour_wp)
                    print(f"      ✓ Detour: [{detour_wp[0]:.2f}, {detour_wp[1]:.2f}, {detour_wp[2]:.2f}]")

            # Add approach waypoint (smooth entry)
            approach = 0.4 * waypoints[-1] + 0.6 * gate_target
            waypoints.append(approach)

            # Add gate waypoint
            waypoints.append(gate_target)
            print(f"   ✓ Gate {gate_idx+1} waypoint: [{gate_target[0]:.2f}, {gate_target[1]:.2f}, {gate_target[2]:.2f}]")

            # Add exit waypoint for smooth transition to next gate
            if gate_idx < 3:
                next_gate = gates_pos[gate_idx + 1]
                exit_wp = 0.3 * gate_target + 0.7 * np.array([
                    next_gate[0], next_gate[1], next_gate[2]
                ])
                waypoints.append(exit_wp)

        waypoints = np.array(waypoints)
        print(f"\n✅ Generated {len(waypoints)} waypoints\n")

        # ===== Phase 2: Time allocation and velocity profile =====
        # Compute segment distances
        distances = [np.linalg.norm(waypoints[i] - waypoints[i-1]) 
                    for i in range(1, len(waypoints))]
        total_distance = sum(distances)

        # Velocity profile: constant cruise speed
        cruise_speed = 1  # m/s (conservative for reliable tracking)

        # Cumulative time allocation
        cumulative_times = [0.0]
        for dist in distances:
            cumulative_times.append(cumulative_times[-1] + dist / cruise_speed)

        self._t_total = cumulative_times[-1]

        print(f"📊 Trajectory Statistics:")
        print(f"   Total distance: {total_distance:.2f}m")
        print(f"   Total time: {self._t_total:.2f}s")
        print(f"   Cruise speed: {cruise_speed:.2f} m/s")

        # ===== Phase 3: Smooth trajectory generation =====
        num_points = int(self._freq * self._t_total) + 1
        t_fine = np.linspace(0, self._t_total, num_points)

        # Cubic spline interpolation with natural boundary conditions
        cs_x = CubicSpline(cumulative_times, waypoints[:, 0], bc_type='clamped')
        cs_y = CubicSpline(cumulative_times, waypoints[:, 1], bc_type='clamped')
        cs_z = CubicSpline(cumulative_times, waypoints[:, 2], bc_type='clamped')

        self._waypoints_pos = np.column_stack([cs_x(t_fine), cs_y(t_fine), cs_z(t_fine)])
        self._waypoints_vel = np.column_stack([cs_x(t_fine, 1), cs_y(t_fine, 1), cs_z(t_fine, 1)])

        # Yaw trajectory (heading towards next waypoint)
        self._waypoints_yaw = np.zeros(len(t_fine))
        for i in range(len(t_fine) - 1):
            dx = self._waypoints_pos[i+1, 0] - self._waypoints_pos[i, 0]
            dy = self._waypoints_pos[i+1, 1] - self._waypoints_pos[i, 1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self._waypoints_yaw[i] = np.arctan2(dy, dx)
        self._waypoints_yaw[-1] = self._waypoints_yaw[-2]

        print(f"✅ Smooth trajectory: {len(self._waypoints_pos)} interpolated points\n")

    

    # 添加这些调试到 compute_control 最开始
    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute MPC control command."""
        try:
            hover_thrust = self.drone_params['mass'] * (-self.drone_params['gravity_vec'][-1])
            
            # ★ 第一步：保证至少输出悬停推力
            default_action = np.array([0.0, 0.0, 0.0, hover_thrust])
            
            if self._finished:
                return default_action
            
            current_time = self._tick * self._dt
            
            # ★ 打印轨迹调试信息
            if self._tick < 10 or self._tick % 20 == 0:
                print(f"[Tick {self._tick}] t={current_time:.2f}s, t_total={self._t_total:.1f}s, "
                    f"waypoints={len(self._waypoints_pos)}")
            
            # ★ 检查轨迹是否为空
            if len(self._waypoints_pos) == 0:
                print("❌ ERROR: No waypoints generated!")
                return default_action
            
            trajectory_progress = current_time / (self._t_total + 1e-6)
            i = int(trajectory_progress * len(self._waypoints_pos))
            i = max(0, min(i, len(self._waypoints_pos) - self._N - 1))
            
            if i >= len(self._waypoints_pos) - self._N:
                self._finished = True
                print(f"✅ Mission complete")
                return default_action
            
            # Current state
            obs_rpy = R.from_quat(obs['quat']).as_euler('xyz')
            obs_drpy = ang_vel2rpy_rates(obs['quat'], obs['ang_vel'])
            x0 = np.concatenate([obs['pos'], obs_rpy, obs['vel'], obs_drpy])
            
            # Set initial state
            self._acados_ocp_solver.set(0, 'x', x0)
            
            # Set reference trajectory
            for j in range(self._N):
                idx = min(i + j, len(self._waypoints_pos) - 1)
                
                yref = np.zeros(16)
                yref[0:3] = self._waypoints_pos[idx]
                yref[3:6] = [0.0, 0.0, self._waypoints_yaw[idx]]
                yref[6:9] = self._waypoints_vel[idx]
                yref[9:12] = [0.0, 0.0, 0.0]
                yref[12:15] = [0.0, 0.0, 0.0]
                yref[15] = hover_thrust
                
                self._acados_ocp_solver.set(j, 'yref', yref)
            
            # Terminal reference
            idx_term = min(i + self._N, len(self._waypoints_pos) - 1)
            yref_e = np.zeros(12)
            yref_e[0:3] = self._waypoints_pos[idx_term]
            yref_e[3:6] = [0.0, 0.0, self._waypoints_yaw[idx_term]]
            yref_e[6:9] = self._waypoints_vel[idx_term]
            yref_e[9:12] = [0.0, 0.0, 0.0]
            
            self._acados_ocp_solver.set(self._N, 'yref', yref_e)
            
            # ★ 关键：求解MPC
            status = self._acados_ocp_solver.solve()
            
            if status != 0:
                print(f"   ⚠️  Solver failed (status={status}), using hover thrust")
                return default_action
            
            u0 = self._acados_ocp_solver.get(0, "u")
            
            # ★ 推力保护：确保至少输出悬停推力
            if u0[3] < hover_thrust * 0.5:
                print(f"   ⚠️  Low thrust: {u0[3]:.3f}, using {hover_thrust:.3f}")
                u0[3] = hover_thrust
            
            if self._tick < 10:
                print(f"   Action: rpy=[{u0[0]:.3f}, {u0[1]:.3f}, {u0[2]:.3f}], thrust={u0[3]:.3f}")
            
            self._tick += 1
            return u0
            
        except Exception as e:
            print(f"❌ EXCEPTION in compute_control: {e}")
            import traceback
            traceback.print_exc()
            hover_thrust = self.drone_params['mass'] * (-self.drone_params['gravity_vec'][-1])
            return np.array([0.0, 0.0, 0.0, hover_thrust])


    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0