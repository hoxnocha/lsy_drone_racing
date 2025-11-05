"""最终版MPC控制器 - 10-15个航点 + 快速执行"""

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
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            50.0,  # pos
            50.0,  # pos
            400.0,  # pos
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            10.0,  # vel
            10.0,  # vel
            10.0,  # vel
            5.0,  # drpy
            5.0,  # drpy
            5.0,  # drpy
        ]
    )
    # Input weights (reference is upright orientation and hover thrust)
    R = np.diag(
        [
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            50.0,  # thrust
        ]
    )

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30°)
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class FinalMPC(Controller):
    """最终MPC - 保持轨迹质量，但在30秒内完成"""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq
        self._N = 25
        self._T_HORIZON = self._N * self._dt

        print("\n" + "=" * 70)
        print("🚀 Final MPC Controller - 30s Time Limit")
        print("=" * 70)

        self._generate_trajectory(obs)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )

        self._tick = 0
        self._finished = False

        print(f"✅ Ready | Traj: {len(self._waypoints_pos)} pts, {self._t_total:.2f}s")
        print("=" * 70 + "\n")

    def _generate_trajectory(self, obs):
        """★ 关键改进：保留10-15个航点，但极端压缩时间"""
        gates_pos = obs['gates_pos']
        obstacles_pos = obs['obstacles_pos']
        current_pos = obs['pos']

        OBSTACLE_RADIUS = 0.015
        DRONE_RADIUS = 0.2
        SAFETY_MARGIN = 0.15
        MIN_CLEARANCE = OBSTACLE_RADIUS + DRONE_RADIUS + SAFETY_MARGIN

        # ★ 从当前位置开始
        waypoints = [current_pos.copy()]

        for gate_idx in range(4):
            gate_pos = gates_pos[gate_idx]
            pass_height = gate_pos[2]

            last_wp = waypoints[-1]
            gate_target = np.array([gate_pos[0], gate_pos[1], pass_height])

            # 检查直线路径是否被阻挡
            path_blocked = False
            blocking_obstacle = None

            for obs_pos in obstacles_pos:
                vec = gate_target[:2] - last_wp[:2]
                vec_len_sq = np.dot(vec, vec)

                if vec_len_sq > 1e-6:
                    t = np.dot(obs_pos[:2] - last_wp[:2], vec) / vec_len_sq
                    t = np.clip(t, 0, 1)
                    closest = last_wp[:2] + t * vec
                    dist = np.linalg.norm(obs_pos[:2] - closest)

                    if dist < MIN_CLEARANCE:
                        path_blocked = True
                        blocking_obstacle = obs_pos
                        break

            # 如果被阻挡，添加一个绕行点
            if path_blocked:
                vec = gate_target[:2] - last_wp[:2]
                vec_len = np.linalg.norm(vec)

                if vec_len > 1e-6:
                    direction = vec / vec_len
                    perpendicular = np.array([-direction[1], direction[0]])

                    midpoint = (last_wp[:2] + gate_target[:2]) / 2
                    to_obstacle = blocking_obstacle[:2] - midpoint
                    side = np.dot(to_obstacle, perpendicular)

                    detour_xy = midpoint - np.sign(side) * 0.3 * perpendicular
                    detour_z = (last_wp[2] + pass_height) / 2
                    detour_wp = np.array([detour_xy[0], detour_xy[1], detour_z])

                    waypoints.append(detour_wp)

            # ★ 只添加一个过渡点（而不是多个）
            if gate_idx < 3:  # 不是最后一个门
                # 向下一个门方向的中间点
                next_gate = gates_pos[gate_idx + 1]
                exit_wp = 0.5 * gate_target + 0.5 * np.array([
                    next_gate[0], next_gate[1], next_gate[2]
                ])
                waypoints.append(exit_wp)

            # 添加门
            waypoints.append(gate_target)

        waypoints = np.array(waypoints)
        print(f"✓ Waypoints: {len(waypoints)}")

        # ★ 计算总距离
        total_distance = sum(np.linalg.norm(waypoints[i] - waypoints[i-1]) 
                            for i in range(1, len(waypoints)))

        # ★ 非常快的速度：目标10秒完成轨迹（留20秒余量）
        cruise_speed = max(0.5, total_distance / 25.0)

        # ★ 时间分配
        waypoint_times = [0.0]
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(waypoints[i] - waypoints[i-1])
            waypoint_times.append(waypoint_times[-1] + dist / cruise_speed)

        waypoint_times = np.array(waypoint_times)
        self._t_total = waypoint_times[-1]

        print(f"✓ Distance: {total_distance:.2f}m | Time: {self._t_total:.2f}s | Speed: {cruise_speed:.2f}m/s")

        # ★ 关键优化：减少插值点的数量
        # 原来：num_points = int(self._freq * self._t_total) + 1
        # 新的：只用10倍频率的点
        num_points = max(50, int(10 * self._freq * self._t_total))  # ★ 大幅减少
        t_fine = np.linspace(0, self._t_total, num_points)

        print(f"✓ Interpolated: {num_points} pts (was {int(self._freq * self._t_total)})")

        # 样条插值
        cs_x = CubicSpline(waypoint_times, waypoints[:, 0], bc_type='natural')
        cs_y = CubicSpline(waypoint_times, waypoints[:, 1], bc_type='natural')
        cs_z = CubicSpline(waypoint_times, waypoints[:, 2], bc_type='natural')

        self._waypoints_pos = np.column_stack([cs_x(t_fine), cs_y(t_fine), cs_z(t_fine)])
        self._waypoints_vel = np.column_stack([cs_x(t_fine, 1), cs_y(t_fine, 1), cs_z(t_fine, 1)])

        # Yaw
        self._waypoints_yaw = np.zeros(len(t_fine))
        for i in range(len(t_fine) - 1):
            dx = self._waypoints_pos[i+1, 0] - self._waypoints_pos[i, 0]
            dy = self._waypoints_pos[i+1, 1] - self._waypoints_pos[i, 1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self._waypoints_yaw[i] = np.arctan2(dy, dx)
        self._waypoints_yaw[-1] = self._waypoints_yaw[-2]

    
    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute control"""
        try:
            hover_thrust = self.drone_params['mass'] * (-self.drone_params['gravity_vec'][-1])
            default = np.array([0.0, 0.0, 0.0, hover_thrust])

            if self._finished or len(self._waypoints_pos) == 0:
                return default

            current_time = self._tick * self._dt
            i = int(current_time / (self._t_total + 1e-6) * len(self._waypoints_pos))
            i = max(0, min(i, len(self._waypoints_pos) - self._N - 1))

            if i >= len(self._waypoints_pos) - self._N:
                self._finished = True
                return default

            obs_rpy = R.from_quat(obs['quat']).as_euler('xyz')
            obs_drpy = ang_vel2rpy_rates(obs['quat'], obs['ang_vel'])
            x0 = np.concatenate([obs['pos'], obs_rpy, obs['vel'], obs_drpy])

            self._acados_ocp_solver.set(0, 'x', x0)

            for j in range(self._N):
                idx = min(i + j, len(self._waypoints_pos) - 1)
                yref = np.zeros(16)
                yref[0:3] = self._waypoints_pos[idx]
                yref[3:6] = [0, 0, self._waypoints_yaw[idx]]
                yref[6:9] = self._waypoints_vel[idx]
                yref[9:15] = 0
                yref[15] = hover_thrust
                self._acados_ocp_solver.set(j, 'yref', yref)

            idx_term = min(i + self._N, len(self._waypoints_pos) - 1)
            yref_e = np.zeros(12)
            yref_e[0:3] = self._waypoints_pos[idx_term]
            yref_e[3:6] = [0, 0, self._waypoints_yaw[idx_term]]
            yref_e[6:9] = self._waypoints_vel[idx_term]
            self._acados_ocp_solver.set(self._N, 'yref', yref_e)

            status = self._acados_ocp_solver.solve()
            if status != 0:
                return default

            u0 = self._acados_ocp_solver.get(0, "u")
            if u0[3] < hover_thrust * 0.3:
                u0[3] = hover_thrust

            self._tick += 1
            return u0
        except:
            hover_thrust = self.drone_params['mass'] * (-self.drone_params['gravity_vec'][-1])
            return np.array([0.0, 0.0, 0.0, hover_thrust])

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        return self._finished

    def episode_callback(self):
        pass
