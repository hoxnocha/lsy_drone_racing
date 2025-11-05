"""改进版的Time-Optimal MPC控制器 - 适应30秒时间限制"""

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


def create_acados_model():
    """Create Acados model for MPC."""
    drone_params = load_params('cf21B500')

    x_sym, u_sym, f_sym = symbolic_dynamics_euler(
        model=drone_params['model'],
        mass=drone_params['mass'],
        Ixx=drone_params['Ixx'],
        Iyy=drone_params['Iyy'],
        Izz=drone_params['Izz'],
        grav=drone_params['grav']
    )

    model = AcadosModel()
    model.name = 'drone_mpc'
    model.x = x_sym
    model.u = u_sym
    model.f_expl_expr = f_sym

    return model, drone_params


class TimeOptimalMPCImproved(Controller):
    """改进的时间最优MPC控制器 - 在30秒内完成赛道"""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize controller."""
        super().__init__(obs, info, config)

        self.drone_params = load_params('cf21B500')
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq
        self._N = 15  # ★ 减小预测步数（从25改为15）以加快求解
        self._T_HORIZON = self._N * self._dt

        print("\n" + "=" * 70)
        print("🚀 改进型 Time-Optimal MPC Controller")
        print("=" * 70)
        print(f"环境频率: {self._freq} Hz")
        print(f"时间限制: 30秒 (1500步)")
        print(f"MPC: N={self._N}, T={self._T_HORIZON:.3f}s")

        self._generate_trajectory(obs)

        self._acados_ocp_solver, self._ocp = self._create_ocp_solver()

        self._tick = 0
        self._finished = False

        print(f"✅ Controller ready")
        print(f"   Trajectory: {len(self._waypoints_pos)} points, {self._t_total:.2f}s")
        print("=" * 70 + "\n")

    def _generate_trajectory(self, obs):
        """生成快速轨迹 - 必须在30秒内完成"""
        gates_pos = obs['gates_pos']
        obstacles_pos = obs['obstacles_pos']
        current_pos = obs['pos']

        OBSTACLE_RADIUS = 0.015
        DRONE_RADIUS = 0.12
        SAFETY_MARGIN = 0.15  # ★ 减小安全裕度
        MIN_CLEARANCE = OBSTACLE_RADIUS + DRONE_RADIUS + SAFETY_MARGIN

        waypoints = [current_pos.copy()]

        # ★ 不添加额外的起飞点，直接向第一个门靠近

        for gate_idx in range(4):
            gate_pos = gates_pos[gate_idx]
            pass_height = gate_pos[2]

            last_wp = waypoints[-1]
            gate_target = np.array([gate_pos[0], gate_pos[1], pass_height])

            # 检测障碍物
            path_blocked = False
            blocking_obstacle = None

            for obs_idx, obs_pos in enumerate(obstacles_pos):
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

            # 如果需要绕行
            if path_blocked:
                vec = gate_target[:2] - last_wp[:2]
                vec_len = np.linalg.norm(vec)

                if vec_len > 1e-6:
                    direction = vec / vec_len
                    perpendicular = np.array([-direction[1], direction[0]])

                    midpoint = (last_wp[:2] + gate_target[:2]) / 2
                    to_obstacle = blocking_obstacle[:2] - midpoint
                    side = np.dot(to_obstacle, perpendicular)

                    # ★ 减小绕行距离
                    detour_offset = 0.35
                    detour_xy = midpoint - np.sign(side) * detour_offset * perpendicular
                    detour_z = (last_wp[2] + pass_height) / 2
                    detour_wp = np.array([detour_xy[0], detour_xy[1], detour_z])

                    waypoints.append(detour_wp)

            # ★ 删除额外的过渡点，直接添加门的位置
            waypoints.append(gate_target)

            # ★ 删除出门点（直接连接到下一个门的方向）

        waypoints = np.array(waypoints)

        # ★ 计算总距离和快速速度
        total_distance = 0
        for i in range(1, len(waypoints)):
            total_distance += np.linalg.norm(waypoints[i] - waypoints[i-1])

        # ★ 关键：提高速度以适应30秒限制
        # 假设需要在20秒内完成轨迹（留10秒余量）
        cruise_speed = min(2.0, total_distance / 20.0)  # ★ 提高速度到最多2.0 m/s

        # ★ 加速响应：前2个航点使用更高速度
        waypoint_times = [0.0]
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(waypoints[i] - waypoints[i-1])

            if i <= 2:
                # 快速起飞和接近第一个门
                speed = cruise_speed * 1.5
            else:
                speed = cruise_speed

            waypoint_times.append(waypoint_times[-1] + dist / speed)

        self._t_total = waypoint_times[-1]

        print(f"📊 轨迹统计:")
        print(f"   航点数: {len(waypoints)}")
        print(f"   总距离: {total_distance:.2f}m")
        print(f"   总时间: {self._t_total:.2f}s (限制: 30s)")
        print(f"   巡航速度: {cruise_speed:.2f} m/s")

        # 插值
        waypoint_times = np.array(waypoint_times)
        num_points = int(self._freq * self._t_total) + 1
        t_fine = np.linspace(0, self._t_total, num_points)

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

    def _create_ocp_solver(self):
        """创建MPC求解器 - 优化参数以适应低频环境"""
        model, _ = create_acados_model()

        ocp = AcadosOcp()
        ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]

        ocp.dims.N = self._N

        # ★ 调整权重：更激进的跟踪
        Q = np.diag([
            100.0, 100.0, 100.0,  # 位置权重高
            10.0, 10.0, 10.0,     # 姿态
            30.0, 30.0, 30.0,     # 速度
            5.0, 5.0, 5.0,        # 角速率
        ])
        R = np.diag([0.05, 0.05, 0.05, 0.05])  # ★ 减小控制惩罚

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

        # 约束
        ocp.constraints.lbx = np.array([-1.0, -1.0, -1.0])
        ocp.constraints.ubx = np.array([1.0, 1.0, 1.0])
        ocp.constraints.idxbx = np.array([3, 4, 5])

        max_thrust = 1.0
        max_rpy_rate = 10.0

        ocp.constraints.lbu = np.array([-max_rpy_rate, -max_rpy_rate, -max_rpy_rate, 0.0])
        ocp.constraints.ubu = np.array([max_rpy_rate, max_rpy_rate, max_rpy_rate, max_thrust])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # ★ 求解器配置：优化速度
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hpipm_mode = 'SPEED'  # ★ 改为SPEED而不是ROBUST
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 3  # ★ 减少迭代
        ocp.solver_options.qp_solver_iter_max = 50  # ★ 减少QP迭代
        ocp.solver_options.tol = 1e-2  # ★ 放宽容差
        ocp.solver_options.tf = self._T_HORIZON

        return AcadosOcpSolver(ocp), ocp

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute MPC control."""
        try:
            hover_thrust = self.drone_params['mass'] * (-self.drone_params['gravity_vec'][-1])
            default_action = np.array([0.0, 0.0, 0.0, hover_thrust])

            if self._finished:
                return default_action

            current_time = self._tick * self._dt

            if len(self._waypoints_pos) == 0:
                return default_action

            trajectory_progress = current_time / (self._t_total + 1e-6)
            i = int(trajectory_progress * len(self._waypoints_pos))
            i = max(0, min(i, len(self._waypoints_pos) - self._N - 1))

            if i >= len(self._waypoints_pos) - self._N:
                self._finished = True
                return default_action

            # State
            obs_rpy = R.from_quat(obs['quat']).as_euler('xyz')
            obs_drpy = ang_vel2rpy_rates(obs['quat'], obs['ang_vel'])
            x0 = np.concatenate([obs['pos'], obs_rpy, obs['vel'], obs_drpy])

            self._acados_ocp_solver.set(0, 'x', x0)

            # References
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

            idx_term = min(i + self._N, len(self._waypoints_pos) - 1)
            yref_e = np.zeros(12)
            yref_e[0:3] = self._waypoints_pos[idx_term]
            yref_e[3:6] = [0.0, 0.0, self._waypoints_yaw[idx_term]]
            yref_e[6:9] = self._waypoints_vel[idx_term]
            yref_e[9:12] = [0.0, 0.0, 0.0]

            self._acados_ocp_solver.set(self._N, 'yref', yref_e)

            # Solve
            status = self._acados_ocp_solver.solve()

            if status != 0:
                return default_action

            u0 = self._acados_ocp_solver.get(0, "u")

            if u0[3] < hover_thrust * 0.3:
                u0[3] = hover_thrust

            self._tick += 1
            return u0

        except Exception as e:
            print(f"❌ Error: {e}")
            hover_thrust = self.drone_params['mass'] * (-self.drone_params['gravity_vec'][-1])
            return np.array([0.0, 0.0, 0.0, hover_thrust])

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        """Step callback."""
        return self._finished

    def episode_callback(self):
        """Episode callback."""
        pass
