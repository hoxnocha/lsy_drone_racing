"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""
from __future__ import annotations

#[obs] dictionary:
#  pos: [-1.4643728   0.7039031   0.01799869]
#  quat: [-0.03229322 -0.04964727  0.02113089  0.9980209 ]
#  vel: [0. 0. 0.]
#  ang_vel: [0. 0. 0.]
#  target_gate: 0
#  gates_pos: [[ 0.5   0.25  0.7 ]
# [ 1.05  0.75  1.2 ]
# [-1.   -0.25  0.7 ]
# [ 0.   -0.75  1.2 ]]
#  gates_quat: [[ 0.0000000e+00  0.0000000e+00 -3.8018841e-01  9.2490906e-01]
# [ 0.0000000e+00  0.0000000e+00  9.2268986e-01  3.8554308e-01]
# [ 0.0000000e+00  0.0000000e+00  9.9999970e-01  7.9632673e-04]
# [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]
#  gates_visited: [False False False False]
#  obstacles_pos: [[ 0.    0.75  1.55]
# [ 1.    0.25  1.55]
# [-1.5  -0.25  1.55]
# [-0.5  -0.75  1.55]]
#  obstacles_visited: [False False False False]


"""Improved MPC for drone racing - Level 0/1/2"""


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
    
    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    
    # Weights - State weights
    Q = np.diag([
        50.0, 50.0, 400.0,    # pos (x, y, z) - z 最重要
        1.0, 1.0, 1.0,        # rpy
        10.0, 10.0, 10.0,     # vel
        5.0, 5.0, 5.0,        # drpy
    ])
    
    # Input weights
    R = np.diag([
        1.0, 1.0, 1.0,        # rpy 命令
        50.0,                 # thrust 命令
    ])
    
    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e
    
    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx = Vx
    
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu
    
    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e
    
    # Set initial references
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))
    
    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])
    
    # Set Input Constraints
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    
    # Initial state constraint
    ocp.constraints.x0 = np.zeros((nx))
    
    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-3
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.regularization_method = 'project'
    ocp.solver_options.tf = Tf
    
    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )
    
    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Improved MPC controller for drone racing."""
    
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller."""
        super().__init__(obs, info, config)
        
        self._N = 40
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt
        self._t_total = 15  
        self._freq = config.env.freq
        
        # ===== 加载无人机参数 =====
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        
        # ===== 关键：生成轨迹（这会设置 waypoints）=====
        self._generate_trajectory_from_gates(obs)
        
        # ===== 创建 MPC 求解器 =====
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        
        # ===== 获取维度信息 =====
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx
        
        # ===== 设置 tick 范围 =====
        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        
        self._config = config
        self._finished = False
        
        print(f"\n 控制器初始化完成")
        print(f"   预测步数: {self._N}, 预测时域: {self._T_HORIZON}s")
        print(f"   总航点数: {len(self._waypoints_pos)}")
        print(f"   Tick 范围: 0 - {self._tick_max}\n")
        
        
    
    
    
    
    def _generate_trajectory_from_gates(self, obs):
        """生成平滑自适应轨迹（Level 2：随机化门和障碍物）"""
        gates_pos = obs['gates_pos']
        obstacles_pos = obs['obstacles_pos']
        current_pos = obs['pos']
    
        OBSTACLE_RADIUS = 0.015
        DRONE_RADIUS = 0.12
        SAFETY_CLEARANCE = 0.25
        MIN_CLEARANCE = OBSTACLE_RADIUS + DRONE_RADIUS + SAFETY_CLEARANCE
    
        print("=" * 70)
        print("🛫 生成平滑适应性轨迹（Level 2 随机化）")
        print("=" * 70)
    
        # ===== 辅助函数：路径安全检测 =====
        def point_to_line_distance(point_xy, line_start_xy, line_end_xy):
            """计算点到线段的最短距离"""
            vec = line_end_xy - line_start_xy
            vec_len_sq = np.dot(vec, vec)
            
            if vec_len_sq < 1e-6:
                return np.linalg.norm(point_xy - line_start_xy)
            
            t = np.dot(point_xy - line_start_xy, vec) / vec_len_sq
            t = np.clip(t, 0, 1)
            closest = line_start_xy + t * vec
            return np.linalg.norm(point_xy - closest)
    
        def is_segment_safe(p1, p2, obstacles_pos, min_dist=MIN_CLEARANCE):
            """检查从p1到p2的路径是否与所有障碍物都保持安全距离"""
            for obs_pos in obstacles_pos:
                dist = point_to_line_distance(obs_pos[:2], p1[:2], p2[:2])
                if dist < min_dist:
                    return False
            return True
    
        def find_smooth_detour(prev_pos, target_pos, obstacles_pos, iterations=3):
            """生成平滑的绕行点（使用多个迭代点）"""
            waypoints = [prev_pos.copy()]
            
            # 分段到达目标，每段检查安全性
            for step in range(1, iterations + 1):
                alpha = step / iterations
                intermediate = prev_pos + alpha * (target_pos - prev_pos)
                
                # 如果这一段不安全，生成绕行点
                if not is_segment_safe(waypoints[-1], intermediate, obstacles_pos):
                    # 在垂直于移动方向上绕行
                    direction = intermediate[:2] - waypoints[-1][:2]
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 0.01:
                        direction = direction / direction_norm
                        perpendicular = np.array([-direction[1], direction[0]])
                        
                        # 尝试两个方向的绕行
                        for detour_offset in [0.35, 0.45, 0.55]:
                            for side in [1, -1]:
                                detour_xy = (waypoints[-1][:2] + intermediate[:2]) / 2
                                detour_xy = detour_xy + side * detour_offset * perpendicular
                                detour_z = (waypoints[-1][2] + intermediate[2]) / 2
                                detour_point = np.array([detour_xy[0], detour_xy[1], detour_z])
                                
                                if is_segment_safe(waypoints[-1], detour_point, obstacles_pos) and \
                                is_segment_safe(detour_point, intermediate, obstacles_pos):
                                    waypoints.append(detour_point)
                                    break
                            else:
                                continue
                            break
                
                waypoints.append(intermediate)
            
            return waypoints[1:]  # 去掉起点
    
        # ===== 生成航点 =====
        waypoints = [current_pos.copy()]
        
        takeoff = current_pos.copy()
        takeoff[2] = 0.4  # 先升到40cm
        waypoints.append(takeoff)
        
        for gate_idx in range(4):
            gate_pos = gates_pos[gate_idx]
        
            # 门的通过高度（保持在门的中心）
            pass_height = gate_pos[2]
        
            last_point = waypoints[-1]
            gate_target = np.array([gate_pos[0], gate_pos[1], pass_height])
            
            # ===== 关键改进：生成平滑的多点轨迹 =====
            # 不直接连接，而是生成平滑的中间点
            
            # 第一步：XY平面靠近（保持高度）
            approach_1_target = np.array([gate_pos[0], gate_pos[1], last_point[2]])
            approach_1 = 0.2 * last_point + 0.8 * approach_1_target
            
            # 检查安全性并添加中间点
            segment_1 = find_smooth_detour(last_point, approach_1, obstacles_pos, iterations=2)
            waypoints.extend(segment_1)
            
            # 第二步：高度调整（同时微调XY）
            approach_2_target = np.array([gate_pos[0], gate_pos[1], pass_height])
            approach_2 = 0.4 * approach_1 + 0.6 * approach_2_target
            
            segment_2 = find_smooth_detour(waypoints[-1], approach_2, obstacles_pos, iterations=2)
            waypoints.extend(segment_2)
            
            # 第三步：通过门
            segment_3 = find_smooth_detour(waypoints[-1], gate_target, obstacles_pos, iterations=1)
            waypoints.extend(segment_3)
        
            # 第四步：离开门（为下一个门做准备）
            if gate_idx < 3:
                next_gate = gates_pos[gate_idx + 1]
                next_height = next_gate[2]
                
                # 中间过渡点
                exit_target = 0.3 * gate_target + 0.7 * np.array([
                    next_gate[0], next_gate[1], next_height
                ])
                
                segment_4 = find_smooth_detour(waypoints[-1], exit_target, obstacles_pos, iterations=2)
                waypoints.extend(segment_4)
    
        waypoints = np.array(waypoints)
        print(f"✅ 生成 {len(waypoints)} 个航点（含平滑过渡）")
        
        # ===== 关键改进：均匀速度分配 =====
        total_distance = 0
        for i in range(1, len(waypoints)):
            total_distance += np.linalg.norm(waypoints[i] - waypoints[i-1])
        
        # 固定均匀速度（关键！）
        cruise_speed = 0.7  # m/s
        total_time = total_distance / cruise_speed
        
        # 均匀时间分配
        num_waypoints = len(waypoints)
        waypoint_times = np.linspace(0, total_time, num_waypoints)
        
        print(f"📊 轨迹统计:")
        print(f"   总距离: {total_distance:.2f}m")
        print(f"   总时间: {total_time:.2f}s")
        print(f"   速度: {cruise_speed:.2f} m/s (均匀)")
        
        # ===== 关键改进：三次样条插值参数 =====
        num_points = int(self._freq * total_time) + 1
        t_fine = np.linspace(0, total_time, num_points)
        
        # 使用 'natural' 而不是 'clamped'，产生更平滑的曲线
        try:
            cs_x = CubicSpline(waypoint_times, waypoints[:, 0], bc_type='natural')
            cs_y = CubicSpline(waypoint_times, waypoints[:, 1], bc_type='natural')
            cs_z = CubicSpline(waypoint_times, waypoints[:, 2], bc_type='natural')
        except:
            # 如果waypoint太少，退回到clamped
            cs_x = CubicSpline(waypoint_times, waypoints[:, 0], bc_type='clamped')
            cs_y = CubicSpline(waypoint_times, waypoints[:, 1], bc_type='clamped')
            cs_z = CubicSpline(waypoint_times, waypoints[:, 2], bc_type='clamped')
        
        self._waypoints_pos = np.column_stack([cs_x(t_fine), cs_y(t_fine), cs_z(t_fine)])
        self._waypoints_vel = np.column_stack([cs_x(t_fine, 1), cs_y(t_fine, 1), cs_z(t_fine, 1)])
        
        # 计算Yaw（朝向下一个点）
        self._waypoints_yaw = np.zeros(len(t_fine))
        for i in range(len(t_fine) - 1):
            dx = self._waypoints_pos[i+1, 0] - self._waypoints_pos[i, 0]
            dy = self._waypoints_pos[i+1, 1] - self._waypoints_pos[i, 1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self._waypoints_yaw[i] = np.arctan2(dy, dx)
        self._waypoints_yaw[-1] = self._waypoints_yaw[-2]
        
        self._t_total = total_time
        
        print(f"✅ 轨迹插值完成：{len(self._waypoints_pos)} 个点\n")

    
    
    
    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute the next control input."""
        
        # 修正：使用真实时间而非tick索引
        current_time = self._tick * self._dt
        
        # 根据当前时间查找轨迹索引
        trajectory_progress = current_time / self._t_total
        trajectory_index = int(trajectory_progress * len(self._waypoints_pos))
        i = min(trajectory_index, len(self._waypoints_pos) - self._N - 1)
        
        # 检查是否完成
        if i >= len(self._waypoints_pos) - self._N:
            self.finished = True
            print(f"✅ 轨迹完成 (time={current_time:.2f}s)")
        
        # 调试信息
        if self._tick % 50 == 0:
            target_gate = obs['target_gate']
            gates_visited = obs['gates_visited']
            print(f"[{self._tick * self._dt:.1f}s] 目标门: {target_gate+1}, 已通过: {sum(gates_visited)}/4")
            print(f"   位置: {obs['pos']}")
            print(f"   轨迹进度: {trajectory_progress*100:.1f}% (index {i}/{len(self._waypoints_pos)})")
        
        # 获取当前状态
        obs_rpy = R.from_quat(obs['quat']).as_euler('xyz')
        obs_drpy = ang_vel2rpy_rates(obs['quat'], obs['ang_vel'])
        
        x0 = np.concatenate([obs['pos'], obs_rpy, obs['vel'], obs_drpy])
        
        self._acados_ocp_solver.set(0, 'lbx', x0)
        self._acados_ocp_solver.set(0, 'ubx', x0)
        
        # 设置参考轨迹
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._waypoints_pos[i:i + self._N]  # position
        yref[:, 5] = self._waypoints_yaw[i:i + self._N]     # yaw
        yref[:, 6:9] = self._waypoints_vel[i:i + self._N]   # velocity
        yref[:, 15] = self.drone_params['mass'] * (-self.drone_params['gravity_vec'][-1])  # hover thrust
        
        for j in range(self._N):
            self._acados_ocp_solver.set(j, 'yref', yref[j])
        
        # Terminal reference
        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = self._waypoints_pos[i + self._N]
        yref_e[5] = self._waypoints_yaw[i + self._N]
        yref_e[6:9] = self._waypoints_vel[i + self._N]
        
        self._acados_ocp_solver.set(self._N, 'yref', yref_e)
        
        # 求解MPC
        status = self._acados_ocp_solver.solve()
        
        if status != 0:
            print(f"⚠️  MPC 求解失败 (status {status})")
        
        u0 = self._acados_ocp_solver.get(0, "u")
        
        return u0

   
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
        """Reset the tick counter."""
        self._tick = 0
