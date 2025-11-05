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
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.nlp_solver_max_iter = 100
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
        
        self._N = 25
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
        
        print(f"\n✅ 控制器初始化完成")
        print(f"   预测步数: {self._N}, 预测时域: {self._T_HORIZON}s")
        print(f"   总航点数: {len(self._waypoints_pos)}")
        print(f"   Tick 范围: 0 - {self._tick_max}\n")
    
    def _generate_trajectory_from_gates(self, obs):
        """根据门和障碍物位置生成安全轨迹"""
        
        gates_pos = obs['gates_pos']
        obstacles_pos = obs['obstacles_pos']
        current_pos = obs['pos']
        
        # ===== 几何参数 =====
        OBSTACLE_RADIUS = 0.015
        DRONE_RADIUS = 0.1
        SAFETY_CLEARANCE = 0.25  # 增加到 25cm
        
        print("\n" + "="*70)
        print("📍 门和障碍物位置")
        print("="*70)
        for i in range(4):
            print(f"门 {i+1}: [{gates_pos[i, 0]:.2f}, {gates_pos[i, 1]:.2f}, {gates_pos[i, 2]:.2f}]")
        for i in range(4):
            print(f"柱子 {i+1}: x={obstacles_pos[i, 0]:.2f}, y={obstacles_pos[i, 1]:.2f}")
        print("="*70)
        
        # ===== 构建航点序列 =====
        waypoints = []
        
        # 起点
        waypoints.append(current_pos.copy())
        
        
        # 依次添加4个门
        for gate_idx in range(4):
            gate_pos = gates_pos[gate_idx].copy()
            
            # 确定通过高度
            if gate_pos[2] < 0.9:  # 矮门
                target_height = gate_pos[2] 
            else:  # 高门
                target_height = gate_pos[2] - 0.5
            
            gate_waypoint = np.array([gate_pos[0], gate_pos[1], target_height])
            
            # 检查碰撞
            last_waypoint = waypoints[-1]
            needs_detour = False
            detour_obstacles = []
            
            for obs_idx in range(4):
                obstacle_xy = obstacles_pos[obs_idx, :2]
                
                path_vec = gate_waypoint[:2] - last_waypoint[:2]
                path_len_sq = np.dot(path_vec, path_vec)
                
                if path_len_sq < 1e-6:
                    distance = np.linalg.norm(obstacle_xy - last_waypoint[:2])
                else:
                    t = np.dot(obstacle_xy - last_waypoint[:2], path_vec) / path_len_sq
                    t = np.clip(t, 0, 1)
                    closest_point = last_waypoint[:2] + t * path_vec
                    distance = np.linalg.norm(obstacle_xy - closest_point)
                
                required_clearance = OBSTACLE_RADIUS + DRONE_RADIUS + SAFETY_CLEARANCE
                
                if distance < required_clearance:
                    needs_detour = True
                    detour_obstacles.append((obs_idx, obstacle_xy, distance))
                    print(f"⚠️  路径接近柱子 {obs_idx+1} (距离 {distance:.3f}m)")
            
            # 添加绕行点
            if needs_detour:
                closest_obs_idx, closest_obs_xy, _ = min(detour_obstacles, key=lambda x: x[2])
                
                path_vec = gate_waypoint[:2] - last_waypoint[:2]
                path_dir = path_vec / (np.linalg.norm(path_vec) + 1e-6)
                perpendicular = np.array([-path_dir[1], path_dir[0]])
                
                path_midpoint = 0.5 * (last_waypoint[:2] + gate_waypoint[:2])
                to_obstacle = closest_obs_xy - path_midpoint
                side = np.dot(to_obstacle, perpendicular)
                
                detour_dir = -np.sign(side) * perpendicular
                detour_distance = 0.40  # 40cm 绕行距离
                
                detour_xy = path_midpoint + detour_dir * detour_distance
                detour_z = 0.5 * (last_waypoint[2] + target_height)
                
                detour_waypoint = np.array([detour_xy[0], detour_xy[1], detour_z])
                waypoints.append(detour_waypoint)
                print(f"   ✓ 绕行点: [{detour_waypoint[0]:.2f}, {detour_waypoint[1]:.2f}, {detour_waypoint[2]:.2f}]")
            
            # 添加第一个门的接近点
            if gate_idx == 0 and len(waypoints) == 1:  # 只有起点和起飞点
                
             approach = 0.6 * last_waypoint + 0.4 * gate_waypoint
             waypoints.append(approach)
             print(f"🎯 接近点: [{approach[0]:.2f}, {approach[1]:.2f}, {approach[2]:.2f}]")
            
            # 添加门通过点
            waypoints.append(gate_waypoint)
            print(f"🚪 门{gate_idx+1}: [{gate_waypoint[0]:.2f}, {gate_waypoint[1]:.2f}, {gate_waypoint[2]:.2f}]")
            
            # 添加退出点
            if gate_idx < 3:
                next_gate_pos = gates_pos[gate_idx + 1]
                next_height = next_gate_pos[2] - 0.12 if next_gate_pos[2] > 0.9 else next_gate_pos[2] - 0.1
                exit_point = 0.6 * gate_waypoint + 0.4 * np.array([
                    next_gate_pos[0], next_gate_pos[1], next_height
                ])
                waypoints.append(exit_point)
        
        waypoints = np.array(waypoints)
        print(f"\n✅ 生成 {len(waypoints)} 个关键航点\n")
        
        
        
        # ===== 三次样条插值 =====
        num_waypoints = len(waypoints)
        waypoint_times = [0]
        
        for i in range(1, num_waypoints):
            distance = np.linalg.norm(waypoints[i] - waypoints[i-1])
            time_increment = distance / 1.0  # 平均速度 1 m/s
            waypoint_times.append(waypoint_times[-1] + time_increment)
        
        waypoint_times = np.array(waypoint_times)
        waypoint_times = waypoint_times / waypoint_times[-1] * self._t_total
        
        # 生成精细时间网格（750个点）
        num_points = int(self._freq * self._t_total) + 1
        t_fine = np.linspace(0, self._t_total, num_points)
        
        # 三次样条
        cs_x = CubicSpline(waypoint_times, waypoints[:, 0], bc_type='clamped')
        cs_y = CubicSpline(waypoint_times, waypoints[:, 1], bc_type='clamped')
        cs_z = CubicSpline(waypoint_times, waypoints[:, 2], bc_type='clamped')
        
        # 生成轨迹
        self._waypoints_pos = np.column_stack([cs_x(t_fine), cs_y(t_fine), cs_z(t_fine)])
        self._waypoints_vel = np.column_stack([cs_x(t_fine, 1), cs_y(t_fine, 1), cs_z(t_fine, 1)])
        
        # 计算偏航角
        self._waypoints_yaw = np.zeros(len(t_fine))
        for i in range(len(t_fine) - 1):
            dx = self._waypoints_pos[i+1, 0] - self._waypoints_pos[i, 0]
            dy = self._waypoints_pos[i+1, 1] - self._waypoints_pos[i, 1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self._waypoints_yaw[i] = np.arctan2(dy, dx)
        self._waypoints_yaw[-1] = self._waypoints_yaw[-2]
        
        print(f"✅ 轨迹插值完成：{len(self._waypoints_pos)} 个点\n")
    
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next control input."""
        
        i = min(self._tick, self._tick_max)
        
        if self._tick >= self._tick_max:
            self._finished = True
        
        # 打印调试信息（每1秒）
        if self._tick % 50 == 0:
            target_gate = obs['target_gate']
            gates_visited = obs['gates_visited']
            print(f"\n[{self._tick/50:.1f}s] 目标门: {target_gate+1}, 已通过: {sum(gates_visited)}/4")
            print(f"   位置: {obs['pos']}")
        
        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)
        
        # Setting state reference
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = self._waypoints_pos[i : i + self._N]  # position
        yref[:, 5] = self._waypoints_yaw[i : i + self._N]    # yaw
        yref[:, 6:9] = self._waypoints_vel[i : i + self._N]  # velocity
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]  # hover thrust
        
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])
        
        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = self._waypoints_pos[i + self._N]
        yref_e[5] = self._waypoints_yaw[i + self._N]
        yref_e[6:9] = self._waypoints_vel[i + self._N]
        
        # ✅ 修复：正确的参数名
        self._acados_ocp_solver.set(self._N, "yref", yref_e)
        
        # Solving problem
        status = self._acados_ocp_solver.solve()
        
        if status != 0:
            print(f"⚠️  MPC 求解失败，状态码: {status}")
        
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
