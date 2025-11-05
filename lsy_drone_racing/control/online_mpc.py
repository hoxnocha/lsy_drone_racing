"""Online Planning MPC - Real-time trajectory optimization without path planner

核心思想：
- 不预先规划整条路径
- 在每个MPC求解步骤中，动态决定MPC预测地平线内的参考轨迹
- 基于当前状态、目标门、和实时障碍物位置
- 使用凸优化自动处理冲突约束
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
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




class OnlinePlanningMPC(Controller):
    """Online Planning MPC - 实时规划不依赖离线路径"""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller."""
        super().__init__(obs, info, config)
        
        self._N = 40
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt
        self._t_total = 15  
        self._freq = config.env.freq
        self._target_gate_idx = 0  # ← 添加这行
        self._gates_pos = obs['gates_pos']  # ← 添加这行
        self._obstacles_pos = obs['obstacles_pos']  # ← 添加这行
        # ===== 加载无人机参数 =====
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        
    
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
        
        
        self._config = config
        self._finished = False
        
        print(f"\n 控制器初始化完成")
        print(f"   预测步数: {self._N}, 预测时域: {self._T_HORIZON}s")
        
    

    def _compute_reference_trajectory(self, current_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ★ 核心：在线计算参考轨迹（不依赖离线路径规划）

        策略：
        1. 在MPC预测地平线内，选择最优的目标点
        2. 参考轨迹为：从当前位置直线向目标
        3. 自动避开实时障碍物
        """
        T = self._T_HORIZON
        N = self._N

        # 初始化参考轨迹
        ref_pos = np.zeros((N + 1, 3))
        ref_vel = np.zeros((N + 1, 3))
        ref_yaw = np.zeros(N + 1)

        # ★ 步骤1：确定目标点
        if self._target_gate_idx < len(self._gates_pos):
            target = self._gates_pos[self._target_gate_idx].copy()
        else:
            # 所有门都通过了，悬停
            target = current_pos.copy()
            self._finished = True

        # ★ 步骤2：生成直线参考轨迹
        direction = target - current_pos
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 0.01:
            direction_normalized = direction / direction_norm
        else:
            direction_normalized = np.array([1.0, 0.0, 0.0])

        # 参考速度：1.0 m/s
        ref_speed = 0.5

        for i in range(N + 1):
            t = (i / N) * T
            alpha = min(1.0, ref_speed * t / max(direction_norm, 0.1))

            # 参考位置：沿直线插值
            ref_pos[i] = current_pos + alpha * direction

            # 参考速度：常数方向速度
            if alpha < 1.0:
                ref_vel[i] = ref_speed * direction_normalized
            else:
                ref_vel[i] = np.array([0.0, 0.0, 0.0])

            # 参考偏航：朝向运动方向
            ref_yaw[i] = np.arctan2(direction_normalized[1], direction_normalized[0])

        # ★ 步骤3：避开实时障碍物（简化版）
        for i in range(N + 1):
            for obs_pos in self._obstacles_pos:
                dist = np.linalg.norm(ref_pos[i, :2] - obs_pos[:2])
                SAFETY_DIST = 0.3

                if dist < SAFETY_DIST:
                    # 远离障碍物
                    away_dir = (ref_pos[i, :2] - obs_pos[:2]) / (dist + 1e-6)
                    ref_pos[i, :2] += away_dir * (SAFETY_DIST - dist)

        return ref_pos, ref_vel, ref_yaw

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """计算控制命令"""
        try:
            # ★ 计算悬停推力
            thrust_hover = self.drone_params['mass'] * (-self.drone_params['gravity_vec'][-1])


            if self._finished:
                return np.array([0.0, 0.0, 0.0, thrust_hover])

            # ★ 检查是否到达目标门
            if self._target_gate_idx < len(self._gates_pos):
                target_pos = self._gates_pos[self._target_gate_idx]
                current_pos = obs['pos']
                dist_to_target = np.linalg.norm(current_pos - target_pos)

                if dist_to_target < 0.5:  # 接近门
                    self._target_gate_idx += 1
                    if self._target_gate_idx >= len(self._gates_pos):
                        self._finished = True

            # ★ 在线计算参考轨迹
            ref_pos, ref_vel, ref_yaw = self._compute_reference_trajectory(obs['pos'])

            # 当前状态
            obs_rpy = R.from_quat(obs['quat']).as_euler('xyz')
            obs_drpy = ang_vel2rpy_rates(obs['quat'], obs['ang_vel'])
            x0 = np.concatenate([obs['pos'], obs_rpy, obs['vel'], obs_drpy])

            # 设置初始状态
            self._acados_ocp_solver.set(0, 'lbx', x0)
            self._acados_ocp_solver.set(0, 'ubx', x0)

            # ★ 设置参考轨迹
            for i in range(self._N):
                yref = np.zeros(16)
                yref[0:3] = ref_pos[i]
                yref[3:5] = [0.0, 0.0]
                yref[5] = ref_yaw[i]
                yref[6:9] = ref_vel[i]
                yref[9:12] = [0.0, 0.0, 0.0]
                yref[12:15] = [0.0, 0.0, 0.0]
                yref[15] = thrust_hover

                self._acados_ocp_solver.set(i, 'yref', yref)

            # 终端参考
            yref_e = np.zeros(12)
            yref_e[0:3] = ref_pos[-1]
            yref_e[3:5] = [0.0, 0.0]
            yref_e[5] = ref_yaw[-1]
            yref_e[6:9] = ref_vel[-1]
            yref_e[9:12] = [0.0, 0.0, 0.0]

            self._acados_ocp_solver.set(self._N, 'yref', yref_e)

            # ★ 求解MPC
            status = self._acados_ocp_solver.solve()

            if status != 0:
                return np.array([0.0, 0.0, 0.0, thrust_hover])

            u = self._acados_ocp_solver.get(0, 'u')

            self._tick += 1
            return u

        except Exception as e:
            print(f"Error: {e}")
            thrust_hover = self.drone_params['mass'] * (-self.drone_params['gravity_vec'][-1])

            return np.array([0.0, 0.0, 0.0, thrust_hover])

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        """Step callback"""
        return self._finished

    def episode_callback(self):
        """Episode callback"""
        pass
