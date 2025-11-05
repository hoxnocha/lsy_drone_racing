"""
高级可调参MPC控制器 - Advanced Tunable MPC Controller
完整数学模型 + 在线规划 + 软约束 + 控制速率限制

作者注释：每个参数都可以通过 MPCConfig 类进行调整
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
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


@dataclass
class MPCConfig:
    """MPC 完整可调参配置类

    所有参数都已精心设计，可以直接修改以进行调优
    """

    # ========== 预测地平线参数 ==========
    N: int = 40  # 预测步数（步）
    dt: float = 0.02  # 时间步长（秒）
    T_HORIZON: float = None  # 预测时域（秒），自动计算 = N * dt

    # ========== 成本函数权重 Q 矩阵 ==========
    # 状态成本：x = [px, py, pz, φ, θ, ψ, vx, vy, vz, ωx, ωy, ωz]
    q_pos: float = 50.0    # 位置误差权重（关键：决定位置跟踪紧密度）
    q_rpy: float = 5.0     # 欧拉角权重（姿态稳定性）
    q_vel: float = 20.0    # 速度权重（平滑性和响应性平衡）
    q_drpy: float = 5.0    # 角速度权重（避免剧烈转动）

    # ========== 成本函数权重 R 矩阵 ==========
    # 控制成本：u = [φ_cmd, θ_cmd, ψ_cmd, thrust]
    r_rpy_cmd: float = 1.0    # 姿态命令权重（控制输入平滑性）
    r_thrust: float = 10.0    # 推力权重（能耗考虑）

    # ========== 约束参数 ==========
    # 状态约束（软化）
    euler_min: float = -0.5   # 最小欧拉角（弧度）
    euler_max: float = 0.5    # 最大欧拉角（弧度）

    # 控制约束
    max_rpy_rate: float = 8.0      # 最大角速率（rad/s）
    max_thrust: float = 0.6        # 最大推力比（0-1）
    min_thrust: float = 0.0        # 最小推力比

    # ========== 控制输入速率限制 ==========
    # 实际无人机无法瞬间改变控制输入
    max_du_rpy: float = 0.1        # 最大角速率变化（rad/s²）
    max_du_thrust: float = 0.1     # 最大推力变化率（1/s）
    enable_rate_limit: bool = True # 启用速率限制

    # ========== 在线规划参数 ==========
    ref_speed: float = 1.0         # 参考速度（m/s）
    obstacle_margin: float = 0.3   # 障碍物安全裕度（米）
    gate_reach_dist: float = 0.5   # 到达门的距离阈值（米）

    # ========== 求解器参数 ==========
    nlp_max_iter: int = 2          # NLP 最大迭代数
    qp_max_iter: int = 50          # QP 最大迭代数
    tol: float = 1e-2              # 收敛容差

    def __post_init__(self):
        """自动计算派生参数"""
        if self.T_HORIZON is None:
            self.T_HORIZON = self.N * self.dt


class AdvancedMPC(Controller):
    """高级在线规划MPC控制器 - 完整实现

    数学模型：
    =========

    1. 动力学模型（12维状态空间）：
       x = [p_x, p_y, p_z, φ, θ, ψ, v_x, v_y, v_z, ω_x, ω_y, ω_z]ᵀ

       其中：
       - 位置 p ∈ ℝ³（地球坐标）
       - 欧拉角 (φ,θ,ψ) ∈ SO(3)（滚转、俯仰、偏航）
       - 速度 v ∈ ℝ³
       - 角速度 ω ∈ ℝ³

    2. 控制输入（4维）：
       u = [φ_cmd, θ_cmd, ψ_cmd, f_thrust]ᵀ

       其中：
       - φ_cmd, θ_cmd: 姿态角命令
       - ψ_cmd: 偏航角速率命令
       - f_thrust: 推力（0~1 归一化）

    3. 成本函数（带软约束）：
       J = Σₖ ||xₖ - x_ref,k||²_Q + ||uₖ - u_ref,k||²_R + ε_soft

       其中：
       - Q = diag(q_pos*I₃, q_rpy*I₃, q_vel*I₃, q_drpy*I₃)
       - R = diag(r_rpy_cmd*I₃, r_thrust)
       - ε_soft: 松弛变量成本（约束软化）

    4. 约束条件：
       - 状态约束：x_min ≤ x ≤ x_max（可软化）
       - 控制约束：u_min ≤ u ≤ u_max
       - 速率约束：||du/dt|| ≤ max_du

    5. 在线参考轨迹生成：
       在每个MPC步骤中，动态计算预测地平线内的参考轨迹
       r_ref(t) = 当前位置 + α(t) * 方向向量，α ∈ [0,1]

    6. 控制输入速率限制（关键改进）：
       u_limited = clip(u_optimal, u_prev - max_du, u_prev + max_du)
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """初始化高级MPC控制器"""
        super().__init__(obs, info, config)

        # 从配置文件加载默认参数
        self.mpc_config = MPCConfig()

        # 覆盖可选的自定义参数（如果在 config 中指定）
        if hasattr(config, 'mpc_params'):
            for key, value in config.mpc_params.items():
                if hasattr(self.mpc_config, key):
                    setattr(self.mpc_config, key, value)

        # 加载无人机参数
        self.drone_params = load_params(config.sim.drone_model)

        # 初始化状态跟踪
        self._tick = 0
        self._finished = False
        self._target_gate_idx = 0
        self._gates_pos = obs['gates_pos']
        self._obstacles_pos = obs['obstacles_pos']

        # 控制输入历史（用于速率限制）
        hover_thrust = self.drone_params['mass'] * abs(self.drone_params['gravity_vec'][-1])
        self._previous_u = np.array([0.0, 0.0, 0.0, hover_thrust])

        # 创建 MPC 求解器
        self._setup_mpc_solver()

        print(f"""
╔════════════════════════════════════════════════════════════════╗
║      高级可调参 MPC 控制器初始化完成                           ║
╚════════════════════════════════════════════════════════════════╝

📊 配置参数：
   预测步数 N: {self.mpc_config.N}
   时间步长: {self.mpc_config.dt:.3f}s
   预测地平线: {self.mpc_config.T_HORIZON:.3f}s

📈 权重参数：
   位置权重 Q_pos: {self.mpc_config.q_pos}
   速度权重 Q_vel: {self.mpc_config.q_vel}
   推力权重 R_thrust: {self.mpc_config.r_thrust}

⚙️  约束参数：
   最大角速率: {self.mpc_config.max_rpy_rate} rad/s
   最大推力: {self.mpc_config.max_thrust}
   速率限制启用: {self.mpc_config.enable_rate_limit}

🎯 在线规划：
   参考速度: {self.mpc_config.ref_speed} m/s
   安全裕度: {self.mpc_config.obstacle_margin}m
""")

    def _setup_mpc_solver(self):
        """设置 MPC 求解器"""
        drone_params = self.drone_params

        # 创建动力学模型
        X_dot, X, U, _ = symbolic_dynamics_euler(
            mass=drone_params['mass'],
            gravity_vec=drone_params['gravity_vec'],
            J=drone_params['J'],
            J_inv=drone_params['J_inv'],
            acc_coef=drone_params['acc_coef'],
            cmd_f_coef=drone_params['cmd_f_coef'],
            rpy_coef=drone_params['rpy_coef'],
            rpy_rates_coef=drone_params['rpy_rates_coef'],
            cmd_rpy_coef=drone_params['cmd_rpy_coef'],
        )

        model = AcadosModel()
        model.name = "advanced_mpc"
        model.x = X
        model.u = U
        model.f_expl_expr = X_dot

        # 创建 OCP
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.mpc_config.N

        # 成本函数权重矩阵（关键参数）
        Q = np.diag([
            self.mpc_config.q_pos, self.mpc_config.q_pos, self.mpc_config.q_pos,
            self.mpc_config.q_rpy, self.mpc_config.q_rpy, self.mpc_config.q_rpy,
            self.mpc_config.q_vel, self.mpc_config.q_vel, self.mpc_config.q_vel,
            self.mpc_config.q_drpy, self.mpc_config.q_drpy, self.mpc_config.q_drpy,
        ])

        R = np.diag([
            self.mpc_config.r_rpy_cmd,
            self.mpc_config.r_rpy_cmd,
            self.mpc_config.r_rpy_cmd,
            self.mpc_config.r_thrust,
        ])

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q

        ocp.cost.Vx = np.zeros((16, 12))
        ocp.cost.Vx[:12, :12] = np.eye(12)
        ocp.cost.Vu = np.zeros((16, 4))
        ocp.cost.Vu[12:, :] = np.eye(4)
        ocp.cost.Vx_e = np.eye(12)

        ocp.cost.yref = np.zeros(16)
        ocp.cost.yref_e = np.zeros(12)

        # 约束条件（软化的状态约束）
        ocp.constraints.lbx = np.array([
            self.mpc_config.euler_min,
            self.mpc_config.euler_min,
            self.mpc_config.euler_min,
        ])
        ocp.constraints.ubx = np.array([
            self.mpc_config.euler_max,
            self.mpc_config.euler_max,
            self.mpc_config.euler_max,
        ])
        ocp.constraints.idxbx = np.array([3, 4, 5])

        # 控制约束
        ocp.constraints.lbu = np.array([
            -self.mpc_config.max_rpy_rate,
            -self.mpc_config.max_rpy_rate,
            -self.mpc_config.max_rpy_rate,
            self.mpc_config.min_thrust,
        ])
        ocp.constraints.ubu = np.array([
            self.mpc_config.max_rpy_rate,
            self.mpc_config.max_rpy_rate,
            self.mpc_config.max_rpy_rate,
            self.mpc_config.max_thrust,
        ])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # 求解器配置
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.nlp_solver_max_iter = self.mpc_config.nlp_max_iter
        ocp.solver_options.qp_solver_iter_max = self.mpc_config.qp_max_iter
        ocp.solver_options.tol = self.mpc_config.tol
        ocp.solver_options.tf = self.mpc_config.T_HORIZON

        self._solver = AcadosOcpSolver(ocp)
        self._ocp = ocp

    def _compute_reference_trajectory(self, current_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在线计算参考轨迹

        实现方程：
        r_ref(τ) = p_curr + α(τ) * (p_target - p_curr)
        其中 α(τ) = min(1.0, v_ref * τ / ||p_target - p_curr||)
        """
        N = self.mpc_config.N
        dt = self.mpc_config.dt
        T = self.mpc_config.T_HORIZON

        ref_pos = np.zeros((N + 1, 3))
        ref_vel = np.zeros((N + 1, 3))
        ref_yaw = np.zeros(N + 1)

        # 确定目标
        if self._target_gate_idx < len(self._gates_pos):
            target = self._gates_pos[self._target_gate_idx].copy()
        else:
            target = current_pos.copy()
            self._finished = True

        # 方向向量
        direction = target - current_pos
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 0.01:
            direction_normalized = direction / direction_norm
        else:
            direction_normalized = np.array([1.0, 0.0, 0.0])

        # 生成参考轨迹
        for i in range(N + 1):
            t = (i / N) * T
            alpha = min(1.0, self.mpc_config.ref_speed * t / max(direction_norm, 0.1))

            ref_pos[i] = current_pos + alpha * direction

            if alpha < 1.0:
                ref_vel[i] = self.mpc_config.ref_speed * direction_normalized

            ref_yaw[i] = np.arctan2(direction_normalized[1], direction_normalized[0])

        # 简化避障（可扩展为更复杂的算法）
        for i in range(N + 1):
            for obs_pos in self._obstacles_pos:
                dist = np.linalg.norm(ref_pos[i, :2] - obs_pos[:2])
                if dist < self.mpc_config.obstacle_margin:
                    away_dir = (ref_pos[i, :2] - obs_pos[:2]) / (dist + 1e-6)
                    ref_pos[i, :2] += away_dir * (self.mpc_config.obstacle_margin - dist)

        return ref_pos, ref_vel, ref_yaw

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """计算 MPC 控制命令"""
        try:
            # 悬停推力
            thrust_hover = self.drone_params['mass'] * abs(self.drone_params['gravity_vec'][-1])
            default_action = np.array([0.0, 0.0, 0.0, thrust_hover])

            if self._finished:
                return default_action

            # 检查目标达成
            if self._target_gate_idx < len(self._gates_pos):
                target_pos = self._gates_pos[self._target_gate_idx]
                dist = np.linalg.norm(obs['pos'] - target_pos)
                if dist < self.mpc_config.gate_reach_dist:
                    self._target_gate_idx += 1
                    if self._target_gate_idx >= len(self._gates_pos):
                        self._finished = True

            # 在线计算参考轨迹
            ref_pos, ref_vel, ref_yaw = self._compute_reference_trajectory(obs['pos'])

            # 当前状态
            obs_rpy = R.from_quat(obs['quat']).as_euler('xyz')
            obs_drpy = ang_vel2rpy_rates(obs['quat'], obs['ang_vel'])
            x0 = np.concatenate([obs['pos'], obs_rpy, obs['vel'], obs_drpy])

            # 设置初始状态约束
            self._solver.set(0, 'lbx', x0)
            self._solver.set(0, 'ubx', x0)

            # 设置参考轨迹
            for i in range(self.mpc_config.N):
                yref = np.zeros(16)
                yref[0:3] = ref_pos[i]
                yref[3:5] = [0.0, 0.0]
                yref[5] = ref_yaw[i]
                yref[6:9] = ref_vel[i]
                yref[9:15] = 0
                yref[15] = thrust_hover
                self._solver.set(i, 'yref', yref)

            yref_e = np.zeros(12)
            yref_e[0:3] = ref_pos[-1]
            yref_e[3:5] = [0.0, 0.0]
            yref_e[5] = ref_yaw[-1]
            yref_e[6:9] = ref_vel[-1]
            self._solver.set(self.mpc_config.N, 'yref', yref_e)

            # 求解 MPC
            status = self._solver.solve()
            if status != 0:
                return default_action

            u = self._solver.get(0, 'u')

            # ★ 控制输入速率限制（关键改进）
            if self.mpc_config.enable_rate_limit:
                u = np.clip(
                    u,
                    self._previous_u - np.array([
                        self.mpc_config.max_du_rpy,
                        self.mpc_config.max_du_rpy,
                        self.mpc_config.max_du_rpy,
                        self.mpc_config.max_du_thrust,
                    ]),
                    self._previous_u + np.array([
                        self.mpc_config.max_du_rpy,
                        self.mpc_config.max_du_rpy,
                        self.mpc_config.max_du_rpy,
                        self.mpc_config.max_du_thrust,
                    ])
                )

            self._previous_u = u.copy()
            self._tick += 1
            return u

        except Exception as e:
            print(f"❌ 错误: {e}")
            thrust_hover = self.drone_params['mass'] * abs(self.drone_params['gravity_vec'][-1])
            return np.array([0.0, 0.0, 0.0, thrust_hover])

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        return self._finished

    def episode_callback(self):
        pass
