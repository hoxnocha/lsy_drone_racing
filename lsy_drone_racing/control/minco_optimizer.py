import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize
import math

class MincoOptimizer:
    def __init__(self, robot_mass=1.0):
        self.mass = robot_mass
        # MINCO s=3 对应 Minimize Snap (位置的4阶导), 产生 7次多项式
        # MINCO s=2 对应 Minimize Jerk (位置的3阶导), 产生 5次多项式 (推荐用于四旋翼)
        self.s = 2 
        self.poly_order = 2 * self.s + 1  # 5次多项式

    def generate_minco_trajectory(self, start_state, end_state, waypoints, times):
        """
        核心函数：给定边界、中间点和时间，利用矩阵逆解算出最优多项式系数。
        :param start_state: [pos, vel, acc] (3x3)
        :param end_state:   [pos, vel, acc] (3x3)
        :param waypoints:   中间固定航点 [N_mid, 3] (对于穿门，这里就是 入点-门点-出点)
        :param times:       每段时长 [N_mid + 1]
        """
        N = len(times)      # 段数
        M = len(waypoints)  # 中间点数 (段数 - 1)
        dim = 3             # x, y, z

        # --- 1. 构建边界条件向量 b ---
        # 初始状态 (P, V, A) 和 结束状态 (P, V, A) 是固定的
        # 中间航点 (P) 也是固定的 (这是您穿门场景的特殊约束)
        # 我们需要求解的是：中间点的 [Velocity, Acceleration] 以及每段的多项式系数
        
        # 这种 "Fixed Waypoints MINCO" 比标准的 "Free Waypoints MINCO" 更简单。
        # 实际上这退化为一个多段多项式插值问题 (Minimum Jerk Spline)。
        
        # 为了代码简单且高性能，我们这里直接使用 osqp-like 的思路或者直接构造线性方程组 Ax=b
        # 这里为了演示方便，使用 scipy 的 BPoly 快速近似 (性能足够且极其稳定)
        # 如果追求极致数学精确性，需要手写构建 6N * 6N 的矩阵求解 (在Python中略显繁琐)
        
        # 使用 OSQP 风格的二次规划或者 SciPy 样条是 Python 的最佳实践：
        
        # 时间节点
        ts = np.concatenate(([0], np.cumsum(times)))
        
        # 构建所有位置点: Start -> Waypoints -> End
        all_pos = np.vstack([start_state[0], waypoints, end_state[0]])
        
        # 边界导数约束
        bc_type = (
            [(1, start_state[1]), (2, start_state[2])],  # 起点 Vel, Acc
            [(1, end_state[1]), (2, end_state[2])]       # 终点 Vel, Acc
        )
        
        # 利用 scipy 的 make_interp_spline 生成 Min-Jerk 轨迹
        # k=5 代表 5次多项式 (Min-Jerk)
        from scipy.interpolate import make_interp_spline
        spline = make_interp_spline(ts, all_pos, k=5, bc_type=bc_type)
        
        return spline

    def calculate_cost(self, times, start_state, end_state, waypoints):
        """
        目标函数：J = Integral(Jerk^2) + rho * Sum(Time)
        """
        # 防止时间为负或过小
        if np.any(times <= 0.01):
            return 1e9
            
        traj = self.generate_minco_trajectory(start_state, end_state, waypoints, times)
        
        # 计算 Jerk 代价 (能量)
        # 解析积分比较复杂，这里用高斯-勒让德积分或简单的采样求和近似
        # 为了速度，我们用采样近似
        cost_jerk = 0.0
        total_time = np.sum(times)
        
        # 采样点数
        num_samples = max(10, int(total_time * 20)) 
        t_samples = np.linspace(0, total_time, num_samples)
        dt = total_time / num_samples
        
        # 获取加加速度 (3阶导)
        jerks = traj.derivative(3)(t_samples) # Shape: [N, 3]
        cost_jerk = np.sum(np.sum(jerks**2, axis=1)) * dt
        
        # 时间代价 (权重 rho 越大，飞得越快)
        rho = 1000.0 
        cost_time = rho * total_time
        
        return cost_jerk + cost_time

    def optimize(self, current_state, door_pos, door_quat, goal_pos):
        """
        主入口：针对穿门任务优化
        """
        # 1. 几何解算 (三明治点)
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(door_quat)
        normal = r.apply([1, 0, 0]) # 假设门朝向 X 轴
        
        p_entry = door_pos - normal * 0.8
        p_door = door_pos
        p_exit = door_pos + normal * 0.8
        
        waypoints = np.array([p_entry, p_door, p_exit])
        
        # 2. 初始时间猜测 (匀速假设)
        # 路径: Start -> Entry -> Door -> Exit -> Goal
        # 段数: 4段
        points = np.vstack([current_state[0], waypoints, goal_pos])
        dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
        v_avg = 1.5 # 期望速度 m/s
        times0 = np.maximum(dists / v_avg, 0.1) # 初始时间
        
        # 3. 优化时间分配
        # 约束：每段至少 0.1秒
        bounds = [(0.1, 5.0) for _ in range(len(times0))]
        
        res = minimize(
            self.calculate_cost, 
            times0, 
            args=(current_state, [goal_pos, [0,0,0], [0,0,0]], waypoints),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-3, 'maxiter': 20} # 限制迭代次数以保证实时性
        )
        
        # 4. 生成最终轨迹
        opt_times = res.x
        final_traj = self.generate_minco_trajectory(
            current_state, 
            [goal_pos, [0,0,0], [0,0,0]], 
            waypoints, 
            opt_times
        )
        
        return final_traj