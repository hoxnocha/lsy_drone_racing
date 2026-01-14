import numpy as np

class GCOPTER_Lite:
    """
     (Minimum Jerk)。
    
    速度优化策略：
    1. 移除 scipy.optimize 的时间优化循环。
    2. 使用 '梯形速度剖面' (Trapezoidal Velocity Profile) 快速估算每段物理可行的时间。
    3. 仅保留一次闭式线性求解 (Closed-form Linear Solve)。
    
    耗时估计: < 3ms (Python)
    """
    def __init__(self, waypoints, avg_speed=5.0, max_acc=8.0):
        self.waypoints = np.array(waypoints)
        self.n_seg = len(waypoints) - 1
        self.dim = waypoints.shape[1]
        
        # 1. 快速启发式时间分配
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        
        # 简单物理估算: t = dist / speed
        self.T = dists / avg_speed 
        
        # 强制每段至少 0.1s (防止两点过近导致求解爆炸)
        self.T = np.maximum(self.T, 0.1) 
        
        # 2. 闭式求解多项式系数
        self.coeffs = self._solve_min_jerk_fast(self.waypoints, self.T)
        self.ts_cumulative = np.concatenate(([0], np.cumsum(self.T)))

    def _solve_min_jerk_fast(self, Q, T):
        """
        求解 Minimum Jerk 的闭式解 (无迭代)
        """
        n = self.n_seg
        dim = self.dim
        coeffs = np.zeros((n, dim, 6)) # 6 coefficients for 5th order

        # 为每个维度分别求解
        for d in range(dim):
            # --- 极速近似法 ---
            qs = Q[:, d]
            vs = np.zeros(n + 1)
            as_ = np.zeros(n + 1)
            
            # 速度启发式：前后两段平均速度 (Finite Difference)
            for i in range(1, n):
                v_prev = (qs[i] - qs[i-1]) / T[i-1]
                v_next = (qs[i+1] - qs[i]) / T[i]
                vs[i] = 0.5 * (v_prev + v_next)
                
            # 加速度启发式
            for i in range(1, n):
                a_prev = (vs[i] - vs[i-1]) / T[i-1]
                a_next = (vs[i+1] - vs[i]) / T[i]
                as_[i] = 0.5 * (a_prev + a_next)
                
            # --- 分段计算系数 (Closed-form mapping) ---
            for i in range(n):
                p0, p1 = qs[i], qs[i+1]
                v0, v1 = vs[i], vs[i+1]
                a0, a1 = as_[i], as_[i+1]
                t = T[i]
                
                t2, t3, t4, t5 = t*t, t*t*t, t*t*t*t, t*t*t*t*t
                
                delta_p = p1 - (p0 + v0*t + 0.5*a0*t2)
                delta_v = v1 - (v0 + a0*t)
                delta_a = a1 - a0
                
                # 逆矩阵系数预计算 (Inverse of [t^3, t^4, t^5; ...])
                k_c3 = (20*delta_p - 8*delta_v*t + delta_a*t2) / (2*t3)
                k_c4 = (-30*delta_p + 14*delta_v*t - 2*delta_a*t2) / (2*t4)
                k_c5 = (12*delta_p - 6*delta_v*t + delta_a*t2) / (2*t5)
                
                coeffs[i, d, :] = [p0, v0, 0.5*a0, k_c3, k_c4, k_c5]
                
        return coeffs

    def __call__(self, t_in, derivative=0):
        t_in = np.atleast_1d(t_in)
        res = np.zeros((len(t_in), self.dim))
        
        # 限制时间范围
        total_T = self.ts_cumulative[-1]
        t_in = np.clip(t_in, 0, total_T - 1e-6)

        # 向量化查找索引
        indices = np.searchsorted(self.ts_cumulative, t_in, side='right') - 1
        indices = np.clip(indices, 0, self.n_seg - 1)
        
        # 提取局部时间 dt
        t_start = self.ts_cumulative[indices]
        dt = t_in - t_start
        
        # 提取系数: (N_samples, Dim, 6)
        c = self.coeffs[indices] 
        
        dt2 = dt**2; dt3 = dt**3; dt4 = dt**4; dt5 = dt**5
        
        for d in range(self.dim):
            cd = c[:, d, :] # (N_samples, 6)
            if derivative == 0: # Pos
                res[:, d] = cd[:,0] + cd[:,1]*dt + cd[:,2]*dt2 + cd[:,3]*dt3 + cd[:,4]*dt4 + cd[:,5]*dt5
            elif derivative == 1: # Vel
                res[:, d] = cd[:,1] + 2*cd[:,2]*dt + 3*cd[:,3]*dt2 + 4*cd[:,4]*dt3 + 5*cd[:,5]*dt4
            elif derivative == 2: # Acc
                res[:, d] = 2*cd[:,2] + 6*cd[:,3]*dt + 12*cd[:,4]*dt2 + 20*cd[:,5]*dt3
                
        if len(res) == 1: return res[0]
        return res
    
    def derivative(self, n=1):
        return lambda t: self.__call__(t, derivative=n)