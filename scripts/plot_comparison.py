"""
Usage:
    python scripts/plot_comparison.py data/final.npz data/edit.npz
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import argparse

def plot_gates_2d(ax, gates_pos, width=0.8):
    """
    在 2D 图上画出门的位置。
    注意：由于录制时可能没存四元数，这里用方形标记代表门中心。
    """
    if gates_pos is None: return
    
    # 画 Gate 中心
    ax.scatter(gates_pos[:, 0], gates_pos[:, 1], c='k', marker='s', s=100, label='Gates', zorder=5)
    
    for i, g in enumerate(gates_pos):
        # 标号
        ax.text(g[0] + 0.2, g[1] + 0.2, f"G{i}", fontsize=9, color='k')

def plot_obstacles_2d(ax, obstacles_pos, radius=0.05, safe_dist=0.15):
    """
    画出障碍物（实心红圆）和安全距离（虚线圆）。
    """
    if obstacles_pos is None: return

    # 只需要加一次 label
    added_label = False
    
    for o in obstacles_pos:
        # 实物 (柱子)
        circ_real = Circle((o[0], o[1]), radius, color='r', alpha=0.5, zorder=4)
        ax.add_patch(circ_real)
        
        # 安全边界 (Safe Dist) - 也就是控制器认为的"碰撞体积"
        circ_safe = Circle((o[0], o[1]), safe_dist, color='r', fill=False, linestyle='--', alpha=0.2, zorder=3)
        ax.add_patch(circ_safe)

        if not added_label:
            circ_real.set_label('Obstacle')
            added_label = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', help="List of .npz files to compare")
    parser.add_argument("--limit", type=int, default=5, help="Max trajectories per controller to plot")
    args = parser.parse_args()

    # 设置一个正方形画布，保证比例对等
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # 经典的 Matplotlib 配色
    linestyles = ['-', '--', '-.', ':']
    
    first_file_loaded = False

    for idx, fpath in enumerate(args.files):
        try:
            data = np.load(fpath, allow_pickle=True)
        except FileNotFoundError:
            print(f"Error: File {fpath} not found.")
            continue

        trajs = data['trajectories']
        success = data['success_flags']
        # 尝试获取控制器名字，如果没有则用文件名
        name = str(data['controller_name']) if 'controller_name' in data else fpath
        
        # 筛选成功的轨迹
        success_indices = np.where(success)[0]
        if len(success_indices) > 0:
            plot_indices = success_indices[:args.limit]
            print(f"[{name}] Plotting {len(plot_indices)} successful trajectories.")
        else:
            print(f"[{name}] No success episodes! Plotting first {args.limit} failed ones.")
            plot_indices = range(min(args.limit, len(trajs)))
        
        c = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]
        
        # --- 画环境 (只画一次，以第一个文件的数据为准) ---
        if not first_file_loaded:
            plot_gates_2d(ax, data['gates_pos'])
            plot_obstacles_2d(ax, data['obstacles_pos'])
            # 标记起飞点
            start_pos = trajs[0][0]
            ax.scatter(start_pos[0], start_pos[1], marker='^', c='g', s=150, label='Start', zorder=10)
            first_file_loaded = True

        # --- 画轨迹 ---
        for i, traj_idx in enumerate(plot_indices):
            traj = trajs[traj_idx]
            
            # 只给第一条轨迹加 Legend 标签，避免 Legend 重复
            label = f"{name} (Traj)" if i == 0 else None
            
            # 绘制 X-Y 平面
            ax.plot(traj[:, 0], traj[:, 1], color=c, linestyle=ls, alpha=0.8, linewidth=1.5, label=label, zorder=6)
            
            # 标记终点位置
            ax.scatter(traj[-1, 0], traj[-1, 1], color=c, marker='x', s=50, zorder=7)

    # --- 设置图表样式 ---
    ax.set_aspect('equal')  # 关键！保证 X 和 Y 轴比例一致，这样圆形才是圆的
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel("X Position [m]", fontsize=12)
    ax.set_ylabel("Y Position [m]", fontsize=12)
    ax.set_title("Drone Trajectory Comparison (Top-Down View)", fontsize=14)
    
    # 自动调整坐标轴范围以包含所有内容
    ax.autoscale_view()
    
    plt.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()