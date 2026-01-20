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
    draw gates on 2d plot
    """
    if gates_pos is None: return
   
    ax.scatter(gates_pos[:, 0], gates_pos[:, 1], c='k', marker='s', s=100, label='Gates', zorder=5)
    
    for i, g in enumerate(gates_pos):
     
        ax.text(g[0] + 0.2, g[1] + 0.2, f"G{i}", fontsize=9, color='k')

def plot_obstacles_2d(ax, obstacles_pos, radius=0.05, safe_dist=0.15):
    
    if obstacles_pos is None: return

    added_label = False
    
    for o in obstacles_pos:
        circ_real = Circle((o[0], o[1]), radius, color='r', alpha=0.5, zorder=4)
        ax.add_patch(circ_real)
        
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
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
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
        name = str(data['controller_name']) if 'controller_name' in data else fpath
        
        success_indices = np.where(success)[0]
        if len(success_indices) > 0:
            plot_indices = success_indices[:args.limit]
            print(f"[{name}] Plotting {len(plot_indices)} successful trajectories.")
        else:
            print(f"[{name}] No success episodes! Plotting first {args.limit} failed ones.")
            plot_indices = range(min(args.limit, len(trajs)))
        
        c = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]
        
        if not first_file_loaded:
            plot_gates_2d(ax, data['gates_pos'])
            plot_obstacles_2d(ax, data['obstacles_pos'])
            # start pos
            start_pos = trajs[0][0]
            ax.scatter(start_pos[0], start_pos[1], marker='^', c='g', s=150, label='Start', zorder=10)
            first_file_loaded = True

   
        for i, traj_idx in enumerate(plot_indices):
            traj = trajs[traj_idx]
            

            label = f"{name} (Traj)" if i == 0 else None
            
            ax.plot(traj[:, 0], traj[:, 1], color=c, linestyle=ls, alpha=0.8, linewidth=1.5, label=label, zorder=6)
            
            ax.scatter(traj[-1, 0], traj[-1, 1], color=c, marker='x', s=50, zorder=7)

    ax.set_aspect('equal')  
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel("X Position [m]", fontsize=12)
    ax.set_ylabel("Y Position [m]", fontsize=12)
    ax.set_title("Drone Trajectory Comparison (Top-Down View)", fontsize=14)
    
    ax.autoscale_view()
    
    plt.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()