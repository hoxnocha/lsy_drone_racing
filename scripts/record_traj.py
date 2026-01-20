"""
Usage:
    python scripts/record_traj.py --controller lsy_drone_racing/control/mpcc_rotor_final.py --out data/final.npz --episodes 20
"""
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lsy_drone_racing.utils import load_config, load_controller

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, required=True, help="Path to the controller python file (relative to project root)")
    parser.add_argument("--config", type=str, default="level0.toml", help="Config file name in config/ dir")
    parser.add_argument("--out", type=str, default="trajectories.npz", help="Output file path")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to record")
    args = parser.parse_args()

    # 1. 加载配置
    # 假设脚本位于 scripts/ 目录，config 位于 config/ 目录
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / args.config
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    config = load_config(config_path)
    
    # 确保不渲染窗口，加快录制速度
    config.sim.render = False


    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    # 使用 JaxToNumpy 包装器，确保 obs/action 格式兼容
    env = JaxToNumpy(env)

    # 3. 加载控制器类
    # sim_4.py 使用 load_controller，它接受一个 Path 对象
    controller_file_path = Path(args.controller)
    if not controller_file_path.is_absolute():
        controller_file_path = (project_root / args.controller).resolve()
    
    if not controller_file_path.exists():
        raise FileNotFoundError(f"Controller file not found: {controller_file_path}")

    ControllerClass = load_controller(controller_file_path)
    print(f"Loaded controller from: {controller_file_path}")

    # 4. 数据容器
    all_trajectories = []
    success_flags = []
    gates_pos = None
    obstacles_pos = None

    print(f"Start recording {args.episodes} episodes...")

    for ep in range(args.episodes):
        obs, info = env.reset()
        
        # 实例化控制器
        controller = ControllerClass(obs, info, config)
        
        # 记录一次环境静态信息（Gate 和 Obstacle 位置）
        if ep == 0:
            gates_pos = obs.get("gates_pos", [])
            obstacles_pos = obs.get("obstacles_pos", [])

        traj = []
        step_cnt = 0
        done = False
        
        # 安全限制，防止死循环
        max_steps = config.env.max_steps if hasattr(config.env, "max_steps") else 2000

        while not done:
            # 记录当前位置
            traj.append(obs["pos"].copy())
            
            # 计算控制
            action = controller.compute_control(obs, info)
            
            # Step 环境
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 回调控制器（更新内部状态）
            controller_finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
            
            step_cnt += 1

            # 终止条件判断
            if terminated or truncated or controller_finished or step_cnt > max_steps:
                done = True

        # 记录最后一步
        traj.append(obs["pos"].copy())
        all_trajectories.append(np.array(traj))
        
        # 判断是否成功
        # 逻辑：target_gate 为 -1 表示通过了所有门
        gates_passed = obs.get("target_gate", 0)
        is_success = (gates_passed == -1)
        
        success_flags.append(is_success)
        
        status_str = "Success" if is_success else "Fail"
        print(f"Episode {ep+1}/{args.episodes}: {status_str}, Steps={step_cnt}")
        
        # 重置控制器状态
        controller.episode_callback()
        controller.episode_reset()

    # 5. 保存数据
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        trajectories=np.array(all_trajectories, dtype=object),
        success_flags=np.array(success_flags),
        gates_pos=gates_pos,
        obstacles_pos=obstacles_pos,
        controller_name=os.path.basename(args.controller)
    )
    print(f"Saved data to {output_path}")
    
    env.close()

if __name__ == "__main__":
    main()