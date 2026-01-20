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

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / args.config
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    config = load_config(config_path)
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

    env = JaxToNumpy(env)
    controller_file_path = Path(args.controller)
    if not controller_file_path.is_absolute():
        controller_file_path = (project_root / args.controller).resolve()
    
    if not controller_file_path.exists():
        raise FileNotFoundError(f"Controller file not found: {controller_file_path}")

    ControllerClass = load_controller(controller_file_path)
    print(f"Loaded controller from: {controller_file_path}")

    all_trajectories = []
    success_flags = []
    gates_pos = None
    obstacles_pos = None

    print(f"Start recording {args.episodes} episodes...")

    for ep in range(args.episodes):
        obs, info = env.reset()
    
        controller = ControllerClass(obs, info, config)
    
        if ep == 0:
            gates_pos = obs.get("gates_pos", [])
            obstacles_pos = obs.get("obstacles_pos", [])

        traj = []
        step_cnt = 0
        done = False
     
        max_steps = config.env.max_steps if hasattr(config.env, "max_steps") else 2000

        while not done:
            traj.append(obs["pos"].copy())
            action = controller.compute_control(obs, info)
            
            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
            step_cnt += 1
            if terminated or truncated or controller_finished or step_cnt > max_steps:
                done = True

        traj.append(obs["pos"].copy())
        all_trajectories.append(np.array(traj))
        
        gates_passed = obs.get("target_gate", 0)
        is_success = (gates_passed == -1)
        
        success_flags.append(is_success)
        
        status_str = "Success" if is_success else "Fail"
        print(f"Episode {ep+1}/{args.episodes}: {status_str}, Steps={step_cnt}")
        
        controller.episode_callback()
        controller.episode_reset()

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