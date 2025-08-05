import sys
sys.path.append('core')
import imageio.v2 as imageio
import numpy as np
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from tqdm import tqdm

import imageio
import json
import os
import random
import torch
from argparse import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference import viz_generated_flow
from inference import FlowPredModel

from policy import MyFlowPolicy

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def run(args):
    result_root = args.result_root
    os.makedirs(result_root, exist_ok=True)

    n_exps = args.n_exps
    cameras = ['corner']
    max_replans = 5

    base_path = "results"
    
    
    flow_pred_model = FlowPredModel(num_points=400, pred_horizon=8, num_sampling_steps=250)

    env_name = args.env_name
    print(env_name)
    benchmark_env = env_dict[env_name]

    succes_rates = []
    reward_means = []
    reward_stds = []
    replans_counters = []

    for camera in cameras:
        success = 0
        rewards = []
        replans_counter = {i: 0 for i in range(max_replans + 1)}
        
        success_rates_path = os.path.join(result_root, f'success_rates')
        pred_flow_path = os.path.join(base_path, f'videos/flow_predicted/{env_name}')
        actuated_video_path = os.path.join(base_path, f'videos/video_actuated/{env_name}')
        os.makedirs(success_rates_path, exist_ok=True)
        os.makedirs(pred_flow_path, exist_ok=True)
        os.makedirs(actuated_video_path, exist_ok=True)
        
        for seed in tqdm(range(n_exps)):
            try: 
            
                env = benchmark_env(camera_name=camera, seed=seed)

                policy = MyFlowPolicy(env, env_name, camera, flow_pred_model, max_replans=max_replans, seed=seed)
        
                rewards.append(policy.episode_return / len(policy.images))

                used_replans = max_replans - policy.replans
            
                ### save sample video
                for idx, pred_flow in enumerate(policy.pred_flows):
                    frames = viz_generated_flow(
                        flows=pred_flow["flows"],
                        initial_frame=pred_flow["image"],
                        normalize=False,
                )
                    imageio.mimsave(f'{pred_flow_path}/episode{seed}_plan{idx}.mp4', frames)
                imageio.mimsave(f'{actuated_video_path}/episode{seed}.mp4', policy.images)
                
                print("test eplen: ", len(policy.images))
                if len(policy.images) <= 500:
                    success += 1
                    if used_replans == -1:
                        used_replans = 0
                    replans_counter[used_replans] += 1
                    print("success, used replans: ", used_replans)
                
            except Exception as e:
                print(e)
                print("something went wrong, skipping this seed")
                continue
            
        rewards = rewards + [0] * (n_exps - len(rewards))
        reward_means.append(np.mean(rewards))
        reward_stds.append(np.std(rewards))

        success_rate = success / n_exps
        succes_rates.append(success_rate)

        replans_counters.append(replans_counter)
        
                
    print(f"Success rates for {env_name}:\n", succes_rates)
    result_dict[env_name] = {
        "success_rates": succes_rates,
        "reward_means": reward_means,
        "reward_stds": reward_stds,
        "replans_counts": replans_counters
    }
    # with open(f"{result_root}/result_dict.json", "w") as f:
    #     json.dump(result_dict, f, indent=4)
    
    with open(f"{success_rates_path}/{env_name}_result.json", "w") as f:
        json.dump(result_dict[env_name], f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="door-open-v2-goal-observable")
    parser.add_argument("--n_exps", type=int, default=25)
    parser.add_argument("--result_root", type=str, default="results")
    args = parser.parse_args()

    try:
        with open(f"{args.result_root}/result_dict.json", "r") as f:
            result_dict = json.load(f)
    except:
        result_dict = {}

    if args.env_name in result_dict.keys():
        print("already done")
    else:
        run(args)
        