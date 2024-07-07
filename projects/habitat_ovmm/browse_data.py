"""
The script converts the trial pickle data to a video.
Usage:
conda activate home-robot
cd $HOME_ROBOT_ROOT
python projects/habitat_ovmm/browse_data.py

please run this code in graphic interface

"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
from natsort import natsorted
from home_robot_sim.env.habitat_objectnav_env.visualizer import Visualizer
from utils.config_utils import (
    create_agent_config,
    create_env_config,
    get_habitat_config,
    get_omega_config,
)
from omegaconf import DictConfig, OmegaConf
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)


def create_video(images, output_file, fps):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video_writer = cv2.VideoWriter(output_file, fourcc, fps, (height, width))
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for image in images:
        video_writer.write(image)
    video_writer.release()

def main(args):
    # get habitat config
    habitat_config, _ = get_habitat_config(
        args.habitat_config_path, overrides=args.overrides
    )

    # get env config
    env_config = get_omega_config(args.env_config_path)

    # merge habitat and env config to create env config
    env_config = create_env_config(
        habitat_config, env_config, evaluation_type=args.evaluation_type
    )
    OmegaConf.set_readonly(env_config, False)
    env_config['VISUALIZE'] = 1

    # load visualizer
    visualizer = Visualizer(env_config)

    output_dir = args.data_dir + "_video"
    os.makedirs(output_dir, exist_ok=True)
    
    # load pickle data
    demo_path = os.path.join(args.data_dir, args.trial, "obs_data.pkl")
    with open(demo_path, "rb") as f:
        demo = pickle.load(f)
    # Record video
    images = []
    for sample in demo:
        obs = sample['obs_data']
        info = sample['info_data']
        action = sample['action_data']
        step = sample['step']
      
        # concate first-person rgb and semantic frame
        semantic = np.concatenate([obs.rgb, np.expand_dims(obs.semantic, axis=2)], axis=2)
        semantic = semantic.astype("uint8")

        image_vis = visualizer.visualize(
            timestep = step,
            semantic_frame = semantic,
            goal_name = obs.task_observations["goal_name"],
            visualize_goal = True,
            third_person_image = obs.third_person_image,     # (512, 512, 3)
            curr_skill = info['curr_skill'],
            # curr_action = info["curr_action"],
        )
        print(f"action: {action}")

        images.append(image_vis)

    output_path = os.path.join(output_dir, f"{args.trial}.mp4")
    create_video(images, output_path, fps=5)
    print(f"Video saved to {output_path}")
    print(f"Total frames: {len(images)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/datasets/rl_agent",
        help="Path to data root dir",
    )
    parser.add_argument(
        "--trial",
        type=str,
        default="104348082_171512994_8067",
        help="trial id to get video from",
    )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="data/datasets/_videos",
    #     help="Path to output video dir",
    # )
    parser.add_argument(
        "--evaluation_type",
        type=str,
        choices=["local", "local_vectorized", "remote"],
        default="local",
    )
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/agent/heuristic_agent.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/env/hssd_demo.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="baseline",
        choices=["baseline", "random", "explore"],
        help="Agent to evaluate",
    )
    parser.add_argument(
        "--force_step",
        type=int,
        default=20,
        help="force to switch to new episode after a number of steps (NOT USED?)",
    )
    parser.add_argument(
        "overrides",
        default=['VISUALIZE=1'],
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    main(args)
