
import time
import os
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from sim_env2 import make_sim_env, BOX_POSE
from constants2 import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS


onscreen_render = True
render_cam_name = 'top'
episode_idx = 1
task_name = 'sim_transfer_cube_scripted'

with open(f'sample_data/{episode_idx}/subtask_info.pkl', 'rb') as file:
    subtask_info = pickle.load(file)

with open(f'sample_data/{episode_idx}/joint_traj.pkl', 'rb') as file:
    joint_traj = pickle.load(file)

env = make_sim_env(task_name)
BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
ts = env.reset()
success = []
episode_replay = [ts]
# setup plotting
if onscreen_render:
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images'][render_cam_name])
    plt.ion()
for t in tqdm(range(len(joint_traj))): # note: this will increase episode length by 1
    
    action = joint_traj[t]
    # print(action)
    ts = env.step(action)
    episode_replay.append(ts)
    if onscreen_render:
        plt_img.set_data(ts.observation['images'][render_cam_name])
        plt.pause(0.002)

episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
if episode_max_reward == env.task.max_reward:
    success.append(1)
    print(f"{episode_idx=} Successful, {episode_return=}")
else:
    success.append(0)
    print(f"{episode_idx=} Failed")

plt.close()