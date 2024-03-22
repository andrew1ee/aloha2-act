import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy, InterACTPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'InterACT':
        state_dim = 7
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [['policy_left_epoch_1000_seed_0.ckpt', 'policy_right_epoch_1000_seed_0.ckpt']]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.isdir(f'{ckpt_dir}/plots'):
        os.makedirs(f'{ckpt_dir}/plots')
    if not os.path.isdir(f'{ckpt_dir}/weights'):
        os.makedirs(f'{ckpt_dir}/weights')
    if not os.path.isdir(f'{ckpt_dir}/ckpts/best'):
        os.makedirs(f'{ckpt_dir}/ckpts/best')
    if not os.path.isdir(f'{ckpt_dir}/videos'):
        os.makedirs(f'{ckpt_dir}/videos')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_left_ckpt_info, best_right_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_left_dict = best_left_ckpt_info
    best_epoch, min_val_loss, best_state_right_dict = best_right_ckpt_info

    # save best checkpoint
    ckpt_left_path = os.path.join(ckpt_dir, f'weights/best/policy_best_left.ckpt')
    torch.save(best_state_left_dict, ckpt_left_path)
    ckpt_right_path = os.path.join(ckpt_dir, f'weights/best/policy_best_right.ckpt')
    torch.save(best_state_right_dict, ckpt_right_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'InterACT':
        policy = InterACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'InterACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_left_path = os.path.join(ckpt_dir, ckpt_name[0])
    ckpt_right_path = os.path.join(ckpt_dir, ckpt_name[1])
    policy_left = make_policy(policy_class, policy_config)
    policy_right = make_policy(policy_class, policy_config)
    loading_status_left = policy_left.load_state_dict(torch.load(ckpt_left_path))
    loading_status_right = policy_right.load_state_dict(torch.load(ckpt_left_path))

    print(loading_status_left)
    print(loading_status_right)
    policy_left.cuda()
    policy_right.cuda()

    policy_left.eval()
    policy_right.eval()
    print(f'Loaded: {ckpt_left_path} and {ckpt_right_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process_left = lambda a: a * stats['action_std'][:7] + stats['action_mean'][:7]
    post_process_right = lambda a: a * stats['action_std'][7:] + stats['action_mean'][7:]

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_left_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_right_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_left = qpos[:,:7]
                qpos_right = qpos[:,7:]
                qpos_left_history[:, t] = qpos_left
                qpos_right_history[:, t] = qpos_right
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                if config['policy_class'] == "InterACT":
                    if t % query_frequency == 0:
                        all_left_actions = policy_left(qpos_left, curr_image)
                        all_right_actions = policy_right(qpos_right, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action_left = all_left_actions[:, t % query_frequency]
                        raw_action_right = all_right_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action_left = raw_action_left.squeeze(0).cpu().numpy()
                raw_action_right = raw_action_right.squeeze(0).cpu().numpy()
                action_left = post_process_left(raw_action_left)
                action_right = post_process_right(raw_action_right)
                target_qpos = np.concatenate([action_left, action_right])

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'videos/video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name[0].split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy, which, feedback=None):
    image_data, qpos_data, action_data, is_pad = data
    if which == 'left':
        qpos_data = qpos_data[:,:7]
        action_data = action_data[:,:,:7]
    elif which == 'right':
        qpos_data = qpos_data[:,7:]
        action_data = action_data[:,:,7:]
    image_data, qpos_data, action_data, is_pad= image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    if feedback is not None:
        feedback = feedback.cuda()
    return_value = policy(qpos_data, image_data, action_data, feedback, is_pad) # TODO remove None
    return return_value

def extract_feedback(forward_dict, which):
    if which == 'left':
        feedback_data = forward_dict
    elif which == 'right':
        feedback_data = forward_dict
    return feedback_data


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy_left = make_policy(policy_class, policy_config).cuda()
    policy_right = make_policy(policy_class, policy_config).cuda()

    optimizer_left = make_optimizer(policy_class, policy_left)
    optimizer_right = make_optimizer(policy_class, policy_right)

    train_left_history = []
    train_right_history = []
    validation_left_history = []
    validation_right_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        tqdm.write(f'\nEpoch {epoch}')
        # validation
        

        # training
        policy_left.train()
        policy_right.train()
        optimizer_left.zero_grad()
        optimizer_right.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            if epoch > 0:  # Skip feedback only for the first epoch
                feedback_from_left = extract_feedback(forward_dict_left, which='left')
                feedback_from_right = extract_feedback(forward_dict_right, which='right')
            else:
                feedback_left = feedback_right = None  # No feedback for the first epoch
            forward_dict_left, left_ahat = forward_pass(data, policy_left, which='left', feedback=feedback_right) ## DATA 나눠야함
            forward_dict_right, right_ahat = forward_pass(data, policy_left, which='right', feedback=feedback_left)

            # backward
            loss_left = forward_dict_left['loss']
            loss_right = forward_dict_right['loss']
            
            loss = loss_left + loss_right

            loss.backward()

            optimizer_left.step()
            optimizer_right.step()

            optimizer_left.zero_grad()
            optimizer_right.zero_grad()

            train_left_history.append(detach_dict(forward_dict_left))
            train_right_history.append(detach_dict(forward_dict_right))
        epoch_left_summary = compute_dict_mean(train_left_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_right_summary = compute_dict_mean(train_right_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])

        epoch_train_loss = epoch_left_summary['loss'] + epoch_right_summary['loss']
        tqdm.write(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_left_summary.items():
            summary_string += f'LEFT: {k}: {v.item():.3f} '
        summary_string += '\n'    
        for k, v in epoch_right_summary.items():
            summary_string += f'RIGHT: {k}: {v.item():.3f} '
        tqdm.write(summary_string)

        
        # if epoch > 0:
        with torch.inference_mode():
            policy_left.eval()
            policy_right.eval()

            epoch_left_dicts = []
            epoch_right_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                if epoch > 0:  # Skip feedback only for the first epoch
                    feedback_from_left = extract_feedback(left_ahat, which='left')
                    feedback_from_right = extract_feedback(left_ahat, which='right')
                else:
                    feedback_left = feedback_right = None  # No feedback for the first epoch
                forward_dict_left, left_ahat = forward_pass(data, policy_left, which='left', feedback=feedback_right) ## DATA 나눠야함
                forward_dict_right, right_ahat = forward_pass(data, policy_left, which='right', feedback=feedback_left)
                epoch_left_dicts.append(forward_dict_left)
                epoch_right_dicts.append(forward_dict_right)

            epoch_left_summary = compute_dict_mean(epoch_left_dicts)
            epoch_right_summary  = compute_dict_mean(epoch_right_dicts)
            validation_left_history.append(epoch_left_summary)
            validation_right_history.append(epoch_right_summary)

            epoch_val_loss = epoch_left_summary['loss'] + epoch_right_summary['loss']
            
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_left_ckpt_info = (epoch, min_val_loss, deepcopy(policy_left.state_dict()))
                best_right_ckpt_info = (epoch, min_val_loss, deepcopy(policy_right.state_dict()))
        tqdm.write(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_left_summary.items():
            summary_string += f'LEFT: {k}: {v.item():.3f} '
        summary_string += '\n'
        for k, v in epoch_right_summary.items():
            summary_string += f'RIGHT: {k}: {v.item():.3f} '
        tqdm.write(summary_string)

        if epoch % 10 == 0:
            plot_history(train_left_history, validation_left_history, epoch, ckpt_dir, seed, 'left')
            plot_history(train_right_history, validation_right_history, epoch, ckpt_dir, seed, 'right')
        if epoch % 100 == 0:
            ckpt_left_path = os.path.join(ckpt_dir, f'weights/policy_left_epoch_{epoch}_seed_{seed}.ckpt')
            ckpt_right_path = os.path.join(ckpt_dir, f'weights/policy_right_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy_left.state_dict(), ckpt_left_path)
            torch.save(policy_right.state_dict(), ckpt_right_path)
        

    ckpt_left_path = os.path.join(ckpt_dir, f'weights/policy_left_last.ckpt')
    ckpt_right_path = os.path.join(ckpt_dir, f'weights/policy_right_last.ckpt')
    torch.save(policy_left.state_dict(), ckpt_left_path)
    torch.save(policy_right.state_dict(), ckpt_right_path)

    best_epoch, min_val_loss, best_state_left_dict = best_left_ckpt_info
    best_epoch, min_val_loss, best_state_right_dict = best_right_ckpt_info
    ckpt_left_path = os.path.join(ckpt_dir, f'weights/policy_left_epoch_{best_epoch}_seed_{seed}.ckpt')
    ckpt_right_path = os.path.join(ckpt_dir, f'weights/policy_right_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_left_dict, ckpt_left_path)
    torch.save(best_state_right_dict, ckpt_right_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_left_history, validation_left_history, num_epochs, ckpt_dir, seed, 'left')
    plot_history(train_right_history, validation_right_history, num_epochs, ckpt_dir, seed, 'right')

    return best_left_ckpt_info, best_right_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed, which):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'plots/train_val_{which}_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))
