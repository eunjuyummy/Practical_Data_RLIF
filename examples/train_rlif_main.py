import os
import pickle

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import gym
import numpy as np
import tqdm
from absl import app, flags
import cloudpickle as pickle

import gymnasium

from ..utils.env_utils import GymnasiumWrapper, TrajSampler, get_d4rl_dataset, evaluate, wrap_gym

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from ..agents.rlpd import RLPDSamplerPolicy, get_rlpd_policy_from_model, SACLearner
from ..agents.iql import get_iql_policy_from_model, IQLSamplerPolicy
from ..utils.dataset_utils import ReplayBuffer
from ..utils.dataset_utils import D4RLDataset

from ..models.model import SamplerPolicy, get_policy_from_model, load_model, evaluate_policy

from ..utils.utils import define_flags_with_default, set_random_seed

#import cv2
import gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from base64 import b64encode
from IPython.display import display, HTML
from collections import deque
from typing import Dict
#import moviepy.editor as mpy
import matplotlib.pyplot as plt
import pandas as pd
import csv
from IPython.display import display, clear_output

FLAGS_DEF = define_flags_with_default(
    project_name="rlif_hopper", 
    env_name="hopper-medium-v2", # hopper-medium-v2 # hopper-expert-v2
    sparse_env='Hopper-v4',
    offline_ratio=0.5,
    seed=43,  
    train_sparse=True, # mod
    dataset_dir='',
    # ./RLIF/experts/rlpd_experts/s24_hopper-expert-v2env/model.pkl # 110%
    # ./RLIF/experts/bc_experts/b750e30dc83a4158a7ded9ccc8ef3298/model.pkl # bc 40%
    expert_dir='./RLIF/experts/bc_experts/b750e30dc83a4158a7ded9ccc8ef3298/model.pkl', 
    ground_truth_agent_dir='./RLIF/experts/rlpd_experts/s24_hopper-expert-v2env/model.pkl',
    intervene_threshold=0.3,
    intervention_strategy='', 
    intervene_n_steps=4,

    eval_episodes=100,
    log_interval=1000,
    eval_interval=10000,
    max_traj_length=200,
    batch_size=256,
    max_steps=int(1e6),
    start_training=0,
    pretrain_steps=0,

    tqdm=True,
    save_video=False,
    save_model=True,
    checkpoint_model=True,
    checkpoint_buffer=True,
    utd_ratio=1,
    binary_include_bc=True,
    )


config_flags.DEFINE_config_file(
    "config",
    "./RLIF/configs/rlpd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def combine(one_dict, other_dict):
    combined = {}
    for k, v in one_dict.items():
        if len(v.shape) > 1:
            tmp = np.vstack((v, other_dict[k]))
        else:
            tmp = np.hstack((v, other_dict[k]))
        combined[k] = tmp
    return combined

def save_to_csv(filename, data):
    fieldnames = data.keys()
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def main(_):
    FLAGS = flags.FLAGS
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    set_random_seed(FLAGS.seed)

    wandb.init(project=FLAGS.project_name, mode='online')
    wandb.config.update(FLAGS)

    exp_prefix = f"s{FLAGS.seed}_{FLAGS.pretrain_steps}pretrain_{FLAGS.utd_ratio}utd_{FLAGS.offline_ratio}offline"
    if hasattr(FLAGS.config, "critic_layer_norm") and FLAGS.config.critic_layer_norm:
        exp_prefix += "_LN"

    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)
    
    if FLAGS.save_model:
        model_dir = os.path.join(log_dir, "model")
        os.makedirs(model_dir, exist_ok=True)


    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    ds = D4RLDataset(env)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    sparse_eval_sampler = TrajSampler(GymnasiumWrapper(gymnasium.make(FLAGS.sparse_env).unwrapped), FLAGS.max_traj_length)

    # load agents
    expert_model_pkl_dir = FLAGS.expert_dir
    if 'iql' in expert_model_pkl_dir:
        saved_ckpt_expert = load_model(expert_model_pkl_dir)
        intervene_policy = get_iql_policy_from_model(eval_env, saved_ckpt_expert)
    elif 'rlpd' in expert_model_pkl_dir:
        saved_ckpt_expert = load_model(expert_model_pkl_dir)
        intervene_policy = get_rlpd_policy_from_model(eval_env, saved_ckpt_expert)
    else:
        saved_ckpt_expert = load_model(expert_model_pkl_dir)
        intervene_policy = get_policy_from_model(eval_env, saved_ckpt_expert)
    
    if FLAGS.ground_truth_agent_dir != '':
        if 'iql' in FLAGS.ground_truth_agent_dir:
            ground_truth_agent = load_model(FLAGS.ground_truth_agent_dir)['iql']
            ground_truth_policy = IQLSamplerPolicy(ground_truth_agent.actor)
            ground_truth_agent_type = 'iql'
        elif 'sac' in FLAGS.ground_truth_agent_dir or 'bc' in FLAGS.ground_truth_agent_dir:
            ground_truth_agent = load_model(FLAGS.ground_truth_agent_dir)['sac']
            ground_truth_policy = SamplerPolicy(ground_truth_agent.policy, ground_truth_agent.train_params['policy'])
            ground_truth_agent_type = 'sac'
        elif 'rlpd' in FLAGS.ground_truth_agent_dir:
            ground_truth_agent = load_model(FLAGS.ground_truth_agent_dir)['rlpd']
            ground_truth_policy = RLPDSamplerPolicy(ground_truth_agent.actor)
            ground_truth_agent_type = 'rlpd'
        else:
            raise ValueError("agent type not supported") 
    else:
        ground_truth_agent = FLAGS.ground_truth_agent_dir
        ground_truth_agent_type = ''


    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    if FLAGS.dataset_dir != '':
            with open(FLAGS.dataset_dir, 'rb') as handle:
                dataset = pickle.load(handle)
    else:
        dataset = get_d4rl_dataset(env)

    dataset['actions'] = np.clip(dataset['actions'], -0.999, 0.999)
    dataset['rewards'] = np.zeros_like(dataset['rewards'])
    dataset['masks'] = 1 - dataset['dones']

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    for i in range(len(dataset['rewards'])):
        replay_buffer.insert(
            dict(
                observations=dataset['observations'][i],
                actions=dataset['actions'][i],
                rewards=0,
                masks=dataset['masks'][i],
                dones=dataset['dones'][i],
                next_observations=dataset['next_observations'][i],
            )
        )
    
    for i in tqdm.tqdm(
        range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        offline_batch = ds.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        batch = {}
        for k, v in offline_batch.items():
            batch[k] = v
            if "antmaze" in FLAGS.env_name and k == "rewards":
                batch[k] -= 1

        agent, update_info = agent.update(batch, FLAGS.utd_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)

            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)

            sampler_policy = RLPDSamplerPolicy(agent.actor)
            sparse_trajs = sparse_eval_sampler.sample(
                    sampler_policy,
                    FLAGS.eval_episodes, deterministic=False
                )
            avg_success = evaluate_policy(sparse_trajs,
                                            success_rate=True,
                                            success_function=lambda t: np.all(t['rewards'][-1:]>=10),
                                            )
            wandb.log({f"offline-evaluation/avg_success": avg_success}, step=i)
        
    all_observations = []
    all_actions = []
    all_rewards = []
    all_masks = []
    all_dones = []
    all_next_observations = []
    all_intervene = []

    observation, done = env.reset(), False
    t = 0
    intervene = False
    prev_intervene = False
    stop_intervene_time = -1
    first_intervene_action_mask = []
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        
        policy_action, agent = agent.sample_actions(observation)

        expert_action = intervene_policy(observation.reshape(1, -1), deterministic=False).reshape(-1)
        ground_truth_action = ground_truth_policy(observation.reshape(1, -1), deterministic=False).reshape(-1)

        if 'ref' in FLAGS.intervention_strategy:
            reference_action = expert_action
        else:
            reference_action = ground_truth_action

        if not intervene:
            if ground_truth_agent_type == 'iql':
                gt_q1, gt_q2 = ground_truth_agent.critic(observation, reference_action)
                gt_q = np.min([gt_q1, gt_q2])
                policy_q1, policy_q2 = ground_truth_agent.critic(observation, policy_action)

                policy_q = np.min([policy_q1, policy_q2])
            elif ground_truth_agent_type == 'sac':
                gt_q1 = ground_truth_agent.qf.apply(ground_truth_agent.train_params['qf1'], observation, reference_action)
                gt_q2 = ground_truth_agent.qf.apply(ground_truth_agent.train_params['qf2'], observation, reference_action)
                gt_q = np.min([gt_q1, gt_q2])

                policy_q1 = ground_truth_agent.qf.apply(ground_truth_agent.train_params['qf1'], observation, policy_action)
                policy_q2 = ground_truth_agent.qf.apply(ground_truth_agent.train_params['qf2'], observation, policy_action)
                policy_q = np.min([policy_q1, policy_q2])
            else:
                gt_qs = ground_truth_agent.critic.apply_fn(
                    {"params": ground_truth_agent.critic.params},
                    observation,
                    reference_action,
                    True,
                )
                gt_q = gt_qs.mean(axis=0)

                policy_qs = ground_truth_agent.critic.apply_fn(
                    {"params": ground_truth_agent.critic.params},
                    observation,
                    policy_action,
                    True,
                )
                policy_q = policy_qs.mean(axis=0)
                
            
            # Intervene (base)
            if policy_q < gt_q * FLAGS.intervene_threshold:
                intervene = np.random.choice([0, 1], p=[0.05, 1-0.05]) 
            else:
                intervene = np.random.choice([0, 1], p=[1-0.05, 0.05]) # default: 개입하지 않을 확률 95%, 개입할 확률 5% 
            '''
            # Intervene (time_factor)
            time_factor = min(i / FLAGS.max_steps, 1) 

            if policy_q < gt_q * (FLAGS.intervene_threshold * time_factor):
                intervene = np.random.choice([0, 1], p=[0.05, 1-0.05]) 
            else:
                intervene = np.random.choice([0, 1], p=[1-0.05, 0.05])
            '''
            intervene = bool(intervene)

            if intervene: 
                stop_intervene_time = t + FLAGS.intervene_n_steps

        if t == stop_intervene_time:
            intervene = False
        

        if intervene:
            if t != 0 and not prev_intervene:
                # append state action pair that led to previous intervention
                first_intervene_action_mask[-1] = 1
            
                replay_buffer.insert(
                   dict(
                        observations=all_observations[-1],
                        actions=all_actions[-1],
                        rewards=-1,
                        masks=all_masks[-1],
                        dones=all_dones[-1],
                        next_observations=all_next_observations[-1],
                    )
                )
            if 'label' in FLAGS.intervention_strategy:
                action = policy_action
            else:
                action = expert_action 
        else:
            action = policy_action

            if t != 0:
                replay_buffer.insert(
                   dict(
                        observations=all_observations[-1],
                        actions=all_actions[-1],
                        rewards=0,
                        masks=all_masks[-1],
                        dones=all_dones[-1],
                        next_observations=all_next_observations[-1],
                    )
                )

        next_observation, _, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        prev_intervene = intervene
        all_observations += [observation]
        all_actions += [action]
        all_rewards += [0]
        all_masks += [mask]
        all_dones += [done]
        all_next_observations += [next_observation]
        first_intervene_action_mask.append(0)
        all_intervene += [intervene]
        t += 1

        observation = next_observation

        if done or t > FLAGS.max_traj_length:
            observation, done = env.reset(), False
            intervene = False
            prev_intervene = False
            stop_intervene_time = -1
            t = 0
            try:
                for k, v in info["episode"].items():
                    decode = {"r": "return", "l": "length", "t": "time"}
                    wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.pretrain_steps)
            except:
                pass

        online_batch = replay_buffer.sample(
            int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
        )
        offline_batch = ds.sample(
            int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
        )

        batch = combine(offline_batch, online_batch)
        #combined_batch = combine(offline_batch, online_batch)

        if "antmaze" in FLAGS.env_name:
            batch["rewards"] -= 1

        agent, update_info = agent.update(batch, FLAGS.utd_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            print("evaluate")
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i + FLAGS.pretrain_steps)
            
            wandb.log({f"evaluation/intervene_rate": np.mean(all_intervene[-FLAGS.eval_interval+1:])}, step=i + FLAGS.pretrain_steps)
            
            sampler_policy = RLPDSamplerPolicy(agent.actor)
            sparse_trajs = sparse_eval_sampler.sample(
                    sampler_policy,
                    FLAGS.eval_episodes, deterministic=False
                )
            avg_success = evaluate_policy(sparse_trajs,
                                            success_rate=True,
                                            success_function=lambda t: np.all(t['rewards'][-1:]>=10),
                                            )
            wandb.log({f"evaluation/avg_success": avg_success}, step=i + FLAGS.pretrain_steps)

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                        pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                except:
                    print("Could not save agent buffer.")
            
            if FLAGS.save_model:
                save_data = {'rlpd': agent}
                with open(os.path.join (model_dir, "model.pkl"), 'wb') as fout:
                    pickle.dump(save_data, fout)

            def save_to_csv(filename, data):
                with open(filename, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow(data)

            def run_evaluation_step(i, FLAGS, model_dir, eval_env, all_intervene, sparse_eval_sampler):
                print(f"============{i}============")
                video_folder = f"results_video_40EAM_{i}" # change the model name

                vec_env = gymnasium.make('Hopper-v4', render_mode="rgb_array", reset_noise_scale=0.005 * i) 
                vec_env = RecordVideo(vec_env, video_folder, episode_trigger=lambda e: True)

                model_path = os.path.join(model_dir, "model.pkl")

                if os.path.exists(model_path):
                    with open(model_path, 'rb') as fin:
                        loaded_data = pickle.load(fin)
                        agent = loaded_data['rlpd']
                        print("Model loaded successfully")
                else:
                    print("Model file not found")
                    return
                
                obs = vec_env.reset()

                for j in range(20):
                    eval_info = evaluate(
                        agent,
                        eval_env,
                        num_episodes=FLAGS.eval_episodes,
                        save_video=FLAGS.save_video,
                    )
                    log_data = {f"test/{k}": v for k, v in eval_info.items()}
                    log_data['step'] = j + FLAGS.pretrain_steps
                    intervene_rate = np.mean(all_intervene[-FLAGS.eval_interval+1:])
                    log_data['test/intervene_rate'] = intervene_rate
                    sampler_policy = RLPDSamplerPolicy(agent.actor)

                    sparse_trajs = sparse_eval_sampler.sample(
                        sampler_policy,
                        FLAGS.eval_episodes, deterministic=False
                    )

                    avg_success = evaluate_policy(sparse_trajs,
                                                    success_rate=True,
                                                    success_function=lambda t: np.all(t['rewards'][-1:]>=10),
                                                    )
                    save_to_csv('evaluation_log_40EAM.csv', log_data) # change the model name

                    for traj in sparse_trajs:
                        vec_env.reset()
                        obs = traj['observations'][0]
                        done = traj['dones'][0]
                        step = 0
                        while not done and step < len(traj['observations']):
                            action = traj['actions'][step]
                            obs, reward, terminated, truncated, info = vec_env.step(action)
                            done = terminated or truncated
                            step += 1
                    
                    video_file = os.path.join(video_folder, "rl-video-episode-0.mp4")
                    if os.path.exists(video_file):
                        with open(video_file, "rb") as video:
                            mp4 = video.read()
                        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
                        
                        display(HTML(f"""
                        <video controls style='margin: auto; display: block'>
                            <source src='{data_url}' type='video/mp4'>
                        </video>
                        """))

            if i >= FLAGS.max_steps:
                for step in [1, 5]: 
                    run_evaluation_step(step, FLAGS, model_dir, eval_env, all_intervene, sparse_eval_sampler)

if __name__ == "__main__":
    app.run(main)
