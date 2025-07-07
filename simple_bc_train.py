import os
import pickle
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from maritime_rl import det_bc  
from imitation.data import types, rollout
from imitation.util import logger as imit_logger

with open('./trajs_times/trajs_times.pkl', 'rb') as f:
    data = pickle.load(f)
trajs = data['trajs'][:50]
times = data['times'][:50]

from build_observations import build_observations
observations = build_observations(trajs, times, N=10, history_length=10)

im_trajs = []
for ship_idx, ship_obs in enumerate(observations):
    if len(ship_obs) < 2:
        continue
        
    T = len(ship_obs) - 1
    obs_list, act_list, rew_list = [], [], []

    for t in range(T):
        obs = ship_obs[t]
        next_obs = ship_obs[t+1]
        
        dx = next_obs['ego'][-1, 0] - obs['ego'][-1, 0]
        dy = next_obs['ego'][-1, 1] - obs['ego'][-1, 1]
        dheading = (next_obs['ego'][-1, 3] - obs['ego'][-1, 3] + np.pi) % (2 * np.pi) - np.pi
        action = np.array([dx, dy, dheading], dtype=np.float32)
        
        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(0.0)
    

    obs_list.append(ship_obs[-1])

    ego_stack = np.stack([o['ego'] for o in obs_list]).astype(np.float32)
    neighbors_stack = np.stack([o['neighbors'] for o in obs_list]).astype(np.float32)
    goal_stack = np.stack([o['goal'] for o in obs_list]).astype(np.float32)

    obs_dict = {
        'ego': ego_stack,
        'neighbors': neighbors_stack,
        'goal': goal_stack
    }

    acts = np.array(act_list)
    rews = np.array(rew_list)

    dict_obs = types.DictObs(obs_dict)

    im_trajs.append(types.TrajectoryWithRew(
        obs=dict_obs,  
        acts=acts,
        infos=None,
        terminal=True,
        rews=rews
    ))

# total_obs = sum(len(traj.obs) for traj in im_trajs) 
# total_actions = sum(len(traj.acts) for traj in im_trajs)
# print(f"Total observations across all trajectories: {total_obs}")
# print(f"Total actions across all trajectories: {total_actions}")


# for idx, traj in enumerate(im_trajs):
#     print(f"Trajectory {idx}: {len(traj.obs)} observations, {len(traj.acts)} actions")


# Flatten trajectories
transitions = rollout.flatten_trajectories(im_trajs)


obs_space = spaces.Dict({
    'ego': spaces.Box(low=-np.inf, high=np.inf, shape=(11, 4), dtype=np.float32),
    'neighbors': spaces.Box(low=-np.inf, high=np.inf, shape=(10, 11, 4), dtype=np.float32),
    'goal': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
})
action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)


tb_logdir = "logs/BC"
custom_logger = imit_logger.configure(
    folder=tb_logdir,
    format_strs=["tensorboard", "stdout"],
)


rng = np.random.default_rng(42)
policy = det_bc.DetPolicy(
    observation_space=obs_space,
    action_space=action_space,
    net_arch=[256],
    features_extractor_class=det_bc.NewCombinedNormExtractor,
)

bc_trainer = det_bc.BC(
    observation_space=obs_space,
    action_space=action_space,
    demonstrations=transitions,
    rng=rng,
    policy=policy,
    l2_weight=0.001,
    batch_size=256,
    custom_logger=custom_logger
)


print(f"Training on {len(transitions.acts)} transitions from {len(im_trajs)} trajectories...")
bc_trainer.train(n_epochs=10)


stats = rollout.rollout_stats(im_trajs)
print("Rollout stats:")
for k, v in stats.items():
    print(f"{k}: {v}")



os.makedirs("new_model_weights", exist_ok=True)
model_path = os.path.join("new_model_weights", "bc_model.th")
th.save(bc_trainer.policy, model_path)
print(f"Model saved to {model_path}")
