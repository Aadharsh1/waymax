import os
import pickle
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
import det_bc
from imitation.data import types, rollout
from imitation.util import logger as imit_logger
from utils import haversine_distance
import argparse
import wandb

region_of_interest = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}
origin_lon, origin_lat = region_of_interest['LON'][0], region_of_interest['LAT'][0]
max_x = haversine_distance(origin_lon, origin_lat, region_of_interest['LON'][1], origin_lat)
max_y = haversine_distance(origin_lon, origin_lat, origin_lon, region_of_interest['LAT'][1])

parser = argparse.ArgumentParser(description="Behavioral Cloning Training for Maritime RL")
parser.add_argument('--normalize', action='store_true', help="Normalize observation space coordinates to [-1, 1]")
args = parser.parse_args()

normalize_xy = args.normalize
print(f"Normalization of observation space: {'Enabled' if normalize_xy else 'Disabled'}")

# Define observation space bounds based on normalization
if normalize_xy:
    obs_low = np.array([-1, -1, 0, 0], dtype=np.float32)
    obs_high = np.array([1, 1, 1000, 2*np.pi], dtype=np.float32)
else:
    obs_low = np.array([0, 0, 0, 0], dtype=np.float32)
    obs_high = np.array([max_x, max_y, 1000, 2*np.pi], dtype=np.float32)

ego_obs_shape = (6, 4)
neighbor_obs_shape = (10, 6, 4)


#Define observation space
obs_space = gym.spaces.Dict({
    'ego': gym.spaces.Box(
        low=np.broadcast_to(obs_low,  ego_obs_shape),
        high=np.broadcast_to(obs_high,  ego_obs_shape),
        shape=ego_obs_shape,
        dtype=np.float32
    ),
    'neighbors': gym.spaces.Box(
        low=np.broadcast_to(obs_low,  neighbor_obs_shape),
        high=np.broadcast_to(obs_high,  neighbor_obs_shape),
        shape=neighbor_obs_shape,
        dtype=np.float32
    ),
    "goal": gym.spaces.Box(
        low=obs_low[:2],
        high=obs_high[:2],
        shape=(2,),
        dtype=np.float32
    )
})


#Define action space
action_space = spaces.Box(
    low=np.array([-max_x/20, -max_y/20, -np.pi/2], dtype=np.float32),
    high=np.array([max_x/20, max_y/20, np.pi/2], dtype=np.float32),
    dtype=np.float32
)

#Load the observations 
with open('observations_original.pkl', 'rb') as f:
	observations = pickle.load(f)


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

total_obs = sum(len(traj.obs) for traj in im_trajs) 
total_actions = sum(len(traj.acts) for traj in im_trajs)
print(f"Total observations across all trajectories: {total_obs}")
print(f"Total actions across all trajectories: {total_actions}")

# Flatten trajectories
transitions = rollout.flatten_trajectories(im_trajs)

lr = 1e-4
epoch = 300
batch = 256
l2_w = 0.001
#l2_orig = 0.001
#batch orig = 256 
run_name = 'bc_original_epoch300'

wandb.init(
    project="ShipNavism",
    name=run_name,
    config={
        "learning_rate": lr,  
        "epochs": epoch,
        "batch_size": batch,
        "l2_weight": l2_w,
        "policy": "DetPolicy",
        "architecture": "CustomExtractor_256",
        "dataset_size": len(transitions)
    }
)

wandb.define_metric("epoch")
wandb.define_metric("rollout/*", step_metric="epoch")
wandb.define_metric("train/*", step_metric="epoch")

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
    l2_weight=l2_w,
    batch_size=batch,
    custom_logger=custom_logger
)

print(f"Training on {len(transitions.acts)} transitions from {len(im_trajs)} trajectories...")
bc_trainer.train(n_epochs=epoch)


stats = rollout.rollout_stats(im_trajs)
print("Rollout stats:")
for k, v in stats.items():
    print(f"{k}: {v}")

model_filename = f"{run_name}.th"
os.makedirs("new_model_weights", exist_ok=True)
model_path = os.path.join("new_model_weights", model_filename)
th.save(bc_trainer.policy, model_path)
print(f"Model saved to {model_path}")

print("Saving model to WandB as an Artifact...")
model_artifact = wandb.Artifact(
    name=run_name, 
    type="model",
    description="Trained Behavioral Cloning policy for ship navigation.",
    metadata=dict(wandb.config) 
)

model_artifact.add_file(model_path)
wandb.log_artifact(model_artifact)
print("Model artifact successfully saved to WandB.")
wandb.finish()
