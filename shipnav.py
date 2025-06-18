DATA_FOLDER = "obs_data"
TARGET_FPS = 10
import pandas as pd
import os
import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
from waymax.datatypes import (
    SimulatorState,
    ShipTrajectory as Trajectory,
    ObjectMetadata,
)
from waymax import dynamics, agents
from waymax.env import MultiAgentEnvironment
from waymax.config import EnvironmentConfig, ObjectType
import dataclasses
import jax
import mediapy
import matplotlib.pyplot as plt
from bc_actor import BCActor
import torch

def unpack_observation_vector(obs_vector):
    ego = obs_vector[:44].reshape(11, 4)
    neighbors = obs_vector[44:484].reshape(10, 11, 4)
    goal = obs_vector[484:486]
    return ego, neighbors, goal

episode_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")])
# print(f"Found {len(episode_files)} episodes:")
# for f in episode_files:
#     print(f"  - {f}")

num_ships = len(episode_files)
dfs = [pd.read_csv(os.path.join(DATA_FOLDER, f)) for f in episode_files]
max_length = max(len(df) for df in dfs)
obj_idx = jnp.arange(num_ships)
# print(f"Maximum episode length: {max_length} steps")

x, y, speed, heading, valid = [], [], [], [], []
goal_positions = []
ego_histories_all = []
neighbor_histories_all = []

for df in dfs:
    episode_x, episode_y, episode_speed, episode_heading, episode_valid = [], [], [], [], []
    ego_histories_this_episode = []
    neighbor_histories_this_episode = []
    episode_goal = None

    for i in range(len(df)):
        row = df.iloc[i]
        obs_vector = row.to_numpy(dtype=np.float32)
        ego, neighbors, goal = unpack_observation_vector(obs_vector)

        if episode_goal is None:
            episode_goal = goal

        latest_ego = ego[-1]
        episode_x.append(latest_ego[0])
        episode_y.append(latest_ego[1])
        episode_speed.append(latest_ego[2])
        episode_heading.append(latest_ego[3])
        episode_valid.append(True)

        ego_histories_this_episode.append(ego)
        neighbor_histories_this_episode.append(neighbors)

    # Padding
    pad_len = max_length - len(df)
    episode_x += [0.0] * pad_len
    episode_y += [0.0] * pad_len
    episode_speed += [0.0] * pad_len
    episode_heading += [0.0] * pad_len
    episode_valid += [False] * pad_len

    ego_histories_this_episode += [np.zeros((11, 4), dtype=np.float32)] * pad_len
    neighbor_histories_this_episode += [np.zeros((10, 11, 4), dtype=np.float32)] * pad_len

    x.append(episode_x)
    y.append(episode_y)
    speed.append(episode_speed)
    heading.append(episode_heading)
    valid.append(episode_valid)
    ego_histories_all.append(ego_histories_this_episode)
    neighbor_histories_all.append(neighbor_histories_this_episode)
    goal_positions.append(episode_goal)

goal_positions = np.array(goal_positions)

x = jnp.array(x)
y = jnp.array(y)
speed = jnp.array(speed)
heading = jnp.array(heading)
valid = jnp.array(valid)
vel_x = speed * jnp.cos(heading)
vel_y = speed * jnp.sin(heading)
ego_histories_all = jnp.array(ego_histories_all)
neighbor_histories_all = jnp.array(neighbor_histories_all)
goal_positions = jnp.array(goal_positions)

timestamps = []
for df in dfs:
    timestamps_episode = np.arange(len(df)) * 10
    timestamps_episode_micro = (timestamps_episode * 1e6).astype(np.int64)
    pad_len = max_length - len(df)
    timestamps_episode_micro = np.pad(timestamps_episode_micro, (0, pad_len), 'constant')
    timestamps.append(timestamps_episode_micro)
timestamps = jnp.array(timestamps)

traj = Trajectory(
    x=x,
    y=y,
    speed=speed,
    yaw=heading,
    vel_x=vel_x,
    vel_y=vel_y,
    valid=valid,
    timestamp_micros=timestamps,
    ego_histories=ego_histories_all,
    neighbor_histories=neighbor_histories_all,
    goals=goal_positions
)


meta = ObjectMetadata(
    ids=jnp.arange(num_ships),
    object_types=jnp.zeros(num_ships, dtype=jnp.int32),
    is_sdc=jnp.array([True] + [False]*(num_ships-1)),
    is_modeled=jnp.ones(num_ships, dtype=bool),
    is_valid=jnp.ones(num_ships, dtype=bool),
    is_controlled = jnp.ones(num_ships, dtype=bool),
    objects_of_interest=jnp.zeros(num_ships, dtype=bool)
)


sim_state = SimulatorState(
    sim_trajectory=traj,
    log_trajectory=traj,
    object_metadata=meta,
    timestep=jnp.array(0),
)

dynamics_model = dynamics.StateDynamics()
env = MultiAgentEnvironment(
    dynamics_model=dynamics_model,
    config=dataclasses.replace(
        EnvironmentConfig(),
        max_num_objects=num_ships,
        controlled_object=ObjectType.VALID
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model_weights/bc_weights.pth"


bc_actor = BCActor(
    model_path=model_path,
    device=device,
    dynamics_model=dynamics_model
)

# actor_const = agents.create_constant_speed_actor(
#     speed=500.0,
#     dynamics_model=dynamics_model,
#     is_controlled_func=lambda state: obj_idx == 0
# )

actor_expert = agents.create_expert_actor(
    dynamics_model=dynamics_model
)

# actor_static = agents.create_expert_actor(
#     dynamics_model=dynamics_model,
#     is_controlled_func=lambda state: obj_idx > 1
# )

actor_list = [
    (actor_expert, lambda state: (state.object_metadata.is_controlled) & (state.object_metadata.ids > 0)),
    (bc_actor, lambda state: (state.object_metadata.is_controlled) & (state.object_metadata.ids == 0))
]

jit_step = jax.jit(env.step)
# jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

jit_select_action_list = []
for actor, _ in actor_list:
    if isinstance(actor, BCActor):
        jit_select_action_list.append(actor.select_action)  
    else:
        jit_select_action_list.append(jax.jit(actor.select_action))


states = [sim_state]
for _ in range(max_length - 1):
    current_state = states[-1]
    outputs = []

    for (jit_select_action, (actor, is_controlled_func)) in zip(jit_select_action_list, actor_list):
        controlled = jnp.where(is_controlled_func(current_state))[0]
        for agent_idx in controlled:
            outputs.append(jit_select_action({}, current_state, int(agent_idx), None))

    action = agents.merge_actions(outputs)
    next_state = jit_step(current_state, action)
    states.append(next_state)


def render_global_state(state, goal_positions, step_idx=None, wake_length=5):
    x = state.sim_trajectory.x
    y = state.sim_trajectory.y
    yaw = state.sim_trajectory.yaw
    valid = state.sim_trajectory.valid

    num_agents = x.shape[0]
    all_x = []
    all_y = []
    for agent_idx in range(num_agents):
        valid_mask = valid[agent_idx]
        x_valid = x[agent_idx][valid_mask]
        y_valid = y[agent_idx][valid_mask]
        non_zero_mask = (x_valid != 0) | (y_valid != 0)
        if jnp.any(non_zero_mask):
            all_x.append(x_valid[non_zero_mask])
            all_y.append(y_valid[non_zero_mask])

    if all_x and any(len(arr) > 0 for arr in all_x):
        all_x_concat = jnp.concatenate(all_x)
        all_y_concat = jnp.concatenate(all_y)
        x_min = min(all_x_concat.min(), goal_positions[:, 0].min()) - 50
        x_max = max(all_x_concat.max(), goal_positions[:, 0].max()) + 50
        y_min = min(all_y_concat.min(), goal_positions[:, 1].min()) - 50
        y_max = max(all_y_concat.max(), goal_positions[:, 1].max()) + 50
    else:
        x_min = goal_positions[:, 0].min() - 100
        x_max = goal_positions[:, 0].max() + 100
        y_min = goal_positions[:, 1].min() - 100
        y_max = goal_positions[:, 1].max() + 100

    fig, ax = plt.subplots(figsize=(10, 8))
    for goal_idx, (gx, gy) in enumerate(goal_positions):
        ax.plot(gx, gy, marker='*', color='gold', markersize=15,
                markeredgecolor='black', markeredgewidth=1)
        ax.text(gx + 20, gy + 20, f'{goal_idx}', color='black',
                fontsize=12, ha='left', va='bottom', fontweight='bold')

    current_step = state.timestep.item()
    for agent_idx in range(num_agents):
        if (current_step < valid.shape[1] and valid[agent_idx, current_step]):
            ship_x = x[agent_idx, current_step]
            ship_y = y[agent_idx, current_step]
            ship_yaw = yaw[agent_idx, current_step]
            if ship_x == 0 and ship_y == 0:
                continue
            goal_x, goal_y = goal_positions[agent_idx]
            dist_to_goal = jnp.sqrt((ship_x - goal_x)**2 + (ship_y - goal_y)**2)
            if dist_to_goal > 50.0:
                ship_length = 100.0 * 1.5
                ship_width = 30.0 * 1.5

                front = (ship_x + ship_length * 0.5 * np.cos(ship_yaw),
                         ship_y + ship_length * 0.5 * np.sin(ship_yaw))
                left_back = (ship_x - ship_length * 0.5 * np.cos(ship_yaw) - ship_width * 0.5 * np.sin(ship_yaw),
                             ship_y - ship_length * 0.5 * np.sin(ship_yaw) + ship_width * 0.5 * np.cos(ship_yaw))
                right_back = (ship_x - ship_length * 0.5 * np.cos(ship_yaw) + ship_width * 0.5 * np.sin(ship_yaw),
                              ship_y - ship_length * 0.5 * np.sin(ship_yaw) - ship_width * 0.5 * np.cos(ship_yaw))

                ship_shape = np.array([front, left_back, right_back, front])
                ax.plot(ship_shape[:, 0], ship_shape[:, 1], color='blue', linewidth=2)
                ax.fill(ship_shape[:, 0], ship_shape[:, 1], color='blue', alpha=0.3)
                ax.text(ship_x + 60, ship_y, f'{agent_idx}', color='black',
                       fontsize=12, ha='left', va='center', fontweight='bold')

                wake_indices = np.arange(max(0, current_step - wake_length), current_step)
                for i, wake_step in enumerate(wake_indices):
                    if (wake_step < valid.shape[1] and valid[agent_idx, wake_step]):
                        wake_x = x[agent_idx, wake_step]
                        wake_y = y[agent_idx, wake_step]

                        if not (wake_x == 0 and wake_y == 0):
                            alpha = (i + 1) / wake_length * 0.6
                            ax.plot(wake_x, wake_y, marker='o', color='lightblue',
                                   alpha=alpha, markersize=3)
            else:
                ax.plot(goal_x, goal_y, marker='o', color='green', markersize=20, alpha=0.7)
                ax.text(goal_x + 30, goal_y + 30, f'{agent_idx}✓',
                       color='green', fontsize=10, ha='left', va='bottom', fontweight='bold')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    title = f'Ship Simulation - Step {step_idx}' if step_idx is not None else 'Ship Simulation'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img

imgs = [
    render_global_state(s, goal_positions=goal_positions, step_idx=i)
    for i, s in enumerate(states)
]

mediapy.write_video("ship_simulation.mp4", imgs, fps=TARGET_FPS)

# ship_idx = 1
# timestep = 5

# print("Ego history at timestep", timestep)
# print(sim_state.sim_trajectory.ego_histories[ship_idx, timestep])
# print("Shape:", sim_state.sim_trajectory.ego_histories[ship_idx, timestep].shape)

# print("\nNeighbor histories at timestep", timestep)
# # print(sim_state.sim_trajectory.neighbor_histories[ship_idx, timestep])
# print("Shape:", sim_state.sim_trajectory.neighbor_histories[ship_idx, timestep].shape)

# print("\nGoal position")
# print(sim_state.sim_trajectory.goals[ship_idx])
