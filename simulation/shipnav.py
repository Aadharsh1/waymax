import os
import pickle
import dataclasses

import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp

import torch
import mediapy
import matplotlib.pyplot as plt

from waymax.datatypes import (
    SimulatorState,
    ShipTrajectory as Trajectory,
    ObjectMetadata,
    Action,
)
from waymax import dynamics, agents, datatypes
from waymax.env import MultiAgentEnvironment, WaymaxGymEnv
from waymax.config import EnvironmentConfig, ObjectType
from waymax.agents.actor_core import WaymaxActorOutput

from maritime_rl.utils import haversine_distance   

from .bc_actor import BCActor   

TARGET_FPS = 10

class CustomShipDynamics(dynamics.StateDynamics):
    def forward(self, action: datatypes.Action,
                trajectory: datatypes.Trajectory,  
                reference_trajectory: datatypes.Trajectory,
                is_controlled: jnp.ndarray,
                timestep: jnp.ndarray,
                allow_object_injection: bool) -> datatypes.Trajectory:

        updated_base_trajectory = super().forward(
            action=action,
            trajectory=trajectory,
            reference_trajectory=reference_trajectory,
            is_controlled=is_controlled,
            timestep=timestep,
            allow_object_injection=allow_object_injection
        )

        return trajectory.replace(
            x=updated_base_trajectory.x,
            y=updated_base_trajectory.y,
            speed=updated_base_trajectory.speed,
            yaw=updated_base_trajectory.yaw,
            vel_x=updated_base_trajectory.vel_x,
            vel_y=updated_base_trajectory.vel_y,
            valid=updated_base_trajectory.valid,
            timestamp_micros=updated_base_trajectory.timestamp_micros,
            ego_histories=trajectory.ego_histories,
            neighbor_histories=trajectory.neighbor_histories,
            goals=trajectory.goals
        )

class CustomEnvironment(MultiAgentEnvironment):
    def __init__(self, dynamics_model, config, trajs, times, overlap_idx, original_ship_indices):  
        super().__init__(dynamics_model, config)
        self.trajs = trajs
        self.times = times
        self.overlap_idx = overlap_idx  
        self.original_ship_indices = original_ship_indices 
        self.ego_start_times = [times[idx][0] for idx in original_ship_indices]
    
    def observe_agent(self, states_history, agent_idx, **kwargs):
        original_agent_idx = self.original_ship_indices[agent_idx]
        
        return super().observe_agent(
            states_history=states_history,
            agent_idx=original_agent_idx,  
            trajs=self.trajs,
            times=self.times,
            overlap_idx=self.overlap_idx,  
            ego_start_time=self.ego_start_times[agent_idx],
            **kwargs
        )

with open('trajs_times/trajs_times_overlap.pkl', 'rb') as f:
    data = pickle.load(f)

trajs = data['trajs']
times = data['times']
overlap_idx = data['overlap_idx']  

with open('observations/observations_original.pkl', 'rb') as f:
    observations = pickle.load(f)

region_of_interest = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}
origin_lon, origin_lat = region_of_interest['LON'][0], region_of_interest['LAT'][0]
max_x = haversine_distance(origin_lon, origin_lat, region_of_interest['LON'][1], origin_lat)
max_y = haversine_distance(origin_lon, origin_lat, origin_lon, region_of_interest['LAT'][1])

SHIP_INDICES = [0, 55]
expert_agent_ids = jnp.array([0])    
bc_agent_ids = jnp.array([1])

observations = [observations[i] for i in SHIP_INDICES]
num_ships = len(observations)
max_length = max(len(ship_obs) for ship_obs in observations)
print(max_length)
ego_histories_all = []
neighbor_histories_all = []
goal_positions = []
x, y, speed, heading, valid = [], [], [], [], []

NUM_NEIGHBORS = 10  
HISTORY_STEPS = 5

for ship_idx in range(num_ships):
    ship_obs = observations[ship_idx]
    # print(len(ship_obs))
    episode_x, episode_y, episode_speed, episode_heading, episode_valid = [], [], [], [], []
    ego_histories_this_episode = []
    neighbor_histories_this_episode = []
    episode_goal = None

    for t in range(len(ship_obs)):
        obs = ship_obs[t]
        ego = obs['ego']
        neighbors = obs['neighbors']
        goal = obs['goal']

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

    # Padding for episodes shorter than max_length
    pad_len = max_length - len(ship_obs)
    episode_x += [0.0] * pad_len
    episode_y += [0.0] * pad_len
    episode_speed += [0.0] * pad_len
    episode_heading += [0.0] * pad_len
    episode_valid += [False] * pad_len

    ego_histories_this_episode += [np.zeros((HISTORY_STEPS + 1, 4), dtype=np.float32)] * pad_len
    neighbor_histories_this_episode += [np.zeros((NUM_NEIGHBORS, HISTORY_STEPS + 1, 4), dtype=np.float32)] * pad_len

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
for length in [len(ship_obs) for ship_obs in observations]:
    timestamps_episode = np.arange(length) * 10
    timestamps_episode_micro = (timestamps_episode * 1e6).astype(np.int64)
    pad_len = max_length - length
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
    is_controlled=jnp.ones(num_ships, dtype=bool),
    objects_of_interest=jnp.zeros(num_ships, dtype=bool)
)

sim_state = SimulatorState(
    sim_trajectory=traj,
    log_trajectory=traj,
    object_metadata=meta,
    timestep=jnp.array(0),
)



dynamics_model = CustomShipDynamics()


env = CustomEnvironment(
    dynamics_model=dynamics_model,
    config=dataclasses.replace(
        EnvironmentConfig(),
        max_num_objects=num_ships,
        controlled_object=ObjectType.VALID
    ),
    trajs=trajs,
    times=times,
    overlap_idx=overlap_idx,  
    original_ship_indices=SHIP_INDICES 
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


actor_expert = agents.create_expert_actor(
    dynamics_model=CustomShipDynamics()
)

bc_actor = BCActor(
    model_path="new_model_weights/bc_original_epoch300.th",
    device=device,
    dynamics_model=dynamics_model,
    environment=env,
    normalize=False,  
    max_x=max_x,
    max_y=max_y
)


actor_list = [
    (actor_expert, lambda state: jnp.isin(state.object_metadata.ids, expert_agent_ids)),
    (bc_actor, lambda state: jnp.isin(state.object_metadata.ids, bc_agent_ids))
]

jit_step = jax.jit(env.step)

jit_select_action_list = []
for actor, _ in actor_list:
    if isinstance(actor, BCActor):
        jit_select_action_list.append(actor.select_action)  
    else:
        jit_select_action_list.append(jax.jit(actor.select_action))

num_agents = num_ships 
print(num_agents)

def no_op_action_for_agent(agent_idx, num_objects):
    action_data = jnp.zeros((num_objects, 5), dtype=jnp.float32)
    valid = jnp.zeros((num_objects, 1), dtype=bool)
    is_ctrl = jnp.zeros((num_objects,), dtype=bool)
    return WaymaxActorOutput(
        actor_state=None,
        action=Action(data=action_data, valid=valid),
        is_controlled=is_ctrl
    )


states = [sim_state]
done_mask = jnp.zeros(num_agents, dtype=bool)  
done_mask_history = []

max_timesteps = 1000
for _ in range(max_timesteps):
    current_state = states[-1]
    outputs = [None] * num_agents

    for agent_idx in range(num_agents):
        for (actor, is_controlled_func), jit_select_action in zip(actor_list, jit_select_action_list):
            if is_controlled_func(current_state)[agent_idx]:
                if not done_mask[agent_idx]:
                    outputs[agent_idx] = jit_select_action({}, current_state, int(agent_idx), states)
                else:
                    outputs[agent_idx] = no_op_action_for_agent(agent_idx, num_agents)
                break  

    action = agents.merge_actions(outputs)
    next_state = jit_step(current_state, action)
    states.append(next_state)
    # Update done_mask for each agent
    terminated = env.termination(next_state)
    truncated = env.truncation(next_state)
    done_mask = jnp.logical_or(done_mask, jnp.logical_or(terminated, truncated))
    done_mask_history.append(done_mask.copy())

    if jnp.all(done_mask):
        break


if len(done_mask_history) == 0:
    done_mask_history = jnp.zeros((1, num_agents), dtype=bool)
else:
    done_mask_history = jnp.stack(done_mask_history, axis=0)
first_done_step = []
for agent_idx in range(num_agents):
    done_steps = jnp.where(done_mask_history[:, agent_idx])[0]
    if len(done_steps) > 0:
        first_done_step.append(int(done_steps[0])+1)
    else:
        first_done_step.append(len(states) - 1)  


metrics = {
    'gc_ade': [],
    'goal_rate': [],
    'near_miss_rate': [],
    'avg_curvature': [],
}

num_steps = len(states)
near_miss_threshold = 555  
goal_threshold = 200  

for agent_idx in range(num_ships):
    # Check if this agent is valid
    valid_mask = traj.valid[agent_idx]
    if not jnp.any(valid_mask):
        # Skip padded agents
        metrics['gc_ade'].append(0.0)
        metrics['goal_rate'].append(False)
        metrics['near_miss_rate'].append(0.0)
        metrics['avg_curvature'].append(0.0)
        continue
    
    # Find the last valid timestep for this agent
    last_valid_idx = int(jnp.where(valid_mask)[0][-1])
    
    # Determine the number of steps to consider (up to done step or last valid)
    T = min(first_done_step[agent_idx] + 1, last_valid_idx + 1)
    
    # GC-ADE Calculation
    expert_x = traj.x[agent_idx]
    expert_y = traj.y[agent_idx]
    sim_x = jnp.array([state.sim_trajectory.x[agent_idx, int(state.timestep)] for state in states[:T]])
    sim_y = jnp.array([state.sim_trajectory.y[agent_idx, int(state.timestep)] for state in states[:T]])
    min_T = min(len(expert_x), len(sim_x))
    gc_ade = jnp.mean(jnp.sqrt((expert_x[:min_T] - sim_x[:min_T])**2 + (expert_y[:min_T] - sim_y[:min_T])**2))
    metrics['gc_ade'].append(float(gc_ade))

    # Goal Reached Calculation (using simulated trajectory's final position)
    final_x = sim_x[-1] if len(sim_x) > 0 else 0.0
    final_y = sim_y[-1] if len(sim_y) > 0 else 0.0
    goal_x, goal_y = goal_positions[agent_idx]
    dist_to_goal = jnp.sqrt((final_x - goal_x)**2 + (final_y - goal_y)**2)
    print(f"Ship {agent_idx}: Final valid position = ({float(final_x):.2f}, {float(final_y):.2f}), "
          f"Goal position = ({float(goal_x):.2f}, {float(goal_y):.2f})")
    print(f"Distance to goal = {float(dist_to_goal):.2f} m")
    goal_reached = dist_to_goal < goal_threshold
    metrics['goal_rate'].append(bool(goal_reached))

    # Near Miss Rate Calculation
    near_miss_count = 0
    for t in range(T):
        if t >= len(states):
            break
            
        # Get dynamic neighbors using observe_agent
        obs = env.observe_agent(states[:t+1], agent_idx)
        neighbors = obs['neighbors']
        
        timestep_idx = int(states[t].timestep)
        x_i = states[t].sim_trajectory.x[agent_idx, timestep_idx]
        y_i = states[t].sim_trajectory.y[agent_idx, timestep_idx]
        
        # Skip if position is (0,0) - likely invalid
        if x_i == 0.0 and y_i == 0.0:
            continue
            
        min_distance = float('inf')
        
        # Use current neighbor positions from observe_agent
        for neighbor in neighbors:
            neighbor_x = neighbor[-1, 0]  # Current neighbor position
            neighbor_y = neighbor[-1, 1]
            if neighbor_x == 0.0 and neighbor_y == 0.0:
                continue
            dist = jnp.sqrt((x_i - neighbor_x)**2 + (y_i - neighbor_y)**2)
            min_distance = min(min_distance, dist)
        
        # Count near miss if closest neighbor distance is below threshold
        if min_distance < near_miss_threshold and min_distance != float('inf'):
            near_miss_count += 1

    # Calculate rate as percentage of timesteps with near misses
    near_miss_rate = 100 * near_miss_count / T if T > 0 else 0.0
    metrics['near_miss_rate'].append(float(near_miss_rate))

    curvatures = []
    dt = 10.0
    for t in range(1, T):
        if t < T - 1:
            x0, y0 = sim_x[t-1], sim_y[t-1]
            x1, y1 = sim_x[t], sim_y[t]
            x2, y2 = sim_x[t+1], sim_y[t+1]
            dx_dt = (x1 - x0) / dt  
            dy_dt = (y1 - y0) / dt
            d2x_dt2 = (x2 - 2*x1 + x0) / dt
            d2y_dt2 = (y2 - 2*y1 + y0) / dt
            velocity_mag_squared = dx_dt**2 + dy_dt**2
            if velocity_mag_squared < 1e-3:
                curvature = 0.0  
            else:
                denominator = velocity_mag_squared ** 1.5
                curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / denominator
            curvatures.append(curvature)
    avg_curvature = jnp.mean(jnp.array(curvatures)) if curvatures else 0.0
    metrics['avg_curvature'].append(float(avg_curvature))
# Display Metrics
for i in range(num_ships):
    print(f"Ship {i} | GC-ADE: {metrics['gc_ade'][i]:.2f} m | "
          f"Goal Reached: {metrics['goal_rate'][i]} | "
          f"Near Miss Rate: {metrics['near_miss_rate'][i]:.3f}% | "
          f"Avg Curvature: {metrics['avg_curvature'][i]:.5f}")





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

filename = 'example_2'

mediapy.write_video(f"visuals/{filename}.mp4", imgs, fps=TARGET_FPS)


##Debugging purposes

# ship_idx = 0
# timestep = 5

# # print("=== LOG TRAJECTORY (Original) ===")
# # print(f"X: {sim_state.log_trajectory.x[ship_idx, timestep-1]}")
# # print(f"Y: {sim_state.log_trajectory.y[ship_idx, timestep-1]}")
# # print(f"Speed: {sim_state.log_trajectory.speed[ship_idx, timestep-1]}")
# # print(f"Yaw: {sim_state.log_trajectory.yaw[ship_idx, timestep-1]}")

# print("\n=== SIM TRAJECTORY (Simulated Previous) ===")
# print(f"X: {states[timestep].sim_trajectory.x[ship_idx, timestep-1]}")
# print(f"Y: {states[timestep].sim_trajectory.y[ship_idx, timestep-1]}")
# print(f"Speed: {states[timestep].sim_trajectory.speed[ship_idx, timestep-1]}")
# print(f"Yaw: {states[timestep].sim_trajectory.yaw[ship_idx, timestep-1]}")

# print("\n=== SIM TRAJECTORY (Simulated Current) ===")
# print(f"X: {states[timestep].sim_trajectory.x[ship_idx, timestep]}")
# print(f"Y: {states[timestep].sim_trajectory.y[ship_idx, timestep]}")
# print(f"Speed: {states[timestep].sim_trajectory.speed[ship_idx, timestep]}")
# print(f"Yaw: {states[timestep].sim_trajectory.yaw[ship_idx, timestep]}")

