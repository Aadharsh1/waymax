import pickle
TARGET_FPS = 10
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
from waymax.datatypes import (
    SimulatorState,
    ShipTrajectory as Trajectory,
    ObjectMetadata,
    Action,
)
from waymax import dynamics, agents
from waymax.env import MultiAgentEnvironment,WaymaxGymEnv
from waymax.config import EnvironmentConfig, ObjectType
import dataclasses
import jax
import mediapy
import matplotlib.pyplot as plt
from bc_actor import BCActor
import torch
from waymax.agents.actor_core import WaymaxActorOutput
from maritime_rl.utils import haversine_distance
from shipnav import CustomShipDynamics

with open('observations.pkl', 'rb') as f:
    observations = pickle.load(f)

region_of_interest = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}
origin_lon, origin_lat = region_of_interest['LON'][0], region_of_interest['LAT'][0]
max_x = haversine_distance(origin_lon, origin_lat, region_of_interest['LON'][1], origin_lat)
max_y = haversine_distance(origin_lon, origin_lat, origin_lon, region_of_interest['LAT'][1])


NUM_SHIPS_TO_SIMULATE = 100
NUM_NEIGHBORS = 10  
HISTORY_STEPS = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dynamics_model = CustomShipDynamics()

bc_actor = BCActor(
    model_path="new_model_weights/epoch_300_10N5H_NF.th",
    device=device,
    dynamics_model=dynamics_model,
    environment=None, 
    normalize=False,  
    max_x=max_x,
    max_y=max_y
)


# actor_expert = agents.create_expert_actor(dynamics_model=dynamics_model)


all_metrics = {
    'gc_ade': [],
    'goal_rate': [],
    'near_miss_rate': [],
    'avg_curvature': [],
}

# Store ship indices for CSV
ship_indices = []

# Simulation parameters
near_miss_threshold = 555  
goal_threshold = 200  

def create_single_agent_state(ship_obs):
    """Create a single-agent simulator state from ship observations"""
    num_ships = 1  # Single agent
    max_length = len(ship_obs)
    
    ego_histories_all = []
    neighbor_histories_all = []
    goal_positions = []
    x, y, speed, heading, valid = [], [], [], [], []
    
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
    
    x.append(episode_x)
    y.append(episode_y)
    speed.append(episode_speed)
    heading.append(episode_heading)
    valid.append(episode_valid)
    ego_histories_all.append(ego_histories_this_episode)
    neighbor_histories_all.append(neighbor_histories_this_episode)
    goal_positions.append(episode_goal)
    
    # Convert to jax arrays
    goal_positions = jnp.array(goal_positions)
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
    
    # Create timestamps
    timestamps_episode = np.arange(len(ship_obs)) * 10
    timestamps_episode_micro = (timestamps_episode * 1e6).astype(np.int64)
    timestamps = jnp.array([timestamps_episode_micro])
    
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
        is_sdc=jnp.array([True]),
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
    
    return sim_state, traj, goal_positions[0]

def simulate_single_agent(ship_idx, ship_obs):
    """Simulate a single agent and return metrics"""
    print(f"Simulating ship {ship_idx}...")
    
    # Create single-agent state
    sim_state, traj, goal_pos = create_single_agent_state(ship_obs)
    
    # Create environment for this single agent
    env = MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(
            EnvironmentConfig(),
            max_num_objects=1,
            controlled_object=ObjectType.VALID
        )
    )
    
    # Set environment for BC actor
    bc_actor.environment = env
    
    jit_step = jax.jit(env.step)
    jit_select_action = bc_actor.select_action
    
    states = [sim_state]
    max_length = len(ship_obs)
    
    for step in range(max_length - 1):
        current_state = states[-1]
        output = jit_select_action({}, current_state, 0, states)
    
        next_state = jit_step(current_state, output.action)
        states.append(next_state)
        
        terminated = env.termination(next_state)
        truncated = env.truncation(next_state)
        if terminated[0] or truncated[0]:
            break
    
    # Calculate metrics for this agent
    metrics = calculate_single_agent_metrics(states, traj, goal_pos, ship_idx)
    
    return metrics

def calculate_single_agent_metrics(states, traj, goal_pos, ship_idx):
    """Calculate metrics for a single agent"""
    T = len(states)
    
    # GC-ADE Calculation
    expert_x = traj.x[0]  # Single agent, so index 0
    expert_y = traj.y[0]
    sim_x = jnp.array([state.sim_trajectory.x[0, int(state.timestep)] for state in states])
    sim_y = jnp.array([state.sim_trajectory.y[0, int(state.timestep)] for state in states])
    
    min_T = min(len(expert_x), len(sim_x))
    gc_ade = jnp.mean(jnp.sqrt((expert_x[:min_T] - sim_x[:min_T])**2 + (expert_y[:min_T] - sim_y[:min_T])**2))
    
    # Goal Reached Calculation
    final_x = sim_x[-1] if len(sim_x) > 0 else 0.0
    final_y = sim_y[-1] if len(sim_y) > 0 else 0.0
    goal_x, goal_y = goal_pos
    dist_to_goal = jnp.sqrt((final_x - goal_x)**2 + (final_y - goal_y)**2)
    goal_reached = dist_to_goal < goal_threshold
    
    # Near Miss Rate Calculation
    near_miss_count = 0
    for t in range(T):
        timestep_idx = int(states[t].timestep)
        x_i = states[t].sim_trajectory.x[0, timestep_idx]
        y_i = states[t].sim_trajectory.y[0, timestep_idx]
        
        neighbors = states[t].sim_trajectory.neighbor_histories[0, timestep_idx]
        min_distance = float('inf')
        
        for neighbor in neighbors:
            neighbor_x = neighbor[-1, 0]
            neighbor_y = neighbor[-1, 1]
            if neighbor_x == 0.0 and neighbor_y == 0.0:
                continue
            dist = jnp.sqrt((x_i - neighbor_x)**2 + (y_i - neighbor_y)**2)
            min_distance = min(min_distance, dist)
        
        if min_distance < near_miss_threshold and min_distance != float('inf'):
            near_miss_count += 1
    
    near_miss_rate = 100 * near_miss_count / T if T > 0 else 0.0
    
    # Average Curvature Calculation
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
    
    return {
        'gc_ade': float(gc_ade),
        'goal_rate': bool(goal_reached),
        'near_miss_rate': float(near_miss_rate),
        'avg_curvature': float(avg_curvature)
    }

# Main simulation loop
print(f"Starting simulation for {NUM_SHIPS_TO_SIMULATE} ships...")
print(f"Total available observations: {len(observations)}")

for ship_idx in range(min(NUM_SHIPS_TO_SIMULATE, len(observations))):
    ship_obs = observations[ship_idx]
    
    # Skip if no observations for this ship
    if len(ship_obs) == 0:
        continue
    
    try:
        metrics = simulate_single_agent(ship_idx, ship_obs)
        
        # Store ship index and metrics
        ship_indices.append(ship_idx)
        all_metrics['gc_ade'].append(metrics['gc_ade'])
        all_metrics['goal_rate'].append(metrics['goal_rate'])
        all_metrics['near_miss_rate'].append(metrics['near_miss_rate'])
        all_metrics['avg_curvature'].append(metrics['avg_curvature'])
        
        # Print individual results
        print(f"Ship {ship_idx} | GC-ADE: {metrics['gc_ade']:.2f} m | "
              f"Goal Reached: {metrics['goal_rate']} | "
              f"Near Miss Rate: {metrics['near_miss_rate']:.3f}% | "
              f"Avg Curvature: {metrics['avg_curvature']:.5f}")
        
    except Exception as e:
        print(f"Error simulating ship {ship_idx}: {e}")
        continue

# Calculate and display average metrics
print("\n" + "="*50)
print("AVERAGE METRICS ACROSS ALL SHIPS:")
print("="*50)

if len(all_metrics['gc_ade']) > 0:
    avg_gc_ade = np.mean(all_metrics['gc_ade'])
    avg_goal_rate = np.mean(all_metrics['goal_rate']) * 100  # Convert to percentage
    avg_near_miss_rate = np.mean(all_metrics['near_miss_rate'])
    avg_curvature = np.mean(all_metrics['avg_curvature'])
    
    print(f"Average GC-ADE: {avg_gc_ade:.2f} m")
    print(f"Average Goal Rate: {avg_goal_rate:.1f}%")
    print(f"Average Near Miss Rate: {avg_near_miss_rate:.3f}%")
    print(f"Average Curvature: {avg_curvature:.5f}")
    
    print(f"\nTotal ships simulated: {len(all_metrics['gc_ade'])}")
    print(f"Ships that reached goal: {sum(all_metrics['goal_rate'])}")
    
    # Create CSV output
    print("\n" + "="*50)
    print("CREATING CSV OUTPUT...")
    print("="*50)
    
    # Create DataFrame with individual ship metrics
    csv_data = []
    for i in range(len(ship_indices)):
        csv_data.append({
            'Ship_Index': ship_indices[i],
            'GC_ADE': all_metrics['gc_ade'][i],
            'Goal_Reached': all_metrics['goal_rate'][i],
            'Near_Miss_Rate': all_metrics['near_miss_rate'][i],
            'Average_Curvature': all_metrics['avg_curvature'][i]
        })
    
    # Add aggregate row
    csv_data.append({
        'Ship_Index': 'AGGREGATE',
        'GC_ADE': avg_gc_ade,
        'Goal_Reached': avg_goal_rate,  # This is already in percentage
        'Near_Miss_Rate': avg_near_miss_rate,
        'Average_Curvature': avg_curvature
    })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    csv_filename = f'simulation_metrics_{NUM_SHIPS_TO_SIMULATE}_ships.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"CSV file saved as: {csv_filename}")
    print(f"Total rows in CSV: {len(df)} (including aggregate)")
    
    # Display first few rows and last row as preview
    print("\nCSV Preview:")
    print(df.head())
    print("...")
    print(df.tail(1))
    
else:
    print("No ships were successfully simulated.")
