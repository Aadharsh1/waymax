import pickle
TARGET_FPS = 10
import pandas as pd
import os
import csv
from datetime import datetime
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
from waymax import datatypes

def save_combined_results_to_csv(results, filename_prefix="simulation_combined"):
    """
    Save both individual and aggregate results to a single CSV file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write aggregate summary first
        writer.writerow(["AGGREGATE METRICS"])
        writer.writerow(["Metric", "Value"])
        agg = results['aggregate_metrics']
        writer.writerow(["Number_of_Ships", agg['num_ships']])
        writer.writerow(["Mean_GC_ADE", round(agg['mean_gc_ade'], 3)])
        writer.writerow(["Std_GC_ADE", round(agg['std_gc_ade'], 3)])
        writer.writerow(["Goal_Success_Rate_Percent", round(agg['goal_success_rate'], 1)])
        writer.writerow(["Mean_Near_Miss_Rate_Percent", round(agg['mean_near_miss_rate'], 3)])
        writer.writerow(["Mean_Curvature", round(agg['mean_curvature'], 6)])
        
        writer.writerow([])  # Empty row separator
        
        # Write individual results
        writer.writerow(["INDIVIDUAL SHIP RESULTS"])
        writer.writerow([
            "Ship_Index", "Actor_Type", "GC_ADE", "Goal_Reached", 
            "Near_Miss_Rate_Percent", "Avg_Curvature", "Dist_To_Goal", 
            "Final_X", "Final_Y", "Goal_X", "Goal_Y"
        ])
        
        for result in results['individual_results']:
            final_x, final_y = result['final_position']
            goal_x, goal_y = result['goal_position']
            
            writer.writerow([
                result['ship_index'], result['actor_type'],
                round(result['gc_ade'], 3), result['goal_reached'],
                round(result['near_miss_rate'], 3), round(result['avg_curvature'], 6),
                round(result['dist_to_goal'], 2), round(final_x, 2), round(final_y, 2),
                round(goal_x, 2), round(goal_y, 2)
            ])
    
    print(f"Combined results saved to: {filename}")
    return filename


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
    def __init__(self, dynamics_model, config, trajs, times, overlap_idx, original_ship_indices):  # Add overlap_idx
        super().__init__(dynamics_model, config)
        self.trajs = trajs
        self.times = times
        self.overlap_idx = overlap_idx  # Add this
        self.original_ship_indices = original_ship_indices 
        self.ego_start_times = [times[idx][0] for idx in original_ship_indices]
    
    def observe_agent(self, states_history, agent_idx, **kwargs):
        original_agent_idx = self.original_ship_indices[agent_idx]
        
        return super().observe_agent(
            states_history=states_history,
            agent_idx=original_agent_idx,  
            trajs=self.trajs,
            times=self.times,
            overlap_idx=self.overlap_idx,  # Add this
            ego_start_time=self.ego_start_times[agent_idx],
            **kwargs
        )


def simulate_single_ship(ship_index, actor_type='bc'):
    """
    Simulate a single ship and return metrics
    """
    print(f"\n=== SIMULATING SHIP {ship_index} ({actor_type.upper()}) ===")
    
    # Load data
    with open('./trajs_times/trajs_times_overlap.pkl', 'rb') as f:  
        data = pickle.load(f)

    trajs = data['trajs']
    times = data['times']
    overlap_idx = data['overlap_idx']  

    with open('observations_original.pkl', 'rb') as f:
        observations = pickle.load(f)

    region_of_interest = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}
    origin_lon, origin_lat = region_of_interest['LON'][0], region_of_interest['LAT'][0]
    max_x = haversine_distance(origin_lon, origin_lat, region_of_interest['LON'][1], origin_lat)
    max_y = haversine_distance(origin_lon, origin_lat, origin_lon, region_of_interest['LAT'][1])

    # Get single ship data
    ship_obs = observations[ship_index]
    max_length = len(ship_obs)
    
    # Build trajectory for single ship
    episode_x, episode_y, episode_speed, episode_heading, episode_valid = [], [], [], [], []
    ego_histories_this_episode = []
    neighbor_histories_this_episode = []
    episode_goal = None

    NUM_NEIGHBORS = 10  
    HISTORY_STEPS = 5

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

    # Create trajectory arrays for single ship
    x = jnp.array([episode_x])
    y = jnp.array([episode_y])
    speed = jnp.array([episode_speed])
    heading = jnp.array([episode_heading])
    valid = jnp.array([episode_valid])
    vel_x = speed * jnp.cos(heading)
    vel_y = speed * jnp.sin(heading)
    ego_histories_all = jnp.array([ego_histories_this_episode])
    neighbor_histories_all = jnp.array([neighbor_histories_this_episode])
    goal_positions = jnp.array([episode_goal])

    # Create timestamps
    timestamps_episode = np.arange(len(ship_obs)) * 10
    timestamps_episode_micro = (timestamps_episode * 1e6).astype(np.int64)
    timestamps = jnp.array([timestamps_episode_micro])

    # Create trajectory
    traj = Trajectory(
        x=x, y=y, speed=speed, yaw=heading, vel_x=vel_x, vel_y=vel_y, valid=valid,
        timestamp_micros=timestamps, ego_histories=ego_histories_all,
        neighbor_histories=neighbor_histories_all, goals=goal_positions
    )

    # Create metadata for single ship
    meta = ObjectMetadata(
        ids=jnp.array([0]),  # Single ship with ID 0
        object_types=jnp.zeros(1, dtype=jnp.int32),
        is_sdc=jnp.array([True]), is_modeled=jnp.array([True]),
        is_valid=jnp.array([True]), is_controlled=jnp.array([True]),
        objects_of_interest=jnp.array([False])
    )

    # Create simulation state
    sim_state = SimulatorState(
        sim_trajectory=traj, log_trajectory=traj,
        object_metadata=meta, timestep=jnp.array(0),
    )

    # ✅ FIXED: Create environment with correct parameters
    dynamics_model = CustomShipDynamics()
    env = CustomEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(
            EnvironmentConfig(),
            max_num_objects=1,  # ✅ Single ship
            controlled_object=ObjectType.VALID
        ),
        trajs=trajs,
        times=times,
        overlap_idx=overlap_idx,  
        original_ship_indices=[ship_index]  # ✅ Single ship index in list
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bc_model_path = "new_model_weights/bc_original_epoch300.th"
    print(f"bc_weights: {bc_model_path.split('/')[-1]}")
    
    # Create actors
    if actor_type == 'bc':
        actor = BCActor(
            model_path=bc_model_path, device=device, dynamics_model=dynamics_model,
            environment=env, normalize=False, max_x=max_x, max_y=max_y
        )
        jit_select_action = actor.select_action
    else:  # expert
        actor = agents.create_expert_actor(dynamics_model=CustomShipDynamics())
        jit_select_action = jax.jit(actor.select_action)

    # No-op action function
    def no_op_action_for_agent(agent_idx, num_objects):
        action_data = jnp.zeros((num_objects, 5), dtype=jnp.float32)
        valid = jnp.zeros((num_objects, 1), dtype=bool)
        is_ctrl = jnp.zeros((num_objects,), dtype=bool)
        return WaymaxActorOutput(
            actor_state=None, action=Action(data=action_data, valid=valid),
            is_controlled=is_ctrl
        )

    # Run simulation
    states = [sim_state]
    done = False
    jit_step = jax.jit(env.step)

    for step in range(max_length - 1):
        current_state = states[-1]
        
        if not done:
            output = jit_select_action({}, current_state, 0, states)
        else:
            output = no_op_action_for_agent(0, 1)

        action = agents.merge_actions([output])
        next_state = jit_step(current_state, action)
        states.append(next_state)
        
        # Check termination
        terminated = env.termination(next_state)
        truncated = env.truncation(next_state)
        done = terminated[0] or truncated[0]

        if done:
            break

    # Calculate metrics
    valid_mask = traj.valid[0]
    last_valid_idx = int(jnp.where(valid_mask)[0][-1])
    T = min(len(states), last_valid_idx + 1)
    
    # GC-ADE Calculation
    expert_x = traj.x[0]
    expert_y = traj.y[0]
    sim_x = jnp.array([state.sim_trajectory.x[0, int(state.timestep)] for state in states[:T]])
    sim_y = jnp.array([state.sim_trajectory.y[0, int(state.timestep)] for state in states[:T]])
    min_T = min(len(expert_x), len(sim_x))
    gc_ade = jnp.mean(jnp.sqrt((expert_x[:min_T] - sim_x[:min_T])**2 + (expert_y[:min_T] - sim_y[:min_T])**2))

    # Goal Reached Calculation
    final_x = sim_x[-1] if len(sim_x) > 0 else 0.0
    final_y = sim_y[-1] if len(sim_y) > 0 else 0.0
    goal_x, goal_y = goal_positions[0]
    dist_to_goal = jnp.sqrt((final_x - goal_x)**2 + (final_y - goal_y)**2)
    goal_reached = dist_to_goal < 200.0  # goal_threshold

    # Near Miss Rate Calculation
    near_miss_count = 0
    near_miss_threshold = 555
    
    for t in range(T):
        if t >= len(states):
            break
            
        obs = env.observe_agent(states[:t+1], 0)
        neighbors = obs['neighbors']
        
        timestep_idx = int(states[t].timestep)
        x_i = states[t].sim_trajectory.x[0, timestep_idx]
        y_i = states[t].sim_trajectory.y[0, timestep_idx]
        
        if x_i == 0.0 and y_i == 0.0:
            continue
            
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

    # Curvature calculation
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

    print(f"Ship {ship_index}: Final position = ({float(final_x):.2f}, {float(final_y):.2f})")
    print(f"Goal position = ({float(goal_x):.2f}, {float(goal_y):.2f})")
    print(f"Distance to goal = {float(dist_to_goal):.2f} m")

    return {
        'ship_index': ship_index,
        'actor_type': actor_type,
        'gc_ade': float(gc_ade),
        'goal_reached': bool(goal_reached),
        'near_miss_rate': float(near_miss_rate),
        'avg_curvature': float(avg_curvature),
        'dist_to_goal': float(dist_to_goal),
        'final_position': (float(final_x), float(final_y)),
        'goal_position': (float(goal_x), float(goal_y))
    }


def run_single_agent_simulations(ship_indices, actor_type='bc'):
    """
    Run single-agent simulations for multiple ships and aggregate results
    
    Args:
        ship_indices: List of ship indices to simulate
        actor_type: 'bc' or 'expert'
    
    Returns:
        dict: Individual and aggregated metrics
    """
    print(f"\n{'='*60}")
    print(f"RUNNING SINGLE-AGENT SIMULATIONS ({actor_type.upper()} ACTOR)")
    print(f"Ship indices: {ship_indices}")
    print(f"{'='*60}")
    
    individual_results = []
    
    for ship_idx in ship_indices:
        result = simulate_single_ship(ship_idx, actor_type)
        individual_results.append(result)
        
        print(f"Ship {ship_idx} | GC-ADE: {result['gc_ade']:.2f} m | "
              f"Goal Reached: {result['goal_reached']} | "
              f"Near Miss Rate: {result['near_miss_rate']:.3f}% | "
              f"Avg Curvature: {result['avg_curvature']:.5f}")
    
    # Calculate aggregate metrics
    gc_ades = [r['gc_ade'] for r in individual_results]
    goal_rates = [r['goal_reached'] for r in individual_results]
    near_miss_rates = [r['near_miss_rate'] for r in individual_results]
    curvatures = [r['avg_curvature'] for r in individual_results]
    
    aggregate_metrics = {
        'mean_gc_ade': np.mean(gc_ades),
        'std_gc_ade': np.std(gc_ades),
        'goal_success_rate': np.mean(goal_rates) * 100,
        'mean_near_miss_rate': np.mean(near_miss_rates),
        'mean_curvature': np.mean(curvatures),
        'num_ships': len(ship_indices)
    }
    
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({actor_type.upper()} ACTOR)")
    print(f"{'='*60}")
    print(f"Number of ships: {aggregate_metrics['num_ships']}")
    print(f"Mean GC-ADE: {aggregate_metrics['mean_gc_ade']:.2f} ± {aggregate_metrics['std_gc_ade']:.2f} m")
    print(f"Goal Success Rate: {aggregate_metrics['goal_success_rate']:.1f}%")
    print(f"Mean Near Miss Rate: {aggregate_metrics['mean_near_miss_rate']:.3f}%")
    print(f"Mean Curvature: {aggregate_metrics['mean_curvature']:.5f}")
    
    return {
        'individual_results': individual_results,
        'aggregate_metrics': aggregate_metrics
    }


# Example usage
if __name__ == "__main__":
    # Specify which ships to simulate
    SHIP_INDICES = list(range(10))
    
    # Run BC actor simulations
    bc_results = run_single_agent_simulations(SHIP_INDICES, actor_type='bc')
    final_csv = save_combined_results_to_csv(bc_results, "bc_simulation_10ships")

    
    # Run expert actor simulations
    # expert_results = run_single_agent_simulations(SHIP_INDICES, actor_type='expert')
    
    # Compare results
    # print(f"\n{'='*60}")
    # print("COMPARISON")
    # print(f"{'='*60}")
    # print(f"BC Actor     - Mean GC-ADE: {bc_results['aggregate_metrics']['mean_gc_ade']:.2f} m")
    # print(f"Expert Actor - Mean GC-ADE: {expert_results['aggregate_metrics']['mean_gc_ade']:.2f} m")
    # print(f"BC Actor     - Goal Success: {bc_results['aggregate_metrics']['goal_success_rate']:.1f}%")
    # print(f"Expert Actor - Goal Success: {expert_results['aggregate_metrics']['goal_success_rate']:.1f}%")
