import pickle 
import numpy as np
from tqdm import tqdm

with open('./trajs_times/trajs_times_tw.pkl', 'rb') as f:
    data = pickle.load(f)
    
trajs = data['trajs']
times = data['times']

def get_padded_history(traj, t, history_length=10):
    start = max(0, t - history_length)
    history = traj[start:t+1]
    pad_len = history_length + 1 - history.shape[0]
    if pad_len > 0:
        padding = np.zeros((pad_len, 4), dtype=traj.dtype)
        history = np.vstack([padding, history])
    return history

def find_temporal_neighbors(all_trajs, all_times, current_ship_idx, t, N, time_tolerance=6):
    if t >= len(all_trajs[current_ship_idx]) or t >= len(all_times[current_ship_idx]):
        return [-1] * N
    
    ego_time = all_times[current_ship_idx][t]
    ego_pos = all_trajs[current_ship_idx][t, :2]
    
    neighbor_candidates = []
    
    for ship_idx, (traj, ship_times) in enumerate(zip(all_trajs, all_times)):
        if ship_idx == current_ship_idx:
            continue
        
        # Find closest time match
        time_diffs = np.abs(ship_times - ego_time)
        min_time_diff_idx = np.argmin(time_diffs)
        min_time_diff = time_diffs[min_time_diff_idx]
        
        if min_time_diff <= time_tolerance:
            ship_pos = traj[min_time_diff_idx, :2]
            distance = np.linalg.norm(ego_pos - ship_pos)
            
            neighbor_candidates.append((ship_idx, min_time_diff_idx, distance, min_time_diff))
    
    neighbor_candidates.sort(key=lambda x: x[2])
    
    nearest_neighbors = [ship_idx for ship_idx, _, _, _ in neighbor_candidates[:N]]
    
    while len(nearest_neighbors) < N:
        nearest_neighbors.append(-1)
    
    return nearest_neighbors

def build_observations(trajs, times, N=10, history_length=10, time_tolerance=6):
    num_ships = len(trajs)
    observations = []
    
    goals = [traj[-1, :2] for traj in trajs]
    
    print(f"Building observations for {num_ships} ships...")
    print(f"Using time tolerance: {time_tolerance}s for 10s timestep intervals")
    
    for ship_idx in range(num_ships):
        ship_obs = []
        traj = trajs[ship_idx]
        ship_times = times[ship_idx]
        T = len(traj)
        
        for t in range(T):
            ego = get_padded_history(traj, t, history_length)
            neighbor_idxs = find_temporal_neighbors(trajs, times, ship_idx, t, N, time_tolerance)
            
            # Enhanced debug output
            if ship_idx < 2 and t < 3:
                current_time = ship_times[t]
                valid_neighbors = [idx for idx in neighbor_idxs if idx != -1]
                if valid_neighbors:
                    neighbor_info = []
                    ego_pos = traj[t, :2]
                    for n_idx in valid_neighbors:
                        neighbor_times = times[n_idx]
                        time_diffs = np.abs(neighbor_times - current_time)
                        best_t = np.argmin(time_diffs)
                        neighbor_pos = trajs[n_idx][best_t, :2]
                        dist = np.linalg.norm(ego_pos - neighbor_pos)
                        time_diff = time_diffs[best_t]
                        neighbor_info.append(f"Ship{n_idx}({dist:.0f}m,Δt={time_diff:.0f}s)")
                    print(f"Ship {ship_idx}, t={t} (time={current_time}): {neighbor_info}")
                else:
                    print(f"Ship {ship_idx}, t={t} (time={current_time}): No neighbors within {time_tolerance}s")
            
            # Build neighbor histories
            neighbors = []
            ego_time = ship_times[t]
            
            for n_idx in neighbor_idxs:
                if n_idx == -1:
                    neighbors.append(np.zeros((history_length+1, 4), dtype=traj.dtype))
                else:
                    neighbor_times = times[n_idx]
                    time_diffs = np.abs(neighbor_times - ego_time)
                    best_neighbor_t = np.argmin(time_diffs)
                    neighbors.append(get_padded_history(trajs[n_idx], best_neighbor_t, history_length))
            
            neighbors = np.stack(neighbors, axis=0)
            
            obs = {
                "ego": ego,
                "neighbors": neighbors,
                "goal": goals[ship_idx],
            }
            ship_obs.append(obs)
        observations.append(ship_obs)
    
    return observations

observations = build_observations(trajs, times, N=10, history_length=10, time_tolerance=6)

# Save observations
with open('observations.pkl', 'wb') as f:
    pickle.dump(observations, f)

# print(f"\nSuccessfully built observations!")
# print(f"Total ships: {len(observations)}")


# print(observations[1][0]['ego'])
# print('')
# print(observations[1][0]['neighbors'])
# print('')
# print(observations[1][0]['goal'])
# print('')
# print(trajs[1])
# print('')
# print(times[1])