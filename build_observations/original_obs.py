import pickle
import numpy as np
from tqdm import tqdm

def get_padded_history(traj, t, history_length=5):
    start = max(0, t - history_length)
    history = traj[start:t+1]
    pad_len = history_length + 1 - history.shape[0]
    if pad_len > 0:
        padding = np.zeros((pad_len, 4), dtype=traj.dtype)
        history = np.vstack([padding, history])
    return history

def find_temporal_neighbors_original(all_trajs, all_times, overlap_idx, 
                                   current_ship_idx, t, N=10):
    if t >= len(all_trajs[current_ship_idx]) or t >= len(all_times[current_ship_idx]):
        return [-1] * N
    
    ego_time = all_times[current_ship_idx][t]
    ego_pos = all_trajs[current_ship_idx][t, :2]
    
    neighbor_candidates = []
    for ship_idx in overlap_idx[current_ship_idx]:
        if ship_idx == current_ship_idx:
            continue
        
        traj = all_trajs[ship_idx]
        ship_times = all_times[ship_idx]
        time_diffs = np.abs(ship_times - ego_time)
        min_time_diff_idx = np.argmin(time_diffs)
        
        ship_pos = traj[min_time_diff_idx, :2]
        distance = np.linalg.norm(ego_pos - ship_pos)
        
        neighbor_candidates.append((ship_idx, min_time_diff_idx, distance))
    neighbor_candidates.sort(key=lambda x: x[2])
    nearest_neighbors = [ship_idx for ship_idx, _, _ in neighbor_candidates[:N]]
    
    while len(nearest_neighbors) < N:
        nearest_neighbors.append(-1)
    
    return nearest_neighbors

def build_observations_original(trajs, times, overlap_idx, N=10, history_length=5):
    num_ships = len(trajs)
    observations = []
    
    goals = [traj[-1, :2] for traj in trajs]
    
    print(f"Building observations for {num_ships} ships using original method...")
    
    for ship_idx in tqdm(range(num_ships)):
        ship_obs = []
        traj = trajs[ship_idx]
        ship_times = times[ship_idx]
        T = len(traj)
        
        for t in range(T):
            ego = get_padded_history(traj, t, history_length)
            neighbor_idxs = find_temporal_neighbors_original(
                trajs, times, overlap_idx, ship_idx, t, N
            )
            
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

with open('trajs_times/trajs_times_overlap.pkl', 'rb') as f:
    data = pickle.load(f)

trajs = data['trajs']
times = data['times']
overlap_idx = data['overlap_idx']

observations = build_observations_original(
    trajs, times, overlap_idx, 
    N=1,  
    history_length=5 
)

filename = 'original_observations'

with open(f'observations/{filename}', 'wb') as f:
    pickle.dump(observations, f)

print(f"Total ships: {len(observations)}")


