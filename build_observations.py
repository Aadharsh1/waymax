import pickle 
import numpy as np

import pickle
import numpy as np

with open('./trajs_times/trajs_times_tw.pkl', 'rb') as f:
    data = pickle.load(f)

trajs = data['trajs']
times = data['times']
overlap_idx = data['overlap_idx']



def get_padded_history(traj, t, history_length=10):
    """
    Returns the last (history_length+1) steps ending at t, zero-padded if needed.
    traj: (T, 4) array
    t: current timestep (int)
    Returns: (history_length+1, 4) array
    """
    start = max(0, t - history_length)
    history = traj[start:t+1]
    # Pad at the front if needed
    pad_len = history_length + 1 - history.shape[0]
    if pad_len > 0:
        padding = np.zeros((pad_len, 4), dtype=traj.dtype)
        history = np.vstack([padding, history])
    return history

def find_nearest_neighbors(all_trajs, all_times, current_ship_idx, t, N, overlap_idx):
    """
    For ship at index current_ship_idx at timestep t, find N nearest other ships.
    Uses overlap_idx to filter ships with temporal overlap.
    Returns: list of neighbor indices (length N, padded with -1 if fewer available)
    """
    num_ships = len(all_trajs)
    ego_pos = all_trajs[current_ship_idx][t, :2]  # (x, y)
    neighbor_indices = []
    dists = []
    # Get list of ships that overlap temporally with current_ship_idx
    overlapping_ships = overlap_idx[current_ship_idx] if current_ship_idx < len(overlap_idx) else []
    
    for idx in overlapping_ships:
        if idx == current_ship_idx:
            continue
        # Only consider ships with index within the current sliced trajs list
        if idx >= 0 and idx < num_ships and t < len(all_trajs[idx]):
            pos = all_trajs[idx][t, :2]
            dist = np.linalg.norm(ego_pos - pos)
            neighbor_indices.append(idx)
            dists.append(dist)
    # Sort by distance
    sorted_neighbors = [x for _, x in sorted(zip(dists, neighbor_indices))]
    # Pad with -1 if fewer than N
    while len(sorted_neighbors) < N:
        sorted_neighbors.append(-1)
    return sorted_neighbors[:N]


def build_observations(trajs, times, overlap_idx, N=10, history_length=10):
    """
    trajs: list of (T_i, 4) arrays
    times: list of (T_i,) arrays
    overlap_idx: list of lists indicating temporally overlapping ships
    Returns: list of list of dicts: obs[ship_idx][timestep] = {ego, neighbors, goal}
    """
    num_ships = len(trajs)
    observations = []

    # Precompute goals for each ship
    goals = [traj[-1, :2] for traj in trajs]

    for ship_idx in range(num_ships):
        ship_obs = []
        traj = trajs[ship_idx]
        T = len(traj)
        for t in range(T):
            # Ego history
            ego = get_padded_history(traj, t, history_length)  # (11, 4)

            # Neighbors
            neighbor_idxs = find_nearest_neighbors(trajs, times, ship_idx, t, N, overlap_idx)
            if ship_idx == 0:  # Only for Ship 0
                valid_neighbors = [idx for idx in neighbor_idxs if idx != -1]
                if valid_neighbors:
                    print(f"Timestep {t}: Found neighbors {valid_neighbors}")
                else:
                    print(f"Timestep {t}: No valid neighbors")
            neighbors = []
            for n_idx in neighbor_idxs:
                if n_idx == -1 or t >= len(trajs[n_idx]):
                    neighbors.append(np.zeros((history_length+1, 4), dtype=traj.dtype))
                else:
                    neighbors.append(get_padded_history(trajs[n_idx], t, history_length))
            neighbors = np.stack(neighbors, axis=0)  # (N, 11, 4)

            # Goal
            goal = goals[ship_idx]  # (2,)

            obs = {
                "ego": ego,
                "neighbors": neighbors,
                "goal": goal
            }
            ship_obs.append(obs)
        observations.append(ship_obs)
    return observations

observations = build_observations(trajs, times, overlap_idx, N=10, history_length=10)

with open('observations.pkl', 'wb') as f:
    pickle.dump(observations, f)


# print(f'trajs: {trajs[0]}\n')

# print(f"len(observations): {len(observations)}\n")

# print(f"ego: {observations[0][40]['ego']}\n")
# print(f"neighbors: {observations[0][20]['neighbors']}\n")
# print(f"goal: {observations[0][0]['goal']}\n")




# print(overlap_idx)