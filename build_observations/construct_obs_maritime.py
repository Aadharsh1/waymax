import pickle
import sys
sys.path.append('./maritime_rl')
from maritime_rl.maritime_env import ShipEnvironment
import maritime_rl.utils 


with open('trajs_times/trajs_times_overlap.pkl', 'rb') as f:
    data = pickle.load(f)
trajs, times, overlap_idx = data['trajs'], data['times'], data['overlap_idx']

region_of_interest = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}

env = ShipEnvironment(
    ship_trajectories=trajs,
    ship_times=times,
    overlap_idx=overlap_idx,
    region_of_interest=region_of_interest,
    ego_pos=0, 
    observation_history_length=5,
    n_neighbor_agents=10,
    normalize_xy=False
)
expert_data = []  
for ego_idx in range(len(trajs)):
    obs, info = env.reset(options={'ego_pos': ego_idx})
    print(f'Construction obs for ship_idx: {ego_idx}')
    actions = info['actions']  

    for t in range(min(len(actions), 1000)):  
        expert_data.append({
            "obs": obs.copy(),
            "action": actions[t].copy(),
            "episode_id": ego_idx
        })

        obs, reward, terminated, truncated, info = env.step(actions[t])
        if terminated or truncated:
            break
with open('expert_obs.pkl', 'wb') as f:
    pickle.dump(expert_data, f)
print(f"Saved expert dataset with {len(expert_data)} transitions.")
