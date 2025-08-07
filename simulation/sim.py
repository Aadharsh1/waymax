import pickle
import numpy as np
import torch as th
import sys
sys.path.append('./maritime_rl')
from maritime_rl import det_bc
from maritime_rl.maritime_env import ShipEnvironment
from maritime_rl.utils import haversine_distance

region_of_interest = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}
origin_lon, origin_lat = region_of_interest['LON'][0], region_of_interest['LAT'][0]
max_x = haversine_distance(origin_lon, origin_lat, region_of_interest['LON'][1], origin_lat)
max_y = haversine_distance(origin_lon, origin_lat, origin_lon, region_of_interest['LAT'][1])

with open('./trajs_times/trajs_times_overlap.pkl', 'rb') as f:
    data = pickle.load(f)
trajs = data['trajs']
times = data['times']
overlap_idx = data['overlap_idx']

model_path = "new_model_weights/bc_expert_epoch300.th"
device = th.device("cuda" if th.cuda.is_available() else "cpu")


class StandaloneBCActor:

    def __init__(self, model_path, device, normalize=False, max_x=None, max_y=None):
        self.device = device
        self.normalize = normalize
        self.max_x = max_x
        self.max_y = max_y
        self.policy = th.load(model_path, map_location=self.device, weights_only=False)
        self.policy.eval()

        if self.normalize and (max_x is None or max_y is None):
            raise ValueError("max_x and max_y must be provided when normalize=True")

    def predict(self, obs_dict, deterministic=True):

        obs_tensor = {}
        for k, v in obs_dict.items():
            if isinstance(v, np.ndarray):
                obs_tensor[k] = th.tensor(v, dtype=th.float32, device=self.device)
            else:
                obs_tensor[k] = v

        with th.no_grad():
            action = self.policy.predict(obs_tensor, deterministic=deterministic)[0]

        if isinstance(action, np.ndarray):
            return action
        return action.cpu().numpy()


def create_ship_environment(ego_ship_idx, normalize=False):
    env = ShipEnvironment(
        ship_trajectories=trajs,
        ship_times=times,
        overlap_idx=overlap_idx,
        region_of_interest=region_of_interest,
        ego_pos=ego_ship_idx,
        observation_history_length=5,
        n_neighbor_agents=10,
        normalize_xy=normalize,
        max_steps=1000,
        second_perts=10,
        drop_neighbor=False,
        use_dis_fea=False,
        use_FoR=False,
    )
    return env


def simulate_ship(ego_ship_idx, normalize=False):

    print(f"\n--- Simulating Ship {ego_ship_idx} ---")
    bc_actor = StandaloneBCActor(
        model_path=model_path,
        device=device,
        max_x=max_x,
        max_y=max_y,
    )
    env = create_ship_environment(ego_ship_idx, normalize=normalize)

    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    max_steps = 1000

    while step_count < max_steps:
        action = bc_actor.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if step_count == 0:
            print(obs)

        current_pos = env.current_state
        goal_pos = info.get('goal_position', None)

        if goal_pos is not None:
            distance_to_goal = np.linalg.norm(
                [current_pos['x'] - goal_pos[0], current_pos['y'] - goal_pos[1]]
            )
        else:
            distance_to_goal = float('nan')

        total_reward += reward
        step_count += 1

        if terminated or truncated:
            break

    reach_goal = info.get('reach_goal', False)
    avg_speed = info.get('avg_speed', float('nan'))
    avg_curvature = info.get('avg_curvature', float('nan'))
    nearmiss_rate = info.get('nearmiss_rate', float('nan'))
    gc_ade = info.get('gc_ade', float('nan'))
    mae_steer = info.get('mae_steer', float('nan'))
    mae_accel = info.get('mae_accel', float('nan'))
    curv_change_rate = info.get('curv_change_rate', float('nan'))
    goal_pos = info.get('goal_position', None)

    print(f"Ship {ego_ship_idx} final position: ({current_pos['x']:.1f}, {current_pos['y']:.1f})")
    print(f"Ship {ego_ship_idx} goal position: ({goal_pos})")
    print(f"Ship {ego_ship_idx} distance to goal: {distance_to_goal:.1f} meters")
    print(f"Ship {ego_ship_idx} goal reached: {reach_goal}")
    print(f"Ship {ego_ship_idx} GC-ADE: {gc_ade}")
    print(f"Ship {ego_ship_idx} Avg curvature: {avg_curvature}")
    print(f"Ship {ego_ship_idx} Near miss rate: {nearmiss_rate:.2f}%")
    # print(f"Ship {ego_ship_idx} MAE steering: {mae_steer}")
    # print(f"Ship {ego_ship_idx} MAE accel: {mae_accel}")
    # print(f"Ship {ego_ship_idx} Curvature change rate: {curv_change_rate:.2f}%")

    return {
        'ship_idx': ego_ship_idx,
        'final_pos': (current_pos['x'], current_pos['y']),
        'distance_to_goal': distance_to_goal,
        'reach_goal': reach_goal,
        'gc_ade': gc_ade,
        'avg_curvature': avg_curvature,
        'nearmiss_rate': nearmiss_rate,
        'mae_steer': mae_steer,
        'mae_accel': mae_accel,
        'curv_change_rate': curv_change_rate,
        'avg_speed': avg_speed,
        'total_reward': total_reward,
        'steps': step_count
    }


def simulate_multiple_ships(ship_indices, normalize=False):
    all_metrics = []

    for idx in ship_indices:
        metrics = simulate_ship(idx, normalize=normalize)
        all_metrics.append(metrics)

    def safe_mean(key):
        vals = [m[key] for m in all_metrics if m.get(key) is not None and not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else float('nan')

    total_simulated = len(all_metrics)
    print("\n=== Aggregate Metrics over all ships ===")
    print(f"Ships simulated: {total_simulated}")
    print(f"Average final distance to goal: {safe_mean('distance_to_goal'):.2f} m")
    print(f"Goal reach rate: {100 * np.mean([m['reach_goal'] for m in all_metrics]):.1f}%")
    print(f"Average GC-ADE: {safe_mean('gc_ade'):.2f}")
    print(f"Average curvature: {safe_mean('avg_curvature'):.5f}")
    print(f"Average near miss rate: {safe_mean('nearmiss_rate'):.2f}%")
    print(f"Average MAE steering: {safe_mean('mae_steer'):.5f}")
    print(f"Average MAE acceleration: {safe_mean('mae_accel'):.5f}")
    print(f"Average curvature change rate: {safe_mean('curv_change_rate'):.2f}%")
    print(f"Average speed: {safe_mean('avg_speed'):.2f}")
    print(f"Average total reward: {safe_mean('total_reward'):.2f}")
    print(f"Average steps taken: {safe_mean('steps'):.2f}")

    return all_metrics


if __name__ == "__main__":
    ship_indices = [0] 
    normalize = False  

    print("Standalone BC Multi-Ship Simulation")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print(f"Region: {region_of_interest}")
    print(f"Max dimensions: ({max_x:.1f} m, {max_y:.1f} m)")
    print(f"Simulating ship indices: {ship_indices}\n")

    simulate_multiple_ships(ship_indices, normalize=normalize)
