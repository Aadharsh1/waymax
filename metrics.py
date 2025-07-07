import pickle
import numpy as np

from shipnav import obs

# Load the original trajectory data for reference
with open('./trajs_times/trajs_times_tw.pkl', 'rb') as f:
    data = pickle.load(f)

trajs = data['trajs']
times = data['times']
overlap_idx = data['overlap_idx']

# Load the saved observations
with open('observations.pkl', 'rb') as f:
    observations = pickle.load(f)

num_ships = len(observations)
print(f"Number of ships in observations: {num_ships}")
print(f"Number of ships in original trajs: {len(trajs)}")


print(observations[0][0]['neighbours'])