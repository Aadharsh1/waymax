import pickle

def extract_observations(input_file, output_file, num_observations=1):
    """
    Extract a specific number of observations from the observations.pkl file
    
    Args:
        input_file (str): Path to the input observations.pkl file
        output_file (str): Path to save the extracted observations
        num_observations (int): Number of observations to extract (default: 1)
    """
    
    # Load the original observations
    with open(input_file, 'rb') as f:
        observations = pickle.load(f)
    
    print(f"Original observations contain {len(observations)} ships")
    
    # Extract the specified number of observations
    extracted_observations = observations[:num_observations]
    
    # Save the extracted observations
    with open(output_file, 'wb') as f:
        pickle.dump(extracted_observations, f)
    
    print(f"Successfully extracted {len(extracted_observations)} observation(s)")
    print(f"Saved to: {output_file}")
    
    # Print some info about the extracted data
    for i, ship_obs in enumerate(extracted_observations):
        print(f"Ship {i}: {len(ship_obs)} timesteps")
        print(f"  - Ego shape: {ship_obs[0]['ego'].shape}")
        print(f"  - Neighbors shape: {ship_obs[0]['neighbors'].shape}")
        print(f"  - Goal: {ship_obs[0]['goal']}")
    
    return extracted_observations

# Usage examples:

# Extract only the first observation (ship 0)
extract_observations('observations.pkl', 'observations_50.pkl', num_observations=50)

# Extract first 5 observations (ships 0-4)
# extract_observations('observations.pkl', 'observations_5.pkl', num_observations=5)

# Extract first 10 observations (ships 0-9)
# extract_observations('observations.pkl', 'observations_10.pkl', num_observations=10)
