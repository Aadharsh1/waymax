import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist

def load_observations():
    """Load both observation files"""
    with open('observations.pkl', 'rb') as f:
        obs_method1 = pickle.load(f)
    
    with open('observations_original.pkl', 'rb') as f:
        obs_method2 = pickle.load(f)
    
    return obs_method1, obs_method2

def compare_observations_comprehensive(obs1, obs2, num_ships_to_analyze=10):
    """Comprehensive comparison between two observation sets"""
    
    print("="*80)
    print("COMPREHENSIVE OBSERVATION COMPARISON")
    print("="*80)
    
    # Basic structure comparison
    print(f"\n📊 BASIC STRUCTURE COMPARISON")
    print(f"Method 1 (time_tolerance): {len(obs1)} ships")
    print(f"Method 2 (overlap_idx): {len(obs2)} ships")
    
    min_ships = min(len(obs1), len(obs2), num_ships_to_analyze)
    
    # Detailed comparison for selected ships
    differences = {
        'ego_differences': [],
        'neighbor_differences': [],
        'goal_differences': [],
        'neighbor_count_differences': [],
        'ship_indices': []
    }
    
    for ship_idx in range(min_ships):
        print(f"\n🚢 SHIP {ship_idx} ANALYSIS")
        print("-" * 50)
        
        ship_obs1 = obs1[ship_idx]
        ship_obs2 = obs2[ship_idx]
        
        min_timesteps = min(len(ship_obs1), len(ship_obs2))
        
        ego_diffs = []
        neighbor_diffs = []
        neighbor_count_diffs = []
        
        for t in range(min_timesteps):
            # Ego comparison
            ego1 = ship_obs1[t]['ego']
            ego2 = ship_obs2[t]['ego']
            ego_diff = np.mean(np.abs(ego1 - ego2))
            ego_diffs.append(ego_diff)
            
            # Neighbors comparison
            neighbors1 = ship_obs1[t]['neighbors']
            neighbors2 = ship_obs2[t]['neighbors']
            
            # Count non-zero neighbors
            non_zero_neighbors1 = np.sum(np.any(neighbors1.reshape(neighbors1.shape[0], -1) != 0, axis=1))
            non_zero_neighbors2 = np.sum(np.any(neighbors2.reshape(neighbors2.shape[0], -1) != 0, axis=1))
            neighbor_count_diffs.append(abs(non_zero_neighbors1 - non_zero_neighbors2))
            
            # Neighbor content difference
            neighbor_diff = np.mean(np.abs(neighbors1 - neighbors2))
            neighbor_diffs.append(neighbor_diff)
        
        # Goal comparison
        goal1 = ship_obs1[0]['goal']  # Goals should be same across timesteps
        goal2 = ship_obs2[0]['goal']
        goal_diff = np.linalg.norm(goal1 - goal2)
        
        # Store results
        differences['ego_differences'].append(np.mean(ego_diffs))
        differences['neighbor_differences'].append(np.mean(neighbor_diffs))
        differences['goal_differences'].append(goal_diff)
        differences['neighbor_count_differences'].append(np.mean(neighbor_count_diffs))
        differences['ship_indices'].append(ship_idx)
        
        # Print summary for this ship
        print(f"  Timesteps compared: {min_timesteps}")
        print(f"  Avg ego difference: {np.mean(ego_diffs):.6f}")
        print(f"  Avg neighbor difference: {np.mean(neighbor_diffs):.6f}")
        print(f"  Goal difference: {goal_diff:.6f}")
        print(f"  Avg neighbor count diff: {np.mean(neighbor_count_diffs):.2f}")
    
    return differences

def visualize_differences(differences, save_plots=True):
    """Create visualizations of the differences"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Observation Methods Comparison', fontsize=16, fontweight='bold')
    
    # Ego differences
    axes[0, 0].bar(differences['ship_indices'], differences['ego_differences'])
    axes[0, 0].set_title('Ego Trajectory Differences')
    axes[0, 0].set_xlabel('Ship Index')
    axes[0, 0].set_ylabel('Mean Absolute Difference')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Neighbor differences
    axes[0, 1].bar(differences['ship_indices'], differences['neighbor_differences'])
    axes[0, 1].set_title('Neighbor Trajectory Differences')
    axes[0, 1].set_xlabel('Ship Index')
    axes[0, 1].set_ylabel('Mean Absolute Difference')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Goal differences
    axes[1, 0].bar(differences['ship_indices'], differences['goal_differences'])
    axes[1, 0].set_title('Goal Position Differences')
    axes[1, 0].set_xlabel('Ship Index')
    axes[1, 0].set_ylabel('Euclidean Distance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Neighbor count differences
    axes[1, 1].bar(differences['ship_indices'], differences['neighbor_count_differences'])
    axes[1, 1].set_title('Neighbor Count Differences')
    axes[1, 1].set_xlabel('Ship Index')
    axes[1, 1].set_ylabel('Avg Count Difference')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('observation_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📈 Visualization saved as 'observation_comparison.png'")
    
    plt.show()

def detailed_neighbor_analysis(obs1, obs2, ship_idx=0, timestep=0):
    """Detailed analysis of neighbor differences for a specific ship and timestep"""
    
    print(f"\n🔍 DETAILED NEIGHBOR ANALYSIS - Ship {ship_idx}, Timestep {timestep}")
    print("="*70)
    
    neighbors1 = obs1[ship_idx][timestep]['neighbors']
    neighbors2 = obs2[ship_idx][timestep]['neighbors']
    
    print(f"Method 1 neighbors shape: {neighbors1.shape}")
    print(f"Method 2 neighbors shape: {neighbors2.shape}")
    
    # Count non-zero neighbors
    non_zero_1 = []
    non_zero_2 = []
    
    for i in range(neighbors1.shape[0]):
        if np.any(neighbors1[i] != 0):
            non_zero_1.append(i)
        if np.any(neighbors2[i] != 0):
            non_zero_2.append(i)
    
    print(f"\nMethod 1 - Non-zero neighbors: {len(non_zero_1)} at indices {non_zero_1}")
    print(f"Method 2 - Non-zero neighbors: {len(non_zero_2)} at indices {non_zero_2}")
    
    # Show actual neighbor values for non-zero neighbors
    print(f"\n📋 NEIGHBOR COMPARISON:")
    max_neighbors = max(len(non_zero_1), len(non_zero_2))
    
    for i in range(min(5, max_neighbors)):  # Show first 5 neighbors
        print(f"\nNeighbor {i}:")
        if i < len(non_zero_1):
            pos1 = neighbors1[non_zero_1[i], -1, :2]  # Current position
            print(f"  Method 1: pos=({pos1[0]:.1f}, {pos1[1]:.1f})")
        else:
            print(f"  Method 1: No neighbor")
            
        if i < len(non_zero_2):
            pos2 = neighbors2[non_zero_2[i], -1, :2]  # Current position
            print(f"  Method 2: pos=({pos2[0]:.1f}, {pos2[1]:.1f})")
        else:
            print(f"  Method 2: No neighbor")

def create_summary_report(differences):
    """Create a summary statistics report"""
    
    df = pd.DataFrame(differences)
    
    print(f"\n📊 SUMMARY STATISTICS REPORT")
    print("="*50)
    
    summary_stats = {
        'Metric': ['Ego Differences', 'Neighbor Differences', 'Goal Differences', 'Neighbor Count Differences'],
        'Mean': [
            np.mean(differences['ego_differences']),
            np.mean(differences['neighbor_differences']),
            np.mean(differences['goal_differences']),
            np.mean(differences['neighbor_count_differences'])
        ],
        'Std': [
            np.std(differences['ego_differences']),
            np.std(differences['neighbor_differences']),
            np.std(differences['goal_differences']),
            np.std(differences['neighbor_count_differences'])
        ],
        'Max': [
            np.max(differences['ego_differences']),
            np.max(differences['neighbor_differences']),
            np.max(differences['goal_differences']),
            np.max(differences['neighbor_count_differences'])
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.to_string(index=False, float_format='%.6f'))
    
    # Save to CSV
    df.to_csv('observation_comparison_detailed.csv', index=False)
    summary_df.to_csv('observation_comparison_summary.csv', index=False)
    print(f"\n💾 Detailed results saved to 'observation_comparison_detailed.csv'")
    print(f"💾 Summary results saved to 'observation_comparison_summary.csv'")

# Main execution
def main():
    # Load observations
    obs1, obs2 = load_observations()
    
    # Compare observations
    differences = compare_observations_comprehensive(obs1, obs2, num_ships_to_analyze=10)
    
    # Visualize differences
    visualize_differences(differences)
    
    # Detailed neighbor analysis for first ship
    detailed_neighbor_analysis(obs1, obs2, ship_idx=0, timestep=1)
    
    # Create summary report
    create_summary_report(differences)
    
    # Quick interpretation
    print(f"\n🎯 QUICK INTERPRETATION:")
    print(f"   • Ego trajectories should be identical (expect ~0 difference)")
    print(f"   • Goal positions should be identical (expect ~0 difference)")
    print(f"   • Neighbor differences show impact of different selection methods")
    print(f"   • Neighbor count differences show how many neighbors differ between methods")

if __name__ == "__main__":
    main()
