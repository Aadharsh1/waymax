import torch
import sys
import numpy as np 
import jax.numpy as jnp
from waymax.datatypes import Action
from waymax.agents.actor_core import WaymaxActorOutput
from maritime_rl import det_bc
sys.modules['det_bc'] = det_bc

class BCActor:
    def __init__(self, model_path, device, dynamics_model, environment, normalize=False, 
                 max_x=None, max_y=None):
        self.device = device
        self.environment = environment
        self.normalize = normalize
        self.max_x = max_x
        self.max_y = max_y
        self.policy = torch.load(model_path, map_location=self.device, weights_only=False)
        self.policy.eval()
        
        self.dynamics_model = dynamics_model
        
        if self.normalize and (max_x is None or max_y is None):
            raise ValueError("max_x and max_y must be provided when normalize=True")

    def _normalize_observation(self, obs):
        """Apply normalization to observation if enabled"""
        if not self.normalize:
            return obs
            
        normalized_obs = obs.copy()
        
        ego = normalized_obs['ego'].copy()
        ego[:, 0] = (ego[:, 0] / self.max_x) * 2 - 1  
        ego[:, 1] = (ego[:, 1] / self.max_y) * 2 - 1  
        normalized_obs['ego'] = ego
        
        neighbors = normalized_obs['neighbors'].copy()
        neighbors[:, :, 0] = (neighbors[:, :, 0] / self.max_x) * 2 - 1
        neighbors[:, :, 1] = (neighbors[:, :, 1] / self.max_y) * 2 - 1
        normalized_obs['neighbors'] = neighbors
        
        goal = normalized_obs['goal'].copy()
        goal[0] = (goal[0] / self.max_x) * 2 - 1
        goal[1] = (goal[1] / self.max_y) * 2 - 1
        normalized_obs['goal'] = goal
        
        return normalized_obs

    def select_action(self, rng_key, state, agent_idx, states_history: list):
        obs = self.environment.observe_agent(
            states_history=states_history,
            agent_idx=agent_idx,
            n_neighbors=10, 
            history_len=5
        )

        # DEBUG: Print raw observation before normalization
        # timestep = int(state.timestep)
        # if timestep == 1:
        #     print(agent_idx)
        #     print("Raw inference observation:")
        #     print("ego :", obs['ego'])
        #     print("neighbor values:\n", obs['neighbors'])
        #     print("Goal:", obs['goal'])
        #     print("-" * 40)

        if self.normalize:
            obs = self._normalize_observation(obs)
        
        obs_dict = {
            'ego': obs['ego'].astype(np.float32),
            'neighbors': obs['neighbors'].astype(np.float32), 
            'goal': obs['goal'].astype(np.float32)
        }
        
        with torch.no_grad():
            action_np = self.policy.predict(obs_dict, deterministic=True)[0]

        dx, dy, dheading = action_np
        # print(dx, dy, dheading)
        
        # Get current state
        timestep = int(state.timestep)
        x_current = state.sim_trajectory.x[agent_idx, timestep]
        y_current = state.sim_trajectory.y[agent_idx, timestep]
        yaw_current = state.sim_trajectory.yaw[agent_idx, timestep]
        
        # Apply relative displacements
        x_new = x_current + dx
        y_new = y_current + dy
        yaw_new = (yaw_current + dheading) % (2 * np.pi)
        
        
        # Calculate velocities
        dt = 10.0
        vel_x = dx / dt
        vel_y = dy / dt
        
        num_objects = state.sim_trajectory.x.shape[0]
       
        action_x = jnp.expand_dims(state.sim_trajectory.x[:, timestep], axis=1)
        action_y = jnp.expand_dims(state.sim_trajectory.y[:, timestep], axis=1)
        action_yaw = jnp.expand_dims(state.sim_trajectory.yaw[:, timestep], axis=1)
        action_vel_x = jnp.expand_dims(state.sim_trajectory.vel_x[:, timestep], axis=1)
        action_vel_y = jnp.expand_dims(state.sim_trajectory.vel_y[:, timestep], axis=1)
        action_valid = jnp.expand_dims(state.sim_trajectory.valid[:, timestep], axis=1)
        
        # Update only the controlled agent
        action_x = action_x.at[agent_idx, 0].set(x_new)
        action_y = action_y.at[agent_idx, 0].set(y_new)
        action_yaw = action_yaw.at[agent_idx, 0].set(yaw_new)
        action_vel_x = action_vel_x.at[agent_idx, 0].set(vel_x)
        action_vel_y = action_vel_y.at[agent_idx, 0].set(vel_y)
        action_valid = action_valid.at[agent_idx, 0].set(True)
        
        # Create proper TrajectoryUpdate action
        from waymax.datatypes import TrajectoryUpdate
        action = TrajectoryUpdate(
            x=action_x,
            y=action_y,
            yaw=action_yaw,
            vel_x=action_vel_x,
            vel_y=action_vel_y,
            valid=action_valid,
        ).as_action()
        
        # Create is_controlled mask
        is_ctrl = jnp.zeros((num_objects,), dtype=bool).at[agent_idx].set(True)
        
        return WaymaxActorOutput(
            actor_state=None,
            action=action,
            is_controlled=is_ctrl
        )

