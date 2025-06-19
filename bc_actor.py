import torch
import torch.nn as nn
from waymax.datatypes import Action
import numpy as np 
import jax.numpy as jnp
from waymax.agents.actor_core import WaymaxActorOutput

class BCPolicy(nn.Module):
    def __init__(self, input_dim=486, hidden_dim=256, output_dim=3):
        super(BCPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class BCActor:
    def __init__(self, model_path, device, dynamics_model):
        self.device = device
        self.model = BCPolicy().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.dynamics_model = dynamics_model

    def select_action(self, rng_key, state, agent_idx, _):
        timestep = int(state.timestep)

        x_current = state.sim_trajectory.x[agent_idx, timestep]
        y_current = state.sim_trajectory.y[agent_idx, timestep]
        yaw_current = state.sim_trajectory.yaw[agent_idx, timestep]

        ego = state.sim_trajectory.ego_histories[agent_idx][timestep]
        neighbors = state.sim_trajectory.neighbor_histories[agent_idx][timestep]
        goal = state.sim_trajectory.goals[agent_idx]

        ego_flat = ego.flatten()
        neighbors_flat = neighbors.flatten()
        goal_flat = goal.flatten()

        obs = jnp.concatenate([ego_flat, neighbors_flat, goal_flat])
        obs_np = np.array(obs)

        with torch.no_grad():
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_tensor = self.model(obs_tensor)
            action_np = action_tensor.cpu().numpy().squeeze()

        dx, dy, dyaw = action_np

        dt = 10.0 

        x_new = x_current + dx
        y_new = y_current + dy
        yaw_new = (yaw_current + dyaw) % (2 * np.pi)

        vel_x = dx / dt
        vel_y = dy / dt

        # # Debug
        # print(f"[DEBUG] timestep {timestep} | dx: {dx}, dy: {dy}, dyaw: {dyaw}")
        # print(f"[DEBUG] x_new: {x_new}, y_new: {y_new}, yaw_new: {yaw_new}")
        # print(f"[DEBUG] vel_x: {vel_x}, vel_y: {vel_y}")

        num_objects = state.sim_trajectory.x.shape[0]

        action_data = jnp.zeros((num_objects, 5), dtype=jnp.float32).at[agent_idx].set(
            jnp.array([x_new, y_new, yaw_new, vel_x, vel_y], dtype=jnp.float32)
        )

        valid = jnp.zeros((num_objects, 1), dtype=bool).at[agent_idx, 0].set(True)
        is_ctrl = jnp.zeros((num_objects,), dtype=bool).at[agent_idx].set(True)

        return WaymaxActorOutput(
            actor_state=None,
            action=Action(data=action_data, valid=valid),
            is_controlled=is_ctrl
        )