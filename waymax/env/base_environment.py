# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core class definitions for MultiAgentEnvironment.

This environment is designed to work with multiple objects (autonomous driving
vehicle and other objects).
"""
import chex
from dm_env import specs
import jax
from jax import numpy as jnp
from waymax import config as _config
from waymax import datatypes
from waymax import dynamics as _dynamics
from waymax import metrics
from waymax import rewards
from waymax.env import abstract_environment
from waymax.env import typedefs as types
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from maritime_rl.utils import haversine_distance


class BaseEnvironment(abstract_environment.AbstractEnvironment):
  """Waymax environment for multi-agent scenarios."""

  def __init__(
      self,
      dynamics_model: _dynamics.DynamicsModel,
      config: _config.EnvironmentConfig,
  ):
    """Constructs a Waymax environment.

    Args:
      dynamics_model: Dynamics model to use which transitions the simulator
        state to the next timestep given an action.
      config: Waymax environment configs.
    """
    self._dynamics_model = dynamics_model
    self._reward_function = rewards.LinearCombinationReward(config.rewards)
    self.config = config

  @property
  def dynamics(self) -> _dynamics.DynamicsModel:
    return self._dynamics_model

  @jax.named_scope('BaseEnvironment.metrics')
  def metrics(self, state: datatypes.SimulatorState) -> types.Metrics:
    """Computes metrics (lower is better) from state."""
    # TODO(b/254483042) Make metric_dict a dataclasses.
    return metrics.run_metrics(
        simulator_state=state, metrics_config=self.config.metrics
    )
  
  def termination(self, state):
    t = int(state.timestep)
    valid_agents = state.sim_trajectory.valid[:, t]
    
    x = state.sim_trajectory.x[:, t]
    y = state.sim_trajectory.y[:, t]
    goals = state.sim_trajectory.goals  
    goal_x = goals[:, 0]
    goal_y = goals[:, 1]
    
    dist = jnp.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
    goal_threshold = 200.0
    terminated = (dist < goal_threshold) & valid_agents
    
    if jnp.any(terminated):
        terminated_agents = jnp.where(terminated)[0]
        print(f"Terminated at timestep {t} for agents: {terminated_agents}")
    
    return terminated
 

  def truncation(self, state):
    t = int(state.timestep)
    max_length = 1000
    
    region_of_interest = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}
    origin_lon, origin_lat = region_of_interest['LON'][0], region_of_interest['LAT'][0]
    max_x = haversine_distance(origin_lon, origin_lat, region_of_interest['LON'][1], origin_lat)
    max_y = haversine_distance(origin_lon, origin_lat, origin_lon, region_of_interest['LAT'][1])
    
    current_x = state.sim_trajectory.x[:, t]
    current_y = state.sim_trajectory.y[:, t]
    valid_agents = state.sim_trajectory.valid[:, t]
    
    min_x, min_y = 0.0, 0.0 
    within_x_bounds = (current_x >= min_x) & (current_x <= max_x)
    within_y_bounds = (current_y >= min_y) & (current_y <= max_y)
    within_region = within_x_bounds & within_y_bounds
    max_length_exceeded = t >= (max_length - 1)
    truncated_per_agent = ((~within_region) | max_length_exceeded) & valid_agents
    
    if jnp.any(truncated_per_agent):
        if max_length_exceeded:
            print(f"Truncation reached at timestep {t} (max length)")
        if jnp.any((~within_region) & valid_agents):
            out_of_bounds_agents = jnp.where((~within_region) & valid_agents)[0]
            print(f"Truncation: agents {out_of_bounds_agents} left region at timestep {t}")
    
    return truncated_per_agent



  def reset(
      self, state: datatypes.SimulatorState, rng: jax.Array | None = None
  ) -> datatypes.SimulatorState:
    """Initializes the simulation state.

    This initializer sets the initial timestep and fills the initial simulation
    trajectory with invalid values.

    Args:
      state: An uninitialized state of shape (...).
      rng: Optional random number generator for stochastic environments.

    Returns:
      The initialized simulation state of shape (...).
    """
    chex.assert_equal(
        self.config.max_num_objects, state.log_trajectory.num_objects
    )

    # Fills with invalid values (i.e. -1.) and False.
    sim_traj_uninitialized = datatypes.fill_invalid_trajectory(
        state.log_trajectory
    )
    state_uninitialized = state.replace(
        timestep=jnp.array(-1), sim_trajectory=sim_traj_uninitialized
    )
    return datatypes.update_state_by_log(
        state_uninitialized, self.config.init_steps
    )

  def observe(self, state: datatypes.SimulatorState) -> types.Observation:
    """Computes the observation for the given simulation state.

    Here we assume that the default observation is just the simulator state. We
    leave this for the user to override in order to provide a user-specific
    observation function. A user can use this to move some of their model
    specific post-processing into the environment rollout in the actor nodes. If
    they want this post-processing on the accelertor, they can keep this the
    same and implement it on the learner side. We provide some helper functions
    at datatypes.observation.py to help write your own observation functions.

    Args:
      state: Current state of the simulator of shape (...).

    Returns:
      Simulator state as an observation without modifications of shape (...).
    """
    return state

  @jax.named_scope('BaseEnvironment.step')
  def step(
      self,
      state: datatypes.SimulatorState,
      action: datatypes.Action,
      rng: jax.Array | None = None,
  ) -> datatypes.SimulatorState:
    """Advances simulation by one timestep using the dynamics model.

    Args:
      state: The current state of the simulator of shape (...).
      action: The action to apply, of shape (..., num_objects). The
        actions.valid field is used to denote which objects are being controlled
        - objects whose valid is False will fallback to default behavior
        specified by self.dynamics.
      rng: Optional random number generator for stochastic environments.

    Returns:
      The next simulation state after taking an action of shape (...).
    """
    is_controlled = _get_control_mask(state, self.config)
    new_traj = self.dynamics.forward(  # pytype: disable=wrong-arg-types  # jax-ndarray
        action=action,
        trajectory=state.sim_trajectory,
        reference_trajectory=state.log_trajectory,
        is_controlled=is_controlled,
        timestep=state.timestep,
        allow_object_injection=self.config.allow_new_objects_after_warmup,
    )
    return state.replace(sim_trajectory=new_traj, timestep=state.timestep + 1)

  @jax.named_scope('BaseEnvironment.reward')
  def reward(
      self, state: datatypes.SimulatorState, action: datatypes.Action
  ) -> jax.Array:
    """Computes the reward for a transition.

    Args:
      state: The state used to compute the reward at state.timestep of shape
        (...).
      action: The action applied to state of shape (..., num_objects, dim).

    Returns:
      An array of rewards of shape (..., num_objects).
    """
    if self.config.compute_reward:
      agent_mask = datatypes.get_control_mask(
          state.object_metadata, self.config.controlled_object
      )
      return self._reward_function.compute(state, action, agent_mask)
    else:
      reward_spec = _multi_agent_reward_spec(self.config)
      return jnp.zeros(state.shape + reward_spec.shape, dtype=reward_spec.dtype)

  def action_spec(self) -> datatypes.Action:
    # Dynamics model class defines specs for a single agent.
    # Need to expand it to multiple objects.
    single_agent_spec = self.dynamics.action_spec()  # rank 1
    data_spec = specs.BoundedArray(
        shape=(self.config.max_num_objects,) + single_agent_spec.shape,
        dtype=single_agent_spec.dtype,
        minimum=jnp.tile(
            single_agent_spec.minimum[jnp.newaxis, :],
            [self.config.max_num_objects, 1],
        ),
        maximum=jnp.tile(
            single_agent_spec.maximum[jnp.newaxis, :],
            [self.config.max_num_objects, 1],
        ),
    )
    valid_spec = specs.Array(
        shape=(self.config.max_num_objects, 1), dtype=jnp.bool_
    )
    return datatypes.Action(data=data_spec, valid=valid_spec)  # pytype: disable=wrong-arg-types  # jax-ndarray

  def reward_spec(self) -> specs.Array:
    return _multi_agent_reward_spec(self.config)

  def discount_spec(self) -> specs.BoundedArray:
    return specs.BoundedArray(
        shape=tuple(), minimum=0.0, maximum=1.0, dtype=jnp.float32
    )

  def observation_spec(self) -> types.Observation:
    raise NotImplementedError()

  def observe_agent(self,
                  states_history: list,
                  agent_idx: int,
                  history_len: int = 10,
                  n_neighbors: int = 10):
    """
    Dynamically builds a gym-compatible observation for an agent by replicating
    the logic from build_observations.py using the simulated state history.
    """
    current_state = states_history[-1]
    current_timestep = int(current_state.timestep)
    
    ego_history = np.zeros((history_len + 1, 4), dtype=np.float32)
    for i in range(history_len + 1):
        hist_step_idx = current_timestep - (history_len - i)
        
        if hist_step_idx >= 0 and hist_step_idx < len(states_history):
            past_state = states_history[hist_step_idx]
            ego_history[i, 0] = past_state.sim_trajectory.x[agent_idx, hist_step_idx]
            ego_history[i, 1] = past_state.sim_trajectory.y[agent_idx, hist_step_idx]
            ego_history[i, 2] = past_state.sim_trajectory.speed[agent_idx, hist_step_idx]
            ego_history[i, 3] = past_state.sim_trajectory.yaw[agent_idx, hist_step_idx]
            
    ego_pos = np.array([
        current_state.sim_trajectory.x[agent_idx, current_timestep],
        current_state.sim_trajectory.y[agent_idx, current_timestep]
    ])
    
    neighbor_candidates = []
    num_total_agents = current_state.sim_trajectory.x.shape[0]
    
    for other_idx in range(num_total_agents):
        if other_idx == agent_idx:
            continue
        
        if not current_state.sim_trajectory.valid[other_idx, current_timestep]:
            continue

        other_pos = np.array([
            current_state.sim_trajectory.x[other_idx, current_timestep],
            current_state.sim_trajectory.y[other_idx, current_timestep]
        ])
        
        distance = np.linalg.norm(ego_pos - other_pos)
        neighbor_candidates.append({'id': other_idx, 'dist': distance})
        
    neighbor_candidates.sort(key=lambda x: x['dist'])
    nearest_neighbor_idxs = [n['id'] for n in neighbor_candidates[:n_neighbors]]
    
    all_neighbors_history = []
    for neighbor_idx in nearest_neighbor_idxs:
        neighbor_history = np.zeros((history_len + 1, 4), dtype=np.float32)
        for i in range(history_len + 1):
            hist_step_idx = current_timestep - (history_len - i)
            if hist_step_idx >= 0 and hist_step_idx < len(states_history):
                past_state = states_history[hist_step_idx]
                if past_state.sim_trajectory.valid[neighbor_idx, hist_step_idx]:
                    neighbor_history[i, 0] = past_state.sim_trajectory.x[neighbor_idx, hist_step_idx]
                    neighbor_history[i, 1] = past_state.sim_trajectory.y[neighbor_idx, hist_step_idx]
                    neighbor_history[i, 2] = past_state.sim_trajectory.speed[neighbor_idx, hist_step_idx]
                    neighbor_history[i, 3] = past_state.sim_trajectory.yaw[neighbor_idx, hist_step_idx]
        all_neighbors_history.append(neighbor_history)
        
    while len(all_neighbors_history) < n_neighbors:
        all_neighbors_history.append(np.zeros((history_len + 1, 4), dtype=np.float32))
    goal = np.array(current_state.sim_trajectory.goals[agent_idx])
    # if current_timestep in [0,1,2]:
    #   print(f'ego history: {ego_history}')
    return {
        'ego': ego_history,
        'neighbors': np.stack(all_neighbors_history, axis=0),
        'goal': goal,
    }

  @property
  def observation_space(self):
      return spaces.Dict({
          'ego': spaces.Box(low=-np.inf, high=np.inf, shape=(11, 4), dtype=np.float32),
          'neighbors': spaces.Box(low=-np.inf, high=np.inf, shape=(10, 11, 4), dtype=np.float32),
          'goal': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
      })


def _get_control_mask(
    state: datatypes.SimulatorState, config: _config.EnvironmentConfig
) -> jax.Array:
  """Gets the control mask for a multi-agent environment."""
  if (
      config.controlled_object == _config.ObjectType.VALID
      and not config.allow_new_objects_after_warmup
  ):
    return datatypes.dynamic_index(
        state.sim_trajectory.valid,
        index=config.init_steps - 1,
        axis=-1,
        keepdims=False,
    )
  else:
    return datatypes.get_control_mask(
        state.object_metadata, config.controlled_object
    )


def _multi_agent_reward_spec(
    config: _config.EnvironmentConfig,
) -> specs.Array:
  """Gets the reward spec for a multi-agent environment."""
  return specs.Array(shape=(config.max_num_objects,), dtype=jnp.float32)


# Add MultiAgentEnvironment as an alias for BaseEnvironment, since
# BaseEnvironment already supports executing multiple agents.
MultiAgentEnvironment = BaseEnvironment
