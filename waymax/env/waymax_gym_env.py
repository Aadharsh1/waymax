import gym
import numpy as np
from waymax.datatypes import Action
import jax.numpy as jnp

class WaymaxGymEnv(gym.Env):
    def __init__(self, waymax_env, initial_state, dt=10.0, goal_threshold=200, max_steps=1000):
        super().__init__()
        self.waymax_env = waymax_env
        self.initial_state = initial_state
        self.dt = dt
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps

        self.num_agents = initial_state.sim_trajectory.x.shape[0]
        self.n_neighbors = initial_state.sim_trajectory.neighbor_histories.shape[2]
        self.features_per_agent = initial_state.sim_trajectory.ego_histories.shape[3]

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_agents, 3), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            'ego': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_agents, self.features_per_agent), dtype=np.float32
            ),
            'neighbors': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_agents, self.n_neighbors, self.features_per_agent), dtype=np.float32
            ),
            'goal': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_agents, 2), dtype=np.float32
            ),
        })

        self.current_state = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        self.current_state = self.waymax_env.reset(self.initial_state)
        self.current_step = 0
        obs = self._extract_observation(self.current_state)
        return obs

    def step(self, action):
        # action: (num_agents, 3) [dx, dy, dyaw] for each agent
        timestep = int(self.current_state.timestep)
        x_current = self.current_state.sim_trajectory.x[:, timestep]
        y_current = self.current_state.sim_trajectory.y[:, timestep]
        yaw_current = self.current_state.sim_trajectory.yaw[:, timestep]

        dx = action[:, 0]
        dy = action[:, 1]
        dyaw = action[:, 2]

        x_new = x_current + dx
        y_new = y_current + dy
        yaw_new = (yaw_current + dyaw) % (2 * np.pi)
        vel_x = dx / self.dt
        vel_y = dy / self.dt

        # Build Waymax action
        data = np.stack([x_new, y_new, yaw_new, vel_x, vel_y], axis=-1)
        data = jnp.array(data, dtype=jnp.float32)
        valid = jnp.ones((self.num_agents, 1), dtype=bool)
        waymax_action = Action(data=data, valid=valid)

        self.current_state = self.waymax_env.step(self.current_state, waymax_action)
        self.current_step += 1

        obs = self._extract_observation(self.current_state)
        reward = self._compute_reward(self.current_state)
        terminated = self._check_terminated(self.current_state)
        truncated = self.current_step >= self.max_steps

        info = {}
        return obs, reward, terminated, truncated, info

    def _extract_observation(self, state):
        timestep = int(state.timestep)
        ego = np.stack([
            state.sim_trajectory.x[:, timestep],
            state.sim_trajectory.y[:, timestep],
            state.sim_trajectory.speed[:, timestep],
            state.sim_trajectory.yaw[:, timestep]
        ], axis=-1).astype(np.float32)

        neighbors = np.array([
            [
                neighbor[-1] if np.any(neighbor) else np.zeros(self.features_per_agent)
                for neighbor in state.sim_trajectory.neighbor_histories[i, timestep]
            ]
            for i in range(self.num_agents)
        ], dtype=np.float32)

        goal = np.array(state.sim_trajectory.goals, dtype=np.float32)
        return {
            'ego': ego,
            'neighbors': neighbors,
            'goal': goal,
        }

    def _check_terminated(self, state):
        timestep = int(state.timestep)
        x = state.sim_trajectory.x[:, timestep]
        y = state.sim_trajectory.y[:, timestep]
        goal = state.sim_trajectory.goals
        dist_to_goal = np.sqrt((x - goal[:, 0])**2 + (y - goal[:, 1])**2)
        return np.all(dist_to_goal < self.goal_threshold)

    def _compute_reward(self, state):
        timestep = int(state.timestep)
        x = state.sim_trajectory.x[:, timestep]
        y = state.sim_trajectory.y[:, timestep]
        goal = state.sim_trajectory.goals
        dist_to_goal = np.sqrt((x - goal[:, 0])**2 + (y - goal[:, 1])**2)
        reward = (dist_to_goal < self.goal_threshold).astype(np.float32)
        return reward
