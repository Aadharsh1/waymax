from collections.abc import Sequence
from typing import Any
import chex
import jax
import jax.numpy as jnp

from waymax.datatypes import operations

_INVALID_FLOAT_VALUE = -1.0
_INVALID_INT_VALUE = -1


@chex.dataclass
class ShipTrajectory:
    """Trajectory format for ship navigation simulation."""

    x: jnp.ndarray
    y: jnp.ndarray
    speed: jnp.ndarray
    yaw: jnp.ndarray
    vel_x: jnp.ndarray
    vel_y: jnp.ndarray
    valid: jnp.ndarray
    timestamp_micros: jnp.ndarray
    ego_histories: jnp.ndarray        
    neighbor_histories: jnp.ndarray   
    goals: jnp.ndarray                 

    @property
    def shape(self) -> tuple[int, ...]:
        return self.x.shape

    @property
    def num_objects(self) -> int:
        return self.shape[-2]

    @property
    def num_timesteps(self) -> int:
        return self.shape[-1]

    @property
    def xy(self) -> jnp.ndarray:
        return jnp.stack([self.x, self.y], axis=-1)

    @property
    def vel_xy(self) -> jnp.ndarray:
        return jnp.stack([self.vel_x, self.vel_y], axis=-1)

    @property
    def computed_speed(self) -> jnp.ndarray:
        speed = jnp.linalg.norm(self.vel_xy, axis=-1)
        return jnp.where(self.valid, speed, _INVALID_FLOAT_VALUE)


    @property
    def vel_yaw(self) -> jnp.ndarray:
        vel_yaw = jnp.arctan2(self.vel_y, self.vel_x)
        return jnp.where(self.valid, vel_yaw, _INVALID_FLOAT_VALUE)

    def __eq__(self, other: Any) -> bool:
        return operations.compare_all_leaf_nodes(self, other)

    def stack_fields(self, field_names: Sequence[str]) -> jnp.ndarray:
        return jnp.stack([getattr(self, f) for f in field_names], axis=-1)

    @classmethod
    def zeros(cls, shape: Sequence[int]) -> "ShipTrajectory":
        """Creates an empty zeroed-out trajectory for the given shape (N, T)."""
        return cls(
            x=jnp.zeros(shape, jnp.float32),
            y=jnp.zeros(shape, jnp.float32),
            speed=jnp.zeros(shape, jnp.float32),
            heading=jnp.zeros(shape, jnp.float32),
            vel_x=jnp.zeros(shape, jnp.float32),
            vel_y=jnp.zeros(shape, jnp.float32),
            valid=jnp.zeros(shape, jnp.bool_),
            timestamp_micros=jnp.zeros(shape, jnp.int64),
            ego_histories=jnp.zeros((shape[0], 11, 4), jnp.float32),
            neighbor_histories=jnp.zeros((shape[0], 10, 11, 4), jnp.float32),
            goals=jnp.zeros((shape[0], 2), jnp.float32),
        )

    def validate(self):
        """Validates shape and types."""
        chex.assert_equal_shape([
            self.x, self.y, self.speed, self.heading,
            self.vel_x, self.vel_y, self.valid, self.timestamp_micros
        ])
        chex.assert_type([
            self.x, self.y, self.speed, self.heading,
            self.vel_x, self.vel_y
        ], jnp.float32)
        chex.assert_type(self.valid, jnp.bool_)
        chex.assert_type(self.timestamp_micros, jnp.int64)
        chex.assert_type(self.ego_histories, jnp.float32)
        chex.assert_type(self.neighbor_histories, jnp.float32)
        chex.assert_type(self.goals, jnp.float32)
