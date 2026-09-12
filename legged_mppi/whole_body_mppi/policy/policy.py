from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

@dataclass
class PolicyBounds:
    # Full quadruped action bounds (12 DoF): [FL, FR, RL, RR] x [hip, thigh, calf]
    act_min: jnp.ndarray
    act_max: jnp.ndarray

    @classmethod
    def quadruped_default(cls) -> "PolicyBounds":
        return cls(
            act_min=jnp.array(
                [-0.863, -0.686, -2.818, -0.863, -0.686, -2.818,
                 -0.863, -0.686, -2.818, -0.863, -0.686, -2.818]
            ),
            act_max=jnp.array(
                [0.863, 4.501, -0.888, 0.863, 4.501, -0.888,
                 0.863, 4.501, -0.888, 0.863, 4.501, -0.888]
            ),
        )

class NeuralControlPolicy(nn.Module):
    """
    JAX/Flax equivalent of the PyTorch MLP policy.

    Input:
      inp: (B, in_dim)

    Output:
      u: (B, act_dim)
    """
    in_dim: int
    act_dim: int = 12
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    bounds: Optional[PolicyBounds] = None
    activation: str = "gelu"
    dropout: float = 0.0

    def setup(self):
        act_map = {
            "relu": nn.relu,
            "tanh": jnp.tanh,
            "gelu": nn.gelu,
            "silu": nn.silu,
        }
        if self.activation.lower() not in act_map:
            raise ValueError(f"Unsupported activation='{self.activation}'")
        self.act_fn = act_map[self.activation.lower()]

        self.hidden_layers = [
            nn.Dense(self.hidden_dim) for _ in range(self.num_hidden_layers)
        ]
        # Small init for final layer (like zero init in your PyTorch code)
        self.out = nn.Dense(
            self.act_dim,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )
        self.drop = nn.Dropout(rate=self.dropout)

    @staticmethod
    def _squash_to_bounds(x: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
        mid = 0.5 * (hi + lo)
        half = 0.5 * (hi - lo)
        return mid + half * jnp.tanh(x)

    def __call__(self, inp: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        x = inp
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.act_fn(x)
            if self.dropout > 0.0:
                x = self.drop(x, deterministic=not train)

        u_raw = self.out(x)

        if self.bounds is None:
            return u_raw

        return self._squash_to_bounds(
            u_raw,
            self.bounds.act_min,
            self.bounds.act_max,
        )
