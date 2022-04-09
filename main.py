import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

def yellow_constant(point, phi, theta):
    return jnp.array([1.0, 1.0, 1.0, 0.0])

def volume_rendering(rng, origin, phi, theta, t_n, t_f, N, query_f):
    ts = random.uniform(
                          rng, 
                          (N,), 
                          minval=jnp.array([t_n+((i-1)/N)*(t_f-t_n)  for i in range(1, N+1)]), 
                          maxval=jnp.array([t_n+((i)/N)*(t_f-t_n)  for i in range(1, N+1)]), 
                        )
    direction = jnp.array([jnp.cos(phi), jnp.sin(phi), jnp.cos(theta)])
    direction = direction / jnp.linalg.norm(direction)
    points = origin + direction * ts[:, None]
    outputs = vmap(query_f)(points, jnp.array(phi)[None].repeat(N), jnp.array(theta)[None].repeat(N))
    densities, colors = outputs[:, 0], outputs[:, 1:]
    deltas = ts[1:] - ts[:-1]
    deltas = jnp.concatenate((deltas, jnp.array([t_f-ts[-1]]),))
    delta_densities = densities*deltas
    cum_delta_densities = jnp.cumsum(delta_densities)
    T = jnp.exp(-cum_delta_densities+delta_densities)
    c = ((T * (1 - jnp.exp(-delta_densities)))[:, None] * colors).sum(axis=0)
    return c

if __name__ == "__main__":
    rng = random.PRNGKey(0)
    print(volume_rendering(rng, jnp.array([0,0,0]), 0.0, jnp.pi*0.4, 0, 1, 100, yellow_constant))
