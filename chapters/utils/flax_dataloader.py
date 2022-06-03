import jax.numpy as jnp
import jax.random as random

def data_loader(rng, x, y, batch_size, shuffle=False):
    steps_per_epoch = len(x) // batch_size

    if shuffle:
        batch_idx = random.permutation(rng, len(x))
    else:
        batch_idx = jnp.arange(len(x))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        yield x[idx], y[idx] 