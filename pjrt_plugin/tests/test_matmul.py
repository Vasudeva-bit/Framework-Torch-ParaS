import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "paras")

import jax
import jax.numpy as jnp
import numpy as np


def test_devices():
    devs = jax.devices()
    assert len(devs) >= 1, f"Expected at least 1 paras device, got {devs}"
    assert any(d.platform == "paras" for d in devs), f"No paras device in {devs}"
    print(f"[PASS] jax.devices() = {devs}")


def test_matmul_cases():
    matmul_jit = jax.jit(lambda a, b: a @ b)
    rng = np.random.default_rng(42)

    cases = [
        ("square_64", (64, 64), (64, 64), False),
        ("rect_128_256_64", (128, 256), (256, 64), False),
        ("trans_rhs_128_256_64", (128, 256), (64, 256), True),
    ]

    for name, shape_a, shape_b, transpose_b in cases:
        a_np = rng.standard_normal(shape_a, dtype=np.float32)
        b_np = rng.standard_normal(shape_b, dtype=np.float32)
        expected = a_np @ (b_np.T if transpose_b else b_np)

        a_jax = jnp.asarray(a_np)
        b_jax = jnp.asarray(b_np)

        for it in range(3):
            if transpose_b:
                out = jax.jit(lambda x, y: x @ y.T)(a_jax, b_jax)
            else:
                out = matmul_jit(a_jax, b_jax)
            out_np = np.asarray(out)
            np.testing.assert_allclose(out_np, expected, rtol=1e-4, atol=1e-4)
        print(f"[PASS] {name} (3 iterations)")


if __name__ == "__main__":
    test_devices()
    test_matmul_cases()
    print("All pjrt_paras v0.1 matmul tests passed!")
