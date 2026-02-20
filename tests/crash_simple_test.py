"""Simple crash-plugin smoke test (no crashes).

Used to validate that per-file JSON/HTML logs and the merged report work.
"""

def test_crash_simple_pass_1():
    assert 1 + 1 == 2

def test_crash_simple_pass_2():
    assert "jax".upper() == "JAX"

def test_crash_simple_pass_3():
    assert True
