from brain import *
import numpy as np
from learning_brain import *
import pytest


Eps = 0.1


def test_mp_sum():
    x = range(1001)
    y = 500 * 1001
    z = mp_sum(x)
    np.testing.assert_equal(y, z)


def test_sin_gradient():
    br = Brain()
    # 1 -> 20 -> 20 -> 3
    br.add_layer(0, 1, 1, 1, (1, 30))
    br.add_layer(0, 1, 1, 1, (30, 30))
    br.add_layer(0, 1, 1, 1, (30, 1), False)
    br.mutate(1, 1)
    # create a basic brain f(x) = th(a + b * x)
    N = 10
    inputs = []
    outputs = []
    for i in range(N):
        x = (i + 1) * (3.1 / N)
        inputs.append(np.array([x]))
        outputs.append(np.array([np.sin(x)]))
    br = learning_brain_by_g_d(br, list(zip(inputs, outputs)), 10)
    br_outputs = []  # br.compute(inputs)[0]
    for x in inputs:
        br_outputs.append(br.compute(x)[0])

    np.testing.assert_allclose(br_outputs, outputs, Eps)
