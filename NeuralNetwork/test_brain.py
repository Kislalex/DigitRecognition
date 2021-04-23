from brain import *
import numpy as np
import pytest


Eps = 0.0001


def test_linear_neuron():
    ln = LinearNeuron(2, 3)
    # 2 3      0
    # 4 5      1
    # 6 7      2
    #      +
    matrix = np.arange(6).reshape((3, 2)) + 2
    column = np.arange(3)
    ln.set_neuron(
        (
            matrix,
            column,
        )
    )
    # 1 2
    inputs = np.arange(2) + 1
    outputs = ln.compute(inputs)
    #
    correct_output = np.array([8, 15, 22])
    np.testing.assert_array_equal(outputs, correct_output)


def test_convolution_neuron():
    cn = ConvolutionNeuron(3, 2)

    #  1  2
    #  3  4
    #  5  6
    matrix = np.arange(6).reshape((3, 2)) + 1
    cn.set_neuron(matrix)

    #  0  1  2  3  4  5
    #  6  7  8  9 10 11
    # 12 13 14 15 16 17 ....
    # ...
    inputs = np.arange(36).reshape(6, 6)
    outputs = cn.compute(inputs)

    correct_output = np.array([[186, 228, 270], [564, 606, 648]])
    np.testing.assert_array_equal(outputs, correct_output)


def test_strided_neuron():
    sn = StridedNeuron(3, 2)

    #  1  2
    #  3  4
    #  5  6
    matrix = np.arange(6).reshape((3, 2)) + 1
    sn.set_neuron(matrix)

    #  0  1  2  3
    #  4  5  6  7
    #  8  9 10 11
    # 12 13 14 15
    inputs = np.arange(16).reshape(4, 4)

    outputs = sn.compute(inputs)

    correct_output = np.array([[128, 149, 170], [212, 233, 254]])
    np.testing.assert_array_equal(outputs, correct_output)


def test_linear_neuron_gradient():
    ln = LinearNeuron(2, 3)
    # 1  3      0
    # 5  7      1
    # 9 11      2
    matrix = 2 * np.arange(6).reshape((3, 2)) + 1
    column = np.arange(3)
    ln.set_neuron(
        (
            matrix,
            column,
        )
    )
    # 1 2
    inputs = np.arange(2) + 1
    # 7
    # 20
    gradient = np.array([7, 20, 33])
    new_gradient, df = ln.compute_gradient(gradient, inputs)
    #
    correct_new_gradient = np.array([404, 524])
    np.testing.assert_array_equal(new_gradient, correct_new_gradient)
    correct_matrix_gradient = np.array([7, 14, 20, 40, 33, 66]).reshape((3, 2))
    np.testing.assert_array_equal(df.matrix, correct_matrix_gradient)


def test_convolution_neuron_gradient():
    cn = ConvolutionNeuron(2, 2)
    #  1  1
    # -1 -1
    matrix = np.array([1, 1, -1, -1]).reshape((2, 2))
    cn.set_neuron(matrix)
    # 1 2 1 1
    # 3 4 1 1
    # 0 0 4 2
    # 0 0 1 3
    inputs = np.array([1, 2, 1, 1, 3, 4, 1, 1, 0, 0, 4, 2, 0, 0, 1, 3]).reshape(4, 4)

    outputs = cn.compute(inputs)
    # -4 0
    #  0 2
    correct_output = np.array([-4, 0, 0, 2]).reshape(2, 2)
    np.testing.assert_array_equal(correct_output, outputs)
    #
    gradient = correct_output
    new_gradient, df = cn.compute_gradient(gradient, inputs)
    #
    correct_matrix_gradient = np.array([4, -4, -10, -10]).reshape((2, 2))
    np.testing.assert_array_equal(df.matrix, correct_matrix_gradient)
    #  -4  -8 0 0
    # -12 -16 0 0
    #   0   0 8 4
    #   0   0 2 6
    correct_new_gradient = np.array(
        [-4, -4, 0, 0, 4, 4, 0, 0, 0, 0, 2, 2, 0, 0, -2, -2]
    ).reshape(4, -1)
    np.testing.assert_array_equal(new_gradient, correct_new_gradient)


def test_strided_neuron_gradient():
    sn = StridedNeuron(2, 2)
    #  1  1
    # -1 -1
    matrix = np.array([1, 1, -1, -1]).reshape((2, 2))
    sn.set_neuron(matrix)
    #  1  2  3
    #  4  5  6
    #  7  8  9
    # 10 11 12
    inputs = np.arange(12).reshape(4, 3) + 1

    outputs = sn.compute(inputs)
    # -6 -6
    # -6 -6
    # -6 -6
    correct_output = np.array([-6, -6, -6, -6, -6, -6]).reshape(3, 2)
    np.testing.assert_array_equal(correct_output, outputs)
    #
    gradient = correct_output
    new_gradient, df = sn.compute_gradient(gradient, inputs)
    #
    correct_matrix_gradient = np.array([-6 * 27, -6 * 33, -6 * 45, -6 * 51]).reshape(
        (2, 2)
    )
    np.testing.assert_array_equal(df.matrix, correct_matrix_gradient)
    #  -4  -8 0 0
    # -12 -16 0 0
    #   0   0 8 4
    #   0   0 2 6
    correct_new_gradient = np.array([-6, -12, -6, 0, 0, 0, 0, 0, 0, 6, 12, 6]).reshape(
        4, -1
    )
    np.testing.assert_array_equal(new_gradient, correct_new_gradient)


def test_brain():
    br = Brain()
    #        1
    #    2   1
    #        1
    #        1
    # x  2   1     0
    #        1
    #        1
    #    2   1
    #        1
    br.add_layer(2, 3, 1, 3, (2, 2))
    br.add_layer(1, 9, 1, 3, (2, 2))
    br.add_layer(0, 1, 9, 1, (36, 3))
    # 5x5 -> 3 * 4x4 -> 9 2x2 -> 10
    inputs = np.arange(25).reshape((5, 5))
    outputs = br.compute(inputs)
    # 0
    correct_output = [np.zeros(3)]
    np.testing.assert_array_equal(correct_output, outputs)
    #
    gradient = np.zeros(3)
    new_gradient, df = br.compute_gradient(gradient, inputs)
    correct_gradient = [np.zeros(25).reshape((5, 5))]
    np.testing.assert_array_equal(new_gradient, correct_gradient)


def test_simple_gradient_actually_works():
    br = Brain()
    br.add_layer(0, 1, 1, 1, (1, 1))
    # create a basic brain f(x) = th(a + b * x)
    inputs = np.array([0.5])
    goal_value = 0.75
    y = np.array([0])
    for i in range(100):
        # get the output
        y = br.compute(inputs)
        # compute the gradients
        gradient = np.array([y[0][0] - goal_value])
        _, df = br.compute_gradient(gradient, inputs)
        br = br + (-1) * df
        # br.apply_gradient(-1)
    assert len(y) == 1
    assert y[0][0] == pytest.approx(goal_value, Eps)


def test_multidim_gradient_actually_works():
    br = Brain()
    # 3 -> 10 -> 10 -> 3
    br.add_layer(0, 1, 1, 1, (3, 10))
    br.add_layer(0, 1, 1, 1, (10, 10))
    br.add_layer(0, 1, 1, 1, (10, 3))
    # create a basic brain f(x) = th(a + b * x)
    inputs = np.array([0.512312, 0.1, 0.2])
    goal_value = np.array([0.7523123, 0.2, 0.3])
    y = np.array([0])
    for i in range(100):
        # get the output
        y = br.compute(inputs)
        # compute the gradients
        gradient = y[0] - goal_value
        _, df = br.compute_gradient(gradient, inputs)
        br = br + (-1) * df
        # br.apply_gradient(-1)
    assert len(y[0]) == len(goal_value)
    np.testing.assert_allclose(y[0], goal_value, Eps)


def test_io_works():
    br = Brain()
    # 3 -> 10 -> 10 -> 3
    br.add_layer(1, 1, 1, 1, (2, 2))
    br.add_layer(2, 2, 1, 2, (3, 3))
    br.add_layer(0, 1, 2, 1, (2, 1), False)
    br.mutate(1.0, 1.0)
    s = "test_brain.txt"
    f = open(s, "wb")
    br.write_to_stream(f)
    f.close()

    f = open(s, "rb")
    br2 = Brain()
    br2.read_from_stream(f)
    inputs = np.arange(36).reshape(6, 6)
    outputs = br.compute(inputs)[0]
    print("One")
    outputs2 = br2.compute(inputs)[0]
    np.testing.assert_allclose(outputs, outputs2, Eps)
