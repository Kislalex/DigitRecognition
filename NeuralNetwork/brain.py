import sys
import random
import math
import numpy as np
import struct


class Neuron:
    def input_count(self):
        pass

    def output_count(self):
        pass

    def copy(self):
        pass

    def set_neuron(self, source):
        pass

    def compute_gradient(self, external_gradient, inputs):
        pass

    def apply_gradient(self, scale):
        pass

    def compute(self, inputs):
        pass

    def mutate(self, bound):
        pass

    def read_from_stream(self, stream):
        pass

    def write_to_stream(self, stream):
        pass


class LinearNeuron(Neuron):
    def __init__(self, inputs, outputs):
        self.N = inputs
        self.M = outputs
        self.matrix = np.zeros((self.M, self.N))
        self.column = np.zeros(self.M)

    def input_count(self):
        return self.N

    def output_count(self):
        return self.M

    def __add__(self, x):
        assert self.N == x.N and self.M == x.M, "Trying to add different sized matrices"
        new_neuron = LinearNeuron(self.N, self.M)
        new_neuron.matrix = self.matrix + x.matrix
        new_neuron.column = self.column + x.column
        return new_neuron

    def __rmul__(self, x):
        new_neuron = LinearNeuron(self.N, self.M)
        new_neuron.matrix = x * self.matrix
        new_neuron.column = x * self.column
        return new_neuron

    def compute_gradient(self, external_gradient, inputs):
        current_shape = inputs.shape
        df = LinearNeuron(self.N, self.M)
        inputs = inputs.reshape(-1)
        assert inputs.shape[0] == self.N, (
            "The input size is wrong :" + str(inputs.shape) + " neq " + str(self.N)
        )
        assert (
            len(external_gradient.shape) == 1
        ), "The gradient is not 1-dimensional " + str(inputs.shape)
        assert external_gradient.shape[0] == self.M, (
            "The gradient size is wrong :"
            + str(external_gradient.shape)
            + " neq "
            + str(self.M)
        )
        # x_1 = a_11 y_1 + a_12 y_2 + b_1
        # x_2 = a_21 y_1 + a_22 y_2 + b_2
        # dx_i/db_j = \delta_ij => dfb_i = dfx_i
        df.column += external_gradient

        # dx/da:
        # dx_1/da_11 = y_1 ,  dx_1/da_12 = y_2
        # dx_2/da_21 = y_1 ,  dx_2/da_22 = y_2
        #
        #
        # dfa_11 = dfx_1 * y_1,  dfa_12 = dfx_1 * y_2,
        # dfa_21 = dfx_2 * y_1,  dfa_22 = dfx_2 * y_2,
        #
        #                             (y_1,
        # dfa = (dfx_1, dfx_2) \tensor y_2)
        df.matrix += np.kron(external_gradient, inputs).reshape((self.M, self.N))
        #
        # dx_i/ dy_j = a_ij
        #
        # dfy_i = sum dfx_j a_ji
        #
        # dfy_1 =                     (a_11, a_12)
        # dfy_2 = (dfx_1, dfx_2) \dot (a_21, a_22)
        new_gradient = np.dot(external_gradient, self.matrix)
        return new_gradient.reshape(current_shape), df

    def compute(self, inputs):
        current_shape = inputs.shape
        inputs = inputs.reshape(-1)
        assert inputs.shape[0] == self.N, (
            "The input size is wrong :" + str(inputs.shape) + " neq " + str(self.N)
        )
        # Compute MATRIX times data
        outputs = np.dot(self.matrix, inputs)
        # Shift by column
        outputs = np.add(outputs, self.column)
        return outputs

    def copy(self):
        new_neuron = LinearNeuron(self.N, self.M)
        new_neuron.matrix = self.matrix.copy()
        new_neuron.column = self.column.copy()
        return new_neuron

    def set_neuron(self, source):
        self.matrix = source[0]
        self.column = source[1]
        self.N = self.matrix.shape[1]
        self.M = self.matrix.shape[0]

    def mutate(self, bound, chance):
        with np.nditer(self.matrix, op_flags=["readwrite"]) as matrix:
            for xij in matrix:
                coin_flip = np.random.random()
                if coin_flip < chance:
                    xij[...] += 2 * bound * (np.random.random() - 0.5)
        with np.nditer(self.column, op_flags=["readwrite"]) as column:
            for xi in column:
                coin_flip = np.random.random()
                if coin_flip < chance:
                    xi[...] += 2 * bound * (np.random.random() - 0.5)

    def read_from_stream(self, stream):
        self.M = struct.unpack("i", stream.read(4))[0]
        self.N = struct.unpack("i", stream.read(4))[0]
        self.matrix = np.fromfile(stream, dtype=float, count=self.M * self.N).reshape(
            (self.M, self.N)
        )
        self.column = np.fromfile(stream, dtype=float, count=self.M)

    def write_to_stream(self, stream):
        stream.write(struct.pack("i", self.M))
        stream.write(struct.pack("i", self.N))
        self.matrix.ravel().tofile(stream)
        self.column.tofile(stream)


class ConvolutionNeuron(Neuron):
    def __init__(self, inputs, outputs):
        self.N = inputs
        self.M = outputs
        self.matrix = np.zeros((self.M, self.N))

    def input_count(self):
        return self.N

    def output_count(self):
        return self.M

    def __add__(self, x):
        assert self.N == x.N and self.M == x.M, "Trying to add different sized matrices"
        new_neuron = ConvolutionNeuron(self.N, self.M)
        new_neuron.matrix = self.matrix + x.matrix
        return new_neuron

    def __rmul__(self, x):
        new_neuron = ConvolutionNeuron(self.N, self.M)
        new_neuron.matrix = x * self.matrix
        return new_neuron

    def compute_gradient(self, external_gradient, inputs):
        # Check that the shape a x b of the source is compatible with the shape n x m of convolution
        assert len(inputs.shape) == 2, "The input is not 2-dimensional " + str(
            inputs.shape
        )
        assert inputs.shape[0] % self.M == 0, (
            "The size of input table is not divisible by the convolution table"
            + str(inputs.shape)
            + " neq "
            + str(self.matrix.shape)
        )
        assert inputs.shape[1] % self.N == 0, (
            "The size of input table is not divisible by the convolution table"
            + str(inputs.shape)
            + " neq "
            + str(self.matrix.shape)
        )
        number_of_tile_rows = inputs.shape[0] // self.M
        number_of_tile_columns = inputs.shape[1] // self.N
        assert external_gradient.shape[0] == number_of_tile_rows, (
            "The size of gradient table is not divisible by the convolution table"
            + str(external_gradient.shape)
            + " neq "
            + str(self.matrix.shape)
        )
        assert external_gradient.shape[1] == number_of_tile_columns, (
            "The size of gradient table is not divisible by the convolution table"
            + str(external_gradient.shape)
            + " neq "
            + str(self.matrix.shape)
        )
        df = ConvolutionNeuron(self.N, self.M)
        new_gradient = (
            np.tensordot(external_gradient, self.matrix, 0)
            .transpose(0, 2, 1, 3)
            .reshape(inputs.shape)
        )
        new_inputs = inputs.reshape(
            number_of_tile_rows, self.M, number_of_tile_columns, self.N
        ).transpose(0, 2, 1, 3)
        df.matrix += np.tensordot(external_gradient, new_inputs)
        return new_gradient, df

    def compute(self, inputs):
        # Check that the shape a x b of the source is compatible with the shape n x m of convolution
        assert len(inputs.shape) == 2, "The input is not 2-dimensional " + str(
            inputs.shape
        )
        assert inputs.shape[0] % self.M == 0, (
            "The size of input table is not divisible by the convolution table"
            + str(inputs.shape)
            + " neq "
            + str(self.matrix.shape)
        )
        assert inputs.shape[1] % self.N == 0, (
            "The size of input table is not divisible by the convolution table"
            + str(inputs.shape)
            + " neq "
            + str(self.matrix.shape)
        )
        number_of_tile_rows = inputs.shape[0] // self.M
        number_of_tile_columns = inputs.shape[1] // self.N
        outputs = np.zeros((number_of_tile_rows, number_of_tile_columns))
        # 1   2  3  4
        # 5   6  7  8
        # 9  10 11 12
        # 13 14 15 16   = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        #
        #
        # 1 2  3 4   9 10  11 12
        # 5 6, 7 8, 13 14, 15 16   = 1 2 5 6 3 4 7 8 9 10 13 14 11 12 15 16
        new_inputs = inputs.reshape(
            number_of_tile_rows, self.M, number_of_tile_columns, self.N
        ).transpose(0, 2, 1, 3)
        outputs = np.tensordot(new_inputs, self.matrix)
        return outputs

    def copy(self):
        new_neuron = ConvolutionNeuron(self.N, self.M)
        new_neuron.matrix = self.matrix.copy()
        return new_neuron

    def set_neuron(self, source):
        self.matrix = source
        self.N = self.matrix.shape[1]
        self.M = self.matrix.shape[0]

    def mutate(self, bound, chance):
        with np.nditer(self.matrix, op_flags=["readwrite"]) as matrix:
            for xij in matrix:
                coin_flip = np.random.random()
                if coin_flip < chance:
                    xij[...] += 2 * bound * (np.random.random() - 0.5)

    def read_from_stream(self, stream):
        self.M = struct.unpack("i", stream.read(4))[0]
        self.N = struct.unpack("i", stream.read(4))[0]
        self.matrix = np.fromfile(stream, dtype=float, count=self.M * self.N).reshape(
            (self.M, self.N)
        )

    def write_to_stream(self, stream):
        stream.write(struct.pack("i", self.M))
        stream.write(struct.pack("i", self.N))
        self.matrix.ravel().tofile(stream)


class StridedNeuron(Neuron):
    def __init__(self, inputs, outputs):
        self.N = inputs
        self.M = outputs
        self.matrix = np.zeros((self.M, self.N))

    def input_count(self):
        return self.N

    def output_count(self):
        return self.M

    def __add__(self, x):
        assert self.N == x.N and self.M == x.M, "Trying to add different sized matrices"
        new_neuron = StridedNeuron(self.N, self.M)
        new_neuron.matrix = self.matrix + x.matrix
        return new_neuron

    def __rmul__(self, x):
        new_neuron = StridedNeuron(self.N, self.M)
        new_neuron.matrix = x * self.matrix
        return new_neuron

    def compute_gradient(self, external_gradient, inputs):
        assert len(inputs.shape) == 2, "The input is not 2-dimensional " + str(
            inputs.shape
        )
        assert inputs.shape[0] > self.M, (
            "The size of input table is too small"
            + str(inputs.shape)
            + " neq "
            + str(self.matrix.shape)
        )
        assert inputs.shape[1] > self.N, (
            "The size of input table is too small"
            + str(inputs.shape)
            + " neq "
            + str(self.matrix.shape)
        )
        assert (
            len(external_gradient.shape) == 2
        ), "The gradient is not 2-dimensional " + str(external_gradient.shape)
        assert external_gradient.shape[0] == inputs.shape[0] - self.M + 1, (
            "The size of gradient table is wrong"
            + str(external_gradient.shape)
            + " neq "
            + str(inputs.shape[0] - self.M + 1)
        )
        assert external_gradient.shape[1] == inputs.shape[1] - self.N + 1, (
            "The size of gradient table is wrong"
            + str(external_gradient.shape)
            + " neq "
            + str(inputs.shape[1] - self.N + 1)
        )
        df = StridedNeuron(self.N, self.M)
        new_gradient = np.zeros(inputs.shape)
        number_of_tile_rows = external_gradient.shape[0]
        number_of_tile_columns = external_gradient.shape[1]
        for i in range(number_of_tile_rows):
            for j in range(number_of_tile_columns):
                # 0...0
                #  ...
                # 10. 0
                # 01. 0
                #  ...
                # 0..01
                #  ...
                # 0...0
                left = np.concatenate(
                    (
                        np.zeros((i, self.M)),
                        np.identity(self.M),
                        np.zeros((number_of_tile_rows - i - 1, self.M)),
                    ),
                    axis=0,
                )
                # 0..1...0..0
                # 0..01..0..0
                # ...........
                # 0..0..10..0
                right = np.concatenate(
                    (
                        np.zeros((self.N, j)),
                        np.identity(self.N),
                        np.zeros((self.N, number_of_tile_columns - j - 1)),
                    ),
                    axis=1,
                )
                #       000
                #  A -> 0A0
                #       000
                extended_matrix = np.dot(left, np.dot(self.matrix, right))
                # ***
                # *A* -> A
                # ***
                essential_part = np.dot(left.T, np.dot(inputs, right.T))

                new_gradient = new_gradient + external_gradient[i][j] * extended_matrix
                df.matrix += external_gradient[i][j] * essential_part
        return new_gradient, df

    def compute(self, inputs):
        assert len(inputs.shape) == 2, "The input is not 2-dimensional " + str(
            inputs.shape
        )
        assert inputs.shape[0] >= self.M, (
            "The size of input table is too small"
            + str(inputs.shape)
            + " neq "
            + str(self.matrix.shape)
        )
        assert inputs.shape[1] >= self.N, (
            "The size of input table is too small"
            + str(size)
            + " neq "
            + str(self.matrix.shape)
        )
        number_of_tile_rows = inputs.shape[0] - self.M + 1
        number_of_tile_columns = inputs.shape[1] - self.N + 1
        outputs = np.zeros((number_of_tile_rows, number_of_tile_columns))
        total_size = self.M * self.N
        for i in range(number_of_tile_rows):
            for j in range(number_of_tile_columns):
                # 0...0
                #  ...
                # 10. 0
                # 01. 0
                #  ...
                # 0..01
                #  ...
                # 0...0
                left = np.concatenate(
                    (
                        np.zeros((i, self.M)),
                        np.identity(self.M),
                        np.zeros((number_of_tile_rows - i - 1, self.M)),
                    ),
                    axis=0,
                )
                # 0..1...0..0
                # 0..01..0..0
                # ...........
                # 0..0..10..0
                right = np.concatenate(
                    (
                        np.zeros((self.N, j)),
                        np.identity(self.N),
                        np.zeros((self.N, number_of_tile_columns - j - 1)),
                    ),
                    axis=1,
                )
                #       000
                #  A -> 0A0
                #       000
                extended_matrix = np.dot(left, np.dot(self.matrix, right))
                outputs[i][j] = np.tensordot(inputs, extended_matrix)
        return outputs

    def copy(self):
        new_neuron = StridedNeuron(self.N, self.M)
        new_neuron.matrix = self.matrix.copy()
        return new_neuron

    def set_neuron(self, source):
        self.matrix = source
        self.N = self.matrix.shape[1]
        self.M = self.matrix.shape[0]

    def mutate(self, bound, chance):
        with np.nditer(self.matrix, op_flags=["readwrite"]) as matrix:
            for xij in matrix:
                coin_flip = np.random.random()
                if coin_flip < chance:
                    xij[...] += 2 * bound * (np.random.random() - 0.5)

    def read_from_stream(self, stream):
        self.M = struct.unpack("i", stream.read(4))[0]
        self.N = struct.unpack("i", stream.read(4))[0]
        self.matrix = np.fromfile(stream, dtype=float, count=self.M * self.N).reshape(
            (self.M, self.N)
        )

    def write_to_stream(self, stream):
        stream.write(struct.pack("i", self.M))
        stream.write(struct.pack("i", self.N))
        self.matrix.ravel().tofile(stream)


class NeuronLayer:
    def __init__(
        self,
        neurons_type,
        count,
        inputs_per_neuron,
        neurons_per_input,
        neurons_size,
        use_activation=True,
    ):
        self.use_activation = use_activation
        self.f = np.tanh
        self.neurons_type = neurons_type
        self.neurons_count = count
        self.neurons_size = neurons_size
        # (A,B) take A inputs, combine, and put into B neurosns
        self.inputs_per_neuron = inputs_per_neuron
        self.neurons_per_input = neurons_per_input
        if neurons_type == 0:
            self.neurons = [
                LinearNeuron(neurons_size[0], neurons_size[1]) for i in range(count)
            ]
        elif neurons_type == 1:
            self.neurons = [
                ConvolutionNeuron(neurons_size[0], neurons_size[1])
                for i in range(count)
            ]
        elif neurons_type == 2:
            self.neurons = [
                StridedNeuron(neurons_size[0], neurons_size[1]) for i in range(count)
            ]
        else:
            raise Exception(
                "Wrong Neuron type. Should be [0,1,2] - got {}".format(neurons_type)
            )

    def __add__(self, x):
        assert self.neurons_count == x.neurons_count, "Trying to add different Layers"
        new_layer = NeuronLayer(
            self.neurons_type,
            self.neurons_count,
            self.inputs_per_neuron,
            self.neurons_per_input,
            self.neurons_size,
            self.use_activation,
        )
        new_layer.neurons = []
        for neuronA, neuronB in zip(self.neurons, x.neurons):
            new_layer.neurons.append(neuronA + neuronB)
        return new_layer

    def __rmul__(self, x):
        new_layer = NeuronLayer(
            self.neurons_type,
            self.neurons_count,
            self.inputs_per_neuron,
            self.neurons_per_input,
            self.neurons_size,
            self.use_activation,
        )
        new_layer.neurons = []
        for neuron in self.neurons:
            new_layer.neurons.append(x * neuron)
        return new_layer

    def read_from_stream(self, stream):
        for neuron in self.neurons:
            neuron.read_from_stream(stream)

    def write_to_stream(self, stream):
        for neuron in self.neurons:
            neuron.write_to_stream(stream)

    def compute(self, inputs):
        assert len(inputs) % self.inputs_per_neuron == 0, (
            "Input number "
            + str(len(inputs))
            + " cannot be combined by "
            + str(self.inputs_per_neuron)
        )
        new_inputs = [
            np.concatenate(inputs[x : x + self.inputs_per_neuron])
            for x in range(0, len(inputs), self.inputs_per_neuron)
        ]
        assert len(new_inputs) * self.neurons_per_input == self.neurons_count, (
            "Input number "
            + str(len(new_inputs))
            + " cannot be used by "
            + str(self.neurons_count)
            + " neurons  by "
            + str(self.neurons_per_input)
        )

        outputs = [
            self.neurons[i].compute(new_inputs[i // self.neurons_per_input])
            for i in range(self.neurons_count)
        ]
        if self.use_activation:
            return [self.f(x) for x in outputs]
        else:
            return outputs

    def compute_gradient(self, external_gradient, inputs, outputs):
        assert len(inputs) % self.inputs_per_neuron == 0, (
            "Input number "
            + str(len(inputs))
            + " cannot be combined by "
            + str(inputs_per_neuron)
        )
        new_inputs = inputs
        if self.inputs_per_neuron > 1:
            new_inputs = [
                np.concatenate(inputs[x : x + self.inputs_per_neuron])
                for x in range(0, len(inputs), self.inputs_per_neuron)
            ]
        assert len(new_inputs) * self.neurons_per_input == self.neurons_count, (
            "Input number "
            + str(len(new_inputs))
            + " cannot be used by "
            + str(self.neurons_count)
            + " neurons  by "
            + str(self.neurons_per_input)
        )
        assert len(external_gradient) == self.neurons_count, (
            "Wrong number of gradients, got "
            + str(len(external_gradient))
            + " but the number of neurons is "
            + str(self.neurons_count)
        )
        df = NeuronLayer(
            self.neurons_type,
            self.neurons_count,
            self.inputs_per_neuron,
            self.neurons_per_input,
            self.neurons_size,
            self.use_activation,
        )
        df.neurons = []
        gradient_before_f = (
            [
                external_gradient[i] * (1 - outputs[i] * outputs[i])
                for i in range(len(external_gradient))
            ]
            if self.use_activation
            else external_gradient
        )

        new_gradient = []
        for i in range(self.neurons_count):
            gradient_i, df_i = self.neurons[i].compute_gradient(
                gradient_before_f[i], new_inputs[i // self.neurons_per_input]
            )
            new_gradient.append(gradient_i)
            df.neurons.append(df_i)

        # We got N gradients for each input ,so we need to add them
        if self.neurons_per_input > 1:
            new_gradient = [
                np.sum(new_gradient[x : x + self.neurons_per_input], axis=0)
                for x in range(0, self.neurons_count, self.neurons_per_input)
            ]
        # We have input into neuron combined from A inputs - we need to split them
        if self.inputs_per_neuron > 1:
            split_gradient = []
            for x in new_gradient:
                split_gradient += np.array_split(x, self.inputs_per_neuron)
            new_gradient = split_gradient
        return new_gradient, df

    def mutate(self, bound, chance):
        for neuron in self.neurons:
            neuron.mutate(bound, chance)

    def copy(self):
        new_layer = NeuronLayer(
            self.neurons_type,
            self.neurons_count,
            self.inputs_per_neuron,
            self.neurons_per_input,
            self.neurons_size,
            self.use_activation,
        )
        new_layer.neurons = []
        for neuron in self.neurons:
            new_layer.neurons.append(neuron.copy())
        return new_layer


class Brain:
    def __init__(self):
        self.neuron_layers = []

    def add_layer(
        self,
        neurons_type,
        count,
        inputs_per_neuron,
        neurons_per_input,
        neurons_size,
        use_activation=True,
    ):
        self.neuron_layers.append(
            NeuronLayer(
                neurons_type,
                count,
                inputs_per_neuron,
                neurons_per_input,
                neurons_size,
                use_activation,
            )
        )

    def __add__(self, x):
        new_brain = Brain()
        for layerA, layerB in zip(self.neuron_layers, x.neuron_layers):
            new_brain.neuron_layers.append(layerA + layerB)
        return new_brain

    def __rmul__(self, x):
        new_brain = Brain()
        for layer in self.neuron_layers:
            new_brain.neuron_layers.append(x * layer)
        return new_brain

    def write_to_stream(self, stream):
        stream.write(struct.pack("i", len(self.neuron_layers)))
        for layer in self.neuron_layers:
            stream.write(struct.pack("i", layer.neurons_type))
            stream.write(struct.pack("i", layer.neurons_count))
            stream.write(struct.pack("i", layer.inputs_per_neuron))
            stream.write(struct.pack("i", layer.neurons_per_input))
            stream.write(struct.pack("i", layer.neurons_size[0]))
            stream.write(struct.pack("i", layer.neurons_size[1]))
            stream.write(struct.pack("i", layer.use_activation))
            layer.write_to_stream(stream)

    def read_from_stream(self, stream):
        count = neurons_type = struct.unpack("i", stream.read(4))[0]
        self.neuron_layers = []
        for i in range(count):
            neurons_type = struct.unpack("i", stream.read(4))[0]
            neurons_count = struct.unpack("i", stream.read(4))[0]
            inputs_per_neuron = struct.unpack("i", stream.read(4))[0]
            neurons_per_input = struct.unpack("i", stream.read(4))[0]
            neurons_size = [
                struct.unpack("i", stream.read(4))[0],
                struct.unpack("i", stream.read(4))[0],
            ]
            use_activation = struct.unpack("i", stream.read(4))[0]
            layer = NeuronLayer(
                neurons_type,
                neurons_count,
                inputs_per_neuron,
                neurons_per_input,
                neurons_size,
                use_activation,
            )
            layer.read_from_stream(stream)
            self.neuron_layers.append(layer)

    def compute(self, inputs):
        current_output = [inputs]
        for layer in self.neuron_layers:
            current_output = layer.compute(current_output)
        return current_output

    def compute_with_history(self, inputs):
        current_output = [inputs]
        history = []
        for layer in self.neuron_layers:
            current_output = layer.compute(current_output)
            if layer.neurons_type > 0:
                history.append(current_output)
        history.append(current_output)
        return history

    def compute_gradient(self, external_gradient, inputs):
        df = Brain()
        inputs_per_layer = [[inputs]]
        current_output = [inputs]
        for layer in self.neuron_layers:
            current_output = layer.compute(current_output)
            inputs_per_layer.append(current_output)
        current_gradient = [external_gradient]
        for i in reversed(range(len(self.neuron_layers))):
            current_gradient, df_i = self.neuron_layers[i].compute_gradient(
                current_gradient, inputs_per_layer[i], inputs_per_layer[i + 1]
            )
            df.neuron_layers.append(df_i)
        df.neuron_layers = reversed(df.neuron_layers)
        return current_gradient, df

    def mutate(self, bound, chance):
        for layer in self.neuron_layers:
            layer.mutate(bound, chance)

    def copy(self):
        new_brain = Brain()
        for layer in self.neuron_layers:
            new_brain.neuron_layers.append(layer.copy())
        return new_brain
