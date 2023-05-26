import numpy as np
from activation_func import activation_func
from Initial_weights import Init_Weights


class MLP:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_func_hidden,
        activation_func_output,
        derivative_act_hidden,
        derivative_act_output,
        bias=True,
        batch_mode=True,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias
        self.batch = batch_mode
        init_weights = Init_Weights(
            self.input_size, self.hidden_size, self.output_size, bias=self.bias
        )
        self.W1, self.W2 = init_weights.fill_equal(v_h=0.5, v_o=0.5)
        """
        self.W1 = np.array([
            [0.22, 0.57],
            [0.7, 0.21],
            [0.18, 0.11],
            [0.38, 0.44]
        ])

        self.W2 = np.array([
            [0.09, 0.81],
            [0.77, 0.86],
            [0.17, 0.29]
        ])
        """
        self.delta_W1 = np.zeros(self.W1.shape)
        self.delta_W2 = np.zeros(self.W2.shape)
        self.act_func_h = activation_func_hidden
        self.act_func_o = activation_func_output
        self.de_act_h = derivative_act_hidden
        self.de_act_o = derivative_act_output
        self.o = 0
        self.h = 0

    def forward(self, X):
        """
        X: input sample
        """
        if self.bias:
            X = np.c_[np.ones(X.shape[0]), X]
        self.z_in = np.dot(X, self.W1)
        self.z = self.act_func_h(self.z_in)
        if self.bias:
            self.z = np.c_[np.ones(self.z.shape[0]), self.z]
        self.y_in = np.dot(self.z, self.W2)
        output = self.act_func_o(self.y_in)
        return output

    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = -1 * self.output_error * self.de_act_o(self.y_in)
        self.sigma_output_delta = self.output_delta.dot(self.W2[1:, :].T)
        self.hidden_delta = self.sigma_output_delta * self.de_act_h(self.z_in)
        if self.batch:
            self.h += X.T.dot(self.hidden_delta)
            self.o += self.z.T.dot(self.output_delta)

    def train(self, X, y, epochs, learning_rate):
        if self.batch:
            if self.print_table:
                print()
            for epoch in range(epochs):
                for i in range(X.shape[0]):
                    inp = X[i].reshape([1, -1])
                    output = self.forward(inp)
                    self.backward(inp, y[i], output)
                self.W1 += (self.h / X.shape[0]) * learning_rate * -1
                self.W2 += (self.o / X.shape[0]) * learning_rate * -1
        else:
            for epoch in range(epochs):
                for i in range(X.shape[0]):
                    inp = X[i].reshape([1, -1])
                    output = self.forward(inp)
                    self.backward(inp, y[i], output)
                    self.W1 += (
                        inp.T.dot(self.hidden_delta) * learning_rate * -1
                    )
                    self.W2 += (
                        self.z.T.dot(self.output_delta) * learning_rate * -1
                    )
        return self.W1, self.W2


def example():
    # Example usage
    X = np.array([[0.25, 0.5, 0.75], [0.75, 0.5, 0.25]])
    y = np.array([[0.25], [0.75]])
    input_size = X.shape[1]
    hidden_size = 2
    output_size = y.shape[1]

    hidden_act = activation_func.bipolar_sigmoid
    de_hidden_act = activation_func.d_bipolar_sigmoid
    output_act = activation_func.bipolar_sigmoid
    de_output_act = activation_func.d_bipolar_sigmoid
    mlp = MLP(
        input_size,
        hidden_size,
        output_size,
        activation_func_hidden=hidden_act,
        activation_func_output=output_act,
        derivative_act_hidden=de_hidden_act,
        derivative_act_output=de_output_act,
        batch_mode=True,
        bias=False,
    )
    updated_weights = mlp.train(X, y, epochs=1, learning_rate=1.0)

    # Print the updated weights
    W1, W2 = updated_weights
    print("Updated Weights (W1):")
    print(W1)
    print("\nUpdated Weights (W2):")
    print(W2)


example()
