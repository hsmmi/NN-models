import numpy as np
from tabulate import tabulate


class hebb_rule:
    def __init__(self, input_size: int, printing=False):
        self.input_size = input_size
        self.w = np.zeros(input_size)
        self.b = 0
        self.a = 1  # Learning rate
        self.table = np.empty((0, 2 * input_size + 6))
        self.printing = printing

    def reset_weights(self):
        self.w = np.zeros(self.input_size)
        self.b = 0

    def train_sample(self, X: np.ndarray, t: int):
        """
        Get one sample and update the weight matrix
        """
        if self.printing:
            row = np.append(X, 1)
            row = np.append(row, t)

        delta_w = self.a * X * t
        delta_b = self.a * t

        self.w += delta_w
        self.b += delta_b

        if self.printing:
            row = np.append(row, delta_w)
            row = np.append(row, delta_b)
            row = np.append(row, self.w)
            row = np.append(row, self.b)

            self.table = np.vstack((self.table, row))

        return delta_w, delta_b

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Get a set of samples and update the weight matrix
        """
        for i in range(len(X)):
            self.train_sample(X[i], y[i])

    def print_table(self):
        # Set columns labels
        labels = ["x" + str(i) for i in range(self.input_size)]
        labels.append("1")
        labels.append("y")
        for i in range(self.input_size):
            labels.append("Δw" + str(i))
        labels.append("Δb")
        for i in range(self.input_size):
            labels.append("w" + str(i))
        labels.append("b")

        print(tabulate(self.table, headers=labels, tablefmt="fancy_grid"))


def test_hebb_rule():
    data = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    labels = np.array([1, -1, -1, -1])

    model = hebb_rule(2, printing=True)
    model.train(data, labels)
    model.print_table()

# test_hebb_rule()