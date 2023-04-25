import numpy as np
from tabulate import tabulate
from activation_func import activation_func
from hebb_rule import hebb_rule
from delta_rule import delta_rule


class HAM:
    def __init__(
        self, input_size: int, output_size, rule, f, printing=False
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.rule = rule
        self.f = f
        self.w = np.zeros((input_size, output_size))
        self.table = np.empty((0, output_size))
        self.printing = printing

    def train_sample(self, S: np.ndarray, T: np.ndarray):
        """
        Get one sample and update the weight matrix
        """
        w = []
        for i in range(self.output_size):
            delta_w, _ = self.rule.train_sample(S, T[i])
            self.rule.reset_weights()
            w.append(delta_w)

        w = np.array(w).T

        if self.printing:
            self.table = np.vstack((self.table, w))

        return w

    def train(self, S: np.ndarray, T: np.ndarray):
        """
        Get a set of samples and update the weight matrix
        """

        for i in range(S.shape[0]):
            self.w += self.train_sample(S[i], T[i])

        if self.printing:
            self.table = np.vstack((self.table, self.w))

    def print_table(self):
        n_sample = int(len(self.table) / self.input_size) - 1
        for i in range(n_sample):
            print(f"w({i+1}):")
            print(
                tabulate(
                    self.table[
                        i * self.input_size : (i + 1) * self.input_size
                    ],
                    tablefmt="fancy_grid",
                )
            )
        print(f"W:")
        print(
            tabulate(
                self.table[n_sample * self.input_size :], tablefmt="fancy_grid"
            )
        )

    def recall(self, X: np.ndarray):
        """
        Get a sample and return the output
        """
        if self.printing:
            print(f"Input: {X}")
            print(f"Weight matrix:\n {self.w}")
            print(f"y_in: {self.w.T @ X}")
        y_in = self.w.T @ X
        y = np.array([self.f(y_in[i]) for i in range(self.output_size)])
        return y


def test_HAM_with_hebb():
    S = np.array([[-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]])
    T = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

    heb = hebb_rule(4)
    model = HAM(4, 2, heb, activation_func.sign, True)

    model.train(S, T)
    model.print_table()

    sample1 = np.array([1, -1, 0, 1])
    print(model.recall(sample1))

    sample2 = np.array([1, -1, 1, -1])
    print(model.recall(sample2))


def test_HAM_with_delta():
    S = np.array([[-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]])
    T = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

    heb = hebb_rule(4)
    model = HAM(4, 2, heb, activation_func.sign, True)

    model.train(S, T, True)
    model.print_table()

    sample1 = np.array([1, -1, 0, 1])
    print(model.recall(sample1))

    sample2 = np.array([1, -1, 1, -1])
    print(model.recall(sample2))


# test_HAM_with_hebb()