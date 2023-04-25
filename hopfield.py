import numpy as np
from tabulate import tabulate


class hopfield:
    def __init__(self, input_size: int, printing: bool = False):
        self.input_size = input_size
        self.w = np.zeros((input_size, input_size))
        self.printing = printing

    def train(self, input_data: np.ndarray):
        """
        Get input data which is include p patterns.
        """
        p = input_data.shape[0]
        for i in range(self.input_size):
            for j in range(i + 1, self.input_size):
                for k in range(p):
                    self.w[i][j] += input_data[k][i] * input_data[k][j]
                self.w[j][i] = self.w[i][j]

        # wp = np.zeros((self.input_size, self.input_size))
        # for i in range(p):
        #     wp += np.outer(input_data[i], input_data[i])
        # wp -= np.eye(self.input_size) * p
        if self.printing:
            self.print_weight()

    def recall_synchrouns(self, input_data: np.ndarray, max_iter: int = 100):
        """
        Recall the input data.
        """
        if self.printing:
            print(f"y(0) = {input_data})")
        for i in range(max_iter):
            output_data = np.dot(input_data, self.w)
            output_data[output_data >= 0] = 1
            output_data[output_data < 0] = -1
            if self.printing:
                print(f"y({i + 1}) = {output_data}")
            if np.all(input_data == output_data):
                break
            input_data = output_data

        return output_data

    def recall_asynchrouns(self, X: np.ndarray, max_iter: int = 100):
        """
        Recall the input data with asynchrouns update.
        """
        Y = X.copy()

        for i in range(max_iter):
            if self.printing:
                print(f"\ny({i}) = {Y})")
            pre_Y = Y.copy()
            rand_index = np.random.permutation(self.input_size)
            if self.printing:
                print(f"\nrandom index: {rand_index}")
            for j in rand_index:
                Y[j] = np.sign(X[j] + self.w.T[j] @ Y)
                if self.printing:
                    print(f"select {j}th neuron")
                    print(f"y({i}) = {Y}")
            if np.all(pre_Y == Y):
                print(f"Converged at {i+1}th iteration\n")
                break

        return Y

    def print_weight(self):
        print("Weight matrix:")
        print(tabulate(self.w, tablefmt="fancy_grid"))

    def energy(self, input_data: np.ndarray):
        """
        Calculate the energy of the input data.
        """
        e = 0
        for i in range(self.input_size):
            for j in range(self.input_size):
                e += -self.w[i][j] * input_data[i] * input_data[j]
        return e


def test_hopfield():
    S = np.array([[1, 1, -1, -1], [-1, -1, 1, 1]])

    model = hopfield(input_size=len(S[0]), printing=True)
    model.train(S)

    sample = np.array([1, 1, 0, -1])
    print(f"Input: {sample}")
    print(f"Output: {model.recall_synchrouns(sample)}")
    print(f"Output: {model.recall_asynchrouns(sample)}")


def test_slide_19():
    """
    The output of synchrouns and asynchrouns method is different.
    """
    S = np.array([[1, 1, -1, -1]])

    model = hopfield(input_size=len(S[0]), printing=True)
    model.train(S)

    sample = np.array([-1, -1, -1, -1])
    print(f"Input: {sample}")
    print(f"Output: {model.recall_synchrouns(sample)}")
    print(f"Output: {model.recall_asynchrouns(sample)}")


# test_hopfield()
test_slide_19()
