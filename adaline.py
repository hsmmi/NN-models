import numpy as np
from delta_rule import delta_rule
from activation_func import activation_func


class adaline:
    def __init__(
        self, sample_size: int, f, d_f, w=None, b=0, a=1, printing=False
    ) -> None:
        self.sample_size = sample_size
        self.f = f
        self.d_f = d_f
        if w is None:
            self.w = np.zeros((sample_size, sample_size))
        else:
            self.w = w
        self.table = np.empty((0, sample_size))
        self.printing = printing

    def train(self, S, t):
        model = delta_rule(
            input_size=2,
            f=activation_func.identity,
            d_f=activation_func.d_identity,
            printing=self.printing,
        )
        model.train(S, t)


model = adaline(
    sample_size=2,
    f=activation_func.identity,
    d_f=activation_func.d_identity,
    printing=True,
)

data = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
labels = np.array([1, -1, -1, -1])

model.train(data, labels)
model.print_table()
