import numpy as np
from activation_func import activation_func
from tabulate import tabulate


class ElmanNet:
    def __init__(self, f_hidden, f_output) -> None:
        self.f_hidden = f_hidden
        self.f_output = f_output

    def feed(self, x, y_0, W_IH, W_OI, W_HO):
        h_in = np.dot(x, W_IH) + np.dot(y_0, W_OI)
        h = self.f_hidden(h_in)
        y_in = np.dot(h, W_HO)
        y = self.f_output(y_in)
        print(
            tabulate(
                [
                    ["x", np.round(x, 3)],
                    ["y_0", np.round(y_0, 3)],
                    ["h_in", np.round(h_in, 3)],
                    ["h", np.round(h, 3)],
                    ["y_in", np.round(y_in, 3)],
                    ["y", np.round(y, 3)],
                ],
                tablefmt="fancy_grid",
            )
        )
        return h, y


f_h = activation_func.bipolar_sigmoid
f_o = activation_func.bipolar_sigmoid

x = np.array([[1, -1, 1]])
y = np.array([[-0.5, 0.5]])
W_IH = np.array(
    [[0.4, -0.2, -0.3, 0.2], [0.3, -0.1, -0.2, 0.3], [-0.2, 0.4, 0.3, 0.1]]
)
W_HO = np.array([[0.1, -0.2], [-0.3, 0.3], [-0.3, 0.1], [0.2, -0.2]])
W_OI = np.array([[0, 0.4, -0.2, 0.3], [0.3, -0.1, 0.2, 0.1]])

Elman = ElmanNet(f_hidden=f_h, f_output=f_o)
h, y = Elman.feed(x=x, y_0=y, W_IH=W_IH, W_OI=W_OI, W_HO=W_HO)
print(f"h = {np.round(h,3)}\ny = {np.round(y,3)}")

Elman = ElmanNet(f_hidden=f_h, f_output=f_o)
h, y = Elman.feed(x=x, y_0=y, W_IH=W_IH, W_OI=W_OI, W_HO=W_HO)
print(f"h = {np.round(h,3)}\ny = {np.round(y,3)}")

Elman = ElmanNet(f_hidden=f_h, f_output=f_o)
h, y = Elman.feed(x=x, y_0=y, W_IH=W_IH, W_OI=W_OI, W_HO=W_HO)
print(f"h = {np.round(h,3)}\ny = {np.round(y,3)}")

Elman = ElmanNet(f_hidden=f_h, f_output=f_o)
h, y = Elman.feed(x=x, y_0=y, W_IH=W_IH, W_OI=W_OI, W_HO=W_HO)
print(f"h = {np.round(h,3)}\ny = {np.round(y,3)}")
