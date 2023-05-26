import numpy as np
from activation_func import activation_func
from pooling import pooling


class convNet:
    def __init__(
        self,
        conv_filter: np.ndarray,
        stride_conv: int,
        pooling_type: pooling,
        pooling_size: int,
        stride_pooling: int,
        act_f: activation_func,
    ):
        """
        Convolutional Neural Network
        :param conv_filter: convolutional filter (1D)
        """
        self.conv_filter = conv_filter[::-1]
        self.stride_conv = stride_conv
        self.pooling_type = pooling_type
        self.pooling_size = pooling_size
        self.stride_pooling = stride_pooling
        self.act_f = act_f

    def conv(self, x: np.ndarray) -> np.ndarray:
        conv_output = []
        for j in range(
            0, len(x) - len(self.conv_filter) + 1, self.stride_conv
        ):
            conv_output.append(
                np.dot(x[j : j + len(self.conv_filter)], self.conv_filter)
            )
        return np.array(conv_output)

    def forward(self, x):
        out_conv = self.conv(x)
        out_H = self.act_f(out_conv)
        out_O = self.pooling_type(
            out_H, self.pooling_size, self.stride_pooling
        )
        return out_H, out_O


def example():
    x = np.array([-2.0, -1.8, -1.4, -0.6, -0.2, 0.4, 0.8, 1.6])

    conv_filter = np.array([0.75, 0.5, 0.75])

    conv1d = convNet(
        conv_filter=conv_filter,
        stride_conv=1,
        pooling_type=pooling.average_pooling,
        pooling_size=2,
        stride_pooling=2,
        act_f=activation_func.Elu,
    )

    out_H, out_O = conv1d.forward(x)

    print(f"out_H: {np.round(out_H,4)}\nout_O: {np.round(out_O,4)}")


example()
