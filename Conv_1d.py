import numpy as np
from activation_func import activation_func


class conv_net1d:
    def __init__(
        self,
        conv_filter,
        stride_conv,
        pooling_type,
        pooling_size,
        stride_pooling,
        act_f,
    ):
        self.conv_filter = conv_filter
        self.stride_conv = stride_conv
        self.pooling_type = pooling_type
        self.pooling_size = pooling_size
        self.stride_pooling = stride_pooling
        self.act_f = act_f

    def conv(self, x):
        j = 0
        convolved_x = []
        for i in range(
            ((len(x) - len(self.conv_filter)) // self.stride_conv) + 1
        ):
            convolved_x.append(
                np.sum(x[j : j + len(conv_filter)] * conv_filter[::-1], axis=0)
            )
            j += self.stride_conv
        return np.array(convolved_x)

    def pooling(self, x):
        pooling_output = []
        if self.pooling_type == "average":
            j = 0
            for i in range(
                ((len(x) - self.pooling_size) // self.stride_pooling) + 1
            ):
                pooling_output.append(np.mean(x[j : j + self.pooling_size]))
                j += self.stride_pooling

        if self.pooling_type == "max":
            j = 0
            for i in range(
                ((len(x) - self.pooling_size) // self.stride_pooling) + 1
            ):
                pooling_output.append(np.max(x[j : j + self.pooling_size]))
                j += self.stride_pooling

        if self.pooling_type == "l2 norm":
            j = 0
            for i in range(
                ((len(x) - self.pooling_size) // self.stride_pooling) + 1
            ):
                pooling_output.append(
                    np.linalg.norm(x[j : j + self.pooling_size])
                )
                j += self.stride_pooling

        return pooling_output

    def forward(self, x):
        out_conv = self.conv(x)
        out_1 = self.act_f(out_conv)
        out_2 = self.pooling(out_1)
        return out_1, out_2


x = np.array([-2.0, -1.8, -1.4, -0.6, -0.2, 0.4, 0.8, 1.6])

conv_filter = np.array([0.75, 0.5, 0.75])

conv1d = conv_net1d(
    conv_filter=conv_filter,
    stride_conv=1,
    pooling_type="average",
    pooling_size=2,
    stride_pooling=2,
    act_f=activation_func.Elu,
)

print(conv1d.forward(x))
