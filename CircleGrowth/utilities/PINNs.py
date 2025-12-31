import torch
import torch.nn as nn
import scipy
import numpy as np

from utilities.dif_operator import DifX, DifY
from utilities.used_funcs import constitutive_law_growth


class PINN(nn.Module):
    def __init__(self, net_u, net_v, args):
        super(PINN, self).__init__()
        self.net_u = net_u
        self.net_v = net_v
        self.args = args

    def forward(self, inputs, dt, delta_t, growth_ratio):
        dif_x = DifX(self.net_u)
        dif_y = DifY(self.net_v)

        # # # obtain partial derivatives of u with respect to x and y
        u, u_x, u_y, u_xx, u_xy, u_yy = dif_x(inputs[0], inputs[1])
        v, v_x, v_y, v_xx, v_xy, v_yy = dif_y(inputs[0])

        # # # Obtain the residuals from the governing equation
        Ein, Egrowth, growth_ratio = constitutive_law_growth([u, u_x, u_y, u_xx, u_xy, u_yy], [v, v_x, v_y, v_xx, v_xy, v_yy],
                                                    inputs[0][..., 0, torch.newaxis], self.args, dt, delta_t, growth_ratio)

        return [Ein, Egrowth, growth_ratio]


