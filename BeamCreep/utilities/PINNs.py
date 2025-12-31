import torch
import torch.nn as nn

from utilities.dif_operator import DifX, DifY
from utilities.used_funcs import constitutive_law, constitutive_law2


class PINN(nn.Module):
    def __init__(self, net_u, net_v, args):
        super(PINN, self).__init__()
        self.net_u = net_u
        self.net_v = net_v
        self.args = args

    def forward(self, inputs, dt=0):
        dif_x = DifX(self.net_u)
        dif_y = DifY(self.net_v)

        # # # obtain the displacment at the right tip of the rod
        u_r = self.net_u(inputs[4]) * inputs[4][..., 0, torch.newaxis]

        # # # obtain partial derivatives of u with respect to x and y
        u, u_x, u_y, u_xx, u_xy, u_yy = dif_x(inputs[0])
        v, v_x, v_y, v_xx, v_xy, v_yy = dif_y(inputs[0])

        # # # Obtain the residuals from the governing equation
        Ein, S11, S22 = constitutive_law2([u, u_x, u_y, u_xx, u_xy, u_yy], [v, v_x, v_y, v_xx, v_xy, v_yy], self.args, dt)

        return [Ein, u_r, S11, S22]


