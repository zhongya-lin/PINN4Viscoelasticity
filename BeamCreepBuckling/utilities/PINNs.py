import torch
import torch.nn as nn

from utilities.dif_operator import DifX, DifY
from utilities.used_funcs import constitutive_law


class PINN(nn.Module):
    def __init__(self, net_u, net_v, args):
        super(PINN, self).__init__()
        self.net_u = net_u
        self.net_v = net_v
        self.args = args

    def forward(self, inputs, dt):
        dif_x = DifX(self.net_u)
        dif_y = DifY(self.net_v)

        # # # obtain the displacment at the right tip of the rod
        u_r = self.net_u(inputs[1]) * inputs[1][..., 0, torch.newaxis] * 1e-3

        # # # obtain partial derivatives of u with respect to x and y
        u, u_x, u_y, u_xx, u_xy, u_yy = dif_x(inputs[0])
        v, v_x, v_y, v_xx, v_xy, v_yy = dif_y(inputs[0])

        # # # Obtain the residuals from the governing equation
        F11, F22, F12, F21, J, Ein = constitutive_law([u_x, u_y, u_xx, u_xy, u_yy], [v_x, v_y, v_xx, v_xy, v_yy],
                                                    self.args, dt)

        return [F11, F22, F12, F21, J, Ein, u_r]


