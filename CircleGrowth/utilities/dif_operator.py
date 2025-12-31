import torch
import scipy
import numpy as np


class DifX(torch.nn.Module):
    def __init__(self, fnn, **kwargs):
        super(DifX, self).__init__(**kwargs)
        self.fnn = fnn

    def forward(self, xy, r_fix):
        x, y = (xy[..., i, torch.newaxis] for i in range(xy.shape[-1]))
        x.requires_grad_()
        y.requires_grad_()

        with torch.autograd.set_detect_anomaly(True):
            with torch.autograd.set_detect_anomaly(True):
                temp = self.fnn(torch.concat([x, y], dim=-1))

                disp_u = temp * (x-r_fix)

                disp_u_x = torch.autograd.grad(disp_u, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
                disp_u_y = torch.autograd.grad(disp_u, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]
            disp_u_xx = torch.autograd.grad(disp_u_x, x, grad_outputs=torch.ones_like(x), retain_graph=True)[0]
            disp_u_xy = torch.autograd.grad(disp_u_x, y, grad_outputs=torch.ones_like(y), retain_graph=True)[0]
            disp_u_yy = torch.autograd.grad(disp_u_y, y, grad_outputs=torch.ones_like(y), retain_graph=True)[0]

        return disp_u, disp_u_x, disp_u_y, disp_u_xx, disp_u_xy, disp_u_yy


class DifY(torch.nn.Module):
    def __init__(self, fnn, **kwargs):
        super(DifY, self).__init__(**kwargs)
        self.fnn = fnn

    def forward(self, xy):
        x, y = (xy[..., i, torch.newaxis] for i in range(xy.shape[-1]))
        x.requires_grad_()
        y.requires_grad_()

        with torch.autograd.set_detect_anomaly(True):
            with torch.autograd.set_detect_anomaly(True):
                temp = self.fnn(torch.concat([x, y], dim=-1))

                disp_v = temp * torch.sin(y) * torch.cos(y)

                disp_v_x = torch.autograd.grad(disp_v, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
                disp_v_y = torch.autograd.grad(disp_v, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]
            disp_v_xx = torch.autograd.grad(disp_v_x, x, grad_outputs=torch.ones_like(x), retain_graph=True)[0]
            disp_v_xy = torch.autograd.grad(disp_v_x, y, grad_outputs=torch.ones_like(y), retain_graph=True)[0]
            disp_v_yy = torch.autograd.grad(disp_v_y, y, grad_outputs=torch.ones_like(y), retain_graph=True)[0]

        return disp_v, disp_v_x, disp_v_y, disp_v_xx, disp_v_xy, disp_v_yy
