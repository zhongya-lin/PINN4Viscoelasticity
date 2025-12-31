import numpy as np
import torch

from utilities.generate_net import GenerateNet
from utilities.PINNs import PINN


def pre_process(args):
    # # # input information
    x_num, y_num = args.x_num, args.y_num
    x_size, y_size = args.x_size, args.y_size
    dx_interval = x_size / (x_num-1)
    dy_interval = y_size / (y_num-1)

    xy_num = x_num * y_num

    # # # Initialize sample points' coordinates
    xy = np.zeros((xy_num, 2)).astype(np.float32)
    for i in range(0, x_num):
        for j in range(0, y_num):
            xy[i*y_num+j, 0] = i * dx_interval
            xy[i*y_num+j, 1] = j * dx_interval
    xy_u = np.hstack([np.linspace(0, x_size, x_num).reshape(x_num, 1).astype(np.float32), (np.ones((x_num, 1))*y_size).astype(np.float32)])
    xy_b = np.hstack([np.linspace(0, x_size, x_num).reshape(x_num, 1).astype(np.float32), np.zeros((x_num,1)).astype(np.float32)])
    xy_l = np.hstack([np.zeros((y_num, 1)).astype(np.float32), np.linspace(0, y_size, y_num).reshape(y_num, 1).astype(np.float32)])
    xy_r = np.hstack([(np.ones((y_num, 1))*x_size).astype(np.float32), np.linspace(0, y_size, y_num).reshape(y_num, 1).astype(np.float32)])
    xy = torch.from_numpy(xy)
    xy_u = torch.from_numpy(xy_u)
    xy_b = torch.from_numpy(xy_b)
    xy_l = torch.from_numpy(xy_l)
    xy_r = torch.from_numpy(xy_r)

    x_train = [xy, xy_u, xy_b, xy_l, xy_r]

    # # # Define the traction boundary conditions
    s_u_x = torch.from_numpy(np.zeros((x_num, 1)).astype(np.float32))
    s_u_y = torch.from_numpy(np.zeros((x_num, 1)).astype(np.float32))
    s_b_x = torch.from_numpy(np.zeros((x_num, 1)).astype(np.float32))
    s_b_y = torch.from_numpy(np.zeros((x_num, 1)).astype(np.float32))
    s_l_x = torch.from_numpy(np.zeros((y_num, 1)).astype(np.float32))
    s_l_y = torch.from_numpy(np.zeros((y_num, 1)).astype(np.float32))
    # s_r_x = torch.cos(xy_r[..., 1, torch.newaxis]/2*torch.pi)
    s_r_x = torch.from_numpy((np.ones((y_num, 1)) * 0.4).astype(np.float32))
    s_r_y = torch.from_numpy(np.zeros((y_num, 1)).astype(np.float32))

    y_train = [s_u_x, s_u_y, s_b_x, s_b_y, s_l_x, s_l_y, s_r_x, s_r_y]

    net_inp_num = args.num_input
    net_out_num = args.num_output
    net_layer = list(args.net_layer)

    net_u = GenerateNet(net_inp_num, net_layer[0], net_out_num)
    net_v = GenerateNet(net_inp_num, net_layer[1], net_out_num)

    # # # build up the PINN
    pinn = PINN(net_u, net_v, args)

    return net_u, net_v, pinn, x_train, y_train, dx_interval, dy_interval

