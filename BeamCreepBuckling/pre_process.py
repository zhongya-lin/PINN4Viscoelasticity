import numpy as np
import torch
import scipy

from utilities.generate_net import GenerateNet
from utilities.PINNs import PINN


def pre_process(args):
    xy_data = scipy.io.loadmat('./data_files/Coord_Beam_10_1.mat')
    xy = xy_data['x'].astype(np.float32)
    xy_out = xy_data['x_out'].astype(np.float32)
    wt = xy_data['w'].astype(np.float32)

    x_train = [torch.from_numpy(xy), torch.from_numpy(xy_out)]

    # # # Define the displacement boundary conditions
    pressure_out = torch.from_numpy((np.ones(xy_out.shape[0]) * -1e-2).astype(np.float32))[..., torch.newaxis]

    y_train = [pressure_out, torch.from_numpy(wt)]

    net_inp_num = args.num_input
    net_out_num = args.num_output
    net_layer = list(args.net_layer)

    net_u = GenerateNet(net_inp_num, net_layer[0], net_out_num)
    net_v = GenerateNet(net_inp_num, net_layer[1], net_out_num)

    # # # # Initial weight
    # net_u.init_weights()
    # net_v.init_weights()

    # # # build up the PINN
    pinn = PINN(net_u, net_v, args)

    return net_u, net_v, pinn, x_train, y_train

