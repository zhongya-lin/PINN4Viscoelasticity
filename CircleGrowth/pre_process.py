import numpy as np
import torch
import scipy

from utilities.generate_net import GenerateNet
from utilities.PINNs import PINN


def pre_process(args):
    xy_data = scipy.io.loadmat('./data_files/Coord_qtr_1-09.mat')
    xy = xy_data['x']
    xy_out = xy_data['x_out']
    wt = xy_data['w'].astype(np.float32)

    # # transform to circular coordinate
    xy_cr = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2).astype(np.float32)
    xy_ct_0 = np.arctan2(xy[:, 1], xy[:, 0]).astype(np.float32)
    xy_ct = np.where(xy_ct_0 > np.pi/2.0, np.pi-xy_ct_0, xy_ct_0)
    xy_c = np.column_stack((xy_cr, xy_ct))
    # # # inner radius
    r_in = np.min(xy_c[:, 0])
    # # # outer radius
    r_out = np.float64(1.0000)
    x_train = [torch.from_numpy(xy_c), torch.tensor(r_out)]
    y_train = [torch.from_numpy(wt)]

    net_inp_num = args.num_input
    net_out_num = args.num_output
    net_layer = list(args.net_layer)

    net_u = GenerateNet(net_inp_num, net_layer[0], net_out_num)
    net_v = GenerateNet(net_inp_num, net_layer[1], net_out_num)

    # # # build up the PINN
    pinn = PINN(net_u, net_v, args)

    return net_u, net_v, pinn, x_train, y_train

