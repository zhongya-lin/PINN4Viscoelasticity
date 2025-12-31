import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utilities.generate_net import GenerateNet
from utilities.PINNs import PINN
from main2 import generate_args


XY_MAT_FILE = './data_files/Coord_qtr_1-09.mat'


def load_network(args):
    inp_num = args.num_input
    out_num = args.num_output
    layers = args.net_layer

    net_u = GenerateNet(inp_num, layers[0], out_num)
    net_v = GenerateNet(inp_num, layers[1], out_num)

    pinn = PINN(net_u, net_v, args)

    pinn.load_state_dict(torch.load(MODEL_EVALUATION, weights_only=True))

    return net_u, net_v, pinn


def post_process(net_u, net_v, pinn, xy_size, ti=0):
    xy_data = scipy.io.loadmat(XY_MAT_FILE)
    xy = xy_data['x'].astype(np.float32)
    # # # circular coordinate
    xy_cr = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2).astype(np.float32)
    xy_ct_0 = np.arctan2(xy[:, 1], xy[:, 0]).astype(np.float32)
    xy_ct = np.where(xy_ct_0 > np.pi/2.0, np.pi-xy_ct_0, xy_ct_0)
    xy_c = np.column_stack((xy_cr, xy_ct))
    xy_tensor = torch.from_numpy(xy_c)
    x, y = (xy_tensor[..., i, torch.newaxis] for i in range(xy_tensor.shape[-1]))

    u = net_u(xy_tensor) * 1e-3
    v = net_v(xy_tensor) * torch.cos(y) * torch.sin(y)
    u = u.detach().numpy()
    v = v.detach().numpy()


    x_u = u * np.cos(xy_ct_0[:, np.newaxis]+v)
    y_v = u * np.sin(xy_ct_0[:, np.newaxis]+v)

    # # # plot figure for displacement u
    if (ti-1) % 4 == 0:
        fig1 = plt.figure(1)
        plt.scatter(xy[:, 0]+x_u[:, 0], xy[:, 1]+y_v[:, 0], s=10, c=u, cmap='jet')
        plt.axis('equal')
        plt.colorbar()
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.savefig('u-1parts_0-05pi_{}.tiff'.format(ti), dpi=300)
        plt.show()

    return None


if __name__ == '__main__':
    args = generate_args()

    for i in range(20):
        MODEL_EVALUATION = r'./model_files/Circle_Creep_{}'.format(str(i+1))
        net_u, net_v, pinn = load_network(args)
        post_process(net_u, net_v, pinn, xy_size=[args.x_size, args.y_size], ti=int(i+1))
