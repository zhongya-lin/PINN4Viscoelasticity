import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utilities.generate_net import GenerateNet
from utilities.PINNs import PINN
from main import generate_args


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


def post_process(net_u, net_v, pinn, xy_size, dxy=0.1, ti=0, gr=np.array([1.0])):
    xy_data = scipy.io.loadmat(XY_MAT_FILE)
    xy = xy_data['x'].astype(np.float32)
    # # circular coordinate
    xy_cr = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2).astype(np.float32)
    xy_ct_0 = np.arctan2(xy[:, 1], xy[:, 0]).astype(np.float32)
    xy_ct = np.where(xy_ct_0 > np.pi/2.0, np.pi-xy_ct_0, xy_ct_0)
    xy_c = np.column_stack((xy_cr, xy_ct))
    r_in = torch.tensor(np.min(xy_c[:, 0]))
    r_out = torch.tensor(1.0)

    xy_tensor = torch.from_numpy(xy_c)
    x, y = (xy_tensor[..., i, torch.newaxis] for i in range(xy_tensor.shape[-1]))

    u = net_u(xy_tensor) * (x-r_out)
    v = net_v(xy_tensor) * torch.cos(y) * torch.sin(y)
    u = u.detach().numpy()
    v = v.detach().numpy()

    u_grow = np.sqrt(1**2 - gr[ti-1]**2*(1**2 - xy_cr**2)) - xy_cr
    u = u + u_grow[:, np.newaxis]

    x_u = u * np.cos(xy_ct_0[:, np.newaxis]+v)
    y_v = u * np.sin(xy_ct_0[:, np.newaxis]+v)

    # # # plot figure for displacement u
    if ti % 3 == 0:
        fig1 = plt.figure(figsize=(4,6))
        color_u = np.append(u, u)
        plt.scatter(xy[:, 1]+y_v[:, 0], xy[:, 0]+x_u[:, 0], s=2.5, c=u, cmap='jet')
        plt.axis('equal')
        plt.xlim((-0.05, 1.05))
        plt.ylim((-1.05, 1.05))
        plt.xticks([0, 0.5, 1])
        plt.yticks([-1, -0.5, 0, 0.5, 1])
        plt.colorbar()
        plt.savefig('u_StopStress2_{}.tiff'.format(ti), dpi=300)
        plt.show()
        print(max(np.sqrt(x_u[:, 0]**2 + y_v[:, 0]**2)))

    return None


if __name__ == '__main__':
    args = generate_args()

    for i in range(50):
        MODEL_EVALUATION = r'./model_files/Circle_Creep_{}'.format(str(i+1))
        net_u, net_v, pinn = load_network(args)
        growth_ratios = np.loadtxt('model_files/GrowthRatio.csv')
        post_process(net_u, net_v, pinn, xy_size=[args.x_size, args.y_size], ti=int(i+1), gr=growth_ratios)

