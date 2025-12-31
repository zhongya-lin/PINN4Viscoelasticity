import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utilities.generate_net import GenerateNet
from utilities.PINNs import PINN
from main import generate_args


def load_network(args):
    inp_num = args.num_input
    out_num = args.num_output
    layers = args.net_layer

    net_u = GenerateNet(inp_num, layers[0], out_num)
    net_v = GenerateNet(inp_num, layers[1], out_num)

    pinn = PINN(net_u, net_v, args)

    pinn.load_state_dict(torch.load(MODEL_EVALUATION, weights_only=True))

    return net_u, net_v, pinn


def post_process(net_u, net_v, pinn, xy_size, dxy=0.1, dt=7.0):
    x_length = xy_size[0]
    y_height = xy_size[1]

    x_points_num = int(x_length / dxy)
    y_points_num = int(y_height / dxy)

    xy = np.zeros((x_points_num * y_points_num, 2)).astype(np.float32)
    k = 0
    for i in range(0, x_points_num):
        for j in range(0, y_points_num):
            xy[k, 0] = i * x_length / (x_points_num - 1)
            xy[k, 1] = j * y_height / (y_points_num - 1)
            k = k + 1

    xy_tensor = torch.from_numpy(xy)

    u = net_u(xy_tensor) * xy_tensor[..., 0, torch.newaxis]
    v = net_v(xy_tensor) * xy_tensor[..., 0, torch.newaxis]
    temp = pinn([xy_tensor for i in range(0, 5)], dt)
    u = u.detach().numpy()
    v = v.detach().numpy()
    s11 = temp[2].detach().numpy()
    s22 = temp[3].detach().numpy()

    # # # plot figure for displacement u
    fig1 = plt.figure(1)
    plt.scatter(xy[:, 0]+u[:, 0], xy[:, 1]+v[:, 0], s=0.2, c=u, cmap='jet')  #, vmin=0, vmax=1.1)
    plt.axis('equal')
    plt.colorbar()
    plt.title('u')
    plt.savefig('u.eps', dpi=300)

    # # # plot figure for displacement v
    fig2 = plt.figure(2)
    plt.scatter(xy[:, 0]+u[:, 0], xy[:, 1]+v[:, 0], s=0.2, c=v, cmap='jet')  #, vmin=-0.1, vmax=0.1)
    plt.axis('equal')
    plt.colorbar()
    plt.title('v')
    plt.savefig('v.eps', dpi=300)
    plt.show()

    return None


MODEL_EVALUATION = r'./model_files/Creep_2D_2025-02-17-15_9'


if __name__ == '__main__':
    args = generate_args()
    net_u, net_v, pinn = load_network(args)
    post_process(net_u, net_v, pinn, xy_size=[args.x_size, args.y_size], dxy=0.05, dt=10.0)

