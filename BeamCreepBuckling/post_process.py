import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utilities.generate_net import GenerateNet
from utilities.PINNs import PINN
from main import generate_args


XY_MAT_FILE = './data_files/Coord_Beam_10_1.mat'


def load_network(args):
    inp_num = args.num_input
    out_num = args.num_output
    layers = args.net_layer

    net_u = GenerateNet(inp_num, layers[0], out_num)
    net_v = GenerateNet(inp_num, layers[1], out_num)

    pinn = PINN(net_u, net_v, args)

    pinn.load_state_dict(torch.load(MODEL_EVALUATION, weights_only=True))

    return net_u, net_v, pinn


def post_process(net_u, net_v, pinn, xy_size, dxy=0.1, ti=0):
    xy_data = scipy.io.loadmat(XY_MAT_FILE)
    xy = xy_data['x'].astype(np.float32)

    xy_tensor = torch.from_numpy(xy)
    x, y = (xy_tensor[..., i, torch.newaxis] for i in range(xy_tensor.shape[-1]))

    u = net_u(xy_tensor) * x * 1e-3
    v = net_v(xy_tensor) * x * 1e-3

    u = u.detach().numpy()
    v = v.detach().numpy()

    # # plot figure for displacement v
    fig2 = plt.figure(2, figsize=(8, 2))
    plt.scatter(xy[:, 0]+u[:,0], xy[:, 1]+v[:,0], s=10, c=v, cmap='jet')#, vmin=0.9, vmax=1.28)
    plt.axis('equal')
    plt.colorbar()
    ttl = 'v-' + str(ti)
    plt.title(ttl)
    # if ti%3==0:
    #     plt.savefig('v_elastic_1e-3_{}.tiff'.format(ti), dpi=300)
    # plt.close()
    print(np.max(v))

    return None


if __name__ == '__main__':
    args = generate_args()

    for i in range(22):
        MODEL_EVALUATION = r'./model_files/Beam_VE_10-1_{}'.format(str(i+1))
        net_u, net_v, pinn = load_network(args)
        post_process(net_u, net_v, pinn, xy_size=[args.x_size, args.y_size], dxy=0.02, ti=int(i+1))

