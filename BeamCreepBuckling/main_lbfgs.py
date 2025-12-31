import datetime
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import torch

from pre_process import pre_process
from utilities.opt_lbfgs import L_BFGS_B
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

MODEL_PATH = r'model_files/'


def generate_args():
    # # Define parameters, including material parameters, mesh parameter, network parameters
    parser = ArgumentParser(description='Process some parameters.')
    # # # Network parameters
    parser.add_argument('--epoch_num', default=1, type=int, help='the training epoch')
    parser.add_argument('--num_input', default=2, type=int, help='the input number')
    parser.add_argument('--num_output', default=1, type=int, help='the output number')
    parser.add_argument('--net_layer', default=[[10, 10, 10], [10, 10, 10]], help='the layer of network')

    # # # mesh parameters
    parser.add_argument('--x_size', default=10.0, type=float, help='the sample size, x-direction')
    parser.add_argument('--y_size', default=1.0, type=float, help='the sample size, y-direction')
    parser.add_argument('--dx_interval', default=0.1, type=float, help='the discretization interval of the x-direction')
    parser.add_argument('--dy_interval', default=0.1, type=float, help='the discretization interval of the y-direction')
    parser.add_argument('--x_num', default=501, type=int, help='point number at the x-direction')
    parser.add_argument('--y_num', default=51, type=int, help='point number at the y-direction')

    # # # material parameters
    parser.add_argument('--modulus1', default=6.0, type=float, help='the maxwell series Young\'s modulus')
    parser.add_argument('--modulus_inf', default=4.0, type=float, help='the steady state Young\'s modulus')
    parser.add_argument('--visco_eta', default=18.0, type=float, help='the viscosity parameter')
    parser.add_argument('--poisson_ratio', default=0.35, type=float, help='the Poisson ratio')
    parser.add_argument('--delta_time', default=2, type=float, help='the increment time')

    parser.add_argument('--condition', default='plain_strain', help='the plain condition')

    packaged_args = parser.parse_args()

    return packaged_args


if __name__ == '__main__':
    print("Start at ", datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
    net_name = MODEL_PATH + 'LBFGS_Beam_HE_10-1_' + datetime.datetime.now().strftime('%Y-%m-%d-%H')

    args = generate_args()

    # pre process
    net_u, net_v, pinn, x_train, y_train = pre_process(args)

    # # # pre-training
    # x_train_0 = x_train
    # y_train_0 = [torch.zeros_like(y_train[0]), y_train[1]]
    # # y_train_0 = [torch.zeros_like(y_train[0]),]
    # opti = Adam(pinn, x_train, y_train_0, 0.0, learning_rate=1e-2, maxiter=500)
    # _, _ = opti.fit()

    # train
    time_start = datetime.datetime.now()

    # # # # relax out of the network
    relaxation_time = 0.0
    for si, relaxation_time in enumerate(np.arange(0, 12, 0.3)):
        print('Time is ', relaxation_time)
        if si == 0:
            iter_num = 2000
        else:
            iter_num = 2000
        opti = L_BFGS_B(pinn, x_train, y_train, relaxation_time, visco=True)
        hist, his_loss = opti.fit()
        torch.save(pinn.state_dict(), net_name+'_'+str(si+1))

        # # plt loss
        fig6 = plt.figure(6, figsize=(3, 3), dpi=300)
        plt.plot(his_loss[0], color='r')
        plt.plot(his_loss[1], color='b')
        plt.plot(his_loss[0] + his_loss[1], color='k')
        plt.yscale('log')
        plt.xlabel('Iteration', fontdict={'fontname': 'Arial'})
        plt.ylabel('Loss', fontdict={'fontname': 'Arial'})
        plt.title('Loss History', fontdict={'fontname': 'Arial'})
        plt.legend(['$L_{ge}$', '$L_{bc}$', 'L'])
        plt.savefig('hist_loss_{}.tiff'.format(si+1), dpi=300, bbox_inches='tight')
        plt.close()

    torch.save(pinn.state_dict(), net_name)

    time_end = datetime.datetime.now()

    T = time_end - time_start

    print('*************************************************\n')
    print('Time end is ', time_end, 'Time cost is', T, 's')
    print('*************************************************\n')
