import torch
import numpy as np
import os
import matplotlib.pyplot as plt

class Adam:
    def __init__(self, pinn, x_train, y_train, dt, learning_rate=0.001, maxiter=1500, visco=False, net_name='model_name'):

        self.pinn = pinn
        self.x_train = [x for x in x_train]
        self.y_train = [y for y in y_train]
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.metrics = ['loss']
        self.iter = 0
        self.his_loss_ge = []
        self.his_loss_bc = []

        self.visco_bool = visco
        self.net_name = net_name
        self.dt = dt
        self.each_iter = 2000

    def pi_loss(self, weights):

        self.set_weights(weights)

        loss, grads, l1, l2 = self.loss_grad(self.x_train, self.y_train)

        self.iter = self.iter + 1.

        if self.iter % 100 == 0:
            print('Iter: %d   L1 = %.4g   L2 = %.4g, L = %.4g' % (self.iter, l1.item(), l2.item(), loss.item()))

        loss = loss.detach().numpy().astype('float64')
        grads = np.concatenate([g.detach().numpy().flatten() for g in grads]).astype('float64')

        self.his_loss_ge.append(l1.detach().numpy())
        self.his_loss_bc.append(l2.detach().numpy())

        return loss, grads

    def loss_grad(self, x, y):
        y_p = self.pinn(x, self.dt)

        loss, l1, l2 = energy_loss(y_p, y)

        grads = torch.autograd.grad(loss, self.pinn.parameters())

        return loss, grads, l1, l2

    def set_weights(self, flat_weights):
        shapes = [w.shape for w in list(self.pinn.parameters())]
        split_ids = np.cumsum([np.prod(shape) for shape in [0] + shapes])

        weights = [flat_weights[from_id:to_id].reshape(shape)
                   for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes)]

        for w, weight in zip(list(self.pinn.parameters()), weights):
            w.data = torch.from_numpy(weight.astype('float32'))

        return None

    def fit(self):
        initial_weights = torch.concatenate([w.flatten() for w in list(self.pinn.parameters())])
        initial_weights_np = initial_weights.detach().numpy().astype('float64')

        print('Optimizer: Adam')
        print('Initializing ...')

        beta1 = 0.9
        beta2 = 0.999
        learning_rate = self.learning_rate
        eps = 1e-10
        zeta = 1e-15
        x0 = initial_weights_np
        x = x0
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        b_w = 0
        loss, g = self.pi_loss(x)
        for i in range(0, self.maxiter):
            if self.__converged1(g, zeta):
                print('Grad == 0, optimizer stop. Iteration steps: {}'.format(i))
                return loss, [np.array(self.his_loss_ge), np.array(self.his_loss_bc)]

            # # # # introduce relaxation, changes of modulus from time self.dt
            if self.visco_bool:
                if i % self.each_iter == 0:
                    self.dt = int(i/self.each_iter) * 1.0

            if (i+1) % self.each_iter == 0:
                print('The time is ', self.dt)
                torch.save(self.pinn.state_dict(), self.net_name+'_'+str(int((i+1)/self.each_iter)))


                ### plot figure for hist_loss
                fig6 = plt.figure(6, figsize=(3, 3), dpi=300)
                plt.plot(self.his_loss_ge, color='r')
                plt.plot(self.his_loss_bc, color='b')
                plt.plot(np.array(self.his_loss_ge) + np.array(self.his_loss_bc), color='k')
                plt.yscale('log')
                plt.xlabel('Iteration', fontdict={'fontname': 'Arial'})
                plt.ylabel('Loss', fontdict={'fontname': 'Arial'})
                plt.title('Loss History', fontdict={'fontname': 'Arial'})
                plt.legend(['$L_{ge}$', '$L_{bc}$', 'L'])
                plt.savefig('hist_loss_{}.tiff'.format(i+1), dpi=300, bbox_inches='tight')
                plt.close()

            m = (1 - beta1) * g + beta1 * m
            v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
            mhat = m / (1 - beta1**(i + 1))  # bias correction.
            vhat = v / (1 - beta2**(i + 1))
            x_new = x - learning_rate * mhat / (np.sqrt(vhat) + eps)
            loss_new, g_new = self.pi_loss(x_new)

            x, loss, g = x_new, loss_new, g_new

        return loss, [np.array(self.his_loss_ge), np.array(self.his_loss_bc)]

    def __converged1(self, grad, zeta):
        if np.linalg.norm(grad, ord=np.inf) < zeta:
            return True
        return False

    def __converged2(self, loss_delta, zeta):
        val = np.abs(loss_delta)
        if val < zeta:
            return True
        return False


def energy_loss(pred, ground):
    ### Internal potential energy
    l1 = torch.sum(pred[5] * ground[1])
    # # # Potential energy of the external force
    l2 = torch.sum(pred[6] * ground[0]) * 1/101

    # # # Final loss
    loss = l1 - l2

    return loss, l1, l2
