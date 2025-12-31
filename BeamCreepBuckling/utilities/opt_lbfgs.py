import torch
import numpy as np
import scipy
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class L_BFGS_B:

    def __init__(self, pinn, x_train, y_train, dt=0, visco=False, factr=10, pgtol=1e-10, m=50, maxls=50, maxfun=40000):

        self.pinn = pinn
        self.x_train = [x for x in x_train]
        self.y_train = [y for y in y_train]
        self.factr = factr
        self.pgtol = pgtol
        self.m = m
        self.maxls = maxls
        self.maxfun = maxfun
        self.metrics = ['loss']
        self.iter = 0
        self.his_loss_ge = []
        self.his_loss_bc = []

        self.visco_bool = visco
        if visco:
            self.dt = dt
        else:
            self.dt = 0

    def pi_loss(self, weights):

        self.set_weights(weights)

        loss, grads, l1, l2 = self.loss_grad(self.x_train, self.y_train)

        self.iter = self.iter + 1.

        if self.iter % 10 == 0:
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

        print('Optimizer: L-BFGS-B')
        print('Initializing ...')
        result = scipy.optimize.fmin_l_bfgs_b(func=self.pi_loss, x0=initial_weights_np,
            factr=self.factr, pgtol=self.pgtol, m=self.m, maxls=self.maxls, maxfun=self.maxfun)

        return result, [np.array(self.his_loss_ge), np.array(self.his_loss_bc)]


def energy_loss(pred, ground):
    ### Internal potential energy
    l1 = torch.sum(pred[5] * ground[1])
    ### Potential energy of the external force
    l2 = torch.sum(pred[6] * ground[0]) * 1/101

    ### Final loss
    loss = l1 - l2

    return loss, l1, l2
