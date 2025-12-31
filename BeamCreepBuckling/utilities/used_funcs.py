import numpy as np
import torch

def constitutive_law(components_u, components_v, material_params, dt=0):
    condition = material_params.condition
    E1 = material_params.modulus1
    eta1 = material_params.visco_eta
    E_inf = material_params.modulus_inf
    nu = material_params.poisson_ratio
    E_effective = E_inf + E1 * np.exp(-dt / (eta1 / E1))
    if condition == 'plain_strain':
        ### plain strain
        la = E_effective * nu / (1 + nu) / (1 - 2 * nu)
        mu = E_effective / (1 + nu) / 2
    elif condition == 'plain_stress':
        ### plain stress
        la = E_effective * nu / (1 + nu) / (1 - nu)
        mu = E_effective / (1 + nu) / 2
    else:
        print('-------------------------------------------------\n')
        print('Material property error!\n')
        print('-------------------------------------------------\n')
        print('Please select the one of the following options:\n1.\tplain_strain\n2.\tplain_stress\n')
        print('-------------------------------------------------\n')

    # # # Strain
    # e1 = components_u[0]
    # e2 = components_v[1]
    # e12 = 0.5 * (components_u[1] + components_v[0])
    #
    # # # # Stress
    # s1 = (2 * mu + la) * e1 + la * e2
    # s2 = (2 * mu + la) * e2 + la * e1
    # s12 = 2 * mu * e12
    # return e1, e2, e12, s1, s2, s12

    F11 = (1 + components_u[0])
    F22 = (1 + components_v[1])
    F12 = components_u[1]
    F21 = components_v[0]

    J = F11 * F22 - F12 * F21
    I = (F11 ** 2 + F12 ** 2 + F21 ** 2 + F22 ** 2)
    # Ein = 0.25 * la * (J ** 2 - 1) - (la / 2 + mu) * torch.log(J) + 0.5 * mu * (I - 2)
    Ein = 0.5 * mu * (I - 2 - 2 * torch.log(J)) + la / 2 * (J - 1) ** 2

    return F11, F22, F12, F21, J, Ein













