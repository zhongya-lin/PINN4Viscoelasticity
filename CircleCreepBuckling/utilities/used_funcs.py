import numpy as np
import torch


def constitutive_law(components_u, components_v, coord_r, material_params, dt):
    condition = material_params.condition
    E1 = material_params.modulus1
    eta1 = material_params.visco_eta
    E_inf = material_params.modulus_inf
    nu = material_params.poisson_ratio
    E_effective = E_inf + E1 * np.exp(-dt / (eta1 / E1))
    if condition == 'plain_strain':
        ### plain strain
        la = E_effective * nu / (1 + nu) / (1 - 2 * nu)
        nu = E_effective / (1 + nu) / 2
    elif condition == 'plain_stress':
        ### plain stress
        la = E_effective * nu / (1 + nu) / (1 - nu)
        nu = E_effective / (1 + nu) / 2
    else:
        print('-------------------------------------------------\n')
        print('Material property error!\n')
        print('-------------------------------------------------\n')
        print('Please select the one of the following options:\n1.\tplain_strain\n2.\tplain_stress\n')
        print('-------------------------------------------------\n')

    # # # Strain
    er = components_u[1]
    et = components_u[0] / coord_r + 1/coord_r * components_v[2]
    ert = 1/coord_r * components_u[2] + components_v[1] - components_v[0] / coord_r

    # # # Stress
    sr = (2 * nu + la) * er + la * et
    st = (2 * nu + la) * et + la * er
    srt = 2 * nu * ert

    return er, et, ert, sr, st, srt


def constitutive_law2(components_u, components_v, coord_r, material_params, dt):
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

    F11 = 1 + components_u[1]
    F22 = 1 + components_u[0] / coord_r + 1/coord_r * components_v[2]
    F12 = components_u[2] / coord_r - components_v[0] / coord_r
    F21 = components_v[1]

    J = F11 * F22 - F12 * F21
    I = (F11 ** 2 + F12 ** 2 + F21 ** 2 + F22 ** 2)
    # Ein = 0.25 * la * (J ** 2 - 1) - 0.5 * la * torch.log(J) + 0.5 * mu * (I - 2 - 2*torch.log(J))  # UJOption=0
    Ein = 0.5 * mu * (I - 2 - 2 * torch.log(J)) + la / 2 * (J - 1) ** 2  # UJOption=1
    # Ein = 0.5 * mu * (I - 2) - mu * torch.log(J) + la / 2 * torch.log(J)**2  # UJOption=2

    return Ein











