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
    e1 = components_u[0]
    e2 = components_v[1]
    e12 = 0.5 * (components_u[1] + components_v[0])

    # # # Stress
    s1 = (2 * nu + la) * e1 + la * e2
    s2 = (2 * nu + la) * e2 + la * e1
    s12 = 2 * nu * e12

    return e1, e2, e12, s1, s2, s12


def constitutive_law2(components_u, components_v, material_params, dt):
    condition = material_params.condition
    E1 = material_params.modulus1
    eta1 = material_params.visco_eta
    E_inf = material_params.modulus_inf
    pr = material_params.poisson_ratio
    E_effective = E_inf + E1 * np.exp(-dt / (eta1 / E1))
    if condition == 'plain_strain':
        ### plain strain
        la = E_effective * pr / (1 + pr) / (1 - 2 * pr)
        mu = E_effective / (1 + pr) / 2
    elif condition == 'plain_stress':
        ### plain stress
        la = E_effective * pr / (1 + pr) / (1 - pr)
        mu = E_effective / (1 + pr) / 2
    else:
        print('-------------------------------------------------\n')
        print('Material property error!\n')
        print('-------------------------------------------------\n')
        print('Please select the one of the following options:\n1.\tplain_strain\n2.\tplain_stress\n')
        print('-------------------------------------------------\n')

    F11 = (1 + components_u[1])
    F22 = (1 + components_v[2])
    F12 = components_u[2]
    F21 = components_v[1]

    BMatrix11 = F11**2+F12**2
    BMatrix12 = F11*F21+F12*F22
    BMatrix21 = F21*F11+F22*F12
    BMatrix22 = F21**2+F22**2

    J = F11 * F22 - F12 * F21
    I = (F11 ** 2 + F12 ** 2 + F21 ** 2 + F22 ** 2)

    # Ein = 0.25 * la * (J ** 2 - 1) - 0.5 * la * torch.log(J) + 0.5 * mu * (I - 2 - 2*torch.log(J))  # UJOption=0
    Ein = 0.5 * mu * (I - 2 - 2 * torch.log(torch.sqrt(J**2))) + la / 2 * (J - 1) ** 2  # UJOption=1
    # Ein = 0.5 * mu * (I - 2) - mu * torch.log(J) + la / 2 * torch.log(J)**2  # UJOption=2

    # # # # UJOption=0
    # S11 = 0.5 * la * (J - 1/J) + mu / J * (BMatrix11 - 1)
    # S22 = 0.5 * la * (J - 1/J) + mu / J * (BMatrix22 - 1)
    # # # UJOption=1
    S11 = la * (J - 1) + mu / J * (BMatrix11 - 1)
    S22 = la * (J - 1) + mu / J * (BMatrix22 - 1)
    # # # # UJOption=2
    # S11 = la * torch.log(J)/J + mu / J * (BMatrix11 - 1)
    # S22 = la * torch.log(J)/J + mu / J * (BMatrix22 - 1)

    return Ein, S11, S22













