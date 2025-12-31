import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()


def constitutive_law_growth(components_u, components_v, coord_r, material_params, dt, delta_t, growth_r):
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

    # # # growth related parameter
    nutrient_density = material_params.nutrient_density
    stop_stress = material_params.stop_stress

    F11 = 1 + components_u[1]
    F22 = 1 + components_u[0] / coord_r + 1/coord_r * components_v[2]
    F12 = components_u[2] / coord_r - components_v[0] / coord_r
    F21 = components_v[1]

    F11 = F11/growth_r
    F22 = F22/growth_r
    F21 = F21/growth_r
    F12 = F12/growth_r

    J = F11 * F22 - F12 * F21

    # # Use left Cauchy-Green tensor
    BMatrix11 = (F11**2+F12**2) #/ J**(2/3)
    BMatrix12 = (F11*F21+F12*F22) #/ J**(2/3)
    BMatrix21 = (F21*F11+F22*F12) #/ J**(2/3)
    BMatrix22 = (F21**2+F22**2) #/ J**(2/3)
    I = BMatrix11 + BMatrix22
    # # Ein = 0.25 * la * (J ** 2 - 1) - 0.5 * la * torch.log(J) + 0.5 * mu * (I - 2 - 2*torch.log(J))  # UJOption=0
    Ein = 0.5 * mu * (I - 2 - 2 * torch.log(J)) + la / 2 * (J - 1) ** 2  # UJOption=1 using this
    # # Ein = 0.5 * mu * (I - 2) - mu * torch.log(J) + la / 2 * torch.log(J)**2  # UJOption=2

    # # # # UJOption=0
    # S11 = 0.5 * la * (J - 1/J) + mu / J * (BMatrix11 - 1)
    # S22 = 0.5 * la * (J - 1/J) + mu / J * (BMatrix22 - 1)
    # # # UJOption=1
    S11 = la * (J - 1) + mu / J * (BMatrix11 - 1)
    S22 = la * (J - 1) + mu / J * (BMatrix22 - 1)
    # # # # UJOption=2
    # # S11 = la * torch.log(J)/J + mu / J * (BMatrix11 - 1)
    # # S22 = la * torch.log(J)/J + mu / J * (BMatrix22 - 1)

    # # # mean for all
    growth_stress = (S11+S22).mean() / 10.0 + stop_stress / 10.0  # # dimensionless by E_zero

    growth_ratio = growth_r * torch.exp(nutrient_density * growth_stress * delta_t)

    Egrowth = 2 * (growth_ratio-1.0) * stop_stress * (growth_ratio-1.0)**2 # current configuration

    return Ein, Egrowth, growth_ratio







