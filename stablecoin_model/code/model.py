# Core hedge-run model structure

import numpy as np

def hedge_demand(theta, cost, chi_h, eta_h):
    return max(0.0, chi_h * theta - eta_h * cost)

def run_demand(pi, omega_s, gamma_r, psi_info, pi0, tau):
    return gamma_r * (1 + psi_info * omega_s) * (1 / (1 + np.exp(-(pi - pi0)/tau)))
