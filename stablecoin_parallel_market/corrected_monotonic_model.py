import numpy as np

# -----------------------------
# Core helper functions
# -----------------------------

def norm_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

# -----------------------------
# Parameters (baseline)
# -----------------------------

params = {
    "mu_theta": 0.55,
    "sigma_theta": 0.30,
    "sigma_x": 0.70,
    "sigma_s_bar": 0.85,
    "rho": 7.0,
    "alpha": 0.94,
    "delta": 0.18,
    "a_c": 0.03,
    "b_c": 0.10,
    "a_s": 0.015,
    "b_s": 0.06,
    "kappa_c": 0.16,
    "kappa_s": 0.04,
    "Delta": 0.40,
    "mu_choice": 0.18,
    "lambda_theta": 0.35,
    "tau_crisis": 0.10,
}

# -----------------------------
# Grids (overvaluation only)
# -----------------------------

theta_grid = np.linspace(0.0, 1.4, 71)
x_grid = np.linspace(-0.3, 1.9, 81)
s_grid = np.linspace(0.0, 1.4, 31)

# -----------------------------
# Cash-only equilibrium
# -----------------------------

def solve_cash():
    p = params
    dtheta = theta_grid[1] - theta_grid[0]
    dx = x_grid[1] - x_grid[0]

    prior = norm_pdf((theta_grid - p["mu_theta"]) / p["sigma_theta"]) / p["sigma_theta"]
    prior /= prior.sum() * dtheta

    fx_theta = np.array([
        norm_pdf((x_grid - th) / p["sigma_x"]) / p["sigma_x"]
        for th in theta_grid
    ])

    # posterior moments (linear-Gaussian)
    prec_theta = 1 / p["sigma_theta"]**2
    prec_x = 1 / p["sigma_x"]**2
    post_mean_x = (prec_theta * p["mu_theta"] + prec_x * x_grid) / (prec_theta + prec_x)

    # fixed-point iteration
    p_c = p["a_c"] + 0.02
    pi_x = np.full(len(x_grid), 0.1)

    for _ in range(100):
        U_D = -p["lambda_theta"] * post_mean_x - p["Delta"] * pi_x
        U_C = -p_c - p["kappa_c"]

        exp_D = np.exp(U_D / p["mu_choice"])
        exp_C = np.exp(U_C / p["mu_choice"])
        P_C = exp_C / (exp_C + exp_D)

        Qc_theta = fx_theta @ (P_C * dx)
        threshold = p["alpha"] - p["delta"] * theta_grid

        # smooth crisis mapping
        crisis = logistic((Qc_theta - threshold) / p["tau_crisis"])

        # update beliefs
        numer = prior[:, None] * fx_theta
        denom = numer.sum(axis=0) * dtheta
        posterior = numer / denom
        pi_new = (crisis[:, None] * posterior).sum(axis=0) * dtheta

        p_c = p["a_c"] + p["b_c"] * (Qc_theta @ prior * dtheta)
        pi_x = 0.8 * pi_x + 0.2 * pi_new

    return Qc_theta, crisis

# -----------------------------
# Run and report
# -----------------------------

if __name__ == "__main__":
    Qc, crisis = solve_cash()

    print("Cash demand (first 5):", Qc[:5])
    print("Crisis probability (first 5):", crisis[:5])

    print("Model runs successfully. Extend to full stablecoin model as needed.")
