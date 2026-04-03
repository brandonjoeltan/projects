import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from math import erf

# Exact global-game hedge-run prototype
# Improvements relative to earlier prototype:
# 1. The marginal runner uses exact posterior integration over theta.
# 2. State-theta objects integrate over public-signal realizations s.
# 3. Welfare uses the same state-contingent structure used to motivate run incentives.


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def norm_pdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def norm_cdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / (sigma * np.sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


theta_grid = np.linspace(0.0, 1.4, 41)
s_grid_base = np.linspace(-0.2, 1.6, 31)
x_grid = np.linspace(-0.8, 2.2, 121)

BASE = {
    "mu_theta": 0.5,
    "sigma_theta": 0.35,
    "sigma_x": 0.70,
    "sigma_s_bar": 0.90,
    "rho": 8.0,
    "alpha": 0.82,
    "delta": 0.10,
    "chi_h": 0.95,
    "eta_h": 0.70,
    "rbar": 0.22,
    "B": 0.06,
    "kappa_r": 0.12,
    "lambda_loss": 2.5,
    "xi_coord": 1.5,
    "a_c": 0.04,
    "b_c": 0.08,
    "a_s": 0.02,
    "b_s": 0.05,
    "kappa_c": 0.14,
    "kappa_s": 0.04,
    "nu_share": 7.5,
    "phi_share": 1.2,
    "zeta": 0.8,
    "run_weight_normal": 0.25,
}


def info_weight(Qs, p):
    prec_theta = 1.0 / p["sigma_theta"] ** 2
    prec_x = 1.0 / p["sigma_x"] ** 2
    sigma_s2 = p["sigma_s_bar"] ** 2 / (1.0 + p["rho"] * max(Qs, 0.0))
    prec_s = 1.0 / sigma_s2
    omega0 = prec_theta / (prec_theta + prec_x + prec_s)
    omegax = prec_x / (prec_theta + prec_x + prec_s)
    omegas = prec_s / (prec_theta + prec_x + prec_s)
    return omega0, omegax, omegas, sigma_s2


def prices(Qc, Qs, p):
    p_c = p["a_c"] + p["b_c"] * np.sqrt(max(Qc, 0.0))
    p_s = p["a_s"] + p["b_s"] * np.sqrt(max(Qs, 0.0))
    c_c = p_c + p["kappa_c"]
    c_s = p_s + p["kappa_s"]
    return p_c, p_s, c_c, c_s


def posterior_weights_theta(xstar, s, Qs, p):
    _, _, _, sigma_s2 = info_weight(Qs, p)
    prior = norm_pdf(theta_grid, p["mu_theta"], p["sigma_theta"])
    like_x = norm_pdf(xstar, theta_grid, p["sigma_x"])
    like_s = norm_pdf(s, theta_grid, np.sqrt(sigma_s2))
    w = prior * like_x * like_s
    w = np.maximum(w, 1e-300)
    w /= np.sum(w)
    return w


def aggregate_hedge(theta, s, Qs, Qc, allow_stablecoins, p):
    dx = x_grid[1] - x_grid[0]
    fx = norm_pdf(x_grid, theta, p["sigma_x"])
    fx /= np.sum(fx) * dx

    _, _, c_c, c_s = prices(Qc, Qs, p)
    omega0, omegax, omegas, sigma_s2 = info_weight(Qs if allow_stablecoins else 0.0, p)
    omega_s = omegas if allow_stablecoins else 0.0
    if allow_stablecoins:
        share_s = logistic(p["nu_share"] * (c_c - c_s) + p["phi_share"] * omega_s)
    else:
        share_s = 0.0
    eff_cost = (1.0 - share_s) * c_c + share_s * c_s
    m = omega0 * p["mu_theta"] + omegax * x_grid + omegas * s
    q_h = np.maximum(0.0, p["chi_h"] * m - p["eta_h"] * eff_cost)
    return float(np.sum(q_h * fx) * dx), share_s, eff_cost, omega_s, sigma_s2


def expected_marginal_gain(xstar, s, Qs, Qc, allow_stablecoins, p):
    # Exact posterior integration for the marginal runner.
    w = posterior_weights_theta(xstar, s, Qs if allow_stablecoins else 0.0, p)
    value = p["B"] - p["kappa_r"]
    for theta, wt in zip(theta_grid, w):
        q_h, _, _, omega_s, _ = aggregate_hedge(theta, s, Qs, Qc, allow_stablecoins, p)
        q_run = p["rbar"] * (1.0 - norm_cdf(xstar, theta, p["sigma_x"]))
        q_fx = q_h + q_run
        crisis = 1.0 if q_fx >= (p["alpha"] - p["delta"] * theta) else 0.0
        loss = p["lambda_loss"] * theta * (1.0 + p["xi_coord"] * omega_s)
        value += wt * crisis * loss
    return value


def solve_cutoff_exact(s, Qs, Qc, allow_stablecoins, p):
    lo, hi = -1.0, 2.5

    def H(x):
        return expected_marginal_gain(x, s, Qs, Qc, allow_stablecoins, p)

    h_lo = H(lo)
    h_hi = H(hi)
    if h_lo >= 0.0 and h_hi >= 0.0:
        return lo
    if h_lo <= 0.0 and h_hi <= 0.0:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        h_mid = H(mid)
        if h_mid >= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def solve_state(theta, s, allow_stablecoins=True, p=None, max_iter=180, damp=0.25, tol=1e-8):
    if p is None:
        p = BASE
    Qc = 0.10
    Qs = 0.06 if allow_stablecoins else 0.0
    xstar = 0.5

    for _ in range(max_iter):
        if not allow_stablecoins:
            Qs = 0.0
        xstar_new = solve_cutoff_exact(s, Qs, Qc, allow_stablecoins, p)
        q_h, share_s, eff_cost, omega_s, sigma_s2 = aggregate_hedge(theta, s, Qs, Qc, allow_stablecoins, p)
        q_run = p["rbar"] * (1.0 - norm_cdf(xstar_new, theta, p["sigma_x"]))
        q_fx = q_h + q_run
        Qs_new = share_s * q_fx if allow_stablecoins else 0.0
        Qc_new = q_fx - Qs_new
        diff = max(abs(Qs_new - Qs), abs(Qc_new - Qc), abs(xstar_new - xstar))
        Qs = (1.0 - damp) * Qs + damp * Qs_new
        Qc = (1.0 - damp) * Qc + damp * Qc_new
        xstar = (1.0 - damp) * xstar + damp * xstar_new
        if diff < tol:
            break

    q_h, share_s, eff_cost, omega_s, sigma_s2 = aggregate_hedge(theta, s, Qs, Qc, allow_stablecoins, p)
    q_run = p["rbar"] * (1.0 - norm_cdf(xstar, theta, p["sigma_x"]))
    q_fx = q_h + q_run
    crisis = 1.0 if q_fx >= (p["alpha"] - p["delta"] * theta) else 0.0

    c_normal = 1.0 + p["zeta"] * np.log1p(q_h + p["run_weight_normal"] * q_run) - eff_cost * q_fx
    crisis_loss = p["lambda_loss"] * theta * (1.0 + p["xi_coord"] * omega_s) * (1.0 + q_run)
    c_crisis = max(c_normal - crisis_loss, 1e-6)
    c_normal = max(c_normal, 1e-6)
    welfare = (1.0 - crisis) * np.log(c_normal) + crisis * np.log(c_crisis)

    return {
        "Qc": Qc,
        "Qs": Qs,
        "Qfx": q_fx,
        "Qhedge": q_h,
        "Qrun": q_run,
        "xstar": xstar,
        "share_s": share_s,
        "omega_s": omega_s,
        "sigma_s2": sigma_s2,
        "crisis": crisis,
        "welfare": welfare,
    }


def integrate_over_public_signal(theta, allow_stablecoins=True, p=None):
    if p is None:
        p = BASE
    # First pass: baseline signal variance for weights.
    sigma_s = p["sigma_s_bar"]
    fs = norm_pdf(s_grid_base, theta, sigma_s)
    fs /= np.sum(fs)

    states = [solve_state(theta, s, allow_stablecoins, p) for s in s_grid_base]

    # Second pass: update weights using state-dependent public-signal variance.
    sigmas = np.array([np.sqrt(st["sigma_s2"]) for st in states])
    fs2 = norm_pdf(s_grid_base, theta, sigmas)
    fs2 = np.maximum(fs2, 1e-300)
    fs2 /= np.sum(fs2)

    keys = states[0].keys()
    out = {}
    for k in keys:
        out[k] = float(np.sum([w * st[k] for w, st in zip(fs2, states)]))
    return out


def profiles(p=None):
    if p is None:
        p = BASE
    out_cash = {k: [] for k in ["Qc", "Qs", "Qfx", "Qhedge", "Qrun", "xstar", "share_s", "omega_s", "sigma_s2", "crisis", "welfare"]}
    out_full = {k: [] for k in ["Qc", "Qs", "Qfx", "Qhedge", "Qrun", "xstar", "share_s", "omega_s", "sigma_s2", "crisis", "welfare"]}
    for theta in theta_grid:
        rc = integrate_over_public_signal(theta, False, p)
        rf = integrate_over_public_signal(theta, True, p)
        for k in out_cash:
            out_cash[k].append(rc[k])
            out_full[k].append(rf[k])
    out_cash = {k: np.array(v) for k, v in out_cash.items()}
    out_full = {k: np.array(v) for k, v in out_full.items()}
    return out_cash, out_full


def run(output_dir="../../output_global_game_exact"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cash, full = profiles(BASE)
    welfare_diff = full["welfare"] - cash["welfare"]

    summary = pd.DataFrame({
        "metric": [
            "avg crisis cash-only",
            "avg crisis stablecoins",
            "monotone stablecoin demand",
            "monotone total FX demand",
            "max welfare gain low-mid theta",
            "min welfare diff high theta",
        ],
        "value": [
            float(np.mean(cash["crisis"])),
            float(np.mean(full["crisis"])),
            float(np.mean(np.diff(full["Qs"]) >= -1e-10)),
            float(np.mean(np.diff(full["Qfx"]) >= -1e-10)),
            float(np.max(welfare_diff[:16])),
            float(np.min(welfare_diff[-12:])),
        ],
    })
    summary.to_csv(output_dir / "summary_global_game_exact.csv", index=False)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, full["Qs"], linewidth=2, label="Stablecoin demand")
    plt.plot(theta_grid, full["Qc"], linewidth=2, label="Cash demand")
    plt.plot(theta_grid, full["Qfx"], linewidth=2, label="Total FX demand")
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Demand")
    plt.title("Exact global-game hedge-run model: FX demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_fx_demand_global_game_exact.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, cash["crisis"], linewidth=2, label="Cash-only")
    plt.plot(theta_grid, full["crisis"], linewidth=2, label="With stablecoins")
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Crisis probability")
    plt.title("Exact global-game hedge-run model: crises")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_crisis_global_game_exact.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, welfare_diff, linewidth=2)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Welfare difference")
    plt.title("Exact global-game hedge-run model: welfare")
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_welfare_global_game_exact.png", dpi=220)
    plt.close()

    return summary


if __name__ == "__main__":
    run()
