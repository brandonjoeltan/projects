import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Revised hedge-run model with structural welfare
# FX demand = hedge demand + run demand
# Welfare is expected utility over normal vs crisis states.

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

theta_grid = np.linspace(0.0, 1.4, 61)

base = {
    "sigma_theta": 0.30,
    "sigma_x": 0.70,
    "sigma_s_bar": 0.90,
    "rho": 8.0,
    "alpha": 0.82,
    "delta": 0.10,
    "tau_crisis": 0.09,
    "chi_h": 0.95,
    "eta_h": 0.70,
    "gamma_r": 0.24,
    "pi0_r": 0.42,
    "tau_r": 0.10,
    "psi_info": 0.70,
    "a_c": 0.04,
    "b_c": 0.08,
    "a_s": 0.02,
    "b_s": 0.05,
    "kappa_c": 0.14,
    "kappa_s": 0.04,
    "nu_share": 7.5,
}

def info_weight(Qs, p):
    prec_theta = 1.0 / p["sigma_theta"]**2
    prec_x = 1.0 / p["sigma_x"]**2
    sigma_s2 = p["sigma_s_bar"]**2 / (1.0 + p["rho"] * max(Qs, 0.0))
    prec_s = 1.0 / sigma_s2
    omega = prec_s / (prec_theta + prec_x + prec_s)
    precision = prec_theta + prec_x + prec_s
    return omega, precision

def solve_state(theta, p, allow_stablecoins=True, max_iter=300, damp=0.25, tol=1e-9):
    Qc, Qs, pi = 0.08, (0.06 if allow_stablecoins else 0.0), 0.12
    for _ in range(max_iter):
        omega_s, precision = info_weight(Qs if allow_stablecoins else 0.0, p)
        p_c = p["a_c"] + p["b_c"] * np.sqrt(max(Qc, 0.0))
        if allow_stablecoins:
            p_s = p["a_s"] + p["b_s"] * np.sqrt(max(Qs, 0.0))
            share_s = logistic(p["nu_share"] * ((p_c + p["kappa_c"]) - (p_s + p["kappa_s"])) + 1.2 * omega_s)
        else:
            p_s = 1e6
            share_s = 0.0
        share_c = 1.0 - share_s
        eff_cost = share_c * (p_c + p["kappa_c"]) + share_s * (p_s + p["kappa_s"])

        q_h = max(0.0, p["chi_h"] * theta - p["eta_h"] * eff_cost)
        q_r = p["gamma_r"] * (1.0 + p["psi_info"] * omega_s) * logistic((pi - p["pi0_r"]) / p["tau_r"])
        Qfx_new = q_h + q_r
        Qs_new = share_s * Qfx_new
        Qc_new = share_c * Qfx_new
        threshold = p["alpha"] - p["delta"] * theta
        pi_new = logistic((Qfx_new - threshold) / p["tau_crisis"])
        diff = max(abs(Qc_new-Qc), abs(Qs_new-Qs), abs(pi_new-pi))
        Qc = (1-damp)*Qc + damp*Qc_new
        Qs = (1-damp)*Qs + damp*Qs_new
        pi = (1-damp)*pi + damp*pi_new
        if diff < tol:
            break
    omega_s, precision = info_weight(Qs if allow_stablecoins else 0.0, p)
    p_c = p["a_c"] + p["b_c"] * np.sqrt(max(Qc, 0.0))
    if allow_stablecoins:
        p_s = p["a_s"] + p["b_s"] * np.sqrt(max(Qs, 0.0))
        share_s = logistic(p["nu_share"] * ((p_c + p["kappa_c"]) - (p_s + p["kappa_s"])) + 1.2 * omega_s)
    else:
        p_s = 1e6
        share_s = 0.0
    share_c = 1.0 - share_s
    eff_cost = share_c * (p_c + p["kappa_c"]) + share_s * (p_s + p["kappa_s"])
    q_h = max(0.0, p["chi_h"] * theta - p["eta_h"] * eff_cost)
    q_r = p["gamma_r"] * (1.0 + p["psi_info"] * omega_s) * logistic((pi - p["pi0_r"]) / p["tau_r"])
    Qfx = q_h + q_r
    return {
        "Qc": Qc, "Qs": Qs, "Qfx": Qfx, "pi": pi, "omega_s": omega_s,
        "precision": precision, "eff_cost": eff_cost, "q_h": q_h, "q_r": q_r
    }

def profiles(p):
    cash = {k: [] for k in ["Qc","Qs","Qfx","pi","omega_s","precision","eff_cost","q_h","q_r"]}
    full = {k: [] for k in ["Qc","Qs","Qfx","pi","omega_s","precision","eff_cost","q_h","q_r"]}
    for th in theta_grid:
        rc = solve_state(th, p, False)
        rf = solve_state(th, p, True)
        for k in cash:
            cash[k].append(rc[k]); full[k].append(rf[k])
    cash = {k: np.array(v) for k,v in cash.items()}
    full = {k: np.array(v) for k,v in full.items()}
    return cash, full

def welfare(theta, st, zeta, lam_loss, xi_coord):
    c_normal = 1.0 + zeta * np.log1p(st["q_h"] + 0.25 * st["q_r"]) - st["eff_cost"] * st["Qfx"]
    crisis_loss = lam_loss * theta * (1.0 + xi_coord * st["omega_s"]) * (1.0 + st["q_r"])
    c_crisis = max(c_normal - crisis_loss, 1e-5)
    c_normal = max(c_normal, 1e-5)
    return (1.0 - st["pi"]) * np.log(c_normal) + st["pi"] * np.log(c_crisis)

def run(output_dir="../../output"):
    p = base.copy()
    zeta, lam_loss, xi_coord = 0.8, 2.5, 1.5
    cash, full = profiles(p)
    welfare_diff = np.array([
        welfare(th, {k: full[k][i] for k in full}, zeta, lam_loss, xi_coord) -
        welfare(th, {k: cash[k][i] for k in cash}, zeta, lam_loss, xi_coord)
        for i, th in enumerate(theta_grid)
    ])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        "metric": [
            "avg crisis probability cash-only",
            "avg crisis probability with stablecoins",
            "share monotone increments in stablecoin demand",
            "share monotone increments in total FX demand",
            "max welfare diff in moderate states",
            "min welfare diff in high-overvaluation states",
        ],
        "value": [
            float(np.mean(cash["pi"])),
            float(np.mean(full["pi"])),
            float(np.mean(np.diff(full["Qs"]) >= -1e-10)),
            float(np.mean(np.diff(full["Qfx"]) >= -1e-10)),
            float(np.max(welfare_diff[:20])),
            float(np.min(welfare_diff[-15:])),
        ]
    })
    summary.to_csv(output_dir / "summary.csv", index=False)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, full["Qs"], linewidth=2, label="Stablecoin demand")
    plt.plot(theta_grid, full["Qc"], linewidth=2, label="Cash demand")
    plt.plot(theta_grid, full["Qfx"], linewidth=2, label="Total FX demand")
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Demand")
    plt.title("Hedge-run model: FX demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_fx_demand.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, cash["pi"], linewidth=2, label="Cash-only")
    plt.plot(theta_grid, full["pi"], linewidth=2, label="With stablecoins")
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Crisis probability")
    plt.title("Hedge-run model: crisis probabilities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_crisis.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, welfare_diff, linewidth=2)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Welfare difference")
    plt.title("Hedge-run model: welfare reversal")
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_welfare.png", dpi=220)
    plt.close()

if __name__ == "__main__":
    run()
