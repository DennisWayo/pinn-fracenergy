import os

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, r2_score


def ensure_fig_dir(fig_dir):
    os.makedirs(fig_dir, exist_ok=True)


def save_fig(fig, fig_dir, filename):
    ensure_fig_dir(fig_dir)
    fig.savefig(os.path.join(fig_dir, filename))
    plt.close(fig)


def configure_plot_style():
    try:
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats("png", "svg")
    except Exception:
        pass

    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _amplitude_calibrate(y_true, y_raw):
    scale = float(np.dot(y_true, y_raw) / (np.dot(y_raw, y_raw) + 1e-12))
    return scale * y_raw, scale


def analytical_baselines(data, num_points=500):
    L_line = np.linspace(0.1, 25.0, num_points)
    E_proxy = (data.P0 * data.W0 / data.mu0) * np.exp(-L_line / 5)

    E_rock = 25e9   # Pa
    nu = 0.25
    height = 10.0   # m; amplitude is calibrated, so this affects scale only.
    E_prime = E_rock / (1 - nu**2)

    # Treat pressure gradient as a length-scaled net pressure proxy.
    P_net_MPa = data.P0 * L_line
    P_net_Pa = P_net_MPa * 1e6
    a = L_line / 2.0

    w_kgd = (4 * (1 - nu**2) / E_rock) * P_net_Pa * a
    w_pkn = (4 * P_net_Pa * height) / E_prime

    E_kgd_raw = (P_net_MPa * w_kgd / data.mu0) * np.exp(-L_line / 5)
    E_pkn_raw = (P_net_MPa * w_pkn / data.mu0) * np.exp(-L_line / 5)

    E_kgd, kgd_scale = _amplitude_calibrate(E_proxy, E_kgd_raw)
    E_pkn, pkn_scale = _amplitude_calibrate(E_proxy, E_pkn_raw)

    return {
        "L": L_line,
        "proxy": E_proxy,
        "kgd": E_kgd,
        "pkn": E_pkn,
        "kgd_scale": kgd_scale,
        "pkn_scale": pkn_scale,
    }


def analytical_baseline_summary(data):
    baselines = analytical_baselines(data)
    rows = []
    for name, key in [
        ("KGD-inspired (simplified)", "kgd"),
        ("PKN-inspired (simplified)", "pkn"),
    ]:
        err = baselines["proxy"] - baselines[key]
        mse = float(np.mean(err ** 2))
        nrmse = float(np.sqrt(mse) / (np.std(baselines["proxy"]) + 1e-12))
        rows.append({
            "Baseline": name,
            "NRMSE vs. proxy": nrmse,
            "R2 vs. proxy": float(r2_score(baselines["proxy"], baselines[key])),
        })
    return rows


def plot_confusion_matrix(name, cm, threshold, fig_dir):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} Confusion Matrix (thr={threshold:.3f})")
    fig.tight_layout()
    save_fig(fig, fig_dir, f"confusion_{name}.png")


def plot_roc_curves(payloads, fig_dir):
    if not payloads:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for name, y_true_bin, y_score, _threshold, _cm in payloads:
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        auc = roc_auc_score(y_true_bin, y_score)
        ax.plot(fpr, tpr, linewidth=1.6, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, fig_dir, "roc_curves.png")


def plot_training_loss(hist_1D, hist_2D, hist_3D, fig_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(hist_1D.history["loss"], label="1D Train", color="#B2182B", linewidth=1.6)
    ax.semilogy(hist_1D.history["val_loss"], label="1D Val", color="#B2182B", linewidth=1.2, linestyle="--")
    ax.semilogy(hist_2D.history["loss"], label="2D Train", color="#2166AC", linewidth=1.6)
    ax.semilogy(hist_2D.history["val_loss"], label="2D Val", color="#2166AC", linewidth=1.2, linestyle="--")
    if hist_3D is not None:
        ax.semilogy(hist_3D.history["loss"], label="3D Train", color="#1B7837", linewidth=1.6)
        ax.semilogy(hist_3D.history["val_loss"], label="3D Val", color="#1B7837", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("PINN Training/Validation Loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, fig_dir, "trainloss.png")


def plot_comparison_line(data, models, fig_dir):
    # Compare predictions along L at fixed W0, P0, mu0
    L_line = np.linspace(0.1, 25.0, 300)
    W_line = np.full_like(L_line, data.W0)
    P_line = np.full_like(L_line, data.P0)
    mu_line = np.full_like(L_line, data.mu0)

    E_line_true = (data.P0 * data.W0 / data.mu0) * np.exp(-L_line / 5)

    X1_line = data.scaler_1D.transform(L_line.reshape(-1, 1))
    X2_line = data.scaler_2D.transform(np.column_stack([L_line, W_line]))
    X3_line = data.scaler_3D.transform(np.column_stack([L_line, W_line, P_line, np.log(mu_line)]))

    pred1_line = models.pinn_1D.predict(X1_line, verbose=0).flatten()
    pred2_line = models.pinn_2D.predict(X2_line, verbose=0).flatten()
    pred1_line = data.y_1D_scaler.inverse_transform(pred1_line.reshape(-1, 1)).ravel()
    pred2_line = data.y_2D_scaler.inverse_transform(pred2_line.reshape(-1, 1)).ravel()
    pred1_line = np.expm1(pred1_line)
    pred2_line = np.expm1(pred2_line)
    pred3_line = None
    if models.pinn_3D is not None:
        pred3_line_log = models.pinn_3D.predict(X3_line, verbose=0).flatten()
        pred3_line = np.expm1(pred3_line_log)

    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.plot(L_line, E_line_true, color="black", linewidth=1.5, label="True (synthetic)")
    ax.plot(L_line, pred1_line, color="#B2182B", linewidth=1.4, label="1D PINN")
    ax.plot(L_line, pred2_line, color="#2166AC", linewidth=1.4, label="2D PINN")
    if pred3_line is not None:
        ax.plot(L_line, pred3_line, color="#1B7837", linewidth=1.4, label="3D PINN")
    ax.set_xlabel("Fracture length L (m)")
    ax.set_ylabel("Energy balance E (a.u.)")
    ax.set_title("True vs. PINN Predictions (Fixed W, P, μ)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_fig(fig, fig_dir, "comparison.png")


def plot_fdm_instability(fig_dir):
    # FDM stability demo (explicit scheme for 1D heat equation)
    alpha = 1.0
    nx = 200
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]

    # Initial condition
    u0 = np.sin(np.pi * x)

    # Stable and unstable time steps
    r_stable = 0.4
    r_unstable = 0.6

    dt_stable = r_stable * dx**2 / alpha
    dt_unstable = r_unstable * dx**2 / alpha
    nt = 200

    def explicit_fd(u_init, dt, nt):
        u = u_init.copy()
        r = alpha * dt / dx**2
        for _ in range(nt):
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
            u = u_new
        return u

    u_stable = explicit_fd(u0, dt_stable, nt)
    u_unstable = explicit_fd(u0, dt_unstable, nt)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, u0, color="black", linewidth=1.2, label="Initial")
    ax.plot(x, u_stable, color="#2166AC", linewidth=1.4, label=f"Stable (r={r_stable})")
    ax.plot(x, u_unstable, color="#B2182B", linewidth=1.4, label=f"Unstable (r={r_unstable})")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("FDM Stability (Explicit Scheme)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, fig_dir, "fdm_instability.png")


def plot_kgd_baseline(data, fig_dir):
    # Analytical trend baselines under simplifying assumptions.
    baselines = analytical_baselines(data)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(baselines["L"], baselines["proxy"], color="black", linewidth=1.5,
            label="Synthetic (fixed W,P,μ)")
    ax.plot(baselines["L"], baselines["kgd"], color="#762A83", linewidth=1.4,
            label="KGD baseline (calibrated)")
    ax.plot(baselines["L"], baselines["pkn"], color="#C99400", linewidth=1.4,
            linestyle="--", label="PKN baseline (calibrated)")
    ax.set_xlabel("Fracture length L (m)")
    ax.set_ylabel("Energy balance E (a.u.)")
    ax.set_title("Synthetic vs. KGD/PKN Baselines")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, fig_dir, "kgd_baseline.png")


def plot_2d_contour(data, models, fig_dir):
    # 2D contour plot (L, W) at fixed P0 and mu0
    num_grid_x = 60
    num_grid_y = 60

    L_vals = np.linspace(0.1, 25.0, num_grid_x)
    W_vals = np.linspace(0.01, 2.5, num_grid_y)
    Xg, Yg = np.meshgrid(L_vals, W_vals)

    grid = np.column_stack([
        Xg.ravel(),
        Yg.ravel()
    ])

    grid_scaled = data.scaler_2D.transform(grid)
    Z_s = models.pinn_2D.predict(grid_scaled, verbose=0).flatten()
    Z_log = data.y_2D_scaler.inverse_transform(Z_s.reshape(-1, 1)).ravel()
    Z = np.expm1(Z_log).reshape(Yg.shape)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    contour = ax.contourf(Xg, Yg, Z, levels=24, cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label("Energy balance E (a.u.)")
    ax.set_xlabel("Fracture length L (m)")
    ax.set_ylabel("Fracture width W (m)")
    ax.set_title("2D PINN Prediction at Fixed P, μ")
    fig.tight_layout()
    save_fig(fig, fig_dir, "contour_map.png")


def plot_3d_surface(data, models, fig_dir):
    if models.pinn_3D is None:
        return
    # 3D surface plot (L, W) at fixed P0 and mu0
    num_grid_x = 60
    num_grid_y = 60

    L_vals = np.linspace(0.1, 25.0, num_grid_x)
    W_vals = np.linspace(0.01, 2.5, num_grid_y)
    Xg, Yg = np.meshgrid(L_vals, W_vals)

    grid = np.column_stack([
        Xg.ravel(),
        Yg.ravel(),
        np.full(Xg.size, data.P0),
        np.full(Xg.size, np.log(data.mu0))
    ])

    grid_scaled = data.scaler_3D.transform(grid)
    Z_log = models.pinn_3D.predict(grid_scaled, verbose=0).flatten()
    Z = np.expm1(Z_log).reshape(Yg.shape)

    fig = plt.figure(figsize=(7.2, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Xg, Yg, Z, cmap="viridis", edgecolor="none", antialiased=True)
    ax.set_xlabel("Fracture length L (m)")
    ax.set_ylabel("Fracture width W (m)")
    ax.set_zlabel("Energy balance E (a.u.)")
    ax.set_title("3D PINN Prediction at Fixed P, μ")
    ax.view_init(elev=25, azim=-135)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="E (a.u.)")
    fig.tight_layout()
    save_fig(fig, fig_dir, "3D_surface.png")


def plot_error_distribution(preds, data, fig_dir):
    err_1D = preds.pred_1D_test - data.y_1D_true_test
    err_2D = preds.pred_2D_test - data.y_2D_true_test
    err_3D = preds.pred_3D_test - data.y_3D_true_test

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.histplot(err_1D, bins=60, color="#B2182B", alpha=0.35, label="1D PINN",
                stat="density", element="step", fill=True)
    sns.histplot(err_2D, bins=60, color="#2166AC", alpha=0.35, label="2D PINN",
                stat="density", element="step", fill=True)
    sns.histplot(err_3D, bins=60, color="#1B7837", alpha=0.35, label="3D PINN",
                stat="density", element="step", fill=True)
    ax.set_xlabel("Prediction error, E_pred - E_true (a.u.)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution: PINN Predictions")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, fig_dir, "error_distribution.png")


def generate_all_plots(data, models, preds, histories, fig_dir, classification_payloads=None):
    configure_plot_style()
    plot_training_loss(histories.hist_1D, histories.hist_2D, histories.hist_3D, fig_dir)
    plot_comparison_line(data, models, fig_dir)
    plot_fdm_instability(fig_dir)
    plot_kgd_baseline(data, fig_dir)
    plot_2d_contour(data, models, fig_dir)
    plot_3d_surface(data, models, fig_dir)
    plot_error_distribution(preds, data, fig_dir)
    if classification_payloads:
        for name, _y_true_bin, _y_score, thr, cm in classification_payloads:
            plot_confusion_matrix(name, cm, thr, fig_dir)
        plot_roc_curves(classification_payloads, fig_dir)
