# pinn-fracenergy
import os
import sys
import time
import gc
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pinn_fracenergy_config import (
    DATA_SEED,
    MODEL_SEED,
    SPLIT_SEED,
    FIG_DIR,
    FAST_RUN,
    RUN_SENSITIVITY,
    RUN_PLOTS,
    USE_EARLY_STOP,
    USE_EARLY_STOP_1D,
    USE_EARLY_STOP_2D,
    USE_EARLY_STOP_3D,
    USE_LR_SCHED,
    RUN_3D_SWEEP,
    SWEEP_SUBPROC,
    RUN_3D_ONLY,
    THREE_D_CONFIG,
    THREE_D_OUT,
    THREE_D_PRED_OUT,
    TARGET_R2_3D,
    TARGET_NRMSE_3D,
    SWEEP_REPEATS,
    SWEEP_SEED_BASE,
    LOG_EVERY,
    EPOCHS_1D,
    EPOCHS_2D,
    EPOCHS_3D,
    BATCH_SIZE_DEFAULT,
    BATCH_SIZE_3D,
    LR_DEFAULT,
    LR_3D,
    CLIPNORM,
    NUM_SAMPLES,
    NOISE_FACTOR,
    VAL_FRAC,
    TEST_FRAC,
)
from pinn_fracenergy_data import prepare_data
from pinn_fracenergy_model import (
    configure_tensorflow,
    detect_hardware,
    create_pinn_model,
    train_pinn,
    evaluate_3d_model,
    predict_3d,
    reset_seeds,
    run_3d_subprocess,
    compute_mse,
    compute_nrmse,
    compute_r2,
    classification_metrics,
    CONFIGS_3D,
)
from pinn_fracenergy_plots import analytical_baseline_summary, generate_all_plots

# Reproducibility for data
np.random.seed(DATA_SEED)

# TensorFlow runtime configuration
configure_tensorflow()
reset_seeds(DATA_SEED)

# Detect hardware (CPU, GPU, or TPU)
print(f"Detected Hardware: {detect_hardware()}")

# -----------------------------
# Synthetic data generation
# -----------------------------

data = prepare_data(
    num_samples=NUM_SAMPLES,
    noise_factor=NOISE_FACTOR,
    split_seed=SPLIT_SEED,
    val_frac=VAL_FRAC,
    test_frac=TEST_FRAC,
)

# Train models (distinct input dimensions)
if RUN_3D_ONLY:
    cfg_name = THREE_D_CONFIG or "C1"
    cfg = CONFIGS_3D.get(cfg_name)
    if cfg is None:
        raise ValueError(f"Unknown 3D config: {cfg_name}")
    print(f"Training 3D PINN ({cfg['name']})...")
    import tensorflow as tf
    tf.keras.backend.clear_session()
    gc.collect()
    reset_seeds()
    pinn_3D, hist_3D, t_train_3D = train_pinn(
        data.X_3D_train_scaled,
        data.y_3D_train_s,
        data.X_3D_val_scaled,
        data.y_3D_val_s,
        num_layers=cfg["num_layers"],
        width=cfg["width"],
        dropout=cfg["dropout"],
        l2=cfg["l2"],
        activation=tf.nn.swish,
        epochs=EPOCHS_3D,
        batch_size=BATCH_SIZE_3D,
        learning_rate=LR_3D,
        early_stop_patience=40,
        label=f"3D-{cfg['name']}",
        use_early_stop=USE_EARLY_STOP_3D,
    )
    metrics_val = evaluate_3d_model(pinn_3D, data.X_3D_val_scaled, data.y_3D_true_val)
    metrics_test = evaluate_3d_model(pinn_3D, data.X_3D_test_scaled, data.y_3D_true_test)
    print(
        f"3D {cfg['name']} Val Metrics: "
        f"R2={metrics_val['r2']:.4f}, NRMSE={metrics_val['nrmse']:.4f}, MSE={metrics_val['mse']:.4f}"
    )
    # Deferred write to include inference timing
    if THREE_D_PRED_OUT:
        pred_test = predict_3d(pinn_3D, data.X_3D_test_scaled)
        np.savez(
            THREE_D_PRED_OUT,
            y_true_test=data.y_3D_true_test,
            y_pred_test=pred_test,
        )
    # Timing for 3D inference in subprocess
    t0 = time.perf_counter()
    _ = predict_3d(pinn_3D, data.X_3D_test_scaled)
    t_inf_3d = time.perf_counter() - t0
    if THREE_D_OUT:
        import json
        with open(THREE_D_OUT, "w", encoding="utf-8") as f:
            json.dump({
                "val": metrics_val,
                "test": metrics_test,
                "train_time": t_train_3D,
                "t_inf_3d": t_inf_3d,
            }, f)
    sys.exit(0)

print("Training 1D PINN...")
pinn_1D, hist_1D, t_train_1D = train_pinn(
    data.X_1D_train_scaled,
    data.y_1D_train_s,
    data.X_1D_val_scaled,
    data.y_1D_val_s,
    dropout=0.0,
    l2=1e-6,
    epochs=EPOCHS_1D,
    batch_size=BATCH_SIZE_DEFAULT,
    label="1D",
    use_early_stop=USE_EARLY_STOP_1D,
)

print("Training 2D PINN...")
pinn_2D, hist_2D, t_train_2D = train_pinn(
    data.X_2D_train_scaled,
    data.y_2D_train_s,
    data.X_2D_val_scaled,
    data.y_2D_val_s,
    dropout=0.0,
    l2=1e-6,
    epochs=EPOCHS_2D,
    batch_size=BATCH_SIZE_DEFAULT,
    label="2D",
    use_early_stop=USE_EARLY_STOP_2D,
)

pred_3D_test = None
pinn_3D = None
hist_3D = None
if RUN_3D_SWEEP:
    configs_3d = [CONFIGS_3D["C1"], CONFIGS_3D["C2"], CONFIGS_3D["C3"], CONFIGS_3D["C4"]]
    best = None
    t_train_3D_total = 0.0
    if SWEEP_SUBPROC:
        script_path = os.path.abspath(__file__)
        for cfg in configs_3d:
            metrics_runs = []
            print(f"Training 3D PINN ({cfg['name']}) [subprocess x{SWEEP_REPEATS}]...")
            for i in range(SWEEP_REPEATS):
                seed = SWEEP_SEED_BASE + i
                metrics = run_3d_subprocess(cfg, script_path, seed=seed)
                metrics = metrics.get("val", metrics)
                metrics["seed"] = seed
                metrics_runs.append(metrics)
                print(
                    f"3D {cfg['name']} seed {seed} Val: "
                    f"R2={metrics['r2']:.4f}, NRMSE={metrics['nrmse']:.4f}, MSE={metrics['mse']:.4f}"
                )
            mean_r2 = float(np.mean([m["r2"] for m in metrics_runs]))
            mean_nrmse = float(np.mean([m["nrmse"] for m in metrics_runs]))
            mean_mse = float(np.mean([m["mse"] for m in metrics_runs]))
            print(
                f"3D {cfg['name']} Mean Val: "
                f"R2={mean_r2:.4f}, NRMSE={mean_nrmse:.4f}, MSE={mean_mse:.4f}"
            )
            if best is None or mean_r2 > best["mean_r2"]:
                best = {
                    "name": cfg["name"],
                    "mean_r2": mean_r2,
                    "mean_nrmse": mean_nrmse,
                    "mean_mse": mean_mse,
                    "runs": metrics_runs,
                }
            if mean_r2 >= TARGET_R2_3D or mean_nrmse <= TARGET_NRMSE_3D:
                best = {
                    "name": cfg["name"],
                    "mean_r2": mean_r2,
                    "mean_nrmse": mean_nrmse,
                    "mean_mse": mean_mse,
                    "runs": metrics_runs,
                }
                print(f"3D Selection: {cfg['name']} met criteria, stopping sweep.")
                break
        chosen = CONFIGS_3D[best["name"]]
        best_run = max(best["runs"], key=lambda m: m["r2"])
        best_seed = best_run["seed"]
        print(
            f"3D Final Selection: {best['name']} (mean R2={best['mean_r2']:.4f}, "
            f"best seed={best_seed})"
        )
        import tensorflow as tf
        tf.keras.backend.clear_session()
        gc.collect()
        reset_seeds(best_seed)
        final_device = os.environ.get("FINAL_3D_DEVICE", "CPU").upper()
        prev_eager = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(os.environ.get("FINAL_3D_EAGER", "1") == "1")
        device_ctx = tf.device("/CPU:0") if final_device == "CPU" else tf.device("/GPU:0")
        with device_ctx:
            pinn_3D, hist_3D, t_train_3D = train_pinn(
                data.X_3D_train_scaled,
                data.y_3D_train_s,
                data.X_3D_val_scaled,
                data.y_3D_val_s,
                num_layers=chosen["num_layers"],
                width=chosen["width"],
                dropout=chosen["dropout"],
                l2=chosen["l2"],
                activation=tf.nn.swish,
                epochs=EPOCHS_3D,
                batch_size=BATCH_SIZE_3D,
                learning_rate=LR_3D,
                early_stop_patience=40,
                label=f"3D-{chosen['name']}",
                use_early_stop=USE_EARLY_STOP_3D,
            )
        tf.config.run_functions_eagerly(prev_eager)
    else:
        import tensorflow as tf
        for cfg in configs_3d:
            print(f"Training 3D PINN ({cfg['name']})...")
            tf.keras.backend.clear_session()
            gc.collect()
            reset_seeds()
            model, history, t_train = train_pinn(
                data.X_3D_train_scaled,
                data.y_3D_train_s,
                data.X_3D_val_scaled,
                data.y_3D_val_s,
                num_layers=cfg["num_layers"],
                width=cfg["width"],
                dropout=cfg["dropout"],
                l2=cfg["l2"],
                activation=tf.nn.swish,
                epochs=EPOCHS_3D,
                batch_size=BATCH_SIZE_3D,
                learning_rate=LR_3D,
                early_stop_patience=40,
                label=f"3D-{cfg['name']}",
                use_early_stop=USE_EARLY_STOP_3D,
            )
            t_train_3D_total += t_train
            metrics = evaluate_3d_model(model, data.X_3D_val_scaled, data.y_3D_true_val)
            print(
                f"3D {cfg['name']} Val Metrics: "
                f"R2={metrics['r2']:.4f}, NRMSE={metrics['nrmse']:.4f}, MSE={metrics['mse']:.4f}"
            )
            if best is None or metrics["r2"] > best["metrics"]["r2"]:
                best = {
                    "model": model,
                    "history": history,
                    "train_time": t_train,
                    "metrics": metrics,
                    "name": cfg["name"],
                }
            if metrics["r2"] >= TARGET_R2_3D or metrics["nrmse"] <= TARGET_NRMSE_3D:
                best = {
                    "model": model,
                    "history": history,
                    "train_time": t_train,
                    "metrics": metrics,
                    "name": cfg["name"],
                }
                print(f"3D Selection: {cfg['name']} met criteria, stopping sweep.")
                break
        pinn_3D = best["model"]
        hist_3D = best["history"]
        t_train_3D = t_train_3D_total
        print(
            f"3D Final Selection: {best['name']} "
            f"(R2={best['metrics']['r2']:.4f}, NRMSE={best['metrics']['nrmse']:.4f})"
        )
else:
    import tensorflow as tf
    print("Training 3D PINN...")
    pinn_3D, hist_3D, t_train_3D = train_pinn(
        data.X_3D_train_scaled,
        data.y_3D_train_s,
        data.X_3D_val_scaled,
        data.y_3D_val_s,
        num_layers=3,
        width=128,
        dropout=0.0,
        l2=1e-6,
        activation=tf.nn.swish,
        epochs=EPOCHS_3D,
        batch_size=BATCH_SIZE_3D,
        learning_rate=LR_3D,
        early_stop_patience=40,
        label="3D",
        use_early_stop=USE_EARLY_STOP_3D,
    )

pred_1D_test_s = pinn_1D.predict(data.X_1D_test_scaled, verbose=0).flatten()
pred_2D_test_s = pinn_2D.predict(data.X_2D_test_scaled, verbose=0).flatten()

pred_1D_test_log = data.y_1D_scaler.inverse_transform(pred_1D_test_s.reshape(-1, 1)).ravel()
pred_2D_test_log = data.y_2D_scaler.inverse_transform(pred_2D_test_s.reshape(-1, 1)).ravel()
pred_1D_test = np.expm1(pred_1D_test_log)
pred_2D_test = np.expm1(pred_2D_test_log)

if pred_3D_test is None and pinn_3D is None:
    raise RuntimeError("3D predictions unavailable (subprocess output missing).")
if pred_3D_test is None:
    pred_3D_test_log = pinn_3D.predict(data.X_3D_test_scaled, verbose=0).flatten()
    pred_3D_test = np.expm1(pred_3D_test_log)


def _monotonic_rate(values, expected):
    diffs = np.diff(np.asarray(values, dtype=float))
    tol = 1e-8 * max(1.0, float(np.nanmax(np.abs(values))))
    if expected == "increasing":
        return float(np.mean(diffs >= -tol))
    if expected == "decreasing":
        return float(np.mean(diffs <= tol))
    raise ValueError(f"Unknown monotonic direction: {expected}")


def _predict_1d_energy(model, L_values):
    X_scaled = data.scaler_1D.transform(np.asarray(L_values).reshape(-1, 1))
    pred_s = model.predict(X_scaled, verbose=0).flatten()
    pred_log = data.y_1D_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    return np.expm1(pred_log)


def _predict_2d_energy(model, L_values, W_values):
    X = np.column_stack([L_values, W_values])
    X_scaled = data.scaler_2D.transform(X)
    pred_s = model.predict(X_scaled, verbose=0).flatten()
    pred_log = data.y_2D_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    return np.expm1(pred_log)


def _predict_3d_energy(model, L_values, W_values, P_values, mu_values):
    X = np.column_stack([L_values, W_values, P_values, np.log(mu_values)])
    X_scaled = data.scaler_3D.transform(X)
    pred_log = model.predict(X_scaled, verbose=0).flatten()
    return np.expm1(pred_log)


def physics_consistency_diagnostics():
    n = 300
    L_line = np.linspace(0.1, 25.0, n)
    W_line = np.linspace(0.01, 2.5, n)
    P_line = np.linspace(1.0, 120.0, n)
    mu_line = np.linspace(0.001, 0.3, n)
    L0 = np.full(n, np.median(data.L))
    W0 = np.full(n, data.W0)
    P0 = np.full(n, data.P0)
    mu0 = np.full(n, data.mu0)

    rows = [
        {
            "Model": "1D",
            "Variable": "L",
            "Expected": "decreasing",
            "Monotonic rate": _monotonic_rate(_predict_1d_energy(pinn_1D, L_line), "decreasing"),
        },
        {
            "Model": "2D",
            "Variable": "L",
            "Expected": "decreasing",
            "Monotonic rate": _monotonic_rate(_predict_2d_energy(pinn_2D, L_line, W0), "decreasing"),
        },
        {
            "Model": "2D",
            "Variable": "W",
            "Expected": "increasing",
            "Monotonic rate": _monotonic_rate(_predict_2d_energy(pinn_2D, L0, W_line), "increasing"),
        },
    ]
    if pinn_3D is not None:
        rows.extend([
            {
                "Model": "3D",
                "Variable": "L",
                "Expected": "decreasing",
                "Monotonic rate": _monotonic_rate(_predict_3d_energy(pinn_3D, L_line, W0, P0, mu0), "decreasing"),
            },
            {
                "Model": "3D",
                "Variable": "W",
                "Expected": "increasing",
                "Monotonic rate": _monotonic_rate(_predict_3d_energy(pinn_3D, L0, W_line, P0, mu0), "increasing"),
            },
            {
                "Model": "3D",
                "Variable": "P",
                "Expected": "increasing",
                "Monotonic rate": _monotonic_rate(_predict_3d_energy(pinn_3D, L0, W0, P_line, mu0), "increasing"),
            },
            {
                "Model": "3D",
                "Variable": "mu",
                "Expected": "decreasing",
                "Monotonic rate": _monotonic_rate(_predict_3d_energy(pinn_3D, L0, W0, P0, mu_line), "decreasing"),
            },
        ])
    return pd.DataFrame(rows)


print("Max true:", np.max(data.y_3D_true_test))
print("Max pred:", np.max(pred_3D_test))
print("Std true:", np.std(data.y_3D_true_test))
print("Std pred:", np.std(pred_3D_test))

mse_1D = compute_mse(data.y_1D_true_test, pred_1D_test)
mse_2D = compute_mse(data.y_2D_true_test, pred_2D_test)
mse_3D = compute_mse(data.y_3D_true_test, pred_3D_test)
nrmse_1D = compute_nrmse(data.y_1D_true_test, pred_1D_test)
nrmse_2D = compute_nrmse(data.y_2D_true_test, pred_2D_test)
nrmse_3D = compute_nrmse(data.y_3D_true_test, pred_3D_test)
r2_1D = compute_r2(data.y_1D_true_test, pred_1D_test)
r2_2D = compute_r2(data.y_2D_true_test, pred_2D_test)
r2_3D = compute_r2(data.y_3D_true_test, pred_3D_test)

# Inference timing

# baseline FDM proxy compute time
start = time.perf_counter()
_ = (data.P_test * data.W_test / data.mu_test) * np.exp(-data.L_test / 5)
t_fdm = time.perf_counter() - start

start = time.perf_counter()
_ = pinn_1D.predict(data.X_1D_test_scaled, verbose=0)
t_inf_1D = time.perf_counter() - start

start = time.perf_counter()
_ = pinn_2D.predict(data.X_2D_test_scaled, verbose=0)
t_inf_2D = time.perf_counter() - start

if pinn_3D is not None:
    start = time.perf_counter()
    _ = pinn_3D.predict(data.X_3D_test_scaled, verbose=0)
    t_inf_3D = time.perf_counter() - start
else:
    t_inf_3D = np.nan

params_1D = pinn_1D.count_params()
params_2D = pinn_2D.count_params()
params_3D = pinn_3D.count_params() if pinn_3D is not None else np.nan

runtime_df = pd.DataFrame({
    "Task": ["FDM baseline (formula)", "PINN training 1D", "PINN training 2D", "PINN training 3D",
             "PINN inference 1D", "PINN inference 2D", "PINN inference 3D"],
    "Time (s)": [t_fdm, t_train_1D, t_train_2D, t_train_3D, t_inf_1D, t_inf_2D, t_inf_3D],
    "Parameters": [np.nan, params_1D, params_2D, params_3D, params_1D, params_2D, params_3D],
})

print("\nPerformance Metrics (Test Set)")
print(f"MSE (1D): {mse_1D:.4f}")
print(f"MSE (2D): {mse_2D:.4f}")
print(f"MSE (3D): {mse_3D:.4f}")
print(f"NRMSE (1D): {nrmse_1D:.4f}")
print(f"NRMSE (2D): {nrmse_2D:.4f}")
print(f"NRMSE (3D): {nrmse_3D:.4f}")
print(f"R2 (1D): {r2_1D:.4f}")
print(f"R2 (2D): {r2_2D:.4f}")
print(f"R2 (3D): {r2_3D:.4f}")

print("\nAnalytical Baseline Metrics (amplitude-calibrated)")
baseline_df = pd.DataFrame(analytical_baseline_summary(data))
print(baseline_df)

print("\nPhysics Consistency Diagnostics (line-out monotonicity)")
physics_df = physics_consistency_diagnostics()
print(physics_df)

print("\nFit Diagnostics (Train vs Val, scaled targets)")


def fit_diagnostics(name, history, y_train_scaled, y_val_scaled):
    train_losses = history.history.get("loss", [])
    val_losses = history.history.get("val_loss", [])
    if not val_losses:
        print(f"{name} Fit Summary: no validation loss recorded.")
        return
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    min_val = float(np.min(val_losses))
    min_epoch = int(np.argmin(val_losses) + 1)
    baseline = float(np.mean((y_val_scaled - np.mean(y_train_scaled)) ** 2))
    gap = final_val - final_train
    print(
        f"{name} Fit Summary: final train={final_train:.6f}, final val={final_val:.6f}, "
        f"min val={min_val:.6f} (epoch {min_epoch}), baseline val={baseline:.6f}, gap={gap:.6f}"
    )
    overfit = (final_val > min_val * 1.1) and (final_train < final_val)
    underfit = final_val >= baseline * 0.95
    if overfit:
        print(f"{name} Diagnosis: possible overfitting after epoch {min_epoch}.")
    elif underfit:
        print(f"{name} Diagnosis: possible underfitting vs mean-baseline.")
    else:
        print(f"{name} Diagnosis: no strong over/underfitting signal (heuristic).")


fit_diagnostics("1D", hist_1D, data.y_1D_train_s, data.y_1D_val_s)
fit_diagnostics("2D", hist_2D, data.y_2D_train_s, data.y_2D_val_s)
if hist_3D is not None:
    fit_diagnostics("3D", hist_3D, data.y_3D_train_s, data.y_3D_val_s)
else:
    print("3D Fit Summary: skipped (subprocess sweep selected model).")

print("\nClassification Metrics (Test Set; threshold = train median)")
classification_rows = []
classification_payloads = []
thr_1D = float(np.median(data.y_1D_true_train))
thr_2D = float(np.median(data.y_2D_true_train))
thr_3D = float(np.median(data.y_3D_true_train))
for name, y_true, y_score, thr in [
    ("1D", data.y_1D_true_test, pred_1D_test, thr_1D),
    ("2D", data.y_2D_true_test, pred_2D_test, thr_2D),
    ("3D", data.y_3D_true_test, pred_3D_test, thr_3D),
]:
    row = classification_metrics(name, y_true, y_score, thr, plot_payloads=classification_payloads)
    if row is not None:
        classification_rows.append(row)
if classification_rows:
    class_df = pd.DataFrame(classification_rows)
    print(class_df)

if RUN_PLOTS:
    models = SimpleNamespace(pinn_1D=pinn_1D, pinn_2D=pinn_2D, pinn_3D=pinn_3D)
    preds = SimpleNamespace(pred_1D_test=pred_1D_test, pred_2D_test=pred_2D_test, pred_3D_test=pred_3D_test)
    histories = SimpleNamespace(hist_1D=hist_1D, hist_2D=hist_2D, hist_3D=hist_3D)
    generate_all_plots(data, models, preds, histories, FIG_DIR, classification_payloads)

print("\nRuntime Summary")
print(runtime_df)

if RUN_SENSITIVITY:
    # Sensitivity Analysis: Varying Batch Sizes & Layers (3D case)
    import tensorflow as tf
    sensitivity_results = []
    for batch_size in [32, 64, 128]:
        for num_layers in [3, 5, 7]:
            model = create_pinn_model(
                input_dim=data.X_3D_train_scaled.shape[1],
                num_layers=num_layers,
                width=128,
                dropout=0.0,
                l2=1e-6,
                activation=tf.nn.swish,
            )
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_3D, clipnorm=CLIPNORM), loss="mse")
            t0 = time.perf_counter()
            model.fit(data.X_3D_train_scaled, data.y_3D_train_s, epochs=10, batch_size=batch_size, verbose=0)
            t1 = time.perf_counter()
            sensitivity_results.append([batch_size, num_layers, t1 - t0])

    sensitivity_df = pd.DataFrame(sensitivity_results, columns=["Batch Size", "Layers", "Training Time (s)"])
    print("\nSensitivity Analysis Results:")
    print(sensitivity_df)
