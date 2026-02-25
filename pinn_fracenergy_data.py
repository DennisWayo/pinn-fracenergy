from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pinn_fracenergy_config import NUM_SAMPLES, NOISE_FACTOR, SPLIT_SEED, VAL_FRAC, TEST_FRAC


def scale_targets(y_train, y_val, y_test):
    scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = scaler.transform(y_test.reshape(-1, 1)).ravel()
    return scaler, y_train_s, y_val_s, y_test_s


def prepare_data(
    num_samples=NUM_SAMPLES,
    noise_factor=NOISE_FACTOR,
    split_seed=SPLIT_SEED,
    val_frac=VAL_FRAC,
    test_frac=TEST_FRAC,
):
    # Sample ranges
    L = np.random.uniform(0.1, 25.0, num_samples)          # fracture length (m)
    W = np.random.uniform(0.01, 2.5, num_samples)          # fracture width (m)
    P = np.random.uniform(1.0, 120.0, num_samples)         # pressure gradient (MPa/m)
    mu = np.random.uniform(0.001, 0.3, num_samples)        # viscosity (Pa·s)

    # Fixed values for slices
    W0 = np.median(W)
    P0 = np.median(P)
    mu0 = np.median(mu)

    # Synthetic energy proxy
    E_3D_true = (P * W / mu) * np.exp(-L / 5)
    E_2D_true = (P0 * W / mu0) * np.exp(-L / 5)
    E_1D_true = (P0 * W0 / mu0) * np.exp(-L / 5)

    # Add noise for robustness
    E_3D_noisy = E_3D_true + noise_factor * np.random.randn(num_samples)
    E_2D_noisy = E_2D_true + noise_factor * np.random.randn(num_samples)
    E_1D_noisy = E_1D_true + noise_factor * np.random.randn(num_samples)

    # Log transform targets to reduce dynamic range
    E_1D_true_log = np.log1p(E_1D_true)
    E_2D_true_log = np.log1p(E_2D_true)
    E_3D_true_log = np.log1p(E_3D_true)
    E_1D_noisy_log = np.log1p(E_1D_noisy)
    E_2D_noisy_log = np.log1p(E_2D_noisy)
    E_3D_noisy_log = np.log1p(E_3D_noisy)

    # Feature matrices
    X_3D = np.column_stack((L, W, P, np.log(mu)))
    X_2D = np.column_stack((L, W))
    X_1D = L.reshape(-1, 1)

    # Train/val/test split (leakage protection)
    indices = np.arange(num_samples)
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_frac + test_frac, random_state=split_seed, shuffle=True
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_frac / (val_frac + test_frac), random_state=split_seed, shuffle=True
    )

    L_train, L_val, L_test = L[train_idx], L[val_idx], L[test_idx]
    W_train, W_val, W_test = W[train_idx], W[val_idx], W[test_idx]
    P_train, P_val, P_test = P[train_idx], P[val_idx], P[test_idx]
    mu_train, mu_val, mu_test = mu[train_idx], mu[val_idx], mu[test_idx]

    X_3D_train, X_3D_val, X_3D_test = X_3D[train_idx], X_3D[val_idx], X_3D[test_idx]
    X_2D_train, X_2D_val, X_2D_test = X_2D[train_idx], X_2D[val_idx], X_2D[test_idx]
    X_1D_train, X_1D_val, X_1D_test = X_1D[train_idx], X_1D[val_idx], X_1D[test_idx]

    y_3D_train_log, y_3D_val_log, y_3D_test_log = (
        E_3D_noisy_log[train_idx],
        E_3D_noisy_log[val_idx],
        E_3D_noisy_log[test_idx],
    )
    y_2D_train_log, y_2D_val_log, y_2D_test_log = (
        E_2D_noisy_log[train_idx],
        E_2D_noisy_log[val_idx],
        E_2D_noisy_log[test_idx],
    )
    y_1D_train_log, y_1D_val_log, y_1D_test_log = (
        E_1D_noisy_log[train_idx],
        E_1D_noisy_log[val_idx],
        E_1D_noisy_log[test_idx],
    )

    y_3D_true_train, y_3D_true_val, y_3D_true_test = (
        E_3D_true[train_idx],
        E_3D_true[val_idx],
        E_3D_true[test_idx],
    )
    y_2D_true_train, y_2D_true_val, y_2D_true_test = (
        E_2D_true[train_idx],
        E_2D_true[val_idx],
        E_2D_true[test_idx],
    )
    y_1D_true_train, y_1D_true_val, y_1D_true_test = (
        E_1D_true[train_idx],
        E_1D_true[val_idx],
        E_1D_true[test_idx],
    )

    # Scale targets (fit on train only) for training stability
    y_1D_scaler, y_1D_train_s, y_1D_val_s, y_1D_test_s = scale_targets(
        y_1D_train_log, y_1D_val_log, y_1D_test_log
    )
    y_2D_scaler, y_2D_train_s, y_2D_val_s, y_2D_test_s = scale_targets(
        y_2D_train_log, y_2D_val_log, y_2D_test_log
    )
    y_3D_train_s = y_3D_train_log
    y_3D_val_s = y_3D_val_log
    y_3D_test_s = y_3D_test_log

    # Normalize features per case (fit on train only)
    scaler_3D = StandardScaler().fit(X_3D_train)
    scaler_2D = StandardScaler().fit(X_2D_train)
    scaler_1D = StandardScaler().fit(X_1D_train)

    X_3D_train_scaled = scaler_3D.transform(X_3D_train)
    X_3D_val_scaled = scaler_3D.transform(X_3D_val)
    X_3D_test_scaled = scaler_3D.transform(X_3D_test)

    X_2D_train_scaled = scaler_2D.transform(X_2D_train)
    X_2D_val_scaled = scaler_2D.transform(X_2D_val)
    X_2D_test_scaled = scaler_2D.transform(X_2D_test)

    X_1D_train_scaled = scaler_1D.transform(X_1D_train)
    X_1D_val_scaled = scaler_1D.transform(X_1D_val)
    X_1D_test_scaled = scaler_1D.transform(X_1D_test)

    data = {
        "L": L,
        "W": W,
        "P": P,
        "mu": mu,
        "W0": W0,
        "P0": P0,
        "mu0": mu0,
        "L_test": L_test,
        "W_test": W_test,
        "P_test": P_test,
        "mu_test": mu_test,
        "X_3D_train_scaled": X_3D_train_scaled,
        "X_3D_val_scaled": X_3D_val_scaled,
        "X_3D_test_scaled": X_3D_test_scaled,
        "X_2D_train_scaled": X_2D_train_scaled,
        "X_2D_val_scaled": X_2D_val_scaled,
        "X_2D_test_scaled": X_2D_test_scaled,
        "X_1D_train_scaled": X_1D_train_scaled,
        "X_1D_val_scaled": X_1D_val_scaled,
        "X_1D_test_scaled": X_1D_test_scaled,
        "y_1D_train_s": y_1D_train_s,
        "y_1D_val_s": y_1D_val_s,
        "y_1D_test_s": y_1D_test_s,
        "y_2D_train_s": y_2D_train_s,
        "y_2D_val_s": y_2D_val_s,
        "y_2D_test_s": y_2D_test_s,
        "y_3D_train_s": y_3D_train_s,
        "y_3D_val_s": y_3D_val_s,
        "y_3D_test_s": y_3D_test_s,
        "y_1D_true_train": y_1D_true_train,
        "y_1D_true_val": y_1D_true_val,
        "y_1D_true_test": y_1D_true_test,
        "y_2D_true_train": y_2D_true_train,
        "y_2D_true_val": y_2D_true_val,
        "y_2D_true_test": y_2D_true_test,
        "y_3D_true_train": y_3D_true_train,
        "y_3D_true_val": y_3D_true_val,
        "y_3D_true_test": y_3D_true_test,
        "y_1D_scaler": y_1D_scaler,
        "y_2D_scaler": y_2D_scaler,
        "scaler_1D": scaler_1D,
        "scaler_2D": scaler_2D,
        "scaler_3D": scaler_3D,
    }
    return SimpleNamespace(**data)
