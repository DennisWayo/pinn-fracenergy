import json
import os
import subprocess
import sys
import tempfile
import time
import gc

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, r2_score

from pinn_fracenergy_config import (
    CLIPNORM,
    LOG_EVERY,
    LR_DEFAULT,
    MODEL_SEED,
    RUN_PLOTS,
    RUN_SENSITIVITY,
    USE_EARLY_STOP,
    USE_LR_SCHED,
)


def configure_tensorflow():
    # Disable eager by default for speed (enable with TF_EAGER_DEBUG=1)
    tf.config.run_functions_eagerly(os.environ.get("TF_EAGER_DEBUG", "0") == "1")
    # Avoid grappler remapper crashes on some GPU stacks
    try:
        tf.config.optimizer.set_jit(False)
        tf.config.optimizer.set_experimental_options({
            "remapping": False,
            "layout_optimizer": False,
        })
    except Exception:
        pass

    # Optional: disable GPU to avoid MPS temp-file growth (set TF_USE_GPU=0)
    if os.environ.get("TF_USE_GPU", "1") == "0":
        tf.config.set_visible_devices([], "GPU")


def detect_hardware():
    device_name = "CPU"
    if tf.config.get_visible_devices("GPU"):
        device_name = "GPU"
    elif "COLAB_TPU_ADDR" in os.environ:
        device_name = "TPU"
    return device_name


# -----------------------------
# PINN model definition
# -----------------------------

def create_pinn_model(input_dim, num_layers=4, width=256, dropout=0.1, l2=1e-4, activation="relu"):
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))
    for _ in range(num_layers):
        model.add(layers.Dense(width, activation=activation, kernel_regularizer=regularizers.l2(l2)))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    return model


class EveryNEpochs(keras.callbacks.Callback):
    def __init__(self, n=20, label=None):
        super().__init__()
        self.n = max(1, int(n))
        self.label = label
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n != 0:
            return
        prefix = f"{self.label} " if self.label else ""
        loss = None if not logs else logs.get("loss")
        val_loss = None if not logs else logs.get("val_loss")
        elapsed = None
        eta = None
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            total_epochs = self.params.get("epochs") if self.params else None
            if total_epochs:
                avg_epoch = elapsed / max(1, (epoch + 1))
                eta = avg_epoch * (total_epochs - (epoch + 1))
        if loss is None:
            msg = f"{prefix}Epoch {epoch + 1}"
        elif val_loss is None:
            msg = f"{prefix}Epoch {epoch + 1}: loss={loss:.6f}"
        else:
            msg = f"{prefix}Epoch {epoch + 1}: loss={loss:.6f}, val_loss={val_loss:.6f}"
        if elapsed is not None:
            msg += f", elapsed={elapsed:.1f}s"
        if eta is not None:
            msg += f", eta={eta:.1f}s"
        print(msg, flush=True)


def train_pinn(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    num_layers=4,
    width=256,
    dropout=0.1,
    l2=1e-4,
    activation="relu",
    batch_size=64,
    epochs=200,
    log_every=LOG_EVERY,
    label=None,
    learning_rate=LR_DEFAULT,
    early_stop_patience=50,
    use_early_stop=None,
    use_lr_sched=None,
):
    model = create_pinn_model(
        input_dim=X_train.shape[1],
        num_layers=num_layers,
        width=width,
        dropout=dropout,
        l2=l2,
        activation=activation,
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=CLIPNORM)
    model.compile(optimizer=optimizer, loss="mse")
    t0 = time.perf_counter()
    callbacks = []
    if log_every:
        callbacks.append(EveryNEpochs(log_every, label=label))
    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)
        if use_early_stop is None:
            use_early_stop = USE_EARLY_STOP
        if use_lr_sched is None:
            use_lr_sched = USE_LR_SCHED
        if use_early_stop:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stop_patience,
                    restore_best_weights=True,
                    min_delta=1e-4,
                )
            )
        if use_lr_sched:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6,
                    verbose=0,
                )
            )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
        **fit_kwargs,
    )
    train_time = time.perf_counter() - t0
    return model, history, train_time


def eval_regression(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    nrmse = np.sqrt(mse) / (np.std(y_true) + 1e-12)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "nrmse": nrmse, "r2": r2}


def reset_seeds(seed=None):
    if seed is None:
        seed = MODEL_SEED
    np.random.seed(seed)
    tf.random.set_seed(seed)


def evaluate_3d_model(model, X_val_scaled, y_true_val):
    pred_log = model.predict(X_val_scaled, verbose=0).flatten()
    pred = np.expm1(pred_log)
    return eval_regression(y_true_val, pred)


def predict_3d(model, X_scaled):
    pred_log = model.predict(X_scaled, verbose=0).flatten()
    return np.expm1(pred_log)


CONFIGS_3D = {
    "C1": {"name": "C1", "num_layers": 3, "width": 128, "dropout": 0.0, "l2": 1e-6},
    "C2": {"name": "C2", "num_layers": 3, "width": 128, "dropout": 0.0, "l2": 1e-6},
    "C3": {"name": "C3", "num_layers": 3, "width": 128, "dropout": 0.0, "l2": 1e-6},
    "C4": {"name": "C4", "num_layers": 3, "width": 128, "dropout": 0.0, "l2": 1e-6},
}


def run_3d_subprocess(cfg, script_path, pred_out=None, seed=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, f"metrics_{cfg['name']}.json")
        env = os.environ.copy()
        env.update({
            "RUN_3D_SWEEP": "0",
            "RUN_3D_ONLY": "1",
            "THREE_D_CONFIG": cfg["name"],
            "THREE_D_OUT": out_path,
            "RUN_PLOTS": "0",
            "RUN_SENSITIVITY": "0",
        })
        if seed is not None:
            env["MODEL_SEED"] = str(seed)
        if pred_out:
            env["THREE_D_PRED_OUT"] = pred_out
        proc = subprocess.run(
            [sys.executable, script_path],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.returncode != 0:
            raise RuntimeError(f"3D subprocess failed for {cfg['name']}")
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)


# Compute MSE against true (noise-free)

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def compute_nrmse(y_true, y_pred):
    mse = compute_mse(y_true, y_pred)
    denom = np.std(y_true) + 1e-12
    return np.sqrt(mse) / denom


def compute_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def classification_metrics(name, y_true, y_score, threshold, plot_payloads=None):
    y_true_bin = (y_true >= threshold).astype(int)
    if len(np.unique(y_true_bin)) < 2:
        print(f"{name} Classification: skipped (single class in y_true).")
        return None
    y_pred_bin = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    try:
        auc = roc_auc_score(y_true_bin, y_score)
    except ValueError:
        auc = np.nan
    tn, fp, fn, tp = cm.ravel()
    print(f"{name} Confusion Matrix (threshold={threshold:.6f}):")
    print(cm)
    if np.isnan(auc):
        print(f"{name} ROC-AUC: n/a")
    else:
        print(f"{name} ROC-AUC: {auc:.4f}")
    if plot_payloads is not None:
        plot_payloads.append((name, y_true_bin, y_score, threshold, cm))
    return {
        "Model": name,
        "Threshold": threshold,
        "ROC-AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }
