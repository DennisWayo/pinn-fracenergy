import os
import warnings

# Quieter TF logs (set before importing TensorFlow)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Mute tf.data eager warning
warnings.filterwarnings(
    "ignore",
    message=r"Even though the `tf\.config\.experimental_run_functions_eagerly` option is set.*",
    category=UserWarning,
    module=r"tensorflow\.python\.data\.ops\.structured_function",
)

# Reproducibility
DATA_SEED = int(os.environ.get("DATA_SEED", "42"))
MODEL_SEED = int(os.environ.get("MODEL_SEED", str(DATA_SEED)))
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", str(DATA_SEED)))

# Figure output directory. Supports both repository layouts:
# - manuscript archive: Python files in codes/, figures at project root
# - code repository: Python files at repository root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == "codes" else BASE_DIR
FIG_DIR = os.environ.get("FIG_DIR", PROJECT_DIR)

# Runtime controls
FAST_RUN = os.environ.get("FAST_RUN", "0") == "1"
RUN_SENSITIVITY = os.environ.get("RUN_SENSITIVITY", "0") == "1"
RUN_PLOTS = os.environ.get("RUN_PLOTS", "1") == "1"
USE_EARLY_STOP = os.environ.get("USE_EARLY_STOP", "1") == "1"
USE_EARLY_STOP_1D = os.environ.get("USE_EARLY_STOP_1D", "0") == "1"
USE_EARLY_STOP_2D = os.environ.get("USE_EARLY_STOP_2D", "0") == "1"
USE_EARLY_STOP_3D = os.environ.get("USE_EARLY_STOP_3D", "1") == "1"
USE_LR_SCHED = os.environ.get("USE_LR_SCHED", "1") == "1"
RUN_3D_SWEEP = os.environ.get("RUN_3D_SWEEP", "1") == "1"
SWEEP_SUBPROC = os.environ.get("SWEEP_SUBPROC", "1") == "1"
RUN_3D_ONLY = os.environ.get("RUN_3D_ONLY", "0") == "1"
THREE_D_CONFIG = os.environ.get("THREE_D_CONFIG", "")
THREE_D_OUT = os.environ.get("THREE_D_OUT", "")
THREE_D_PRED_OUT = os.environ.get("THREE_D_PRED_OUT", "")
TARGET_R2_3D = float(os.environ.get("TARGET_R2_3D", "0.85"))
TARGET_NRMSE_3D = float(os.environ.get("TARGET_NRMSE_3D", "0.60"))
SWEEP_REPEATS = int(os.environ.get("SWEEP_REPEATS", "2"))
SWEEP_SEED_BASE = int(os.environ.get("SWEEP_SEED_BASE", "100"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "20"))
EPOCHS_1D = int(os.environ.get("EPOCHS_1D", "400"))
EPOCHS_2D = int(os.environ.get("EPOCHS_2D", "400"))
EPOCHS_3D = int(os.environ.get("EPOCHS_3D", "600"))
BATCH_SIZE_DEFAULT = int(os.environ.get("BATCH_SIZE", "64"))
BATCH_SIZE_3D = int(os.environ.get("BATCH_SIZE_3D", str(BATCH_SIZE_DEFAULT)))
LR_DEFAULT = float(os.environ.get("LR", "0.0001"))
LR_3D = float(os.environ.get("LR_3D", "0.00005"))
CLIPNORM = float(os.environ.get("CLIPNORM", "1.0"))

# Data generation
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "2000"))
NOISE_FACTOR = float(os.environ.get("NOISE_FACTOR", "0.03"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.15"))
TEST_FRAC = float(os.environ.get("TEST_FRAC", "0.15"))
