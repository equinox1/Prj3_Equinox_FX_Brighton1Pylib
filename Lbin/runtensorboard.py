from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import subprocess
import os

# Define global log directory
global_logdir = r"C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\Logdir"
global_logdir=r"C:\WinRunMnt1\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\PythonLib\tsModelData\tboard_logs"
logdir = os.path.join(global_logdir, "hparams_tuning")
logdir=global_logdir

# Ensure the log directory exists
os.makedirs(logdir, exist_ok=True)

# Start TensorBoard in a non-blocking way
try:
    subprocess.Popen([
        "tensorboard",
        "--logdir", global_logdir,
        "--bind_all",
        "--load_fast=false",
        "--reload_interval=10"
    ])
except FileNotFoundError:
    print("Error: TensorBoard executable not found. Ensure TensorBoard is installed and available in PATH.")

# Define hyperparameters and metrics
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
METRIC_ACCURACY = 'accuracy'

# Configure hyperparameter logging
with tf.summary.create_file_writer(logdir).as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='tsNeuro Accuracy')],
    )
