from keras_tuner.engine.oracle import Oracle
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.engine import trial as trial_lib
from tsMqlSetup import CMqlSetup
import os
import logging

# -- start of logging setup --
from tsMqlSetup import CMqlSetup
# ✅ Logger and Logdir Setup
setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=8,
    num_threads=1
)
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})
from tsMqlSetup import CMqlSetup
gtuner_model = app_params.get('gtuner_model', 'pytorch')  # or "tensorflow"
backend = tune_params.get('backend', gtuner_model)  # or "tensorflow"
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')
tunerlogfile = xerces_logfile
global_logdir, global_logfile = setup_config.set_log_dir(logdir=None, logfile=tunerlogfile, servername=xerces_servername,ltuner=gtuner_model)
logger = setup_config.setup_global_logger(global_logfile)
# -- end of logging setup ----

class CustomOracle(Oracle):
    def __init__(self, objective="val_loss", max_trials=50,log='tslog', seed=42):
        super().__init__(objective=objective, max_trials=max_trials, seed=seed)

        setup_config = CMqlSetup(
            loglevel='INFO',
            warn='ignore',
            precision='mixed_bfloat16',
            tfdebug=False,
            num_cores=48,
            num_threads=4
        )

        self.global_logdir = global_logdir
        self.global_logfile = global_logfile
        
        self._trials = {}

    def populate_space(self, trial_id):
        hp = HyperParameters()
        logger.info(f"[CustomOracle] Populating hyperparameters for trial_id: {trial_id}")

        hp.Float("lr", 1e-4, 1e-2, sampling="log", default=1e-3)
        hp.Choice("optimizer", ["Adam", "RMSprop", "SGD"], default="Adam")
        hp.Int("epochs", 5, 20, step=5, default=10)

        hp.Int("n_units1", 64, 256, step=64, default=128)
        hp.Int("n_units2", 64, 256, step=64, default=128)

        hp.Int("lstm_units", 32, 128, step=32, default=64)

        hp.Int("cnn_filters", 16, 128, step=16, default=32)
        hp.Int("cnn_kernel_size", 2, 5, step=1, default=3)

        hp.Int("trans_heads", 2, 8, step=2, default=4)
        hp.Int("trans_ff_dim", 64, 512, step=64, default=128)

        hp.Choice("loss", ["mse", "mae", "binary_crossentropy"], default="mse")
        hp.Choice("metric", ["mse", "mae", "accuracy"], default="mse")

        return hp

    def create_trial(self, tuner_id):
        trial_id = f"{len(self._trials):02d}"
        hp = self.populate_space(trial_id)
        trial = trial_lib.Trial(
            hyperparameters=hp,
            trial_id=trial_id,
            status=trial_lib.TrialStatus.RUNNING,
        )
        self._trials[trial_id] = trial
        logger.info(f"[CustomOracle] Created trial {trial_id} with hyperparameters: {hp.values}")
        return trial


    def score_trial(self, trial_id, result):
        if trial_id in self._trials:
            self._trials[trial_id].score = result
            self._trials[trial_id].status = trial_lib.TrialStatus.COMPLETED
