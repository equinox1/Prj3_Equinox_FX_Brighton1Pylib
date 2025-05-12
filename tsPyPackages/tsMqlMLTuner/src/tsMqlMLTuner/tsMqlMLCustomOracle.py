from keras_tuner.engine.oracle import Oracle
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.engine import trial as trial_lib
from tsMqlSetup import CMqlSetup
import os
import logging

logger = logging.getLogger(__name__)

class CustomOracle(Oracle):
    def __init__(self, objective="val_loss", max_trials=50, seed=42):
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
        )

        setup_config = CMqlSetup(
            loglevel='INFO',
            warn='ignore',
            precision='mixed_bfloat16',
            tfdebug=False,
            num_cores=48,
            num_threads=4
        )
        xerces_servername = "WINSVRXERCES01"
        xerces_logfile = 'tsneuropredict_app.log'
        self.global_logdir, self.global_logfile = setup_config.set_log_dir(
            logdir=None,
            logfile=xerces_logfile,
            servername=xerces_servername
        )
        print(f"[CustomOracle] Logdir: {self.global_logdir}")

        self._directory = os.path.join(self.global_logdir, "oracle_dir")
        self._project_name = os.path.join(self.global_logdir, "oracle_project")
        self._trials = {}

    def populate_space(self, trial_id):
        hp = HyperParameters()
        hp.Choice('model_type', ['cnn', 'lstm', 'gru', 'transformer'])
        hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd', 'nadam'])
        hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')
        hp.Choice('loss', ['mse', 'mae'])
        hp.Choice('metric', ['mse', 'mae'])
        hp.Int('epochs', 5, 50)

        model_type = hp.get('model_type')

        if model_type == 'cnn':
            hp.Int('num_cnn_layers', 1, 3)
            for i in range(3):
                hp.Int(f'cnn_filters_{i}', 32, 256, step=32)
                hp.Int(f'cnn_kernel_size_{i}', 2, 5)
                hp.Choice(f'cnn_activation_{i}', ['relu', 'tanh'])

        elif model_type == 'lstm':
            hp.Int('num_lstm_layers', 1, 2)
            for i in range(2):
                hp.Int(f'lstm_units_{i}', 32, 256, step=32)

        elif model_type == 'gru':
            hp.Int('num_gru_layers', 1, 2)
            for i in range(2):
                hp.Int(f'gru_units_{i}', 32, 256, step=32)

        elif model_type == 'transformer':
            hp.Int('num_transformer_blocks', 1, 2)
            for i in range(2):
                hp.Int(f'key_dim_{i}', 32, 128, step=32)
                hp.Int(f'num_heads_{i}', 2, 8, step=2)

        hp.Int('dense_1_units', 32, 256, step=32)

        # Save Trial object locally if needed
        self.trials[trial_id] = trial_lib.Trial(
            trial_id=trial_id,
            hyperparameters=hp,
            status="RUNNING"
        )

        # Return a dict for KerasTuner engine
        return {
            "trial_id": trial_id,
            "hyperparameters": hp,
            "status": "RUNNING"
        }

    def score_trial(self, trial_id, result):
        if trial_id in self._trials:
            self._trials[trial_id].score = result
            self._trials[trial_id].status = "COMPLETED"

    def get_trial(self):
        trial_id = self._generate_trial_id()
        trial_dict = self.create_trial(trial_id)
        self._trials[trial_id] = trial_dict
        return {
            "trial_id": trial_dict["trial_id"],
            "hyperparameters": trial_dict["hyperparameters"].values
        }