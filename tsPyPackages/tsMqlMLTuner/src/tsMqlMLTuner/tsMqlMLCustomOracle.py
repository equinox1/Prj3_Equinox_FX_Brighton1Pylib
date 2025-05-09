from keras_tuner.engine.oracle import Oracle
from keras_tuner.engine.trial import Trial
from keras_tuner.engine.hyperparameters import HyperParameters


class CustomOracle(Oracle):
    def __init__(self, objective="val_loss", max_trials=50, seed=42):
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
        )
        # Required for directory/project structure (though unused in your setup)
        self._directory = "oracle_dir"
        self._project_name = "oracle_project"
        self._trials = {}  # ✅ Initialize the trials dictionary

    def populate_space(self, trial_id):
        hp = HyperParameters()

        # General settings
        hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd', 'nadam'])
        hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])
        hp.Choice('loss', ['mse', 'mae', 'binary_crossentropy'])
        hp.Choice('metric', ['mse', 'mae', 'accuracy'])
        hp.Int('epochs', min_value=5, max_value=50, step=1)

        # CNN
        hp.Int('num_cnn_layers', 1, 3)
        for i in range(3):
            hp.Int(f'cnn_filters_{i}', 32, 256, step=32)
            hp.Int(f'cnn_kernel_size_{i}', 2, 5)
            hp.Choice(f'cnn_activation_{i}', ['relu', 'tanh'])

        # LSTM
        hp.Int('num_lstm_layers', 1, 2)
        for i in range(2):
            hp.Int(f'lstm_units_{i}', 32, 256, step=32)

        # GRU
        hp.Int('num_gru_layers', 1, 2)
        for i in range(2):
            hp.Int(f'gru_units_{i}', 32, 256, step=32)

        # Transformer
        hp.Int('num_transformer_blocks', 1, 2)
        for i in range(2):
            hp.Int(f'key_dim_{i}', 32, 128, step=32)
            hp.Int(f'num_heads_{i}', 2, 8, step=2)

        # Dense
        hp.Int('dense_1_units', 32, 256, step=32)

        self._trials[trial_id] = Trial(hyperparameters=hp)
        self._trials[trial_id].status = "RUNNING"
        return {
            "trial_id": trial_id,
            "hyperparameters": hp,
            "status": "RUNNING",
        }

       

    def score_trial(self, trial_id, result):
        trial = self._trials.get(trial_id)
        if trial:
            trial.score = result
            trial.status = "COMPLETED"
