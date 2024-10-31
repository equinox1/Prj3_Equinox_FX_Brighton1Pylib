import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import os

class CMdtuner:
    def __init__(self, **kwargs):
        # Load dataset and model configuration parameters
        self.X_train = kwargs['X_train'] if 'X_train' in kwargs else None
        self.y_train = kwargs['y_train'] if 'y_train' in kwargs else None
        self.X_test = kwargs['X_test'] if 'X_test' in kwargs else None
        self.y_test = kwargs['y_test'] if 'y_test' in kwargs else None

        # Load model configuration parameters
        self.cnn_model = kwargs['cnn_model'] if 'cnn_model' in kwargs else False
        self.lstm_model = kwargs['lstm_model'] if 'lstm_model' in kwargs else False
        self.gru_model = kwargs['gru_model'] if 'gru_model' in kwargs else False
        self.transformer_model = kwargs['transformer_model'] if 'transformer_model' in kwargs else False

        # Load hyperparameter tuning configuration parameters
        self.objective = kwargs['objective'] if 'objective' in kwargs else 'val_loss'
        self.max_epochs = kwargs['max_epochs'] if 'max_epochs' in kwargs else 10
        self.epochs = kwargs['epochs'] if 'epochs' in kwargs else 10
        self.factor = kwargs['factor'] if 'factor' in kwargs else 3
        self.seed = kwargs['seed'] if 'seed' in kwargs else 42
        self.hyperband_iterations = kwargs['hyperband_iterations'] if 'hyperband_iterations' in kwargs else 1
        self.tune_new_entries = kwargs['tune_new_entries'] if 'tune_new_entries' in kwargs else True
        self.allow_new_entries = kwargs['allow_new_entries'] if 'allow_new_entries' in kwargs else True
        self.max_retries_per_trial = kwargs['max_retries_per_trial'] if 'max_retries_per_trial' in kwargs else 3
        self.max_consecutive_failed_trials = kwargs['max_consecutive_failed_trials'] if 'max_consecutive_failed_trials' in kwargs else 3
        self.validation_split = kwargs['validation_split'] if 'validation_split' in kwargs else 0.2
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 32
        self.dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.3
        self.oracle = kwargs['oracle'] if 'oracle' in kwargs else 'hyperband'
        self.activation1 = kwargs['activation1'] if 'activation1' in kwargs else 'relu'
        self.activation2 = kwargs['activation2'] if 'activation2' in kwargs else 'relu'
        self.activation3 = kwargs['activation3'] if 'activation3' in kwargs else 'relu'
        self.activation4 = kwargs['activation4'] if 'activation4' in kwargs else 'linear'
        self.hypermodel = kwargs['hypermodel'] if 'hypermodel' in kwargs else None
        self.max_model_size = kwargs['max_model_size'] if 'max_model_size' in kwargs else 1000000
        self.optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else 'adam'
        self.loss = kwargs['loss'] if 'loss' in kwargs else 'mean_squared_error'
        self.metrics = kwargs['metrics'] if 'metrics' in kwargs else ['mean_absolute_error']
        self.distribution_strategy = kwargs['distribution_strategy'] if 'distribution_strategy' in kwargs else None
        self.directory = kwargs['directory'] if 'directory' in kwargs else None       
        self.basepath = kwargs['basepath'] if 'basepath' in kwargs else None
        self.project_name = kwargs['project_name'] if 'project_name' in kwargs else None
        self.logger = kwargs['logger'] if 'logger' in kwargs else None
        self.tuner_id = kwargs['tuner_id'] if 'tuner_id' in kwargs else None
        self.overwrite = kwargs['overwrite'] if 'overwrite' in kwargs else False
        self.executions_per_trial = kwargs['executions_per_trial'] if 'executions_per_trial' in kwargs else 1

        # Load checkpoint configuration parameters
        self.chk_fullmodel = kwargs['chk_fullmodel'] if 'chk_fullmodel' in kwargs else False
        self.chk_verbosity = kwargs['chk_verbosity'] if 'chk_verbosity' in kwargs else 1
        self.chk_mode = kwargs['chk_mode'] if 'chk_mode' in kwargs else 'auto'
        self.chk_monitor = kwargs['chk_monitor'] if 'chk_monitor' in kwargs else 'val_loss'
        self.chk_sav_freq = kwargs['chk_sav_freq'] if 'chk_sav_freq' in kwargs else 'epoch'
        self.chk_patience = kwargs['chk_patience'] if 'chk_patience' in kwargs else 0
        self.checkpoint_filepath = kwargs['checkpoint_filepath'] if 'checkpoint_filepath' in kwargs else None
        self.modeldatapath = kwargs['modeldatapath'] if 'modeldatapath' in kwargs else None


        # Define inputs
        if 'inputs' in kwargs:
            self.inputs = kwargs['inputs']
        else:
            if self.X_train is not None and hasattr(self.X_train, 'shape') and len(self.X_train.shape) >= 3:
                self.inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
            else:
                raise ValueError("Either 'inputs' or 'X_train' with a defined shape must be provided.")
        
        # Define and configure the tuner
        self.tuner = kt.Hyperband(
            hypermodel=self.build_model,
            objective=self.objective,
            max_epochs=self.max_epochs,
            factor=self.factor,
            directory=self.basepath,
            project_name=self.project_name,
            overwrite=self.overwrite
        )

        # Display search space summary and begin tuning
        self.tuner.search_space_summary()
        self.tuner.search(self.X_train, self.y_train,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          validation_data=(self.X_test, self.y_test),
                          callbacks=self.get_callbacks())

    def build_model(self, hp):
        # Ensure that at least one model branch is selected
        if not any([self.cnn_model, self.lstm_model, self.gru_model, self.transformer_model]):
            raise ValueError("At least one model branch (cnn_model, lstm_model, gru_model, or transformer_model) must be specified.")

        # Model branches
        x_cnn, x_lstm, x_gru, x_trans = None, None, None, None

        # CNN branch
        if self.cnn_model:
            x_cnn = Conv1D(filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32, default=64),
                           kernel_size=hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1, default=3),
                           activation='relu')(self.inputs)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Flatten()(x_cnn)

        # LSTM branch
        if self.lstm_model:
            x_lstm = LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32, default=64), return_sequences=True)(self.inputs)
            x_lstm = LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32, default=64))(x_lstm)

        # GRU branch
        if self.gru_model:
            x_gru = GRU(units=hp.Int('gru_units', min_value=32, max_value=128, step=32, default=64), return_sequences=True)(self.inputs)
            x_gru = GRU(units=hp.Int('gru_units', min_value=32, max_value=128, step=32, default=64))(x_gru)

        # Transformer branch
        if self.transformer_model:
            x_trans = MultiHeadAttention(num_heads=hp.Int('num_heads', min_value=2, max_value=4, step=1, default=2),
                                         key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32, default=64))(self.inputs, self.inputs)
            x_trans = LayerNormalization(epsilon=1e-6)(x_trans)
            x_trans = GlobalAveragePooling1D()(x_trans)

        # Concatenate branches and build model
        branches = [x for x in [x_cnn, x_lstm, x_gru, x_trans] if x is not None]
        if len(branches) > 1:
            combined = Concatenate()(branches)
        else:
            combined = branches[0]
            
        x = Dense(50, activation=self.activation1)(combined)
        x = Dropout(0.3)(x)
        output = Dense(1, activation=self.activation2)(x)

        model = Model(inputs=self.inputs, outputs=output)
        model = self.compile_model(model, hp)

        return model

    def compile_model(self, model, hp):
        model.compile(
            optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )
        return model
    
    # Get callbacks with adjusted checkpoint filepath
    def get_callbacks(self):
        checkpoint_filepath = os.path.join(self.basepath, 'best_model.weights.h5')  # Use .weights.h5 extension
        return [
            EarlyStopping(monitor=self.chk_monitor, patience=self.chk_patience),
            ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, save_best_only=True, monitor=self.chk_monitor, verbose=1)
        ]
