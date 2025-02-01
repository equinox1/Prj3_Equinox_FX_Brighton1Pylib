import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout,
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras_tuner as kt

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Enable XLA for performance boost
tf.config.optimizer.set_jit(True)


class CMdtuner:
    def __init__(self, **kwargs):
        # Initialize hypermodel parameters
        self.hypermodel_params = kwargs.get('hypermodel_params', {})
        self.traindataset = kwargs.get('traindataset')
        self.valdataset = kwargs.get('valdataset')
        self.testdataset = kwargs.get('testdataset')

        if self.traindataset is not None:
            self.traindataset = self.traindataset.map(self.cast_to_float32)
        if self.valdataset is not None:
            self.valdataset = self.valdataset.map(self.cast_to_float32)
        if self.testdataset is not None:
            self.testdataset = self.testdataset.map(self.cast_to_float32)

        # Model configuration
        self.cnn_model = kwargs.get('cnn_model', False)
        self.lstm_model = kwargs.get('lstm_model', False)
        self.gru_model = kwargs.get('gru_model', False)
        self.transformer_model = kwargs.get('transformer_model', False)
        self.multi_inputs = kwargs.get('multi_inputs', False)
        self.multi_outputs = kwargs.get('multi_outputs', False)
        self.multi_branches = kwargs.get('multi_branches', False)
        self.tunemode = kwargs.get('tunemode', False)
        self.tunemodeepochs = kwargs.get('tunemodeepochs', False)
        self.modelsummary = kwargs.get('modelsummary', False)
        
        # Data shapes
        self.data_input_shape = kwargs.get('data_input_shape')
        self.batch_size = kwargs.get('batch_size', 32)
        self.dropout = kwargs.get('dropout', 0.3)
        self.steps_per_execution = kwargs.get('steps_per_execution', 32)

        # Training configurations
        self.objective = kwargs.get('objective', 'val_loss')
        self.max_epochs = kwargs.get('max_epochs', 10)
        self.min_epochs = kwargs.get('min_epochs', 1)
        self.step = kwargs.get('step', 1)
        self.factor = kwargs.get('factor', 3)
        self.checkpoint_filepath = kwargs.get('checkpoint_filepath', 'best_model.keras')
        self.step = kwargs.get('step', 1)
        self.basepath = kwargs.get('basepath', 'tuner_results')
        self.project_name = kwargs.get('project_name', 'cm_tuning')
        self.num_trials = kwargs.get('num_trials', 3)
        self.overwrite = kwargs.get('overwrite', True)
        self.tune_new_entries = kwargs.get('tune_new_entries', True)
        self.allow_new_entries = kwargs.get('allow_new_entries', True)
        self.max_retries_per_trial = kwargs.get('max_retries_per_trial', 9)
        self.max_consecutive_failed_trials = kwargs.get('max_consecutive_failed_trials', 3)
        self.hyperband_iterations = kwargs.get('hyperband_iterations', 1)
        

        #unit for LSTM and GRU
        self.unitmin = kwargs.get('unitmin', 32)
        self.unitmax = kwargs.get('unitmax', 128)
        self.unitstep = kwargs.get('unitstep', 32)
        self.unitdefault = kwargs.get('unitdefault', 64)

        #debugging
        self.tf1 = kwargs.get('tf1', False)
        self.tf2 = kwargs.get('tf2', False)

        # Output dimensions
        self.output_dim = kwargs.get('output_dim', 1)

        # Ensure the base path exists
        os.makedirs(self.basepath, exist_ok=True)

        # TensorFlow debugging (optional)
        self.enable_debugging(kwargs)

        # Prepare input shapes
        self.prepare_shapes()

        # Validate configurations
        self.validate_config()

        # Initialize the tuner
        self.initialize_tuner()

    @staticmethod
    def cast_to_float32(x, y):
        return tf.cast(x, tf.float32), y

    def enable_debugging(self, kwargs):
        if kwargs.get('tf1', False):
            tf.debugging.set_log_device_placement(True)
        if kwargs.get('tf2', False):
            tf.debugging.enable_check_numerics()

    def prepare_shapes(self):
        if not self.data_input_shape:
            raise ValueError("Data input shape must be specified.")
        self.main_input_shape = self.get_shape(self.data_input_shape)

    @staticmethod
    def get_shape(data_shape):
        if len(data_shape) not in [2, 3]:
            raise ValueError(f"Unsupported input shape: {data_shape}. Must be 2D or 3D.")
        return tuple(data_shape)

    def validate_config(self):
        if not (self.cnn_model or self.lstm_model or self.gru_model or self.transformer_model):
            raise ValueError("At least one model type (CNN, LSTM, GRU, Transformer) must be enabled.")

    def initialize_tuner(self):
        hp = kt.HyperParameters()
        
        # Tune the number of epochs
        hp.Int('epochs', min_value=self.min_epochs, max_value=self.max_epochs, step=self.step)

        print(f"Tuning Max epochs between {self.min_epochs} and {self.max_epochs}")

        self.tuner = kt.Hyperband(
            hypermodel=self.build_model,
            hyperparameters=hp,  # Pass the HyperParameters object
            hyperband_iterations=self.hyperband_iterations,
            objective=self.objective,
            max_epochs=self.max_epochs,  # Ensure max_epochs is properly set
            factor=self.factor,
            directory=self.basepath,
            project_name=self.project_name,
            overwrite=self.overwrite,
            tune_new_entries=self.tune_new_entries,
            allow_new_entries=self.allow_new_entries,
            max_retries_per_trial=self.max_retries_per_trial,
            max_consecutive_failed_trials=self.max_consecutive_failed_trials,
        )

    self.tuner.search_space_summary()
        
    def build_model(self, hp):
        # Shared Input Logic
        shared_input = Input(shape=self.main_input_shape, name='shared_input') if not self.multi_inputs else None
        inputs = [] if self.multi_inputs else [shared_input]
        branches = []

        # CNN Branch
        if self.cnn_model:
            cnn_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='cnn_input')
            if self.multi_inputs: inputs.append(cnn_input)
            if self.tunemode:
                cnn_branch = Conv1D(
                    filters=hp.Int('cnn_filters', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.unitdefault),
                    kernel_size=hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1, default=3),
                    activation=hp.Choice('activation', ['relu', 'tanh', 'selu', 'elu', 'linear', 'sigmoid', 'softmax', 'softplus'])
                )(cnn_input)
            else:
                cnn_branch = Conv1D(128, 5, activation="relu")(cnn_input)

            cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
            cnn_branch = Flatten()(cnn_branch)
            branches.append(cnn_branch)

        # Transformer Branch
        if self.transformer_model:
            transformer_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='transformer_input')
            if self.multi_inputs: inputs.append(transformer_input)
            if self.tunemode:
                transformer_branch = MultiHeadAttention(
                    num_heads=hp.Int('num_heads', min_value=1, max_value=4, step=1, default=2),
                    key_dim=hp.Int('key_dim', min_value=1, max_value=4, step=1, default=2)
                )(transformer_input, transformer_input)
            else:
                transformer_branch = MultiHeadAttention(num_heads=2, key_dim=1)(transformer_input, transformer_input)
            transformer_branch = LayerNormalization()(transformer_branch)
            transformer_branch = GlobalAveragePooling1D()(transformer_branch)
            
            # Adding Feed-Forward Network (FFN)
            transformer_branch = Dense(64, hp.Choice('activation', ['relu', 'tanh', 'selu', 'elu', 'linear', 'sigmoid', 'softmax', 'softplus']))(transformer_branch)
            transformer_branch = Dense(32, hp.Choice('activation', ['relu', 'tanh', 'selu', 'elu', 'linear', 'sigmoid', 'softmax', 'softplus']))(transformer_branch)
            branches.append(transformer_branch)

        
        # LSTM Branch
        if self.lstm_model:
            lstm_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='lstm_input')
            if self.multi_inputs: inputs.append(lstm_input)
            if self.tunemode:
                lstm_branch = LSTM(
                    units=hp.Int('lstm_units', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.unitdefault),
                    activation=hp.Choice('activation', ['relu', 'tanh', 'selu', 'elu', 'linear', 'sigmoid', 'softmax', 'softplus'])
                )(lstm_input)
            else:
                lstm_branch = LSTM(96)(lstm_input)
            branches.append(lstm_branch)
           

        # GRU Branch
        if self.gru_model:
            gru_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='gru_input')
            if self.multi_inputs: inputs.append(gru_input)
            if self.tunemode:
                gru_branch = GRU(
                     units=hp.Int('gru_units',min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.unitdefault),
                    activation=hp.Choice('activation', ['relu', 'tanh', 'selu', 'elu', 'linear', 'sigmoid', 'softmax', 'softplus'])
                )(gru_input)
            else:
                gru_branch = GRU(64)(gru_input)
            branches.append(gru_branch)

        # Concatenate
        # Concatenate the branches if multiple branches are used
        concatenated = Concatenate()(branches) if self.multi_branches else branches[0]
        dense_1 = Dense(50, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(concatenated)
        dropout = Dropout(0.5)(dense_1)
        output = Dense(1, activation="sigmoid")(dropout)
 
        # Set Input 
        print(f"Including {len(inputs)} input(s) and {len(branches)} branch(es).")
       
        if len(inputs) == 2:
            print(f"Input shape1: {inputs[0].shape}, input shape2: {inputs[1].shape}")
            model = Model(inputs=inputs, outputs=output)          
        elif len(inputs) == 3:
            print(f"Input shape1: {inputs[0].shape}, input shape2: {inputs[1].shape}, input shape3: {inputs[2].shape}")
            model = Model(inputs=inputs, outputs=output)         
        elif len(inputs) == 4:
            print(f"Input shape1: {inputs[0].shape}, input shape2: {inputs[1].shape}, input shape3: {inputs[2].shape}, input shape4: {inputs[3].shape}")
            model = Model(inputs=inputs, outputs=output)
        elif len(inputs) == 1:
            print(f"Input shape1: {inputs[0].shape}")
            model = Model(inputs=inputs[0], outputs=output)
        else:
            model = Model(inputs=shared_input, outputs=output)
           
        # Print branch shapes
        if len(branches) == 2:
            print(f"Branch shape1: {branches[0].shape}, branch shape2: {branches[1].shape}")
        elif len(branches) == 3:
            print(f"Branch shape1: {branches[0].shape}, branch shape2: {branches[1].shape}, branch shape3: {branches[2].shape}")
        elif len(branches) == 4:
            print(f"Branch shape1: {branches[0].shape}, branch shape2: {branches[1].shape}, branch shape3: {branches[2].shape}, branch shape4: {branches[3].shape}")
        elif len(branches) == 1:
            print(f"Branch shape1: {branches[0].shape}")

        print(f"Output shape: {output.shape}")
        model.compile(
            optimizer=self.get_optimizer(
                hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd', 'nadam', 'adadelta', 'adagrad', 'adamax', 'ftrl']),
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])
            ),
            steps_per_execution=self.steps_per_execution,
            loss=hp.Choice('loss', ['binary_crossentropy', 'mse', 'mae', 'mape', 'msle', 'poisson', 'kld', 'cosine_similarity']),
            metrics=['accuracy', 'mae', 'mse', 'mape', 'msle', 'poisson', 'cosine_similarity']
            
        )
        # Print model summary
        if self.modelsummary:
            model.summary()
         

        return model


    def get_optimizer(self, optimizer_name, learning_rate):
        optimizers = {
            'adam': Adam,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'sgd': tf.keras.optimizers.SGD,
            'nadam': tf.keras.optimizers.Nadam,
            'adadelta': tf.keras.optimizers.Adadelta,
            'adagrad': tf.keras.optimizers.Adagrad,
            'adamax': tf.keras.optimizers.Adamax,
            'ftrl': tf.keras.optimizers.Ftrl
        }
        return optimizers[optimizer_name](learning_rate=learning_rate)


    def get_callbacks(self):
        return [
            EarlyStopping(monitor=self.objective, patience=3, verbose=1),
            ModelCheckpoint(filepath=self.checkpoint_filepath, save_best_only=True, verbose=1),
            TensorBoard(log_dir=os.path.join(self.basepath, 'logs'))
        ]

    def run_search(self):
    try:
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        tuned_epochs = best_hps.get('epochs', self.max_epochs)

        print(f"Running tuner search with tuned epochs: {tuned_epochs}")

        self.tuner.search(
            self.traindataset,
            validation_data=self.valdataset,
            callbacks=self.get_callbacks(),
            epochs=tuned_epochs,  # Use the tuned epochs
        )
    except Exception as e:
        print(f"Error during tuning: {e}")

    

    def export_best_model(self, ftype='tf'):

        try:
            best_model = self.tuner.get_best_models(num_models=1)[0]
            export_path = os.path.join(self.basepath, self.project_name, 'best_model')
            print(f"Exporting best model to {export_path}")
            if ftype == 'h5':
                best_model.save(export_path + '.h5')
                print(f"Model saved to {export_path}.h5")
            else:
                best_model.save(export_path + '.keras')
                print(f"Model saved to {export_path}.keras")
        except IndexError:
            print("No models found to export.")
        except Exception as e:
            print(f"Error saving the model: {e}")


    def check_and_load_model(self,lpbasepath,ftype='tf'):
        """
        Check if the model file exists and load it.
        """
        print(f"Checking for model file at basepath {lpbasepath}")
        print(f"Project name: {self.project_name}")

        if ftype == 'h5':
            modelext = 'h5'
            model_path = os.path.join(lpbasepath, self.project_name)
            print(f"Model path: {model_path}")
        else:
            modelext = 'keras'
            model_path = os.path.join(lpbasepath, self.project_name)
            print(f"Model path: {model_path}")

        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
                print(model.summary())
                return model
            else:
                print(f"Model file does not exist at {model_path}")
                return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


    def run_prediction(self, test_data, batch_size=None):
        try:
            best_model = self.tuner.get_best_models(num_models=1)[0]
            if isinstance(test_data, tf.data.Dataset):
                test_data = test_data.map(self.cast_to_float32)
            predictions = best_model.predict(test_data, batch_size=batch_size or self.batch_size)
            return predictions
        except IndexError:
            print("No models found. Ensure tuning has been run successfully.")
        except Exception as e:
            print(f"Error during prediction: {e}")


import os
import pathlib
import numpy as np
from datetime import date

class CMdtunerHyperModel:
    def __init__(self, **kwargs):
        self.input_width = kwargs.get('input_width', 24)
        self.shift = kwargs.get('shift', 24)
        self.label_width = kwargs.get('label_width', 1)
        self.train_df = kwargs.get('train_df', None)
        self.val_df = kwargs.get('val_df', None)
        self.test_df = kwargs.get('test_df', None)
        self.label_columns = kwargs.get('label_columns', None)
        self.batch_size = kwargs.get('batch_size', 8)
        self.column_indices = {}
        self.label_columns_indices = {}
        self.total_window_size = self.input_width + self.shift
        self.objective = kwargs.get('objective', 'val_loss')
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.min_epochs = kwargs.get('min_epochs', 10)
        self.factor = kwargs.get('factor', 10)
        self.seed = kwargs.get('seed', 42)
        np.random.seed(self.seed)  # Ensuring reproducibility
        self.hyperband_iterations = kwargs.get('hyperband_iterations', 1)
        self.tune_new_entries = kwargs.get('tune_new_entries', False)
        self.allow_new_entries = kwargs.get('allow_new_entries', False)
        self.max_retries_per_trial = kwargs.get('max_retries_per_trial', 5)
        self.max_consecutive_failed_trials = kwargs.get('max_consecutive_failed_trials', 3)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.epochs = kwargs.get('epochs', 2)
        self.dropout = kwargs.get('dropout', 0.2)
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.loss = kwargs.get('loss', 'mean_squared_error')
        self.metrics = kwargs.get('metrics', ['mean_squared_error'])
        self.directory = kwargs.get('directory', None)
        self.logger = kwargs.get('logger', None)
        self.tuner_id = kwargs.get('tuner_id', None)
        self.overwrite = kwargs.get('overwrite', True)
        self.executions_per_trial = kwargs.get('executions_per_trial', 1)
        self.chk_fullmodel = kwargs.get('chk_fullmodel', True)
        self.chk_verbosity = kwargs.get('chk_verbosity', 1)
        self.chk_mode = kwargs.get('chk_mode', 'min')
        self.chk_monitor = kwargs.get('chk_monitor', 'val_loss')
        self.chk_sav_freq = kwargs.get('chk_sav_freq', 'epoch')
        self.chk_patience = kwargs.get('chk_patience', 3)
        self.modeldatapath = kwargs.get('modeldatapath', None)
        self.project_name = kwargs.get('project_name', "prjEquinox1_prod.keras")
        self.today = kwargs.get('today', None)
        self.random = kwargs.get('random', None)
        self.baseuniq = kwargs.get('baseuniq', None)
        self.basepath = kwargs.get('basepath', None)
        self.checkpoint_filepath = kwargs.get('checkpoint_filepath', None)
        self.unitmin = kwargs.get('unitmin', 32)
        self.unitmax = kwargs.get('unitmax', 512)
        self.unitstep = kwargs.get('unitstep', 32)
        self.defaultunits = kwargs.get('defaultunits', 128)
        self.num_trials = kwargs.get('num_trials', 3)
        self.steps_per_execution = kwargs.get('steps_per_execution', 50)

    def get_hypermodel_params(self, basepath=None, **kwargs):
        today_date = date.today().strftime('%Y-%m-%d %H:%M:%S')
        random_seed = self.seed  # Using self.seed for consistency
        base_path = pathlib.Path(basepath) if basepath else pathlib.Path(os.getcwd())
        project_name = self.project_name or "prjEquinox1_prod.keras"

        subdir = base_path / 'tshybrid_ensemble_tuning_prod' / '1'
        subdir.mkdir(parents=True, exist_ok=True)

        return {
            'directory': str(subdir),
            'basepath': str(subdir),
            'checkpoint_filepath': str(base_path / 'tshybrid_ensemble_tuning_prod' / self.project_name),
            'objective': self.objective,
            'max_epochs': self.max_epochs,
            'min_epochs': self.min_epochs,
            'factor': self.factor,
            'seed': self.seed,
            'hyperband_iterations': self.hyperband_iterations,
            'tune_new_entries': self.tune_new_entries,
            'allow_new_entries': self.allow_new_entries,
            'max_retries_per_trial': self.max_retries_per_trial,
            'max_consecutive_failed_trials': self.max_consecutive_failed_trials,
            'validation_split': self.validation_split,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'dropout': self.dropout,
            'optimizer': self.optimizer,
            'loss': self.loss,
            'metrics': self.metrics,
            'logger': self.logger,
            'tuner_id': self.tuner_id,
            'overwrite': self.overwrite,
            'executions_per_trial': self.executions_per_trial,
            'chk_fullmodel': self.chk_fullmodel,
            'chk_verbosity': self.chk_verbosity,
            'chk_mode': self.chk_mode,
            'chk_monitor': self.chk_monitor,
            'chk_sav_freq': self.chk_sav_freq,
            'chk_patience': self.chk_patience,
            'modeldatapath': self.modeldatapath,
            'project_name': self.project_name,
            'steps_per_execution': self.steps_per_execution,
            'unitmin': self.unitmin,
            'unitmax': self.unitmax,
            'unitstep': self.unitstep,
            'defaultunits': self.defaultunits,
            'num_trials': self.num_trials
        }
