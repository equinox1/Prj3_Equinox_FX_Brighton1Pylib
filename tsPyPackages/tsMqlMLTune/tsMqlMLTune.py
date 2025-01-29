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

tf.config.optimizer.set_jit(True)  # Enable XLA


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

        # Training configurations
        self.objective = kwargs.get('objective', 'val_loss')
        self.max_epochs = kwargs.get('max_epochs', 1)
        self.min_epochs = kwargs.get('min_epochs', 1)
        self.step = kwargs.get('step', 1)
        self.factor = kwargs.get('factor', 3)
        self.checkpoint_filepath = kwargs.get('checkpoint_filepath', 'best_model.keras')
        self.step = kwargs.get('step', 1)
        self.basepath = kwargs.get('basepath', 'tuner_results')
        self.project_name = kwargs.get('project_name', 'cm_tuning')
        self.num_trials = kwargs.get('num_trials', 3)

        # Activation functions
        self.activation1 = kwargs.get('activation1', 'relu')
        self.activation2 = kwargs.get('activation2', 'linear')

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
        if self.tunemodeepochs:
            hp = kt.HyperParameters()
            max_epochs = hp.Int('epochs', min_value=self.min_epochs, max_value=self.max_epochs, step=self.step)
        else:
            max_epochs = self.max_epochs

        self.tuner = kt.Hyperband(
            hypermodel=self.build_model,
            objective=self.objective,
            max_epochs=max_epochs,
            factor=self.factor,
            directory=self.basepath,
            project_name=self.project_name,
            overwrite=True
        )
        self.tuner.search_space_summary()
        # Display the model summary
        

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
                    filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32, default=64),
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
            branches.append(transformer_branch)

        # LSTM Branch
        if self.lstm_model:
            lstm_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='lstm_input')
            if self.multi_inputs: inputs.append(lstm_input)
            if self.tunemode:
                lstm_branch = LSTM(
                    units=hp.Int('lstm_units', min_value=32, max_value=128, step=32, default=64),
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
                     units=hp.Int('gru_units', min_value=32, max_value=128, step=32, default=64),
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
            loss=hp.Choice('loss', ['binary_crossentropy', 'mse']),
            metrics=['accuracy']
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
            # Define an additional HyperParameter for epochs
            epoch_hp = kt.HyperParameters()
            epoch_hp.Int('epochs', self.min_epochs, self.max_epochs)

            # Run the hyperparameter tuning search
            self.tuner.search(
                self.traindataset,
                validation_data=self.valdataset,
                callbacks=self.get_callbacks(),
                epochs=epoch_hp.get('epochs')
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
