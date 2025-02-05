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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras_tuner as kt
import os
import pathlib
import numpy as np
from datetime import date
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        
        self.keras_tuner = kwargs.get('keras_tuner', 'hyperband') # bayesian, random, hyperband

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
        self.executions_per_trial = kwargs.get('executions_per_trial', 1)

        # Training configurations
        self.objective = kwargs.get('objective', 'val_loss')
        self.max_epochs = kwargs.get('max_epochs', 10)
        self.min_epochs = kwargs.get('min_epochs', 1)
        self.step = kwargs.get('step', 1)
        self.factor = kwargs.get('factor', 3)
        self.checkpoint_filepath = kwargs.get('checkpoint_filepath', 'best_model.keras')
        self.basepath = kwargs.get('basepath', 'tuner_results')
        self.project_name = kwargs.get('project_name', 'cm_tuning')
        self.num_trials = kwargs.get('num_trials', 3)
        self.overwrite = kwargs.get('overwrite', True)
        self.tune_new_entries = kwargs.get('tune_new_entries', True)
        self.allow_new_entries = kwargs.get('allow_new_entries', True)
        self.max_retries_per_trial = kwargs.get('max_retries_per_trial', 9)
        self.max_consecutive_failed_trials = kwargs.get('max_consecutive_failed_trials', 3)
        self.hyperband_iterations = kwargs.get('hyperband_iterations', 1)
        self.verbose = kwargs.get('verbose', 1)
        self.patience = kwargs.get('patience', 10)
        self.restore_best_weights = kwargs.get('restore_best_weights', True)
        self.save_best_only = kwargs.get('save_best_only', True)
        
        
        # Optimizer configurations
        self.scaler = None  # Initialize scaler

        # model scaling
        self.all_modelscale = kwargs.get('allmodelscale', 1)
        self.cnn_modelscale = kwargs.get('cnnmodelscale', 1)
        self.lstm_modelscale = kwargs.get('lstmmodelscale', 1)
        self.gru_modelscale = kwargs.get('grumodelscale', 1)
        self.trans_modelscale = kwargs.get('transmodelscale', 1)
        self.transh_modelscale = kwargs.get('transhmodelscale', 1)
        self.transff_modelscale = kwargs.get('transffmodelscale', 1)
        self.dense_modelscale = kwargs.get('densemodelscale', 1)
        
        #unit for LSTM and GRU
        self.unitmin = kwargs.get('unitmin', 32/self.all_modelscale)
        self.unitmax = kwargs.get('unitmax', 128/self.all_mmodelscale)
        self.unitstep = kwargs.get('unitstep', 32/self.all_mmodelscale)
        self.unitdefault = kwargs.get('unitdefault', 64/self.all_gru_mmodelscale)
       
        # Transformer configurations
        self.trans_dim_min = kwargs.get('trans_dim_min', 32/self.trans_modelscale)
        self.trans_dim_max = kwargs.get('trans_dim_max', 256/self.trans_modelscale)
        self.trans_dim_step = kwargs.get('trans_dim_step', 32/trans_modelscale)
        self.trans_dim_default =kwargs.get('trans_dim_default', 64/trans_modelscale)

        
        self.lstm_units_min = kwargs.get('lstm_units_min', 32/self.lstm_modelscale)
        self.lstm_units_max = kwargs.get('lstm_units_max', 128/self.lstm_modelscale)
        self.lstm_units_step = kwargs.get('lstm_units_step', 32/self.lstm_modelscale)
        self.lstm_units_default = kwargs.get('lstm_units_default', 64/self.lstm_modelscale)

        self.gru_units_min = kwargs.get('gru_units_min', 32/self.gru_modelscale)
        self.gru_units_max = kwargs.get('gru_units_max', 128/self.gru_modelscale)
        self.gru_units_step = kwargs.get('gru_units_step', 32/self.gru_modelscale)
        self.gru_units_default = kwargs.get('gru_units_default', 64/self.gru_modelscale)

        self.cnn_units_min = kwargs.get('cnn_units_min', 32/cnn_modelscale)
        self.cnn_units_max = kwargs.get('cnn_units_max', 128/cnn_modelscale)
        self.cnn_units_step = kwargs.get('cnn_units_step', 32/cnn_modelscale)
        self.cnn_units_default = kwargs.get('cnn_units_default', 64/cnn_modelscale)

        self.trans_heads_min = kwargs.get('trans_heads_min', 2/transh_modelscale)
        self.trans_heads_max = kwargs.get('trans_heads_max', 8/transh_modelscale)
        self.trans_heads_step = kwargs.get('trans_heads_step', 2/transh_modelscale)

        self.trans_ff_min = kwargs.get('trans_ff_min', 64/transff_modelscale) 
        self.trans_ff_max = kwargs.get('trans_ff_max',512/transff_modelscale)
        self.trans_ff_step = kwargs.get('trans_ff_step', 64/transff_modelscale) 

        self.dense_units_min = kwargs.get('dense_units_min', 32/dense_modelscale)
        self.dense_units_max = kwargs.get('dense_units_max', 128/dense_modelscale)
        self.dense_units_step = kwargs.get('dense_units_step', 32/dense_modelscale)
        

        #debugging
        self.tf1 = kwargs.get('tf1', False)
        self.tf2 = kwargs.get('tf2', False)

        # Output dimensions
        self.output_dim = kwargs.get('output_dim', 1)

        # Ensure the base path exists
        os.makedirs(self.basepath, exist_ok=True)

        # Enable XLA for performance boost
        tf.config.optimizer.set_jit(True)

        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')

        # Distributed Training Strategy
        self.strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {self.strategy.num_replicas_in_sync}")

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
        
        hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd', 'nadam', 'adadelta', 'adagrad', 'adamax', 'ftrl'])
        hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])
        hp.Choice('loss', ['binary_crossentropy', 'mse', 'mae', 'mape', 'msle', 'poisson', 'kld', 'cosine_similarity'])
        hp.Choice('activation', ['relu', 'tanh', 'selu', 'elu', 'linear', 'sigmoid', 'softmax', 'softplus'])
        hp.Choice('dense_1_activation', ['relu', 'tanh'])
        hp.Choice('metric', ['accuracy', 'mae', 'mse', 'mape', 'msle', 'poisson', 'cosine_similarity'])

        hp.Float('l2_reg', min_value=1e-6, max_value=1e-2, sampling='log', default=1e-4)

        if self.tunemodeepochs:
            hp.Int('epochs', min_value=self.min_epochs, max_value=self.max_epochs, step=1)
        else:
            hp.Fixed('epochs', self.min_epochs)

        if self.tunemode:
            hp.Int('cnn_filters', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.unitdefault)
            hp.Int('cnn_kernel_size', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.unitdefault)
        else:
            hp.Int('cnn_filters', min_value=2, max_value=5, step=1, default=3)
            hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1, default=3)
                
        
        hp.Int('lstm_units', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.unitdefault)
        hp.Int('gru_units', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.unitdefault)
        hp.Int('cnn_units', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.unitdefault)
        hp.Int('dense_1_units', min_value=self.dense_units_min, max_value=self.dense_units_max, step=self.dense_units_step)

        num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=3, step=1)
        hp.Int('trans_dim', min_value=self.trans_dim_min, max_value=self.trans_dim_max, step=self.trans_dim_step, default=self.trans_dim_default)
                
        logging.info(f"Tuning Max epochs between {self.min_epochs} and {self.max_epochs}")
        print("Tuner mode: ", self.tunemode, "Tuner mode epochs: ", self.tunemodeepochs)

        tuner_classes = {
            'random': kt.RandomSearch,
            'hyperband': kt.Hyperband,
            'bayesian': kt.BayesianOptimization
        }
        print("Tuner Service Checker:", self.keras_tuner)
        if self.keras_tuner in tuner_classes:
            print("Tuner Service is:", self.keras_tuner)
        try:
            if self.keras_tuner in tuner_classes:
                self.tuner = tuner_classes[self.keras_tuner](
                    hypermodel=self.build_model,
                    hyperparameters=hp,  # Pass the HyperParameters object
                    hyperband_iterations=self.hyperband_iterations,
                    objective=self.objective,
                    max_epochs=self.max_epochs,
                    factor=self.factor,
                    directory=self.basepath,
                    project_name=self.project_name,
                    overwrite=self.overwrite,
                    tune_new_entries=self.tune_new_entries,
                    allow_new_entries=self.allow_new_entries,
                    max_retries_per_trial=self.max_retries_per_trial,
                    max_consecutive_failed_trials=self.max_consecutive_failed_trials,
                    executions_per_trial=self.executions_per_trial,
                )
                self.tuner.search_space_summary()  # Only call if tuner initialized
            else:
                raise ValueError(f"Unsupported keras_tuner type: {self.keras_tuner}")
                assert hasattr(self, 'tuner'), "Tuner was not initialized!"
        except Exception as e:
            logging.error(f"Error initializing tuner: {e}")
            self.tuner = None  # Set to None to avoid future errors


        
    def build_model(self, hp):
        # Shared Input Logic
        shared_input = Input(shape=self.main_input_shape, name='shared_input') if not self.multi_inputs else None
        inputs = [] if self.multi_inputs else [shared_input]
        branches = []
        print(f"Shared input shape: {shared_input.shape} self.multi_inputs {self.multi_inputs}")

        # CNN Branch
        if self.cnn_model:
            cnn_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='cnn_input')
            if self.multi_inputs: inputs.append(cnn_input)
            print(f"Input shape cnn: {cnn_input.shape}")
                
            if self.tunemode:
                cnn_branch = Conv1D(
                    filters=hp.get('cnn_filters'),
                    kernel_size=hp.get('cnn_kernel_size'),
                    activation=hp.get('activation') # Tunable activation
                )(cnn_input)
            else:
                cnn_branch = Conv1D(
                   filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32, default=64),
                    kernel_size=hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1, default=3),
                    activation="relu"
                )(cnn_input)

            cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
            cnn_branch = Flatten()(cnn_branch)
            branches.append(cnn_branch)

        # LSTM Branch
        if self.lstm_model:
            lstm_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='lstm_input')
            print(f"Input shape lstm: {lstm_input.shape}")
            if self.multi_inputs: inputs.append(lstm_input)
          
            if self.tunemode:
                lstm_branch = LSTM(units=hp.get('lstm_units'), activation=hp.get('activation'))(lstm_input)
            else:
                lstm_branch = LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32, default=64))(lstm_input)
            
            branches.append(lstm_branch)

        # GRU Branch
        if self.gru_model:
            gru_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='gru_input')
            if self.multi_inputs: inputs.append(gru_input)
            print(f"Input shape gru : {gru_input.shape}")
            if self.tunemode:
                gru_branch = GRU(units=hp.get('gru_units'), activation=hp.get('activation'))(gru_input)
            else:
                 gru_branch = GRU(units=hp.Int('gru_units', min_value=32, max_value=128, step=32, default=64))(gru_input)
            branches.append(gru_branch)

        # Transformer Branch
        if self.transformer_model:
            transformer_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='transformer_input')
            if self.multi_inputs: inputs.append(transformer_input)
            print(f"Input shape trans: {transformer_input.shape}")
            transformer_branch = transformer_input

            num_transformer_blocks = hp.get('num_transformer_blocks')
            trans_dim = hp.get('trans_dim') # Shared transformer dimension

            for i in range(num_transformer_blocks):
                transformer_branch = self.transformer_block(transformer_branch, hp, i, trans_dim) # Pass trans_dim

            transformer_branch = GlobalAveragePooling1D()(transformer_branch)

            # FFN after transformer blocks
            transformer_branch = Dense(64, activation=hp.get('activation'))(transformer_branch)
            transformer_branch = Dropout(self.dropout)(transformer_branch) # Dropout in FFN
            branches.append(transformer_branch)

        # Concatenate the branches if multiple branches are used
        concatenated = Concatenate()(branches) if self.multi_branches else branches[0]
        dense_1 = Dense(
            units=hp.Int('dense_1_units', min_value=self.dense_units_min, max_value=self.dense_units_max, step=self.dense_units_step),
            activation=hp.Choice('dense_1_activation', ['relu', 'tanh']),
            kernel_regularizer=tf.keras.regularizers.l2(hp.get('l2_reg'))
        )(concatenated)
        dropout = Dropout(self.dropout)(dense_1)
        output = Dense(self.output_dim, activation="linear" if self.output_dim > 1 else "sigmoid")(dropout)  # Linear for multi-output regression

        # Set Input 
        print(f"Including {len(inputs)} input(s) and {len(branches)} branch(es).")
       
        if len(inputs) == 1:
            print(f"Input shape1: {inputs[0].shape}")
            model = Model(inputs=inputs, outputs=output)
        elif len(inputs) == 2:
            print(f"Input shape1: {inputs[0].shape}, input shape2: {inputs[1].shape}")
            model = Model(inputs=inputs, outputs=output)          
        elif len(inputs) == 3:
            print(f"Input shape1: {inputs[0].shape}, input shape2: {inputs[1].shape}, input shape3: {inputs[2].shape}")
            model = Model(inputs=inputs, outputs=output)         
        elif len(inputs) == 4:
            print(f"Input shape1: {inputs[0].shape}, input shape2: {inputs[1].shape}, input shape3: {inputs[2].shape}, input shape4: {inputs[3].shape}")
            model = Model(inputs=inputs, outputs=output)
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
        
        tf.keras.backend.clear_session()
        print("Ran swith clear session")
        model.compile(
            optimizer=self.get_optimizer(hp.get('optimizer'), hp.get('learning_rate')),
            loss=hp.get('loss') if self.tunemode else self.loss,  # Use tunable loss or default
            steps_per_execution=self.steps_per_execution,
            metrics=[hp.get('metric')] if self.tunemode else [self.metric]  # Use tunable metric or default
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
            EarlyStopping(monitor=self.objective, patience=self.patience, verbose=self.verbose, restore_best_weights=self.restore_best_weights), # restore_best_weights added
            ModelCheckpoint(filepath=self.checkpoint_filepath, save_best_only=self.save_best_only, verbose=self.verbose),
            TensorBoard(log_dir=os.path.join(self.basepath, 'logs')),
            ReduceLROnPlateau(monitor=self.objective, factor=0.1, patience=self.patience, min_lr=1e-6, verbose=self.verbose) # Learning rate scheduler
        ]

    def transformer_block(self, inputs, hp, block_num,dim):
        key_dim = hp.Int(f'key_dim_{block_num}', min_value=self.trans_dim_min, max_value=self.trans_dim_max, step=self.trans_dim_step)
        num_heads = hp.Int(f'num_heads_{block_num}', min_value=self.trans_heads_min, max_value=self.trans_heads_max, step=self.trans_heads_step)
        ff_dim = hp.Int(f'ff_dim_{block_num}', min_value=self.trans_ff_min, max_value=self.trans_ff_max, step=self.trans_ff_step)
        activation = hp.Choice(f'activation_{block_num}', ['relu', 'tanh', 'gelu'])  # Configurable activation

        # Multi-Head Attention
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
        attn_out = LayerNormalization()(inputs + attn_out)

        # Feed-Forward Network
        ffn_out = Dense(ff_dim, activation=activation)(attn_out)
        ffn_out = Dropout(self.dropout)(ffn_out) # Dropout in FFN
        ffn_out = Dense(inputs.shape[-1])(ffn_out)  # Project back to original dimension
        ffn_out = LayerNormalization()(attn_out + ffn_out) # Residual connection and Layer Normalization
        ffn_out = Dropout(self.dropout)(ffn_out) # Dropout after the block

        return ffn_out


    def run_search(self):
        logging.info("Running tuner search...")
        try:
            self.tuner.search(
                self.traindataset,
                validation_data=self.valdataset,
                epochs=self.max_epochs,
                verbose=self.verbose,
                callbacks=self.get_callbacks()
            )

            best_hps = self.tuner.get_best_hyperparameters(num_trials=1)
            if not best_hps:
                raise ValueError("No hyperparameters found. Ensure tuning has been run successfully.")
            
            logging.info(f"Best hyperparameters: {best_hps[0].values}")

        except Exception as e:
            logging.error(f"Error during tuning: {e}")


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
                model.summary()
                return model
            else:
                print(f"Model file does not exist at {model_path}")
                return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_best_hyperparameters(self):
        try:
            return self.tuner.get_best_hyperparameters(num_trials=1)[0]
        except Exception as e:
            print(f"Error retrieving best hyperparameters: {e}")
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
            return None
