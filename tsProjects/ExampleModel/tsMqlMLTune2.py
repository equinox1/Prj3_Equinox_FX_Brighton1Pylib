import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import os
from keras_tuner import HyperParameters

from tensorflow.keras.callbacks import TensorBoard




class CMdtuner:
    def __init__(self, **kwargs):
        # Load dataset and model configuration parameters
        
        self.traindataset = kwargs['traindataset'] if 'traindataset' in kwargs else None
        self.valdataset = kwargs['valdataset'] if 'valdataset' in kwargs else None
        self.testdataset = kwargs['testdataset'] if 'testdataset' in kwargs else None

        # Load model configuration parameters
        self.cnn_model = kwargs['cnn_model'] if 'cnn_model' in kwargs else False
        self.lstm_model = kwargs['lstm_model'] if 'lstm_model' in kwargs else False
        self.gru_model = kwargs['gru_model'] if 'gru_model' in kwargs else False
        self.transformer_model = kwargs['transformer_model'] if 'transformer_model' in kwargs else False

        # Load hyperparameter tuning configuration parameters
        self.objective = kwargs['objective'] if 'objective' in kwargs else 'val_loss'
        self.max_epochs = kwargs['max_epochs'] if 'max_epochs' in kwargs else 10
        self.min_epochs = kwargs['min_epochs'] if 'min_epochs' in kwargs else 1
        self.epochs = kwargs['epochs'] if 'epochs' in kwargs else 10
        self.factor = kwargs['factor'] if 'factor' in kwargs else 3
        self.seed = kwargs['seed'] if 'seed' in kwargs else 42
        self.hyperband_iterations = kwargs['hyperband_iterations'] if 'hyperband_iterations' in kwargs else 1
        self.tune_new_entries = kwargs['tune_new_entries'] if 'tune_new_entries' in kwargs else True
        self.allow_new_entries = kwargs['allow_new_entries'] if 'allow_new_entries' in kwargs else True
        self.max_retries_per_trial = kwargs['max_retries_per_trial'] if 'max_retries_per_trial' in kwargs else 3
        self.max_consecutive_failed_trials = kwargs['max_consecutive_failed_trials'] if 'max_consecutive_failed_trials' in kwargs else 30
        print("self.max_consecutive_failed_trials", self.max_consecutive_failed_trials)
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
        self.step = kwargs['step'] if 'step' in kwargs else 5
        self.multiactivate = kwargs['multiactivate'] if 'multiactivate' in kwargs else False
        self.tf1 = kwargs['tf1'] if 'tf1' in kwargs else False
        self.tf2 = kwargs['tf2'] if 'tf2' in kwargs else False
        self.shape = kwargs['shape'] if 'shape' in kwargs else 0
        self.tensorshape = kwargs['tensorshape'] if 'tensorshape' in kwargs else False

        # Enable tensorflow debugging
        if self.tf1: tf.debugging.set_log_device_placement(self.tf1)
        if self.tf2: tf.debugging.enable_check_numerics()
        
        # Ensure the output directory exists
        os.makedirs(self.basepath, exist_ok=True)

        # Define inputs
        self.inputs = kwargs.get('inputs')
        

        print("tunemodel self.inputs", self.inputs)
        print("tunemodel self.inputs[0] Rows:  ", self.inputs[0])
        print("tunemodel self.inputs[1] Batch size: None is dynamic: ", self.inputs[1])
        print("tunemodel self.inputs[2] Timesteps or Sequence Length):", self.inputs[2])
        print("tunemodel self.inputs[3] Features or Observations per Timestep:", self.inputs[3])
        if len(self.inputs) > 4 :
            print("tunemodel self.inputs[4] Channels:", self.inputs[4])
        else: print("tunemodel self.inputs[4] Channels: None:")
        
        if len(self.inputs) > 4:
            print("tunemodel self.inputs[4] Channels:", self.inputs[4])
        
      
        # Define the tuner
        
        rows = self.inputs[0] if len(self.inputs) > 0 else 1
        batches = self.inputs[1] if len(self.inputs) > 1 else None
        timesteps = self.inputs[2]
        features = self.inputs[3]
        channels = self.inputs[4] if len(self.inputs) > 4 else 1
        print("tunemodel inputs", self.inputs)
        print("tunemodel rows", rows), print("tunemodel batches", batches), print("tunemodel timesteps", timesteps), print("tunemodel features", features)
        # Correct the input shape to include the channels dimension
        if self.shape in {1, 2, 3, 4, 5, 6, 7}:
            shapes = [(features,), (timesteps, features), (batches, timesteps, features), 
                    (rows, batches, timesteps, features), (1, timesteps, features), 
                    (1, 1, timesteps, features), (batches, timesteps, features)]
            self.inputs = shapes[self.shape - 1]
            print("tunemodel self.inputs", self.inputs)



        print("tunemodel self.inputs", self.inputs)
        input_tensor = Input(shape=(self.inputs))  # The shape does not include the batch size
        self.inputs = input_tensor

        if self.tensorshape:
            self.traindataset = self.traindataset.map(lambda x, y: (tf.reshape(x, (-1,timesteps,features)), y))



        # Define and configure the tuner
        self.tuner = kt.Hyperband(
            hypermodel=self.build_model,
            objective=self.objective,
            max_epochs=self.max_epochs,
            factor=self.factor,
            directory=self.basepath,
            project_name=self.project_name,
            overwrite=self.overwrite,
        )

        # Configure the tuner's oracle
        self.tuner.oracle.max_fail_streak = self.max_consecutive_failed_trials
        # Display search space summary and begin tuning
        self.tuner.search_space_summary()

        
        self.tuner.search(
                        self.traindataset,
                        validation_data=self.valdataset,
                        epochs=HyperParameters().Int('epochs', min_value=self.min_epochs, max_value=self.max_epochs, step=self.step),
                        batch_size=hp.Int('batch_size', min_value=16, max_value=128, step=16),
                        callbacks=self.get_callbacks(),
        )
  #


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
            seq_len = timesteps
            model_dim = key_dim
            inputs_projected += self.positional_encoding(seq_len, model_dim)

            key_dim = hp.Int('key_dim', min_value=32, max_value=128, step=32, default=64)
            num_heads = hp.Int('num_heads', min_value=2, max_value=4, step=1, default=2)

            # Project inputs if necessary
            if self.inputs.shape[-1] != key_dim:
                inputs_projected = Dense(units=key_dim)(self.inputs)
            else:
                inputs_projected = self.inputs

            x_trans = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs_projected, inputs_projected)
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
        if self.multiactivate:
            output = Dense(1, activation=hp.Choice('output_activation', [self.activation2, self.activation3, self.activation4]))(x)
        else:
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

    def get_callbacks(self):
        checkpoint_filepath = os.path.join(self.basepath, 'best_model.keras')
        return [
            EarlyStopping(monitor=self.objective, patience=self.chk_patience, verbose=1),
            ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, verbose=1)
        ]
   

    def positional_encoding(seq_len, model_dim):
        positions = tf.range(0, seq_len, dtype=tf.float32)[:, tf.newaxis]
        dims = tf.range(0, model_dim, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000, (2 * (dims // 2)) / tf.cast(model_dim, tf.float32))
        angle_rads = positions * angle_rates
        angle_rads[:, 0::2] = tf.sin(angle_rads[:, 0::2])  # Apply sin to even indices
        angle_rads[:, 1::2] = tf.cos(angle_rads[:, 1::2])  # Apply cos to odd indices
        return angle_rads
    
    