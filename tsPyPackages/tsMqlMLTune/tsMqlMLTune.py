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
import os
from keras_tuner import HyperParameters

class CMdtuner:
    def __init__(self, **kwargs):
        # Load dataset and model configuration parameters
        self.traindataset = kwargs.get('traindataset')
        self.valdataset = kwargs.get('valdataset')
        self.testdataset = kwargs.get('testdataset')

        # Load model configuration parameters
        self.cnn_model = kwargs.get('cnn_model', False)
        self.lstm_model = kwargs.get('lstm_model', False)
        self.gru_model = kwargs.get('gru_model', False)
        self.transformer_model = kwargs.get('transformer_model', False)

        # Load hyperparameter tuning configuration parameters
        self.objective = kwargs.get('objective', 'val_loss')
        self.max_epochs = kwargs.get('max_epochs', 10)
        self.min_epochs = kwargs.get('min_epochs', 1)
        self.epochs = kwargs.get('epochs', 10)
        self.factor = kwargs.get('factor', 3)
        self.seed = kwargs.get('seed', 42)
        self.hyperband_iterations = kwargs.get('hyperband_iterations', 1)
        self.tune_new_entries = kwargs.get('tune_new_entries', True)
        self.allow_new_entries = kwargs.get('allow_new_entries', True)
        self.max_retries_per_trial = kwargs.get('max_retries_per_trial', 3)
        self.max_consecutive_failed_trials = kwargs.get('max_consecutive_failed_trials', 30)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.batch_size = kwargs.get('batch_size', 32)
        self.dropout = kwargs.get('dropout', 0.3)
        self.oracle = kwargs.get('oracle', 'hyperband')
        self.activation1 = kwargs.get('activation1', 'relu')
        self.activation2 = kwargs.get('activation2', 'relu')
        self.activation3 = kwargs.get('activation3', 'relu')
        self.activation4 = kwargs.get('activation4', 'linear')
        self.hypermodel = kwargs.get('hypermodel')
        self.max_model_size = kwargs.get('max_model_size', 1000000)
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.loss = kwargs.get('loss', 'mean_squared_error')
        self.metrics = kwargs.get('metrics', ['mean_absolute_error'])
        self.distribution_strategy = kwargs.get('distribution_strategy')
        self.directory = kwargs.get('directory')
        self.basepath = kwargs.get('basepath')
        self.project_name = kwargs.get('project_name')
        self.logger = kwargs.get('logger')
        self.tuner_id = kwargs.get('tuner_id')
        self.overwrite = kwargs.get('overwrite', False)
        self.executions_per_trial = kwargs.get('executions_per_trial', 1)

        # Load checkpoint configuration parameters
        self.chk_fullmodel = kwargs.get('chk_fullmodel', False)
        self.chk_verbosity = kwargs.get('chk_verbosity', 1)
        self.chk_mode = kwargs.get('chk_mode', 'auto')
        self.chk_monitor = kwargs.get('chk_monitor', 'val_loss')
        self.chk_sav_freq = kwargs.get('chk_sav_freq', 'epoch')
        self.chk_patience = kwargs.get('chk_patience', 0)
        self.checkpoint_filepath = kwargs.get('checkpoint_filepath')
        self.modeldatapath = kwargs.get('modeldatapath')
        self.step = kwargs.get('step', 5)
        self.multiactivate = kwargs.get('multiactivate', False)
        self.tf1 = kwargs.get('tf1', False)
        self.tf2 = kwargs.get('tf2', False)
        self.shape = kwargs.get('shape', 0)
        self.tensorshape = kwargs.get('tensorshape', False)

        # Enable tensorflow debugging
        if self.tf1:
            tf.debugging.set_log_device_placement(self.tf1)
        if self.tf2:
            tf.debugging.enable_check_numerics()

        # Ensure the output directory exists
        os.makedirs(self.basepath, exist_ok=True)

        # Define inputs (as a list of shape components or a shape tuple)
        self.inputs = kwargs.get('inputs')

        print("tunemodel self.inputs", self.inputs)
        print("tunemodel self.inputs[0] Rows:  ", self.inputs[0])
        print("tunemodel self.inputs[1] Batch size: None is dynamic: ", self.inputs[1]," variable self.batch_size:", self.batch_size)
        print("tunemodel self.inputs[2] Timesteps or Sequence Length):", self.inputs[2])
        print("tunemodel self.inputs[3] Features or Observations per Timestep:", self.inputs[3])
        if len(self.inputs) > 4 :
            print("tunemodel self.inputs[4] Channels:", self.inputs[4])
        else:
            print("tunemodel self.inputs[4] Channels: None:")

        if len(self.inputs) > 4:
            print("tunemodel self.inputs[4] Channels:", self.inputs[4])

        # Prepare shapes (local variables)
        rows = self.inputs[0] if len(self.inputs) > 0 else 1
        batches = self.inputs[1] if len(self.inputs) > 1 else None
        timesteps = self.inputs[2]
        features = self.inputs[3]
        channels = self.inputs[4] if len(self.inputs) > 4 else 1

        print("tunemodel inputs", self.inputs)
        print("tunemodel rows", rows)
        print("tunemodel batches", batches, "batch_size", self.batch_size)
        print("tunemodel timesteps", timesteps)
        print("tunemodel features", features)

        # Save as class attributes if needed in build_model
        self.rows = rows
        self.batches = batches
        self.timesteps = timesteps
        self.features = features
        self.channels = channels

        # Correct the input shape if self.shape is in a certain set
        if self.shape in {1, 2, 3, 4, 5, 6, 7}:
            shapes = [
                (features,),
                (timesteps, features),
                (batches, timesteps, features),
                (rows, batches, timesteps, features),
                (1, timesteps, features),
                (1, 1, timesteps, features),
                (self.batch_size, timesteps, features),
            ]
            self.inputs = shapes[self.shape - 1]
            print("tunemodel self.inputs", self.inputs)

        # Prepare the dataset for training
        input_tensor = Input(shape=self.inputs)  # batch size not included here
        self.inputs = input_tensor

        if self.tensorshape:
            # If you need to reshape each batch item dynamically
            self.traindataset = self.traindataset.map(
                lambda x, y: (tf.reshape(x, (-1, timesteps, features)), y)
            )

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

        # Display search space summary
        self.tuner.search_space_summary()

        # Start the hyperparameter search
        hp = HyperParameters()
        self.tuner.search(
            self.traindataset,
            validation_data=self.valdataset,
            epochs=hp.Int('epochs', min_value=self.min_epochs, max_value=self.max_epochs, step=self.step),
            batch_size=hp.Int('batch_size', min_value=16, max_value=128, step=16),
            callbacks=self.get_callbacks(),
        )

    def build_model(self, hp):
        """
        Builds a model depending on which branches (CNN, LSTM, GRU, Transformer) are set to True.
        """
        if not any([self.cnn_model, self.lstm_model, self.gru_model, self.transformer_model]):
            raise ValueError("At least one model branch must be specified (CNN, LSTM, GRU, or Transformer).")

        # Because we are inside build_model, self.inputs is a Keras Input object.
        # Rely on its static shape if needed:
        input_shape = self.inputs.shape

        # Retrieve shapes from instance variables
        rows = self.rows
        batches = self.batches
        timesteps = self.timesteps
        features = self.features
        channels = self.channels
        batch_size = self.batch_size

        # Model branches
        x_cnn, x_lstm, x_gru, x_trans = None, None, None, None

        # CNN branch
        if self.cnn_model:
            x_cnn = Conv1D(
                filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32, default=64),
                kernel_size=hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1, default=3),
                activation='relu'
            )(self.inputs)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Flatten()(x_cnn)

        # LSTM branch
        if self.lstm_model:
            x_lstm = LSTM(
                units=hp.Int('lstm_units', min_value=32, max_value=128, step=32, default=64),
                return_sequences=True
            )(self.inputs)
            x_lstm = LSTM(
                units=hp.Int('lstm_units', min_value=32, max_value=128, step=32, default=64)
            )(x_lstm)

        # GRU branch
        if self.gru_model:
            x_gru = GRU(
                units=hp.Int('gru_units', min_value=32, max_value=128, step=32, default=64),
                return_sequences=True
            )(self.inputs)
            x_gru = GRU(
                units=hp.Int('gru_units', min_value=32, max_value=128, step=32, default=64)
            )(x_gru)

        # Transformer branch
        if self.transformer_model:
            # For time series, typically:
            seq_len = self.timesteps  # timesteps
            key_dim = hp.Int('key_dim', min_value=32, max_value=128, step=32, default=64)
            num_heads = hp.Int('num_heads', min_value=2, max_value=4, step=1, default=2)

            # If needed, project inputs from 'features' to 'key_dim'.
            if self.inputs.shape[-1] != key_dim:
                inputs_projected = Dense(units=key_dim)(self.inputs)
                print("Trans: self inputs:",self.inputs, ", self.inputs[-1]" ,self.inputs[-1] ,"key_dim", key_dim," not equal so projected", inputs_projected)
            else:
                inputs_projected = self.inputs
                print("Trans: inputs_projected", inputs_projected)
                
                

            # Add positional encoding:
            pos_encoding = self.positional_encoding(seq_len, key_dim)
            pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # shape (1, seq_len, key_dim)
            inputs_projected = inputs_projected + pos_encoding

            # Multi-head attention
            x_trans = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
                inputs_projected, inputs_projected
            )
            x_trans = LayerNormalization(epsilon=1e-6)(x_trans)
            # Pool along the time dimension
            x_trans = GlobalAveragePooling1D()(x_trans)

        # Concatenate branches and build output
        branches = [x for x in [x_cnn, x_lstm, x_gru, x_trans] if x is not None]
        if len(branches) > 1:
            combined = Concatenate()(branches)
        else:
            combined = branches[0]

        x = Dense(50, activation=self.activation1)(combined)
        x = Dropout(0.3)(x)

        if self.multiactivate:
            # Let the HP choose which final activation to use
            output = Dense(
                1,
                activation=hp.Choice('output_activation', [
                    self.activation2, self.activation3, self.activation4
                ])
            )(x)
        else:
            output = Dense(1, activation=self.activation2)(x)
        
        print("create model with inputs", self.inputs, "output", output)
        model = Model(inputs=self.inputs, outputs=output)
        return self.compile_model(model, hp)

    def compile_model(self, model, hp):
        model.compile(
            optimizer=Adam(
                learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
            ),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )
        return model

    def get_callbacks(self):
        """Returns the list of callbacks used during model training."""
        checkpoint_filepath = os.path.join(self.basepath, 'best_model.keras')
        
        # You can customize your log directory here
        log_dir = os.path.join(self.basepath, 'logs')
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        return [
            EarlyStopping(monitor=self.objective, patience=self.chk_patience, verbose=1),
            ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, verbose=1),
            tensorboard_callback
        ]

    @staticmethod
    def positional_encoding(seq_len, model_dim):
        """
        Generates a standard sine/cosine positional encoding for a sequence of length `seq_len`
        and embedding dimension `model_dim`.

        Returns a tensor of shape `[seq_len, model_dim]`.
        """
        positions = tf.range(0, seq_len, dtype=tf.float32)[:, tf.newaxis]  # (seq_len, 1)
        dims = tf.range(0, model_dim, dtype=tf.float32)[tf.newaxis, :]      # (1, model_dim)
        angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(model_dim, tf.float32))
        angle_rads = positions * angle_rates

        # Apply sin to even indices and cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        # Initialize final encoding matrix of zeros
        pos_encoding = tf.zeros_like(angle_rads)
        # Assign sines and cosines
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            indices=[[i, j] for i in range(seq_len) for j in range(0, model_dim, 2)],
            updates=tf.reshape(sines, [-1])
        )
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            indices=[[i, j] for i in range(seq_len) for j in range(1, model_dim, 2)],
            updates=tf.reshape(cosines, [-1])
        )

        return pos_encoding
