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
        # ---------------------
        # 1) Read kwargs
        # ---------------------
        self.traindataset = kwargs.get('traindataset')
        self.valdataset = kwargs.get('valdataset')
        self.testdataset = kwargs.get('testdataset')
        self.trainshape = kwargs.get('trainshape')
        self.valshape = kwargs.get('valshape')
        self.testshape = kwargs.get('testshape')

        self.cnn_model = kwargs.get('cnn_model', False)
        self.lstm_model = kwargs.get('lstm_model', False)
        self.gru_model = kwargs.get('gru_model', False)
        self.transformer_model = kwargs.get('transformer_model', False)

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
        self.tensorshape = kwargs.get('tensorshape', False)
        self.shape = kwargs.get('shape', 2)
        
        self.cnn_shape = kwargs.get('cnn_shape')
        self.lstm_shape = kwargs.get('lstm_shape')
        self.gru_shape = kwargs.get('gru_shape')
        self.transformer_shape = kwargs.get('transformer_shape')

        self.inputs = None
        self.cnn_inputs = None
        self.lstm_inputs = None
        self.gru_inputs = None
        self.transformer_inputs = None
        
        self.multi_inputs = kwargs.get('multi_inputs', False)
        self.traindatainput1 = kwargs.get('traindatainput1', 70)
        self.traindatainput2 = kwargs.get('traindatainput2', 3)
        self.traindatainput3 = kwargs.get('traindatainput3', 7)

        # If user provided something like `(24, 7)`, store in self.input_shape.
        self.input_shape = kwargs.get('inputs', None)

        # Enable TF debugging if requested
        if self.tf1:
            tf.debugging.set_log_device_placement(self.tf1)
        if self.tf2:
            tf.debugging.enable_check_numerics()

        # Ensure base path exists
        os.makedirs(self.basepath, exist_ok=True)

        # ------------- 2) Prepare shapes -------------
        self.prepare_shapes()

        # ------------- 3) Create the Keras Input layers -------------
        # For a multi-branch approach, each branch can have its own Input.
        if self.input_shape is not None:
            self.inputs = Input(shape=self.input_shape, name="main_input")

        if self.cnn_model and (self.cnn_input_shape is not None):
            self.cnn_inputs = Input(shape=self.cnn_input_shape, name="cnn_input")

        if self.lstm_model and (self.lstm_input_shape is not None):
            self.lstm_inputs = Input(shape=self.lstm_input_shape, name="lstm_input")

        if self.gru_model and (self.gru_input_shape is not None):
            self.gru_inputs = Input(shape=self.gru_input_shape, name="gru_input")

        if self.transformer_model and (self.transformer_input_shape is not None):
            self.transformer_inputs = Input(
                shape=self.transformer_input_shape, name="transformer_input"
            )

        # If user wants dynamic reshaping in the dataset (be careful with extra dims!)
        # For example, if your data truly needs shaping into (timesteps, features):
        if self.tensorshape and self.traindataset is not None and self.valdataset is not None:
            # Example logic: ensure data is (batch, timesteps, features)
            timesteps = self.input_shape[0] if len(self.input_shape) >= 1 else 24
            features = self.input_shape[1] if len(self.input_shape) >= 2 else 7

            def reshape_fn(x, y):
                # Removes any extra dims, sets shape to (timesteps, features)
                x_reshaped = tf.reshape(x, (tf.shape(x)[0], timesteps, features))
                return x_reshaped, y

            self.traindataset = self.traindataset.map(reshape_fn)
            self.valdataset = self.valdataset.map(reshape_fn)

        # --------------- 4) Define the Tuner ---------------
        self.tuner = kt.Hyperband(
            hypermodel=self.build_model,  
            objective=self.objective,
            max_epochs=self.max_epochs,
            factor=self.factor,
            directory=self.basepath,
            project_name=self.project_name,
            overwrite=self.overwrite,
        )
        self.tuner.oracle.max_fail_streak = self.max_consecutive_failed_trials

        # Display search space summary
        self.tuner.search_space_summary()

        # 5) Start search
        hp = HyperParameters()
        self.tuner.search(
            self.traindataset,
            validation_data=self.valdataset,
            epochs=hp.Int('epochs', min_value=self.min_epochs, max_value=self.max_epochs, step=self.step),
            batch_size=hp.Int('batch_size', min_value=16, max_value=128, step=16),
            callbacks=self.get_callbacks(),
        )

    # ----------------------------------------
    # Build the model (KerasTuner callback)
    # ----------------------------------------
    def build_model(self, hp):
        """Builds a model depending on which branches (CNN, LSTM, GRU, Transformer) are True."""
        if not any([self.cnn_model, self.lstm_model, self.gru_model, self.transformer_model]):
            raise ValueError("At least one model branch must be specified (CNN, LSTM, GRU, or Transformer).")

        branches = []

        # ----- CNN branch -----
        if self.cnn_model and self.cnn_inputs is not None:
            x_cnn = Conv1D(
                filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32, default=64),
                kernel_size=hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1, default=3),
                activation='relu'
            )(self.cnn_inputs)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Flatten()(x_cnn)
            branches.append(x_cnn)

        # ----- LSTM branch -----
        if self.lstm_model and self.lstm_inputs is not None:
            x_lstm = LSTM(
                units=hp.Int('lstm_units', min_value=32, max_value=128, step=32, default=64),
                return_sequences=True
            )(self.lstm_inputs)
            x_lstm = LSTM(
                units=hp.Int('lstm_units', min_value=32, max_value=128, step=32, default=64)
            )(x_lstm)
            branches.append(x_lstm)

        # ----- GRU branch -----
        if self.gru_model and self.gru_inputs is not None:
            x_gru = GRU(
                units=hp.Int('gru_units', min_value=32, max_value=128, step=32, default=64),
                return_sequences=True
            )(self.gru_inputs)
            x_gru = GRU(
                units=hp.Int('gru_units', min_value=32, max_value=128, step=32, default=64)
            )(x_gru)
            branches.append(x_gru)

        # ----- Transformer branch -----
        if self.transformer_model and self.transformer_inputs is not None:
            key_dim = hp.Int('key_dim', min_value=32, max_value=128, step=32, default=64)
            num_heads = hp.Int('num_heads', min_value=2, max_value=4, step=1, default=2)

            seq_len = self.transformer_input_shape[0]  
            feat_dim = self.transformer_input_shape[1] if len(self.transformer_input_shape) >= 2 else 7

            # If needed, project to key_dim
            if feat_dim != key_dim:
                inputs_projected = Dense(units=key_dim)(self.transformer_inputs)
            else:
                inputs_projected = self.transformer_inputs

            # Positional encoding
            pos_encoding = self.positional_encoding(seq_len, key_dim)
            pos_encoding = tf.expand_dims(pos_encoding, axis=0)
            inputs_projected = inputs_projected + pos_encoding

            x_trans = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
                inputs_projected, inputs_projected
            )
            x_trans = LayerNormalization(epsilon=1e-6)(x_trans)
            x_trans = GlobalAveragePooling1D()(x_trans)
            branches.append(x_trans)

        # If multiple branches, concatenate their outputs
        if len(branches) > 1:
            combined = Concatenate()(branches)
        else:
            combined = branches[0]

        # Dense + Dropout
        x = Dense(50, activation=self.activation1)(combined)
        x = Dropout(self.dropout)(x)

        # Output layer with optional multiple activation choices
        if self.multiactivate:
            output = Dense(
                1,
                activation=hp.Choice('output_activation', [
                    self.activation2, self.activation3, self.activation4
                ])
            )(x)
        else:
            output = Dense(1, activation=self.activation2)(x)

        # Gather all "active" inputs
        all_inputs = []
        if self.inputs is not None:
            all_inputs.append(self.inputs)
        if self.cnn_model and self.cnn_inputs is not None:
            all_inputs.append(self.cnn_inputs)
        if self.lstm_model and self.lstm_inputs is not None:
            all_inputs.append(self.lstm_inputs)
        if self.gru_model and self.gru_inputs is not None:
            all_inputs.append(self.gru_inputs)
        if self.transformer_model and self.transformer_inputs is not None:
            all_inputs.append(self.transformer_inputs)

        # ----- Correct multi-input usage -----
        # If user truly wants separate branches for separate data, do NOT
        # concatenate input layers. Just pass them as a list to Model(...).
        # Or if we have only one input, pass the single one.
        if self.multi_inputs and len(all_inputs) > 1:
            # Multi-input multi-branch
            model = Model(inputs=all_inputs, outputs=output)
            print("Model with multiple inputs created.")
        else:
            # Single-input model (most common case)
            # If multiple branches but from the *same* input, we only pass one Input
            # or you can specify e.g. all_inputs[0] if that's your "main" input
            model = Model(inputs=all_inputs[0], outputs=output)
            print("Model with single input created.")

        return self.compile_model(model, hp)

    def compile_model(self, model, hp):
        model.compile(
            optimizer=Adam(
                learning_rate=hp.Float(
                    'lr', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3
                )
            ),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )
        return model

    def get_callbacks(self):
        checkpoint_filepath = os.path.join(self.basepath, 'best_model.keras')
        log_dir = os.path.join(self.basepath, 'logs')
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        return [
            EarlyStopping(monitor=self.objective, patience=self.chk_patience, verbose=1),
            ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_best_only=True,
                verbose=1,
                monitor=self.chk_monitor
            ),
            tensorboard_callback
        ]

    @staticmethod
    def positional_encoding(seq_len, model_dim):
        positions = tf.range(0, seq_len, dtype=tf.float32)[:, tf.newaxis]  # (seq_len, 1)
        dims = tf.range(0, model_dim, dtype=tf.float32)[tf.newaxis, :]     # (1, model_dim)
        angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(model_dim, tf.float32))
        angle_rads = positions * angle_rates

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.zeros_like(angle_rads)
        seq_indices_even = [[i, j] for i in range(seq_len) for j in range(0, model_dim, 2)]
        seq_indices_odd = [[i, j] for i in range(seq_len) for j in range(1, model_dim, 2)]

        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            indices=seq_indices_even,
            updates=tf.reshape(sines, [-1])
        )
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            indices=seq_indices_odd,
            updates=tf.reshape(cosines, [-1])
        )
        return pos_encoding

    # ----------------------------------------
    # prepare_shapes() and get_predefined_shape()
    # ----------------------------------------
    def prepare_shapes(self):
        valid_shapes = range(1, 10)
        print("prepare_shapes() called.")
        print("Initial self.input_shape:", self.input_shape)
        print(f"Requested main shape index: {self.shape}")
        print(f"Requested CNN shape index: {self.cnn_shape}")
        print(f"Requested LSTM shape index: {self.lstm_shape}")
        print(f"Requested GRU shape index: {self.gru_shape}")
        print(f"Requested Transformer shape index: {self.transformer_shape}")
        print(f"Requested batch_size: {self.batch_size}")

        # Update from trainshape if provided
        if self.trainshape[0]:
            self.traindatainput1 = self.trainshape[0]
        if self.trainshape[1]:
            self.traindatainput2 = self.trainshape[1]
        if self.trainshape[2]:
            self.traindatainput3 = self.trainshape[2]

        print(f"Train data shape: {self.traindatainput1}, {self.traindatainput2}, {self.traindatainput3}")

        # If we have something like (rows, batches, timesteps, features, channels)
        # or (timesteps, features)
        if self.input_shape is not None and len(self.input_shape) >= 4:
            self.rows = self.input_shape[0]
            self.batches = self.input_shape[1]
            self.timesteps = self.input_shape[2]
            self.features = self.input_shape[3]
            if len(self.input_shape) > 4:
                self.channels = self.input_shape[4]
            else:
                self.channels = 1
            print(f"Parameter shapes: Rows: {self.rows}, Batches: {self.batches}, "
                  f"Timesteps: {self.timesteps}, Features: {self.features}, Channels: {self.channels}")
        elif self.input_shape is not None and len(self.input_shape) == 2:
            # e.g., (timesteps, features)
            self.timesteps = self.input_shape[0]
            self.features = self.input_shape[1]
            self.rows = None
            self.batches = None
            self.channels = 1
        else:
            # fallback defaults
            self.rows = None
            self.batches = None
            self.timesteps = 24
            self.features = 7
            self.channels = 1

        # If user didn't provide a shape at all
        if self.input_shape is None:
            self.input_shape = (self.timesteps, self.features)

        # 1) Main shape
        if (self.shape in valid_shapes):
            shape_tuple = self.get_predefined_shape(
                idx=self.shape,
                batch_size=self.batch_size,
                rows=self.rows,
                batches=self.batches,
                timesteps=self.timesteps,
                features=self.features,
                channels=self.channels
            )
            self.input_shape = shape_tuple
            print(f"[SHAPE] Final main input_shape = {self.input_shape}")

        # 2) CNN shape
        self.cnn_input_shape = None
        if self.cnn_model and (self.cnn_shape in valid_shapes):
            shape_tuple = self.get_predefined_shape(
                idx=self.cnn_shape,
                batch_size=self.batch_size,
                rows=self.rows,
                batches=self.batches,
                timesteps=self.timesteps,
                features=self.features,
                channels=self.channels
            )
            self.cnn_input_shape = shape_tuple
            print(f"[CNN] Final cnn_input_shape = {self.cnn_input_shape}")

        # 3) LSTM shape
        self.lstm_input_shape = None
        if self.lstm_model and (self.lstm_shape in valid_shapes):
            shape_tuple = self.get_predefined_shape(
                idx=self.lstm_shape,
                batch_size=self.batch_size,
                rows=self.rows,
                batches=self.batches,
                timesteps=self.timesteps,
                features=self.features,
                channels=self.channels
            )
            self.lstm_input_shape = shape_tuple
            print(f"[LSTM] Final lstm_input_shape = {self.lstm_input_shape}")

        # 4) GRU shape
        self.gru_input_shape = None
        if self.gru_model and (self.gru_shape in valid_shapes):
            shape_tuple = self.get_predefined_shape(
                idx=self.gru_shape,
                batch_size=self.batch_size,
                rows=self.rows,
                batches=self.batches,
                timesteps=self.timesteps,
                features=self.features,
                channels=self.channels
            )
            self.gru_input_shape = shape_tuple
            print(f"[GRU] Final gru_input_shape = {self.gru_input_shape}")

        # 5) Transformer shape
        self.transformer_input_shape = None
        if self.transformer_model and (self.transformer_shape in valid_shapes):
            shape_tuple = self.get_predefined_shape(
                idx=self.transformer_shape,
                batch_size=self.batch_size,
                rows=self.rows,
                batches=self.batches,
                timesteps=self.timesteps,
                features=self.features,
                channels=self.channels
            )
            self.transformer_input_shape = shape_tuple
            print(f"[TRANSFORMER] Final transformer_input_shape = {self.transformer_input_shape}")

    def get_predefined_shape(
        self,
        idx: int,
        batch_size: int,
        rows: int,
        batches: int,
        timesteps: int,
        features: int,
        channels: int
    ):
        """
        Return a shape tuple for the given index.
        Usually, do NOT include the batch dimension in these shapes,
        because Keras Input(...) expects (None, *shape).
        """
        shapes = [
            (features,),                           # shape=1
            (timesteps, features),                 # shape=2
            (batches, timesteps, features),        # shape=3
            (rows, batches, timesteps, features),  # shape=4
            (self.traindatainput1, timesteps, features),  # shape=5
            (self.traindatainput1, batches, timesteps, features),  # shape=6
            (batch_size, timesteps, features),     # shape=7 (rarely used in Input)
            (batches, timesteps, features, channels),  # shape=8
            (batches, self.traindatainput1, timesteps, features),  # shape=9
        ]
        return shapes[idx - 1]
