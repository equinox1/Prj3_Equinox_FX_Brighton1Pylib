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
        self.X_train = kwargs.get('X_train', None)
        self.y_train = kwargs.get('y_train', None)
        self.X_test = kwargs.get('X_test', None)
        self.y_test = kwargs.get('y_test', None)
        
        # Set model component flags
        self.cnn_model = kwargs.get('cnn_model', False)
        self.lstm_model = kwargs.get('lstm_model', False)
        self.gru_model = kwargs.get('gru_model', False)
        self.transformer_model = kwargs.get('transformer_model', False)
        
        # Set training and tuning parameters
        self.objective = kwargs.get('objective', 'val_loss')
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.factor = kwargs.get('factor', 3)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.overwrite = kwargs.get('overwrite', True)
        self.project_name = kwargs.get('project_name', 'tuning_project')
        self.basepath = kwargs.get('basepath', './')
        self.chk_monitor = kwargs.get('chk_monitor', 'val_loss')
        self.chk_patience = kwargs.get('chk_patience', 5)
        
        # Default inputs if not provided
        if 'inputs' in kwargs:
            self.inputs = kwargs['inputs']
        else:
            # Fall back to an Input layer shape based on X_train if provided
            if self.X_train is not None:
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

        self.tuner.search_space_summary()
        self.tuner.search(self.X_train, self.y_train,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          validation_data=(self.X_test, self.y_test),
                          callbacks=self.get_callbacks())

        
    def build_model(self, hp):
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

        # Concatenate and build the model
        combined = Concatenate()([x for x in [x_cnn, x_lstm, x_gru, x_trans] if x is not None])
        x = Dense(50, activation='relu')(combined)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='linear')(x)

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
    
    # Example fix in the get_callbacks method
    def get_callbacks(self):
        checkpoint_filepath = os.path.join(self.basepath, 'best_model.weights.h5')  # Use .weights.h5 extension
        return [
                EarlyStopping(monitor=self.chk_monitor, patience=self.chk_patience),
                ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, save_best_only=True, monitor=self.chk_monitor, verbose=1)
    ]

