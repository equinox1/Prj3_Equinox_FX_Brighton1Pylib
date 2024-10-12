import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from keras_tuner import HyperModel, RandomSearch
import numpy as np

# Transformer Block definition (used later in the transformer model)
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define a HyperModel class for the Keras Tuner
class HybridEnsembleHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        ### Base Models ###
        
        # LSTM Model
        def create_lstm_model():
            # Ensure input_shape is a tuple of integers
            if isinstance(self.input_shape, tuple):
                # Ensure all elements in the tuple are integers
                self.input_shape = tuple(int(dim) for dim in self.input_shape)
            else:
                raise ValueError(f"input_shape must be a tuple, got {type(self.input_shape)}")
            
            inputs = Input(shape=self.input_shape)
            units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
            x = LSTM(units)(inputs)
            x = Dense(hp.Int('lstm_dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)
            return Model(inputs, x)
        
        # 1D CNN Model
        def create_cnn_model():
            inputs = Input(shape=self.input_shape)
            filters = hp.Int('cnn_filters', min_value=32, max_value=128, step=32)
            kernel_size = hp.Choice('cnn_kernel_size', values=[3, 5])
            x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Flatten()(x)
            x = Dense(hp.Int('cnn_dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)
            return Model(inputs, x)
        
        # GRU Model
        def create_gru_model():
            inputs = Input(shape=self.input_shape)
            units = hp.Int('gru_units', min_value=32, max_value=128, step=32)
            x = GRU(units)(inputs)
            x = Dense(hp.Int('gru_dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)
            return Model(inputs, x)
        
        # Transformer Model
        def create_transformer_model():
            inputs = Input(shape=self.input_shape)
            embed_dim = hp.Int('transformer_embed_dim', min_value=32, max_value=128, step=32)
            num_heads = hp.Int('transformer_num_heads', min_value=2, max_value=8, step=2)
            ff_dim = hp.Int('transformer_ff_dim', min_value=32, max_value=128, step=32)
            x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(inputs, training=True)
            x = GlobalAveragePooling1D()(x)
            x = Dense(hp.Int('transformer_dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)
            return Model(inputs, x)
        
        ### Hybrid Ensemble Model ###
        # Instantiate base models
        lstm_model = create_lstm_model()
        cnn_model = create_cnn_model()
        gru_model = create_gru_model()
        transformer_model = create_transformer_model()
        
        # Concatenate the outputs from each base model
        combined_output = concatenate([lstm_model.output, cnn_model.output, gru_model.output, transformer_model.output])
        
        # Add Dense layers to learn from the combined outputs
        x = Dense(hp.Int('ensemble_dense_units', min_value=64, max_value=256, step=64), activation='relu')(combined_output)
        x = Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1))(x)
        output = Dense(1, activation='sigmoid')(x)
        
        # Create and compile the model
        model = Model(inputs=[lstm_model.input, cnn_model.input, gru_model.input, transformer_model.input], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    

class CMdtuner:
    def __init__(self, input_shape, X_train, y_train, lp_objective, lp_max_trials, lp_executions_per_trial, lp_directory, lp_project_name, lp_validation_split, lp_epochs, lp_batch_size):
        self.input_shape = input_shape
        self.X_train = X_train
        self.y_train = y_train
        self.lp_objective = 'val_accuracy' #lp_objective
        self.lp_max_trials = lp_max_trials
        self.lp_executions_per_trial = lp_executions_per_trial
        self.lp_directory = lp_directory
        self.lp_project_name = lp_project_name
        self.lp_validation_split = lp_validation_split
        self.lp_epochs = lp_epochs
        self.lp_batch_size = lp_batch_size
        self.hypermodel = HybridEnsembleHyperModel(self.input_shape)  # Make sure to assign this later
        print("init tuner class Input shape:", self.input_shape)
        print("init tuner end!")
    def run_tuner(self):
        # Ensure lp_objective is correctly formatted
        if isinstance(self.lp_objective, str):
            objective = self.lp_objective
        elif isinstance(self.lp_objective, Objective):
            objective = self.lp_objective
        else:
            raise TypeError(f"Invalid type for lp_objective: expected str or Objective, got {type(self.lp_objective)}")

        # Ensure hypermodel is defined
        if not hasattr(self, 'hypermodel') or self.hypermodel is None:
            raise ValueError("hypermodel must be defined before running the tuner.")

        # Convert directory and project_name to strings to avoid the TypeError
        directory = str(self.lp_directory)
        project_name = str(self.lp_project_name)

        # Initialize the tuner
        tuner = RandomSearch(
            hypermodel=self.hypermodel,  # Ensure hypermodel is passed
            objective=objective,
            max_trials=self.lp_max_trials,
            executions_per_trial=self.lp_executions_per_trial,
            directory=directory,  # Ensure this is a string
            project_name=project_name  # Ensure this is a string
        )
        return tuner


