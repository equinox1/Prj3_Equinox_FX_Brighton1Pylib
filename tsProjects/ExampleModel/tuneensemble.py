import tensorflow as tf
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D

from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import LayerNormalization
from keras._tf_keras.keras.layers import Input,LSTM,GRU,Conv1D, MaxPooling1D, Flatten, Dense,Dropout, concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from keras_tuner import HyperModel
from keras_tuner import RandomSearch

from keras_tuner import HyperModel
from keras_tuner import RandomSearch
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

    def call(self, inputs, training):
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
            x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(inputs)
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
            x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(inputs)
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

# Set up the Keras Tuner
def run_tuner(input_shape, X_train, y_train):
    tuner = RandomSearch(
        HybridEnsembleHyperModel(input_shape=input_shape),
        objective='val_accuracy',
        max_trials=10,  # Number of hyperparameter sets to try
        executions_per_trial=1,  # Number of models to build and evaluate for each trial
        directory='hybrid_ensemble_tuning',
        project_name='hybrid_ensemble_model'
    )
    
    # Train the tuner
    tuner.search([X_train, X_train, X_train, X_train], y_train, validation_split=0.2, epochs=10, batch_size=32)
    
    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Print the best hyperparameters
    print(f"Best hyperparameters: {best_hp.values}")
    
    # Return the best model
    return tuner.get_best_models(num_models=1)[0]

# Example data
X_train = np.random.rand(1000, 100, 1)  # 1000 samples, 100 time steps, 1 feature
y_train = np.random.randint(2, size=(1000,))  # Binary target

# Define input shape
input_shape = (100, 1)

# Run the tuner
best_model = run_tuner(input_shape, X_train, y_train)

# Print the summary of the best model
best_model.summary()

# Optionally: train the best model on the full dataset
# best_model.fit([X_train, X_train, X_train, X_train], y_train, epochs=10, batch_size=32)
