import tensorflow as tf
# Ensure compatibility with TensorFlow v1 functions
tf.compat.v1.reset_default_graph()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Reshape
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
    def __init__(self, input_shape, input_tensor1, input_tensor2, input_tensor3, input_tensor4):
        self.input_shape = input_shape
        self.input_tensor1 = input_tensor1
        self.input_tensor2 = input_tensor2
        self.input_tensor3 = input_tensor3
        self.input_tensor4 = input_tensor4
        
        if not isinstance(self.input_shape, tuple):
            raise ValueError(f"input_shape must be a tuple, got {type(self.input_shape)}")

        print("Hybrid init Input shape:", input_shape, "Init input:", self.input_shape)
        
    def build(self, hp):
        ### Base Models ###       
        # LSTM Model
        def create_lstm_model(hp):
            inputs = Input(shape=self.input_shape)
            units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
            x = LSTM(units, return_sequences=True)(inputs)
            x = Dense(hp.Int('lstm_dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)
            return Model(inputs, x)  
        
        # 1D CNN Model
        def create_cnn_model(hp):
            inputs = Input(shape=self.input_shape)
            filters = hp.Int('cnn_filters', min_value=32, max_value=128, step=32)
            kernel_size = hp.Choice('cnn_kernel_size', values=[3, 5])
            x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
            x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
            x = Flatten()(x)
            x = Dense(hp.Int('cnn_dense_units', min_value=32, max_value=128, step=32), activation='sigmoid')(x)
            return Model(inputs, x)

        # GRU Model
        def create_gru_model(hp):
            inputs = Input(shape=self.input_shape)
            units = hp.Int('gru_units', min_value=32, max_value=128, step=32)
            x = GRU(units)(inputs)
            x = Dense(hp.Int('gru_dense_units', min_value=32, max_value=128, step=32), activation='sigmoid')(x)
            return Model(inputs, x)
               
        # Transformer Model
        def create_transformer_model(hp):
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
        lstm_model = create_lstm_model(hp)
        cnn_model = create_cnn_model(hp)
        gru_model = create_gru_model(hp)
        transformer_model = create_transformer_model(hp)
        
        # Adjust the output shapes
        lstm_output = Flatten()(lstm_model.output)
        cnn_output = Flatten()(cnn_model.output)
        gru_output = Flatten()(gru_model.output)
        transformer_output = Flatten()(transformer_model.output)
        
        print("All the same: lstm", lstm_output)
        print("All the same: cnn",  cnn_output)
        print("All the same: gru",  gru_output)
        print("All the same: tran", transformer_output)
        
        # Concatenate the outputs
        combined_output = Concatenate([lstm_output, cnn_output, gru_output, transformer_output])
    
        # Assuming tensor_inputs is a list of tensors
        tensor_inputs = [lstm_output, cnn_output, gru_output, transformer_output]

        # Ensure all elements in tensor_inputs are tensors
        #tensor_inputs = [tensor for tensor in tensor_inputs if isinstance(tensor, tf.Tensor)]
        
        # Debug prints to check the shapes and types of tensors
        for i, tensor in enumerate(tensor_inputs):
            if tensor is None:
               print(f"tensor_inputs[{i}] is None")
        else:
               print(f"tensor_inputs[{i}] shape: {tensor.shape}, type: {type(tensor)}")

        # Ensure all tensors are properly defined before concatenation
        if all(tensor is not None for tensor in tensor_inputs):
            concatenated = Concatenate()(tensor_inputs)
        else:
            raise ValueError("One or more tensors in tensor_inputs are None")
        
        
        # Concatenate the tensor inputs
        concatenated = Concatenate()(tensor_inputs)
  

        # Reshape if needed
        reshaped = Reshape((np.prod(concatenated.shape[1:]),))(concatenated)

        # Add Dense layers to learn from the combined outputs
        x = Dense(hp.Int('ensemble_dense_units', min_value=64, max_value=256, step=64), activation='relu')(reshaped)
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
        self.lp_objective = lp_objective
        self.lp_max_trials = lp_max_trials
        self.lp_executions_per_trial = lp_executions_per_trial
        self.lp_directory = lp_directory
        self.lp_project_name = lp_project_name
        self.lp_validation_split = lp_validation_split
        self.lp_epochs = lp_epochs
        self.lp_batch_size = lp_batch_size

    def run_tuner(self):
        tuner = RandomSearch(
           # HybridEnsembleHyperModel(input_shape=self.input_shape, input_tensor1=self.X_train, input_tensor2=self.X_train, input_tensor3=self.X_train, input_tensor4=self.X_train),
            HybridEnsembleHyperModel(input_shape=self.input_shape, input_tensor1=self.input_shape, input_tensor2=self.input_shape, input_tensor3=self.input_shape, input_tensor4=self.input_shape), 
            objective=self.lp_objective,  # Objective to optimize
            max_trials=self.lp_max_trials,  # Number of hyperparameter sets to try
            executions_per_trial=self.lp_executions_per_trial,  # Number of models to build and evaluate for each trial
            directory=self.lp_directory,
            project_name=self.lp_project_name
        )
        print("init tuner class Input shape:", self.input_shape)
        print("init tuner end!")
    
        # Train the tuner
        tuner.search([self.X_train, self.X_train, self.X_train, self.X_train], self.y_train, validation_split=self.lp_validation_split, epochs=self.lp_epochs, batch_size=self.lp_batch_size)
    
        # Get the best hyperparameters
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    
        # Print the best hyperparameters
        print(f"Best hyperparameters: {best_hp.values}")
    
        # Return the best model
        return tuner.get_best_models(num_models=1)[0]