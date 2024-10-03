#+------------------------------------------------------------------+
#|                                                    tsMqlmod1.pyw
#|                                                    Tony Shepherd |
#|                                                  www.equinox.com |
#+------------------------------------------------------------------+
#property copyright "Tony Shepherd"
#property link      "www.equinox.com"
#property version   "1.01"
#+-------------------------------------------------------------------
# Classes for MQL
#+-------------------------------------------------------------------
import numpy as np
import pandas as pd
import tabulate
#+-------------------------------------------------------------------
# import ai packages scikit-learns
#+-------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
#+-------------------------------------------------------------------
# import keras package
#+-------------------------------------------------------------------
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
#======================================================
# import ai packages tensorflow and keras libraries
#======================================================
import tensorflow as tf
import keras_tuner as kt

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D

#-----------------------------------------------i
# Class CMqlmlsetup
#+--------------------------------------------------
class CMqlmlsetup:
    def __init__(self, hp ,df,X=None, y=None, X_train=None, X_train_scaled=None, X_test=None, X_test_scaled=None, y_train=None, y_test=None):
        if X is None:
            X = []
        if y is None:
            y = []
        if X_train is None:
            X_train = []
        if X_train_scaled is None:
            X_train_scaled = []
        if X_test is None:
            X_test = []
        if X_test_scaled is None:
            X_test_scaled = []
        if y_train is None:
            y_train = []
        if y_test is None:
            y_test = []
        self._df = df
    @property
    def df(self):
        return self.df

    @df.setter
    def dfx(self, value):
        self._df = value
    @property
    def X(self):
        return self.X
    @X.setter
    def X(self, value):
        value = self._X = []
        self._X = value
    @property
    def y(self):
        return self.y
    @y.setter
    def y(self, value):
        value = self._y = []
        self._y = value
    @property
    def X_train(self):
        return self.X_train
    @X_train.setter
    def X_train(self, value):
        self._X_train = value
    @property
    def X_test(self):
        return self.X_test
    @X_test.setter
    def X_test(self, value):
        self._X_test = value
    @property
    def X_train_scaled(self):
        return self.X_train_scaled
    @X_train_scaled.setter

    def X_train_scaled(self, value):
        scaler = StandardScaler()
        value = scaler.fit_transform(self.X_train)
        self._X_train_scaled = value

    @property
    def X_test_scaled(self):
        return self.X_test_scaled
    @X_test_scaled.setter
    def X_test_scaled(self, value):
        scaler = StandardScaler()
        value = scaler.transform(self.X_test)
        self._X_test_scaled = value

    @property
    def y_train(self):
        return self.y_train
    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def y_test(self):
        return self.y_test
    @y_test.setter
    def y_test(self, value):
        self._y_test = value
    

#--------------------------------------------------------------------
# create method  "dl_split_data_sets".
# class:cmqlmlsetup
# usage: mql data
# \pdlsplit data

#--------------------------------------------------------------------
    def dl_split_data_sets(self,df, X, y, test_size=0.2, shuffle = False, prog = 1):
        # sourcery skip: instance-method-first-arg-name
        
        X = df[['close']]
        y = df['target']
        # Split the data into training and testing sets
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        X_train,y_train,X_test,y_test =  train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        if prog == 1:
            return X_train
        if prog == 2:
            return y_train
        if prog == 3:
            return X_test
        if prog == 4:
            return y_test
#--------------------------------------------------------------------
# create method  "dl_train_model_scaled".
# class:cmqlmlsetup
# usage: mql data
# \param  var
#--------------------------------------------------------------------
    def dl_train_model_scaled(df):
        # sourcery skip: instance-method-first-arg-name
        # meta names
        scaler = StandardScaler()
        return scaler.fit_transform(df)

#--------------------------------------------------------------------
# create method  "dl_test_model_scaled".
# class:cmqlmlsetup
# usage: mql data
# \param  var
#----------test----------------------------------------------------
    def dl_test_model_scaled(df):
        # sourcery skip: instance-method-first-arg-name
        scaler = StandardScaler()
        return scaler.fit_transform(df)

#--------------------------------------------------------------------
# create method  "dl_train_model".
# class:cmqlmlsetup  usage: mql data
# \param  var
#--------------------------------------------------------------------
    def dl_train_model(self,lp_model,lp_X_train_scaled, lp_y_train, epoch = 1, batch_size = 256, validation_split = 0.2, verbose =1):
        # sourcery skip: instance-method-first-arg-name
        lp_X_train_scaled= lp_X_train_scaled[:len(lp_y_train)]  # Truncate 'x' to match 'y'
        lp_model.fit(
            lp_X_train_scaled,
            lp_y_train,
            epochs=epoch,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return lp_model


#--------------------------------------------------------------------
# create method  "dl_predict_network"
# class:cmqlmlsetup
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_predict_values(self,df, model, seconds = 60):
        # sourcery skip: instance-method-first-arg-name
        # Use the model to predict the next N instances
        X_predict=[]
        X_predict_scaled=[]
        predictions = pd.DataFrame()
        predictions=[]
        scaler = StandardScaler()
        # Empty DataFrame
        print("dftail:",df.tail(seconds)[['close']].values)
        X_predict = df.tail(seconds)[['close']].values
        scaler.fit(X_predict)
        X_predict_scaled = scaler.transform(X_predict)
        return model.predict(X_predict_scaled)

#--------------------------------------------------------------------
# create method  "dl_build_neuro_general".
# class: cmqlmlsetup
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------

#Key Parameters
#units: The number of neurons or units in the dense layer. This is the dimensionality of the output space.
#activation: The activation function to apply to the output. 

#Common activation functions include:
#====================================
#relu (Rectified Linear Unit)
#sigmoid
#softmax
#tanh
#linear (no activation)
#use_bias: Boolean, whether to include a bias term in the layer (default: True).

#kernel_initializer: Specifies the initializer for the kernel weights matrix, which
#determines how the weights are initialized before training.
#bias_initializer: Specifies the initializer for the bias vector.
#kernel_regularizer: Optional regularizer function applied to the kernel weights matrix 
#(for regularization like L1 or L2).
#bias_regularizer: Optional regularizer applied to the bias vector.
#activity_regularizer: Regularizer applied to the output (activation) of the layer.
#kernel_constraint: Constraint function applied to the kernel weights matrix.
#bias_constraint: Constraint function applied to the bias vector.

    def dl_build_neuro_network(self, lp_k_reg, X_train, y_train, optimizer='adam', lp_act1='relu', lp_act2='linear', lp_metric='accuracy', lp_loss='mean_squared_error', cells1=128, cells2=256, cells3=128, cells4=64, cells5=1):
        bmodel = tf.keras.models.Sequential([
        tf.keras.layers.Dense(cells1, activation=lp_act1, input_shape=(X_train.shape[1],), kernel_regularizer=tf.l2(lp_k_reg)),
        tf.keras.layers.Dense(cells2, activation=lp_act1, kernel_regularizer=tf.l2(lp_k_reg)),
        tf.keras.layers.Dense(cells3, activation=lp_act1, kernel_regularizer=tf.l2(lp_k_reg)),
        tf.keras.layers.Dense(cells4, activation=lp_act1, kernel_regularizer=tf.l2(lp_k_reg)),
        tf.keras.layers.Dense(cells5, activation=lp_act2)
        ])
        bmodel.compile(optimizer=optimizer, loss=lp_loss, metrics=[lp_metric])
        return bmodel


#Key Parameters
#1. **LSTM (Long Short-Term Memory)**: LSTMs are particularly well-suited for time series forecasting due to their ability to remember long-term dependencies. They can capture patterns over time, making them a popular choice for forex prediction.
#2. **GRU (Gated Recurrent Unit)**: Similar to LSTMs but with a simpler structure, GRUs can also perform well in time series forecasting. They may train faster than LSTMs while still capturing the necessary temporal patterns.
#3. **CNN (Convolutional Neural Networks)**: While traditionally used for image data, CNNs can be effective for time series by treating the data as a one-dimensional image. They can capture local patterns in the data.
#4. **1D CNN-LSTM Hybrid**: Combining CNNs with LSTMs can leverage the strengths of both architectures, where CNN layers extract features from the time series data, and LSTM layers capture temporal dependencies.
#5. **Transformer Models**: Transformers have gained popularity in various fields, including time series forecasting. They can model relationships in sequential data without relying on recurrent connections, which can sometimes lead to better performance.
#6. **ARIMA (AutoRegressive Integrated Moving Average)**: While not a deep learning model, ARIMA is a classic statistical approach that can be effective for forecasting time series data, including forex.
#7. **Ensemble Models**: Combining predictions from multiple models can often yield better results. You can create an ensemble of different architectures to improve robustness and accuracy.

#--------------------------------------------------------------------
# create method  "dl_build_neuro_ensemble".
# class: cmqlmlsetup
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
#Base Models:

#LSTM Model: Captures long-term temporal dependencies.
#1D CNN Model: Extracts local spatial features.
#GRU Model: Another RNN variant that captures temporal patterns.
#Transformer Model: Uses self-attention to capture global dependencies in the sequence.

#Transformer Block:

#This block consists of multi-head attention and feedforward layers wrapped with layer normalization 
#and residual connections. It is used inside the Transformer model to process the input sequences.

#Ensemble Process:

#The outputs of all four base models (LSTM, CNN, GRU, Transformer) are concatenated using the concatenate() function.
#After concatenation, a fully connected (dense) layer learns from the combined features, followed by a dropout layer for
#regularization and an output layer with a sigmoid activation for binary classification.
#Input Shape:

#The input shape is defined as (100, 1), meaning 100 time steps with 1 feature per time step. Modify this based
#on your actual data.

#Training:

#During training, all models receive the same input X_train, which may be adjusted in real use cases if your 
#models expect different types of inputs.

#Customization:

#You can increase/decrease the number of units in the LSTM, GRU, CNN, or Transformer layers.
#Use a different loss function for multi-class classification, such as categorical_crossentropy.
#Add more models to the ensemble or adjust the dropout rates.

#Why Use an Ensemble?
#Combining different architectures allows the model to leverage the strengths of each type:
#LSTM: Handles sequential dependencies well.
#CNN: Good at feature extraction, even from time series.
#GRU: Efficient with fewer parameters than LSTM.
#Transformer: Captures long-range dependencies using attention mechanisms.
#This ensemble approach can help in complex sequence modeling tasks where combining models improves accuracy and generalization.

    def dl_build_neuro_ensemble (self,lp_seq,lp_filt,lp_pool,lp_ksizelp_k_reg, X_train, y_train,optimizer='adam',lp_act1 = 'relu', lp_act2 = 'linear',lp_act3 = 'sigmoid',lp_metric = 'accuracy', lp_loss1='mean_squared_error',lp_loss2 = 'binary_crossentropy',cells1=128,cells2=256, cells3=128,cells4=64,cells5=1):
        # Generate some dummy data
        X_train = np.random.rand(1000, 100, 1)  # 1000 samples, 100 time steps, 1 feature
        y_train = np.random.randint(2, size=(1000,))  # Binary target

    ### Base Models ###
    # LSTM Model
    def create_lstm_model(input_shape):
        inputs = Input(shape=input_shape)
        x = LSTM(50)(inputs)
        x = Dense(50, activation='relu')(x)
        return Model(inputs, x)

    # 1D CNN Model
    def create_cnn_model(input_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(50, activation='relu')(x)
        return Model(inputs, x)

    # GRU Model
    def create_gru_model(input_shape):
        inputs = Input(shape=input_shape)
        x = GRU(50)(inputs)
        x = Dense(50, activation='relu')(x)
        return Model(inputs, x)

# Transformer Encoder Layer
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

    # Transformer Model
    def create_transformer_model(input_shape):
        inputs = Input(shape=input_shape)
        x = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=32)(inputs)
        x = GlobalAveragePooling1D()(x)
        x = Dense(50, activation='relu')(x)
        return Model(inputs, x)

    ### Ensemble Model ###
    def create_ensemble_model(input_shape):
        # Define the base models
        lstm_model = create_lstm_model(input_shape)
        cnn_model = create_cnn_model(input_shape)
        gru_model = create_gru_model(input_shape)
        transformer_model = create_transformer_model(input_shape)

        # Get outputs from base models
        lstm_output = lstm_model.output
        cnn_output = cnn_model.output
        gru_output = gru_model.output
        transformer_output = transformer_model.output

        # Concatenate outputs
        combined = concatenate([lstm_output, cnn_output, gru_output, transformer_output])

        # Add a dense layer to learn from combined features
        x = Dense(100, activation='relu')(combined)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)

        # Define the ensemble model
        model = Model(inputs=[lstm_model.input, cnn_model.input, gru_model.input, transformer_model.input], outputs=output)
        return model

    # Create and compile the ensemble model
    #input_shape = (100, 1)  # Adjust based on your data shape
    #ensemble_model = create_ensemble_model(input_shape)
    #ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Print model summary
    #ensemble_model.summary()

    # Train the ensemble model
    #ensemble_model.fit([X_train, X_train, X_train, X_train], y_train, epochs=10, batch_size=32)
    #return ensemble_model