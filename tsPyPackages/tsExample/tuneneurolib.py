#--------------------------------------------------------------------
# create method  "dl_model_tune"
# class:cmqlmlsetup
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
# Define a function to create a model, required for KerasRegressor
    def dl_model_tune(k_reg, lp_X_train, optimizer='adam', loss='mean_squared_error',learning_rate=0.001):
        # sourcery skip: instance-method-first-arg-name
        tmodel = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(lp_X_train.shape[1],), kernel_regularizer=l2(k_reg)),
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(k_reg)),
                tf.keras.layers.Dense(1, activation='linear')
            ])
        tmodel.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
        return tmodel