
import tensorflow as tf
from tensorflow import keras

import keras_tuner as kt

import tsMqlML
from tsMqlML import CMqlmlsetup

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values between 0 and 1
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0


def lmodel_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  k_reg=0.01
  hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units1, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(k_reg)))

  #hp_units2 = hp.Int('units2', min_value=32, max_value=512, step=32)
  #model.add(keras.layers.Dense(units=hp_units2, activation='relu'))

  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  fn_loss ='mean_squared_error'

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=fn_loss,
                metrics=['mean_squared_error'])  # For regression

  return model

m1 = CMqlmlsetup
model_builder = m1.model_builder
model_builder = lmodel_builder

tuner = kt.Hyperband(model_builder,
                     objective='val_loss',  # Use 'val_loss' for regression
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)