import keras_tuner as kt
from tensorflow import keras
import numpy as np

# Step 1: Define a model building function for tuning
def build_model(hp):
    model = keras.Sequential()
    
    # Tune the number of layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(keras.layers.Dense(
            units=hp.Int('units_' + str(i), min_value=32, max_value=128, step=32),
            activation='relu'))
    
    model.add(keras.layers.Dense(10, activation='softmax'))  # Output layer for classification
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    return model

# Step 2: Set up the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='tune_model')

# Step 3: Define the training data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Step 4: Perform hyperparameter search
tuner.search(x_train, y_train, epochs=5, validation_split=0.2)

# Step 5: Get the best model(s) and form an ensemble
best_models = tuner.get_best_models(num_models=3)

# Step 6: Ensemble model predictions (simple averaging example)
def ensemble_predict(models, data):
    predictions = [model.predict(data) for model in models]
    return np.mean(predictions, axis=0)

# Predict using the ensemble
ensemble_preds = ensemble_predict(best_models, x_test)
