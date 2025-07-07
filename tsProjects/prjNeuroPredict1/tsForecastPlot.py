import os
import numpy as np
import matplotlib.pyplot as plt
import logging

def run_best_model_forecast_plot(tuner, x_forecast, y_true_forecast, title="Forecast vs Actual"):
    """
    Load best model from tuner, predict on forecast window, and plot results.

    Args:
        tuner: The KerasTuner tuner instance (after search).
        x_forecast: Input features for prediction (numpy array or tf.Tensor).
        y_true_forecast: Ground truth values (numpy array or tf.Tensor).
        title: Title for the plot.
    """
    logger = logging.getLogger(__name__)
    logger.info("Retrieving best model from tuner.")

    best_models = tuner.get_best_models(num_models=1)
    if not best_models:
        logger.error("No best models found in tuner.")
        return

    best_model = best_models[0]
    logger.info("Predicting on forecast window.")

    forecast = best_model.predict(x_forecast)

    if len(forecast.shape) > 1 and forecast.shape[1] == 1:
        forecast = forecast[:, 0]
    if len(y_true_forecast.shape) > 1 and y_true_forecast.shape[1] == 1:
        y_true_forecast = y_true_forecast[:, 0]

    plt.figure(figsize=(12, 5))
    plt.plot(y_true_forecast, label="Actual")
    plt.plot(forecast, label="Forecast", linestyle="--")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()