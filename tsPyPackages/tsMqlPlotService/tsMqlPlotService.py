# filename: tsMqlPlotService.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

# Set up logging for this module
logger = logging.getLogger(__name__)

class PlottingService:
    """
    A service class for generating various plots related to model predictions.
    Plots are saved to a specified log directory.
    """
    def __init__(self, log_dir: Path):
        """
        Initializes the PlottingService.

        :param log_dir: The base directory where plots should be saved.
        """
        self.log_dir = log_dir / "plots" # Create a 'plots' subdirectory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PlottingService initialized. Plots will be saved to: {self.log_dir}")

        # Set seaborn style for better aesthetics
        sns.set_style("whitegrid")

    def _save_plot(self, fig, filename: str):
        """
        Saves the given matplotlib figure to the log directory.

        :param fig: The matplotlib figure object to save.
        :param filename: The name of the file (e.g., "my_plot.png").
        """
        file_path = self.log_dir / filename
        try:
            fig.savefig(file_path, bbox_inches='tight', dpi=300)
            logger.info(f"Plot saved successfully: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {e}", exc_info=True)
        finally:
            plt.close(fig) # Close the figure to free up memory

    def plot_predictions(self, actual_values: np.ndarray, predicted_values: np.ndarray, title: str, filename: str):
        """
        Generates and saves a plot comparing actual vs. predicted values over time/index.

        :param actual_values: NumPy array of actual values.
        :param predicted_values: NumPy array of predicted values.
        :param title: Title of the plot.
        :param filename: Name of the file to save the plot (e.g., "predictions.png").
        """
        if len(actual_values) != len(predicted_values):
            logger.error("Actual and predicted values must have the same length for plot_predictions.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual_values, label='Actual Values', color='blue', alpha=0.7)
        ax.plot(predicted_values, label='Predicted Values', color='red', alpha=0.7, linestyle='--')
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.legend()
        self._save_plot(fig, filename)

    def plot_residuals(self, residuals: np.ndarray, title: str, filename: str):
        """
        Generates and saves a histogram and scatter plot of residuals.

        :param residuals: NumPy array of residuals (actual - predicted).
        :param title: Title of the plot.
        :param filename: Name of the file to save the plot (e.g., "residuals.png").
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram of residuals
        sns.histplot(residuals, kde=True, ax=axes[0], color='green')
        axes[0].set_title(f"{title} - Histogram")
        axes[0].set_xlabel("Residual Value")
        axes[0].set_ylabel("Frequency")

        # Scatter plot of residuals vs. index
        axes[1].scatter(range(len(residuals)), residuals, alpha=0.6, color='purple')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axes[1].set_title(f"{title} - Scatter Plot")
        axes[1].set_xlabel("Sample Index")
        axes[1].set_ylabel("Residual Value")

        fig.suptitle(title)
        plt.tight_layout()
        self._save_plot(fig, filename)

    def plot_scatter(self, actual_values: np.ndarray, predicted_values: np.ndarray, title: str, filename: str):
        """
        Generates and saves a scatter plot of actual vs. predicted values.

        :param actual_values: NumPy array of actual values.
        :param predicted_values: NumPy array of predicted values.
        :param title: Title of the plot.
        :param filename: Name of the file to save the plot (e.g., "scatter.png").
        """
        if len(actual_values) != len(predicted_values):
            logger.error("Actual and predicted values must have the same length for plot_scatter.")
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(actual_values, predicted_values, alpha=0.6, color='blue')
        
        # Add a perfect prediction line (y=x)
        min_val = min(actual_values.min(), predicted_values.min())
        max_val = max(actual_values.max(), predicted_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

        ax.set_title(title)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()
        ax.set_aspect('equal', adjustable='box') # Ensure equal scaling
        self._save_plot(fig, filename)

# Example usage (for testing purposes, will not run when imported)
if __name__ == "__main__":
    # Dummy data for testing
    np.random.seed(42)
    test_actual = np.random.rand(100) * 100
    test_predicted = test_actual + np.random.randn(100) * 10

    # Create a dummy log directory for testing
    test_log_dir = Path("./test_plot_output")
    if test_log_dir.exists():
        import shutil
        shutil.rmtree(test_log_dir)
    test_log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging for this test script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Test plots will be saved to: {test_log_dir}")

    plot_service = PlottingService(log_dir=test_log_dir)

    # Test plot_predictions
    plot_service.plot_predictions(test_actual, test_predicted, "Test Predictions", "test_predictions.png")

    # Test plot_residuals
    test_residuals = test_actual - test_predicted
    plot_service.plot_residuals(test_residuals, "Test Residuals", "test_residuals.png")

    # Test plot_scatter
    plot_service.plot_scatter(test_actual, test_predicted, "Test Scatter", "test_scatter.png")

    logger.info("All test plots attempted. Check the 'test_plot_output' directory.")
