import logging
import os

# ----- Global Logging Configuration -----
global_logdir = r"C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\Logdir"
try:
    os.makedirs(global_logdir, exist_ok=True)
except OSError as e:
    print(f"Error creating log directory: {e}")
    global_logdir = os.getcwd()  # Fallback to current working directory

#global_logfile = os.path.join(global_logdir, 'tsneuropredict_app.log')
global_logfile = os.path.join(global_logdir, 'tstest.log')

# Set up the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()

try:
    fh = logging.FileHandler(global_logfile, mode='w')
except OSError as e:
    print(f"Error creating log file: {e}")
    fh = logging.FileHandler('fallback.log', mode='w')  # Fallback to a local log file

formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info("Logging configured successfully with FileHandler.")
logger.info("Logfile: %s", global_logfile)


def submodule():
    # Get a logger for this module
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Ensure the logger level is set appropriately


    """Logs messages from a submodule context."""
    logger.debug("sub:Debug message")
    logger.info("sub:Info message")
    logger.warning("sub:Warning message")
    logger.error("sub:Error message")
    logger.critical("sub:Critical message")


def main():
    """Main function that logs messages and calls the submodule."""
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    submodule()


if __name__ == "__main__":
    main()
