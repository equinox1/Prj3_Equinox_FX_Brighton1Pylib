import os
from tsMqlPlatformPkg import platform_checker, PLATFORM_DEPENDENCIES, logger, config


class CPlatformHelper:
    """Class to store and manage platform dependencies."""
    def __init__(self):
        self.platform_name = platform_checker.get_platform()
        logger.info("Starting application...")

        # Get platform from the checker
        platform_name = self.platform_name
        print(f"Running on: {platform_name}")

        # Check if platform settings were modified
        default_platform = config.get("default_platform")
        if default_platform:
            print(f"Config-set default platform: {default_platform}")

        # Override platform using an environment variable (Example)
        # export FORCE_PLATFORM=Linux  # (Linux/MacOS users)
        # set FORCE_PLATFORM=Linux  # (Windows users)
        os.environ['FORCE_PLATFORM'] = 'Windows'

        if platform_checker.is_windows():
            mt5 = PLATFORM_DEPENDENCIES["mt5"]
            onnx = PLATFORM_DEPENDENCIES["onnx"]
            logger.info("MetaTrader5 and ONNX are ready for use.")
            print("MetaTrader5 and ONNX are available.")
        else:
            logger.info("No additional dependencies are needed for this platform.")
