from tsMqlPlatform import platform_checker, PLATFORM_DEPENDENCIES, logger, config

logger.info("Starting application...")

class RunPlatform:
    def __init__(self):
        self.debug = config.get("debug", False)
        self.platform_name = platform_checker.get_platform()

        if self.debug:
            print(f"Running on: {self.platform_name}")

        self.mt5 = None  # Initialize mt5 attribute
        self.onnx = None  # Initialize onnx attribute

        if platform_checker.is_windows():
            if PLATFORM_DEPENDENCIES is not None:
                self.mt5 = PLATFORM_DEPENDENCIES.get("mt5")  # Assign to class attributes
                self.onnx = PLATFORM_DEPENDENCIES.get("onnx")

                if self.mt5 and self.onnx:
                    logger.info("MetaTrader5 and ONNX are ready for use.")
                    if self.debug:
                        print("MetaTrader5 and ONNX are available.")
                else:
                    logger.error("MetaTrader5 or ONNX dependencies are missing.")
            else:
                logger.error("PLATFORM_DEPENDENCIES is not initialized.")
        else:
            logger.info("No additional dependencies are needed for this platform.")
