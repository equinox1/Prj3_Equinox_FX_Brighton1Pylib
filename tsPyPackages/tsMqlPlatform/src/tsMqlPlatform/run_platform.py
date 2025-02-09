from tsMqlPlatform import platform_checker, PLATFORM_DEPENDENCIES, logger, config

logger.info("Starting application...")

class RunPlatform:
    def __init__(self):
        # Get platform from the checker
        platform_name = platform_checker.get_platform()
        print(f"Running on: {platform_name}")

        # Check if platform settings were modified
        default_platform = config.get("default_platform")
        if default_platform:
            print(f"Config-set default platform: {default_platform}")

        # Override platform using an environment variable (Example)
        # export FORCE_PLATFORM=Linux  # (Linux/MacOS users)
        # set FORCE_PLATFORM=Linux  # (Windows users)

        if platform_checker.is_windows():
            if PLATFORM_DEPENDENCIES:
                mt5 = PLATFORM_DEPENDENCIES.get("mt5")
                onnx = PLATFORM_DEPENDENCIES.get("onnx")
                if mt5 and onnx:
                    logger.info("MetaTrader5 and ONNX are ready for use.")
                    print("MetaTrader5 and ONNX are available.")
                else:
                    logger.error("MetaTrader5 or ONNX dependencies are missing.")
            else:
                logger.error("PLATFORM_DEPENDENCIES is not initialized.")
        else:
            logger.info("No additional dependencies are needed for this platform.")
