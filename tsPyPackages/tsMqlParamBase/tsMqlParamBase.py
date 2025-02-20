import logging
from pathlib import Path

class BaseParamBase:
    """Base class for parameter management with default values and validation."""
    
    DEFAULT_PARAMS = {}

    def __init__(self, **overrides):
        """Initialize parameters with defaults and allow overrides."""
        if not self.DEFAULT_PARAMS:
            raise ValueError("DEFAULT_PARAMS must be defined in a subclass.")
        
        self.params = {key: overrides.get(key, default) for key, default in self.DEFAULT_PARAMS.items()}
        logging.info(f"Initialized {self.__class__.__name__} with parameters: {self.params}")

    def get_params(self):
        """Return a dictionary of all parameters."""
        return self.params.copy()

    def set_param(self, key, value):
        """Set a parameter dynamically, ensuring it's a known parameter."""
        if key in self.DEFAULT_PARAMS:
            self.params[key] = value
            logging.info(f"Parameter '{key}' set to {value}.")
        else:
            logging.warning(f"Unknown parameter '{key}' - Cannot set.")

    def reset_to_defaults(self):
        """Reset all parameters to their default values."""
        self.params = self.DEFAULT_PARAMS.copy()
        logging.info("Parameters reset to default values.")

    def save_params(self, file_path="params.json"):
        """Save parameters to a JSON file."""
        import json
        path = Path(file_path)
        with path.open("w") as f:
            json.dump(self.params, f, indent=4)
        logging.info(f"Parameters saved to {path.absolute()}.")

    def load_params(self, file_path="params.json"):
        """Load parameters from a JSON file."""
        import json
        path = Path(file_path)
        if path.exists():
            with path.open("r") as f:
                self.params.update(json.load(f))
            logging.info(f"Parameters loaded from {path.absolute()}.")
        else:
            logging.warning(f"Parameter file '{file_path}' not found.")

