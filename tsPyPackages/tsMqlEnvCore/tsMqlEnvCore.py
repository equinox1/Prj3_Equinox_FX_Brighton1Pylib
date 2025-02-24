import logging
from pathlib import Path


class CEnvCore:
    """Common functionality shared across parameter classes."""
    
    DEFAULT_PARAMS = {}  # Ensure this is a class attribute
    
    def __init__(self, custom_params=None):
        self.params = self.DEFAULT_PARAMS.copy()  # Initialize instance params from the class attribute

        if custom_params:
            self.params.update(custom_params)

    def get(self, key, default=None):
        return self.params.get(key, default)

    def get_param(self, key, default=None):
        return self.params.get(key, default)


    def set(self, key, value):
        self.params[key] = value

    def all_params(self):
        return self.params  # Fixed missing `return`