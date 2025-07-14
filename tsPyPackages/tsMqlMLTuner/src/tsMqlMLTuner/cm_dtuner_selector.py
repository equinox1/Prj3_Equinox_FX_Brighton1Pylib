# Patched CMdtunerSelector worker tuning loop
import logging
import os
import torch
from urllib.parse import urlparse


from .tsMqlMLTunerMod import CMdtuner, get_callbacks # Import get_callbacks
from .tsMqlMLTunerModTorch import CMdtunerTorch
from .tsMqlMLOracleClient import OracleClient

logger = logging.getLogger(__name__)
class CMdtunerSelector:
    def __init__(self, backend='keras', tuner_id=None, oracle_client=None, **kwargs):
        self.backend = backend
        self.tuner_id = tuner_id
        self.kwargs = kwargs.copy()

        self.oracle_client = oracle_client or self.kwargs.pop('oracle_client', None)
        if self.oracle_client is None:
            logger.warning("No oracle_client provided to CMdtunerSelector. Attempting to initialize...")
            oracle_url = self.kwargs.get("oracle_url") or os.environ.get("ORACLE_URL")
            logger.debug(f"oracle_url from kwargs: {self.kwargs.get('oracle_url')}")
            logger.debug(f"oracle_url from env: {os.environ.get('ORACLE_URL')}")
            if not oracle_url:
                raise RuntimeError("Oracle URL is missing. Cannot initialize OracleClient.")
            try:
                parsed = urlparse(oracle_url)
                # Ensure the scheme is present. If not, prepend 'http://'.
                # This prevents 'http://http://' if the URL already has a scheme.
                if not parsed.scheme:
                    fixed_url = f"http://{oracle_url}"
                else:
                    fixed_url = oracle_url # Use the URL as is if it already has a scheme

                # Pass the corrected URL to the OracleClient's 'url' parameter
                from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
                # Pass the timeout argument to OracleClient.__init__
                self.oracle_client = OracleClient(url=fixed_url, timeout=30.0)
                logger.info(f"Oracle client initialized from URL: {fixed_url} with timeout 30s")
            except Exception as e:
                logger.error(f"Failed to auto-initialize OracleClient: {e}")
                raise

        self.kwargs.pop('tuner_id', None)
        self.is_chief = self.kwargs.pop('is_chief', True) # Get is_chief from kwargs, default to True

        self.tuner = None

        if backend == 'pytorch':
            from .tsMqlMLTunerModTorch import CMdtunerTorch
            self.tuner = CMdtunerTorch(
                tuner_id=tuner_id,
                oracle_client=self.oracle_client,
                is_chief=self.is_chief, # Pass the correct is_chief flag
                **self.kwargs
            )
        # Corrected: Handle 'tensorflow' backend by mapping it to CMdtuner
        elif backend == 'keras' or backend == 'tensorflow':
            from .tsMqlMLTunerMod import CMdtuner
            self.tuner = CMdtuner(
                tuner_id=tuner_id,
                oracle_client=self.oracle_client,
                is_chief=self.is_chief, # Pass the correct is_chief flag
                **self.kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def run(self):
        logger.info(f"Running tuner for backend: {self.backend}")
        # The run method should delegate to the tuner's run method,
        # which will handle chief/worker logic internally.
        self.tuner.run() # Call the tuner's run method directly

    def get_best_model(self):
        if hasattr(self.tuner, "get_best_model"):
            return self.tuner.get_best_model()
        else:
            raise AttributeError("Current tuner does not support get_best_model().")

    def get_model_dir(self):
        if hasattr(self.tuner, "get_model_dir"):
            return self.tuner.get_model_dir()
        else:
            raise AttributeError("Current tuner does not support get_model_dir().")

    @property
    def app_params(self):
        # This property should return the app_params that were passed in kwargs
        # This is used by tsNeuroPredictWinMql_chief.py for model saving paths
        return self.kwargs.get("hypermodel_params", {}).get("app", {})