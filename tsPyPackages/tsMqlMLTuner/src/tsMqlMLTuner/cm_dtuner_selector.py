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
                if not parsed.scheme:
                    oracle_url = f"http://{oracle_url}"
                    parsed = urlparse(oracle_url)
                fixed_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
                self.oracle_client = OracleClient(url=fixed_url, tuner_id=tuner_id or "default_tuner")
                logger.info(f"Oracle client initialized from URL: {self.oracle_client.url}")
            except Exception as e:
                logger.error(f"Failed to initialize OracleClient: {e}")
                raise

        # IMPORTANT: Remove various arguments from kwargs that are expected as explicit parameters
        # by CMdtunerTorch/CMdtuner, or are handled elsewhere.
        self.kwargs.pop('oracle_url', None)
        self.kwargs.pop('project_name', None)
        self.kwargs.pop('log_dir', None)
        self.kwargs.pop('max_trials', None)
        self.kwargs.pop('overwrite', None)
        self.kwargs.pop('oracle_directory', None)

        # Extract dataset and model-related arguments explicitly
        train_data = self.kwargs.pop('train_data', None)
        val_data = self.kwargs.pop('val_data', None)
        test_data = self.kwargs.pop('test_data', None)
        train_dataset = self.kwargs.pop('train_dataset', None)
        val_dataset = self.kwargs.pop('val_dataset', None)
        test_dataset = self.kwargs.pop('test_dataset', None)

        # Extract the missing positional arguments
        dataset_params = self.kwargs.pop('dataset_params', {})
        base_path = self.kwargs.pop('base_path', "") # Changed default from None to ""
        model_id = self.kwargs.pop('model_id', None)


        if backend == 'pytorch':
            from .tsMqlMLTunerModTorch import CMdtunerTorch
            self.tuner = CMdtunerTorch(
                tuner_id=tuner_id,
                oracle_client=self.oracle_client,
                train_data=train_data or train_dataset,
                val_data=val_data or val_dataset,
                dataset_params=dataset_params, # Pass explicitly
                base_path=base_path,         # Pass explicitly
                model_id=model_id,           # Pass explicitly
                **self.kwargs # Pass remaining kwargs
            )
        elif backend == 'keras':
            from .tsMqlMLTunerMod import CMdtuner
            self.tuner = CMdtuner(
                tuner_id=tuner_id,
                oracle_client=self.oracle_client,
                train_data=train_data or train_dataset,
                val_data=val_data or val_dataset,
                dataset_params=dataset_params, # Pass explicitly
                base_path=base_path,         # Pass explicitly
                model_id=model_id,           # Pass explicitly
                **self.kwargs # Pass remaining kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def run(self):
        logger.info(f"Running tuner for backend: {self.backend}")
        self._run_chief(self.tuner)

    def _run_chief(self, tuner):
        logger.info("🚀 Chief starting distributed tuning...")
        try:
            if not self.oracle_client:
                raise RuntimeError("Oracle client is not initialized. Cannot proceed with tuning.")
            tuner.run()
        except Exception as e:
            logger.exception(f"❌ Error during model training and tuning: {e}")
        logger.info("✅ Chief finished tuning")

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
        return self.kwargs
