# Patched CMdtunerSelector worker tuning loop
import logging
import os
import torch
from urllib.parse import urlparse
from pathlib import Path

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
                oracle_url = oracle_url.replace("http://http://", "http://").replace("https://http://", "http://").replace("http://https://", "https://")
                parsed = urlparse(oracle_url)
                if not parsed.scheme:
                    oracle_url = f"http://{oracle_url}"
                    parsed = urlparse(oracle_url)
                fixed_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
                from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient
                self.oracle_client = OracleClient(fixed_url)
                logger.info(f"Oracle client initialized from URL: {fixed_url}")
            except Exception as e:
                logger.error(f"Failed to auto-initialize OracleClient: {e}")
                raise

        self.kwargs.pop('tuner_id', None)
        self.kwargs.pop('is_chief', None)

        self.tuner = None

        log_dir_root = self.kwargs.get('hypermodel_params', {}).get('base', {}).get(
            'mp_glob_base_log_path',
            r"C:\\WinRunMnt1\\8.0 Projects\\8.3 ProjectModelsEquinox\\EQUINRUN\\Logdir"
        )
        log_dir_root = Path(log_dir_root)
        self.kwargs['log_dir'] = str(log_dir_root)

        if backend == 'pytorch':
            from .tsMqlMLTunerModTorch import CMdtunerTorch
            self.tuner = CMdtunerTorch(
                tuner_id=tuner_id,
                oracle_client=self.oracle_client,
                is_chief=True,
                **self.kwargs
            )
        elif backend == 'keras':
            from .tsMqlMLTunerMod import CMdtuner
            self.tuner = CMdtuner(
                tuner_id=tuner_id,
                oracle_client=self.oracle_client,
                is_chief=True,
                **self.kwargs
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
        return self.kwargs.get("hypermodel_params", {}).get("app", {})
