from .tsMqlMLTunerMod import CMdtuner
from .tsMqlMLTunerModTorch import PyTorchTuner
import logging
# Get a logger for this module
logger = logging.getLogger(__name__)


class CMdtunerSelector:
    def __init__(self, **kwargs):
        backend = kwargs.get("hypermodel_params", {}).get("mltune", {}).get("backend", "tensorflow").lower()
        if backend == "pytorch":
            self.tuneobj = PyTorchTuner(**kwargs)
        elif backend == "tensorflow":
            self.tuneobj = CMdtuner(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def run(self):
        return self.tuneobj.run()

    def run_search(self):
        return self.tuneobj.run_search()

    def export_best_model(self, ftype='tf'):
        if hasattr(self.tuneobj, 'export_best_model'):
            return self.tuneobj.export_best_model(ftype=ftype)
        return None

    def check_and_load_model(self, *args, **kwargs):
        if hasattr(self.tuneobj, 'check_and_load_model'):
            return self.tuneobj.check_and_load_model(*args, **kwargs)
        return None