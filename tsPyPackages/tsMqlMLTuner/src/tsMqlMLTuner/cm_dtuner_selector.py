class CMdtunerSelector:
    def __init__(self, **kwargs):
        backend = kwargs.get("hypermodel_params", {}).get("mltune", {}).get("backend", "tensorflow").lower()
        if backend == "pytorch":
            from .tsMqlMLTunerModTorch import PyTorchTuner
            self.impl = PyTorchTuner(**kwargs)
        elif backend == "tensorflow":
            from .tsMqlMLTunerMod import CMdtuner as TensorflowTuner
            self.impl = TensorflowTuner(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def run(self):
        return self.impl.run()

    def run_search(self):
        return self.impl.run_search()

    def export_best_model(self, ftype='tf'):
        if hasattr(self.impl, 'export_best_model'):
            return self.impl.export_best_model(ftype=ftype)
        return None

    def check_and_load_model(self, *args, **kwargs):
        if hasattr(self.impl, 'check_and_load_model'):
            return self.impl.check_and_load_model(*args, **kwargs)
        return None
