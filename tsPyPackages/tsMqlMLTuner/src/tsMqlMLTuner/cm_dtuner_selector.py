# cm_dtuner_selector.py
class CMdtunerSelector:
    def __init__(self, **kwargs):
        backend = kwargs.get("hypermodel_params", {}).get("mltune", {}).get("backend", "tensorflow").lower()
        if backend == "pytorch":
            from pytorch_tuner import PyTorchTuner
            self.impl = PyTorchTuner(**kwargs)
        elif backend == "tensorflow":
            from tsMqlMLTunerMod import CMdtuner as TensorflowTuner
            self.impl = TensorflowTuner(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def run(self):
        return self.impl.run()

    def build_model(self, hp):
        return self.impl.build_model(hp)
