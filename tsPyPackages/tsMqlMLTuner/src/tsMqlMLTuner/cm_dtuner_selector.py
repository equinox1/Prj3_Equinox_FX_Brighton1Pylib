from .tsMqlMLTunerMod import CMdtuner
from .tsMqlMLTunerModTorch import PyTorchTuner
import logging
# -- start of logging setup --
from tsMqlSetup import CMqlSetup
# ✅ Logger and Logdir Setup
setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=8,
    num_threads=1
)
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})
from tsMqlSetup import CMqlSetup
gtuner_model = app_params.get('gtuner_model', 'pytorch')  # or "tensorflow"
backend = tune_params.get('backend', gtuner_model)  # or "tensorflow"
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')
tunerlogfile = xerces_logfile
global_logdir, global_logfile = setup_config.set_log_dir(logdir=None, logfile=tunerlogfile, servername=xerces_servername,ltuner=gtuner_model)
logger = setup_config.setup_global_logger(global_logfile)
# -- end of logging setup ----

class CMdtunerSelector:
    def __init__(self, **kwargs):
        backend = kwargs.get("hypermodel_params", {}).get("mltune", {}).get("backend", "tensorflow").lower()
        logger.info(f"Tunerselector Using backend: {backend}")
        print(f"Tunerselector Using backend: {backend}")
        # Initialize the appropriate tuner based on the backend
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