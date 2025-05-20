
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient

import logging
import os
import pathlib
import uuid  # Ensure uuid is imported for use in get_callbacks


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
logger = setup_config.setup_global_logger(global_logfile, force_reset=True)
# -- end of logging setup ----

# Platform imports
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
pchk         = run_platform.RunPlatform()
os_platform  = platform_checker.get_platform()
loadmql      = pchk.check_mql_state()


class PyTorchTuner:
    def __init__(self, **kwargs):
        self.params = kwargs.get("hypermodel_params", {})
        self.train_data = kwargs.get("traindataset")
        self.val_data = kwargs.get("valdataset")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.oracle = kwargs.get("oracle", OracleClient())
        self.logdev = kwargs.get("logdev", False)

    def prepare_data(self):
        def extract_xy(data):
            if isinstance(data, (list, tuple)) and len(data) == 2:
                x, y = data
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                if isinstance(y, np.ndarray):
                    y = torch.tensor(y, dtype=torch.float32)
                return x, y
            elif isinstance(data, TensorDataset):
                x_all, y_all = [], []
                for x, y in data:
                    x_all.append(x.unsqueeze(0))
                    y_all.append(y.unsqueeze(0))
                return torch.cat(x_all, dim=0), torch.cat(y_all, dim=0)
            raise ValueError("Unsupported dataset format. Expected (x, y) tuple or TensorDataset.")

        X_train, y_train = extract_xy(self.train_data)
        X_val, y_val = extract_xy(self.val_data)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, num_workers=4, pin_memory=True)
        return train_loader, val_loader

    def build_model(self, hp, input_dim):
        n_units1 = hp.get('n_units1', 64)
        n_units2 = hp.get('n_units2', 64)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, n_units1),
            nn.ReLU(),
            nn.Linear(n_units1, n_units2),
            nn.ReLU(),
            nn.Linear(n_units2, 1)
        )
        return model.to(self.device)

    def objective_from_hp(self, hp):
        if self.logdev:
             print(f"[PyTorchTuner] 💻 Using device: {self.device}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_loader, val_loader = self.prepare_data()
        sample_batch = next(iter(train_loader))[0]
        input_dim = int(np.prod(sample_batch.shape[1:]))

        print(f"[PyTorchTuner] Input batch shape: {sample_batch.shape}, computed input_dim: {input_dim}")

        model = self.build_model(hp, input_dim)
        print(f"[PyTorchTuner] 🧠 Model initialized on device: {next(model.parameters()).device}")
        optimizer_name = hp.get("optimizer", "Adam")
        lr = hp.get("lr", 1e-3)
        optimizer = getattr(optim, optimizer_name, optim.Adam)(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        try:
            for epoch in range(5):
                model.train()
                for xb, yb in train_loader:
                    if self.logdev:
                        print(f"[PyTorchTuner] 📦 Training batch shape: {xb.shape}, target shape: {yb.shape}")
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    preds = model(xb).squeeze()
                    loss = loss_fn(preds, yb)
                    loss.backward()
                    optimizer.step()

            model.eval()
            if self.device == "cuda":
                print(f"[PyTorchTuner] 🔋 GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                print(f"[PyTorchTuner] 🔋 GPU memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds = model(xb).squeeze()
                    loss = loss_fn(preds, yb)
                    val_losses.append(loss.item())

            return np.mean(val_losses)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("[PyTorchTuner] ⚠️ Trial failed due to CUDA OOM")
            raise e

    def export_best_model(self, ftype='onnx'):
        if self.best_model is None:
            logger.warning("No model available for export.")
            return

        base = self.params.get("base", {})
        modeldata_path = base.get("mp_glob_sub_ml_src_modeldata", "./")
        model_name = base.get("mp_glob_sub_ml_model_name", "best_model")
        mp_symbol = base.get("lp_app_primary_symbol", "EURUSD")
        ml_data_type = base.get("mp_ml_data_type", "data")

        onnx_path = os.path.join(modeldata_path, f"model_{mp_symbol}_{ml_data_type}.onnx")
        dummy_input = torch.randn(1, self.best_input_dim).to(self.device)

        try:
            torch_onnx_export(self.best_model, dummy_input, onnx_path, input_names=['input'], output_names=['output'], opset_version=17)
            logger.info(f"ONNX model exported to {onnx_path}")
            onnx.checker.check_model(onnx.load(onnx_path))
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")


    def run_search(self):
        tuner_id = os.environ.get("TUNER_ID", "worker")
        print(f"[PyTorchTuner] 🚀 Starting distributed search as {tuner_id}")

        while True:
            try:
                trial_resp = self.oracle.get_trial()
                trial = trial_resp.get("trial") if isinstance(trial_resp, dict) else trial_resp

                if not trial or "trial_id" not in trial:
                    print("[PyTorchTuner] 💤 No more trials available. Exiting.")
                    break

                trial_id = trial["trial_id"]
                hp = trial.get("hyperparameters", {})

                if not isinstance(hp, dict) or not hp:
                    print(f"[PyTorchTuner] ⚠️ Trial {trial_id} has invalid or empty hyperparameters. Marking as FAILED.")
                    self.oracle.update_trial_status(trial_id, "FAILED")
                    continue

                print(f"[PyTorchTuner] 🔍 Running trial {trial_id} with hyperparameters: {hp}")
                val_loss = self.objective_from_hp(hp)

                print(f"[PyTorchTuner] ✅ Trial {trial_id} completed. val_loss = {val_loss:.5f}")
                self.oracle.report_trial_result(trial_id, val_loss)

            except Exception as e:
                trial_name = trial_id if "trial_id" in locals() else "[UNKNOWN]"
                print(f"[PyTorchTuner] ❌ Exception during trial {trial_name}: {e}")
                if "trial_id" in locals():
                    self.oracle.update_trial_status(trial_id, "FAILED")
                break

        print("[PyTorchTuner] 🧹 Distributed search complete.")
        return True
