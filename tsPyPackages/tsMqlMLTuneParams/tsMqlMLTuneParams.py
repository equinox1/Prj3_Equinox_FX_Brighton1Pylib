from dataclasses import dataclass, field
import os
import tensorflow as tf
from datetime import date
import logging
from tsMqlPlatform import run_platform, platform_checker, logger


# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tsMqlDataParams")

# Run platform checker once
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform}, MQL state: {loadmql}")

@dataclass
class BaseTuneModel:
    """Base class for tuner models with shared parameters."""
    tunemode: str = "Hyperband"
    tunemodeepochs: int = 100
    today: str = field(default_factory=lambda: date.today().strftime('%Y-%m-%d %H:%M:%S'))
    seed: int = 42
    batch_size: int = 32
    max_epochs: int = 100
    dropout: float = 0.2
    cnn_model: bool = True
    lstm_model: bool = True
    gru_model: bool = True
    transformer_model: bool = True
    multiactivate: bool = True
    multi_branches: bool = True
    validation_split: float = 0.2
    optimizer: str = "adam"
    loss: str = "mean_squared_error"
    metrics: list = field(default_factory=lambda: ["mean_squared_error"])
    objective: str = "val_loss"
    
    def get_params(self):
        return {key: getattr(self, key) for key in self.__dict__}

@dataclass
class CMdtunerHyperModel(BaseTuneModel):
    """Hyperparameter tuning model class."""
    tuner_id: int = None
    train_dataset: object = None
    val_dataset: object = None
    test_dataset: object = None
    input_shape: tuple = None
    multi_outputs: bool = False
    num_trials: int = 3
    max_retries_per_trial: int = 5
    unitmin: int = 32
    unitmax: int = 512
    unitstep: int = 32
    defaultunits: int = 128
    allow_new_entries: bool = True
    base_checkpoint_filepath: str = None
    chk_monitor: str = "val_loss"
    chk_patience: int = 3

@dataclass
class CMqlEnvTuneML(BaseTuneModel):
    """ML environment class for tuning."""
    globalenv: object = None
    tuner_id: int = 1
    train_split: float = 0.7
    validation_split: float = 0.2
    test_split: float = 0.1
    modeldatapath: str = None
    input_width: int = 24
    shift: int = 24
    total_window_size: int = field(init=False)
    label_width: int = 1
    
    def __post_init__(self):
        self.total_window_size = self.input_width + self.shift
        
# Instantiate global environment
hypermodel = CMdtunerHyperModel()
