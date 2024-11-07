from cv_expt.base.defs.defs import ImageChannel
from pydantic import BaseModel
from pathlib import Path
from typing import List, Tuple, Optional


class DataConfig(BaseModel):
    data_path: Path = Path("assets/training_data")
    input_size: Tuple[int, int] = (224, 224)
    label_path: Optional[Path] = None
    train_size: float = 0.8
    seed: int = 42
    shuffle: bool = True
    image_channels: ImageChannel = ImageChannel.RGB
    image_extensions: List[str] = ["jpg", "jpeg", "png"]
    max_data: int = -1  # -1 means all data
    samples_per_epoch: int = -1 # -1 means all data of respective split

    class Config:
        arbitrary_types_allowed = True


class TrainerConfig(BaseModel):
    result_dir: Path
    expt_name: str
    run_name: str
    log_every: int = 1
    chkpt_every: int = 0
    best_model_name: str = "best_model.pth"
    device: str = "cuda"
    epochs: int = 100
    log_display_every: int = 10
    display_samples_per_epoch: int = 2
    batch_size: int = 32
    shuffle: bool = True
    show_images: bool = False
    plot_order: Tuple[int, int] = (-1, 3)
    number_of_workers: int = 0
    
    @property
    def run_dir(self):
        return self.result_dir / self.expt_name / self.run_name

    class Config:
        arbitrary_types_allowed = True


class BaseLoggerConfig(BaseModel):
    log_level: str = "DEBUG"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_path: Path = Path("logs")
    log_file: str = "app.log"
    name: str = "CVExpt"
    log_write_mode: str = "w"  # 'w' for overwrite, 'a' for append
    metric_file: str = "metrics.csv"
    log_in_console: bool = True

class LocalLoggerConfig(BaseModel):
    log_level: str = "DEBUG"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_path: Path = Path("logs")
    log_file: str = "logs.log"
    name: str = "LocalLogger"
    log_write_mode: str = 'w'