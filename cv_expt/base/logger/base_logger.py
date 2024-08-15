import os
import sys
import cv2
from rich.console import Console
from rich.logging import RichHandler
import logging
from pydantic import BaseModel
from typing import Optional
from csv_logger import CsvLogger
from settings import BASE_DIR, LOGGER_DIR, IMAGE_LOGGER_DIR
from pathlib import Path


# use rich library to log messages
# csv-logger # package hera hai
# rich console logger le message log garne
# csv-logger le locally metrics log garne

"""
class LogConfig(BaseModel):
    wandb: bool = False
    as_csv: bool = True
    log_file: Optional[str] = None
    pass


class Logger:
    def __init__(self, log_config: LogConfig):
        if log_config.wandb:
            import wandb
            wandb.init()
            self.logger = wandb
        if log_config.as_csv:
            self.csv_logger = CSVLogger()
        if log_config.log_file:
            pass
            # yeha rich logger ko code aauxa
        else:
            pass
            
        pass
    
    def log_metric(self, metric: str, step:int, value: float):
        if self.logger:
            self.logger.log({metric: value}, step=step)
        if self.csv_logger:
            self.csv_logger.log_metric(metric, step, value)
        pass
    def log_message(self, message: str, step:int=-1):
        # rich le log garxa
        pass
    def log_image(self, image: np.ndarray, img_name: str):
        simply image write garne
        pass

"""

rich_txt_log = LOGGER_DIR / "rich_logger.log"


class LogConfig(BaseModel):
    wandb: bool = False
    as_csv: bool = True
    log_file: Optional[str] = None
    


class Logger:
    def __init__(self, log_config: LogConfig):
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(RichHandler())

        if log_config.wandb:
            import wandb

            self.wandb = wandb.init()

        if log_config.as_csv:
            self.csv_logger = CsvLogger()
        else:
            self.csv_logger = None

        if log_config.log_file:
            file_handler = logging.FileHandler(log_config.log_file)  # to log in file
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
        else:
            self.log_file = None

    def log_metric(self, epoch: int, metric: str, step: int, value: float):
        """Log the metric value and epoch to the logger and csv file"""
        log_message = f"Epoch: {epoch}, Step: {step}, {metric}: {value}"

        # Log the message with INFO level
        if self.logger:
            self.logger.log(logging.INFO, log_message)

        if self.csv_logger:
            self.csv_logger.log_metric(metric, step, value)

    # log message using rich logger
    def log_message(self, message: str, step: int = -1):
        """Log the message to the logger"""
        log_message = f"Step: {step if step != -1 else 'N/A'} | {message}"
        if self.logger:
            self.logger.log(logging.INFO, message)

        self.console.print(log_message)

    def log_image(
        self, image: np.ndarray, img_name: str, image_logger_dir: Optional[Path] = None
    ):
        """Log the image to the logger"""
        if image_logger_dir is None:
            image_logger_dir = IMAGE_LOGGER_DIR
        image_logger_dir.mkdir(parents=True, exist_ok=True)

        image_path = image_logger_dir / img_name

        cv2.imwrite(str(image_path), image)

        if self.wandb:
            self.wandb.log({img_name: wandb.Image(str(image_path))})

        self.console.print(f"Image logged at: {image_path}")
        self.logger.log(logging.INFO, f"Image logged at: {image_path}")




if __name__ == '__main__':
    log_config = LogConfig(wandb=False, as_csv=False, log_file=rich_txt_log)
    logger = Logger(log_config)

