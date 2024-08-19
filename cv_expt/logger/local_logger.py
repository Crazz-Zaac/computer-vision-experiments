import logging
from rich.logging import RichHandler
from pydantic import BaseModel
from pathlib import Path


class LocalLoggerConfig(BaseModel):
    log_level: str = "DEBUG"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_path: Path = Path("logs")
    log_file: str = "app.log"
    name: str = "LocalLogger"
    log_write_mode: str = 'w'


class LocalLogger(logging.Logger):
    def __init__(self, 
                 config: LocalLoggerConfig):
        super().__init__(config.name, level=getattr(logging, config.log_level.upper()))
        self.config = config
        self._setup_logger()

    def _setup_logger(self):
        # Create log directory if it doesn't exist
        self.config.log_path.mkdir(parents=True, exist_ok=True)

        # Create a RichHandler for colorful console output
        console_handler = RichHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level.upper(), "DEBUG"))

        # Create a FileHandler for logging to a file
        file_handler = logging.FileHandler(self.config.log_path / self.config.log_file, mode=self.config.log_write_mode)
        file_handler.setLevel(getattr(logging, self.config.log_level.upper(), "DEBUG"))

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter(self.config.log_format)
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add both handlers to the logger
        self.addHandler(console_handler)
        self.addHandler(file_handler)
    
    def log_file_path(self):
        return self.config.log_path / self.config.log_file
