from cv_expt.base.configs.configs import BaseLoggerConfig

import logging
from rich.logging import RichHandler
import numpy as np
from matplotlib import pyplot as plt
import csv
from typing import Union, List, Dict, Tuple


class BaseLogger(logging.Logger):

    def __init__(self, config: BaseLoggerConfig):
        """
        A custom logger class that extends the Python logging.Logger class.
        This class is used to log messages to the console and a file.

        Args:
        config: Configuration for the logger.
        """
        super().__init__(config.name, level=getattr(logging, config.log_level.upper()))
        self.config = config
        self._setup_logger()
        self.metric_history:Dict = {}

    def _setup_logger(self):
        # Create log directory if it doesn't exist
        self.config.log_path.mkdir(parents=True, exist_ok=True)

        # Create a RichHandler for colorful console output
        console_handler = RichHandler()
        console_handler.setLevel(
            getattr(logging, self.config.log_level.upper(), self.config.log_level.upper())
        )

        # Create a FileHandler for logging to a file
        file_handler = logging.FileHandler(
            self.config.log_path / self.config.log_file, mode=self.config.log_write_mode
        )
        file_handler.setLevel(getattr(logging, self.config.log_level.upper(), self.config.log_level.upper()))

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter(self.config.log_format)
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add both handlers to the logger
        self.addHandler(console_handler)
        self.addHandler(file_handler)

    def log_metric(
        self, metric_name: str, metric_value: Union[int, float], step: int = 1
    ):
        """
        Logs metric to both the log file and a CSV file.

        Args:
        metric_name: The name of the metric.
        metric_value: The value of the metric.
        step: The step or iteration at which the metric is logged.
        """
        # Log to standard logger
        self.info(f"Step: {step}, {metric_name}: {metric_value}")

        # Log to CSV file
        csv_path = self.config.log_path / self.config.metric_file

        # Write to CSV file
        file_exists = csv_path.exists()
        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write header if file is new
                writer.writerow(["Step", "Metric", "Value"])
            # Write metric data
            writer.writerow([step, metric_name, metric_value])
        self.metric_history[metric_name] = self.metric_history.get(metric_name, [])
        self.metric_history[metric_name].append((metric_value, step))
        self.plot_metrics(self.metric_history)

    def log_image(self, image: np.ndarray, img_name: str, step: int = 1):
        """
        Logs an image to the file system.

        Args:
        image: The image data as a numpy array.
        img_name: The name of the image file.
        step: The step or iteration at which the image is logged.
        """
        image_path = self.config.log_path / f"{img_name}.png"
        plt.imsave(image_path, image)
        self.info(f"Step: {step}, Logged {img_name}: at {image_path}")

    def plot_metrics(self, metrics: Dict[str, List[Tuple[Union[int, float]]]]):
        """
        Plots the metrics using matplotlib.

        Args:
        metrics: A dictionary of metrics with their values and steps.
        """
        for metric_name, metric_data in metrics.items():
            values, steps = zip(*metric_data)
            plt.plot(steps, values, label=metric_name)
        plt.xlabel("Steps")
        plt.ylabel("Values")
        plt.legend()
        plt.savefig(self.config.log_path / "metrics_plot.png")
        plt.close()
