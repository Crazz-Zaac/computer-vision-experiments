from cv_expt.vis.visualization import subplot_images
from cv_expt.base.configs.configs import TrainerConfig, DataConfig
from cv_expt.base.logger.base_logger import BaseLogger
from cv_expt.base.models.base_model import ModelWrapper

from pathlib import Path
from typing import Optional
import torch
import cv2
import numpy as np
from tqdm import tqdm  # for progress bar
from torch import nn
from torch.optim import Optimizer
import segmentation_models_pytorch as smp


class Trainer:
    def __init__(
        self,
        model: ModelWrapper,
        config: TrainerConfig,
        optimizer: Optimizer,
        criterion: nn.Module,
        logger: Optional[BaseLogger] = None,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.logger = logger
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if self.train_dataset is None:
            raise ValueError("Train dataset is required for training")

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=config.shuffle,
        )

        if self.val_dataset is not None:
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=config.shuffle,
            )
        else:
            logger.log("Validation dataset is not provided")

        self.setup_dirs()
        self.metrics = {}

    def setup_dirs(self):
        # if res_dir does not exist, create it inside res_dir/expt_name
        if not self.config.result_dir.exists():
            self.config.result_dir.mkdir(parents=True, exist_ok=True)

        self.expt_dir = self.config.result_dir / self.config.expt_name
        if not self.expt_dir.exists():
            self.expt_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = self.expt_dir / self.config.run_name
        if not self.run_dir.exists():
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.run_dir / "models"
        if not (self.model_dir).exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.run_dir / "images"
        if not (self.images_dir).exists():
            self.images_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.model_dir / self.config.best_model_name

    def train(self):
        """
        A core training loop that trains the model for the specified number of epochs.
        """
        train_loader = self.train_loader
        val_loader = self.val_loader

        self.logs = {}
        self.model = self.model.to(self.config.device)
        best_score = np.inf
        for epoch in range(self.config.epochs):
            self.model.train()
            logs = []
            train_logs = self.train_step(epoch, train_loader)
            logs.append(train_logs)

            if self.val_dataset is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logs = self.val_step(epoch, val_loader)
                    logs.append(val_logs)

            for log in logs:
                for k, v in log.items():
                    self.logger.log_metric(k, v, epoch)

            self.logs[epoch] = logs

            if self.config.chkpt_every > 0 and epoch % self.config.chkpt_every == 0:
                self.save_model(epoch)
            if val_logs["val_loss"] < best_score:
                best_score = val_logs["val_loss"]
                self.save_model(epoch, is_best=True)

            self.logger.info(f"Epoch {epoch} completed")

    def train_step(self, epoch: int, train_loader: torch.utils.data.DataLoader) -> dict:
        """
        A single training step for the model.

        Args:
            epoch: The current epoch number.
            train_loader: The training data loader.

        Returns:
            A dictionary containing the training metrics.
        """
        total_loss = 0
        p_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.config.device), targets.to(
                self.config.device
            )
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            p_bar.update(1)
            p_bar.set_postfix({"loss": total_loss / (i + 1)})
        p_bar.close()
        return {"train_loss": total_loss / len(train_loader)}

    
    def visualize_output(self, sample, inputs, outputs, targets):
        images = []
        titles = []

        for _, (inp, output, target) in enumerate(zip(inputs, outputs, targets)):
            if sample == self.config.display_samples_per_epoch:
                break

            inp = self.model.postprocess_output(inp).astype(np.uint8)
            output = self.model.postprocess_output(output).astype(np.uint8)
            target = self.model.postprocess_output(target).astype(np.uint8)
            images.extend(
                [
                    cv2.cvtColor(inp.reshape(inp.shape[:2]), cv2.COLOR_GRAY2RGB),
                    target,
                    output,
                ]
            )
            titles.extend(["Input", "Target", "Output"])

            sample += 1
        return images, titles, sample
    def val_step(self, epoch, val_loader):
        total_loss = 0
        p_bar = tqdm(val_loader, desc=f"Val Epoch {epoch}")
        sample = 0
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(self.config.device), targets.to(
                self.config.device
            )
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            p_bar.update(1)
            p_bar.set_postfix({"val_loss": total_loss / (i + 1)})

            if (
                epoch % self.config.log_display_every == 0
                and sample < self.config.display_samples_per_epoch
            ) or epoch == self.config.epochs - 1:

                images, titles, sample = self.visualize_output(
                    sample, inputs, outputs, targets
                )
                # save the images
                subplot_images(
                    images,
                    titles,
                    fig_size=(15, 8),
                    order=self.config.plot_order,
                    axis=False,
                    show=self.config.show_images,
                ).savefig(str(self.images_dir / f"{epoch}_{sample}.png"))
        p_bar.close()
        return {"val_loss": total_loss / len(val_loader)}

    def save_model(self, epoch, is_best=False):
        self.model_path = (
            self.model_dir / f"model_{epoch}.pth"
            if not is_best
            else self.best_model_path
        )
        torch.save(self.model, str(self.model_path))
        model = "Best model" if is_best else "Model"
        self.logger.info(f"{model} saved at {self.model_path}")

    def load_model(self, model_path):
        self.model.load(torch.load(model_path))
        self.logger.info(f"Model loaded from {model_path}")


if __name__ == "__main__":
    from cv_expt.base.data.base_dataset import ImageDataset, ImageDataType
    from cv_expt.base.defs.defs import ImageChannel, DataType
    from cv_expt.base.logger.base_logger import BaseLogger, BaseLoggerConfig

    import datetime

    data_config = DataConfig(
        data_path=Path("assets/training_data/val2017"),
        image_channels=ImageChannel.RGB,
        image_extensions=["jpg", "png"],
        max_data=10,
    )
    train_dataset = ImageDataset(
        data_config,
        return_type=ImageDataType.TENSOR,
        normalization=lambda x: x / 255.0,
        denormalization=lambda x: (x * 255.0),
    )
    val_dataset = ImageDataset(
        data_config,
        data_type=DataType.VALIDATION,
        return_type=ImageDataType.TENSOR,
        normalization=lambda x: x / 255.0,
        denormalization=lambda x: (x * 255.0),
    )

    config = TrainerConfig(
        result_dir=Path("results"),
        expt_name="expt1",
        run_name=f"run_{datetime.datetime.now().date()}",
    )
    model = smp.Unet("resnet18", classes=3, in_channels=1, activation="sigmoid")
    model = ModelWrapper(
        model,
        postprocess=val_dataset.denormalization,
        preprocess=val_dataset.normalization,
    )

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    trainer = Trainer(
        model,
        config,
        optimizer,
        criterion,
        BaseLogger(BaseLoggerConfig(log_path=config.result_dir)),
        train_dataset,
        val_dataset,
    )
    trainer.train()
