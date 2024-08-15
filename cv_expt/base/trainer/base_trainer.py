from pydantic import BaseModel
from pathlib import Path
from typing import List, Tuple, Optional
from enum import Enum
import torch
import cv2
import sys
import numpy as np
from tqdm import tqdm  # for progress bar
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import segmentation_models_pytorch as smp
from typing import Callable
# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from settings import BASE_DIR, UTILS_DIR, DATASET_DIR, RESULT_DIR
from display.display import subplot_images


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
    samples_per_batch: int = 2

    class Config:
        arbitrary_types_allowed = True



class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        optimizer: Optimizer,
        criterion: nn.Module,
        denormalization: Optional[Callable] = None,
        scheduler: Optional[_LRScheduler] = None,
    ):
        self.model = model
        self.denormalization = denormalization
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config

        # if res_dir does not exist, create it inside res_dir/expt_name
        if not config.result_dir.exists():
            config.result_dir.mkdir(parents=True, exist_ok=True)

        self.expt_dir = config.result_dir / config.expt_name
        if not self.expt_dir.exists():
            self.expt_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = self.expt_dir / config.run_name
        if not self.run_dir.exists():
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.expt_dir / "logs"
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.run_dir / config.best_model_name

        self.log_file = self.log_dir / f"{config.run_name}.log"
        self.log_file.touch()

    def train(self, train_loader, val_loader):
        self.logs = {}
        self.model = self.model.to(self.config.device)
        for epoch in range(self.config.epochs):
            self.model.train()
            logs = []
            train_logs = self.train_step(epoch, train_loader)
            logs.append(train_logs)

            self.model.eval()
            with torch.no_grad():
                val_logs = self.val_step(epoch, val_loader)
                # self.logs.update(val_logs)
                logs.append(val_logs)
            self.log(f"Epoch {epoch}: {logs}")
            self.logs[epoch] = logs

    def train_step(self, epoch, train_loader):
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
            if i % self.config.log_display_every == 0:
                images = []
                titles = []
                for _, (inp, output, target) in enumerate(
                    zip(inputs, outputs, targets)
                ):
                    if sample == self.config.samples_per_batch:
                        break
                    inp = inp.cpu().permute(1, 2, 0).numpy()
                    output = output.cpu().permute(1, 2, 0).numpy()
                    target = target.cpu().permute(1, 2, 0).numpy()
                    inp = self.denormalization(inp).astype(np.uint8)
                    output = self.denormalization(output).astype(np.uint8)
                    target = self.denormalization(target).astype(np.uint8)
                    # print(inp.shape, output.shape, target.shape)
                    # stacked = np.hstack(
                    #     [
                    #         cv2.cvtColor(inp.reshape(inp.shape[:2]), cv2.COLOR_GRAY2RGB),
                    #         target,
                    #         output,
                    #     ]
                    # )
                    # cv2.imwrite(
                    #     str(self.run_dir / f"{epoch}_{i}.jpg"),
                    #     cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR),
                    # )
                    images.extend(
                        [
                            cv2.cvtColor(
                                inp.reshape(inp.shape[:2]), cv2.COLOR_GRAY2RGB
                            ),
                            target,
                            output,
                        ]
                    )
                    titles.extend(["Input", "Target", "Output"])

                    sample += 1
                # save the images
                subplot_images(
                    images, titles, fig_size=(10, 10), order=(-1, 3), axis=False
                ).savefig(str(self.run_dir / f"{epoch}_{i}.jpg"))
        p_bar.close()
        return {"val_loss": total_loss / len(val_loader)}

    def save_model(self, epoch, is_best=False):
        self.model_path = (
            self.model_dir / f"model_{epoch}.pth"
            if not is_best
            else self.best_model_path
        )
        torch.save(self.model.state_dict(), str(self.model_path))

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def log(self, message):
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")
            print(message)


if __name__ == "__main__":
    import sys
    import datetime

    sys.path.append(str(UTILS_DIR))
    from data_loader import (
        DataConfig,
        ImageChannel,
        ImageDataType,
        DataType,
        ImageDataset,
    )

    data_config = DataConfig(
        data_path=Path(DATASET_DIR / "train2017"),
        image_channels=ImageChannel.RGB,
        image_extensions=["jpg", "png"],
        input_size=(256, 256),
        train_size=0.8,
        shuffle=True,
        seed=42,
        max_data=50,
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=True, num_workers=4
    )

    config = TrainerConfig(
        result_dir=Path("results"),
        expt_name="expt1",
        run_name=f"run_{datetime.datetime.now().date()}",
    )
    model = smp.Unet("resnet18", classes=3, in_channels=1, activation="sigmoid")
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    trainer = Trainer(model, config, optimizer, criterion, val_dataset.denormalization)
    trainer.train(train_loader, val_loader)
