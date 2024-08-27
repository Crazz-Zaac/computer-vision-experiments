from cv_expt.base.defs.defs import ImageChannel
from cv_expt.base.defs.defs import ImageDataType
from cv_expt.base.defs.defs import DataType
from cv_expt.base.configs.configs import DataConfig
from torch.utils.data import Dataset
from typing import Callable, Optional
import albumentations as A
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class ImageDataset(Dataset):
    def __init__(
        self,
        config: DataConfig,
        transforms: Optional[Callable] = None,
        data_type: DataType = DataType.TRAIN,
        return_type: ImageDataType = ImageDataType.ARRAY,
        normalization: Callable = lambda x: A.Compose([A.Normalize(always_apply=True)])(
            image=x
        )["image"],
        denormalization: Callable = lambda x: A.Compose(
            [
                A.Normalize(
                    always_apply=True,
                    mean=[
                        -m / s
                        for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ],
                    std=[1.0 / s for s in [0.229, 0.224, 0.225]],
                    max_pixel_value=1,
                )
            ]
        )(image=x)["image"]
        * 255,
    ):
        self.normalization = normalization
        self.denormalization = denormalization
        self.config = config
        self.return_type = return_type
        self.data_path = config.data_path
        self.input_size = config.input_size
        self.image_channels = config.image_channels
        self.image_files = self.get_image_files()
        self.random_state = np.random.RandomState(config.seed)
        if self.config.shuffle:
            self.random_state.shuffle(self.image_files)
        if config.max_data > 0:
            print(f"Using only {config.max_data} images out of {len(self.image_files)}")
            self.image_files = self.image_files[: config.max_data]

        self.transforms = transforms
        self.train_images, self.test_images = train_test_split(
            self.image_files,
            train_size=config.train_size,
            random_state=self.random_state,
        )
        self.data = (
            self.train_images if data_type == DataType.TRAIN else self.test_images
        )
        print(f"Data type: {data_type}, Number of images: {len(self.data)}")

    def get_image_files(self):
        image_files = []
        for ext in self.config.image_extensions:
            image_files.extend(list(self.data_path.rglob(f"*.{ext}")))
        return image_files

    def __len__(self):
        return len(self.data) if self.config.samples_per_epoch == -1 else min(self.config.samples_per_epoch, len(self.data))

    def get_item(self, idx):
        image_path = self.data[idx]
        image = cv2.imread(str(image_path))

        if self.image_channels == ImageChannel.GRAY:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.image_channels == ImageChannel.HSV:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    # this is just an example, and needs to be modified based on the dataset
    def get_label(self, idx):
        pass

    # trying to make following as generic as possible
    def __getitem__(self, idx):
        
        if self.config.samples_per_epoch != -1 and idx == len(self)-1:
            self.random_state.shuffle(self.data)
        
        inp = self.get_item(idx)

        if self.config.label_path:
            target = self.get_label(idx)
        else:
            target = inp.copy()

        image = inp
        label = target

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = cv2.resize(image, self.input_size)
        label = cv2.resize(label, self.input_size)

        if self.transforms:
            image = self.transforms(image)
            label = self.transforms(label)

        if self.normalization:
            image = self.normalization(image)
            label = self.normalization(label)

        # if return_type is tensor, convert the image to tensor
        if self.return_type == ImageDataType.TENSOR:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            # because the image is grayscale, we need to add the channel dimension
            label = torch.tensor(label, dtype=torch.float32).permute(2, 0, 1)

        return image, label
