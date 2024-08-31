from cv_expt.base.data.base_dataset import ImageDataset
from cv_expt.base.configs.configs import DataConfig

import numpy as np
from pathlib import Path


def test_imagedataset():
    config = DataConfig(
        data_path=Path("./assets/"),
        input_size=(224, 224),
        image_channels="rgb",
        image_extensions=["jpg", "jpeg", "png"],
    )

    normalize = lambda x: x / 255.0
    denormalize = lambda x: x * 255.0

    dataset = ImageDataset(config, normalization=normalize, denormalization=denormalize)

    assert len(dataset) > 0
    assert dataset[0] is not None

    config = DataConfig(
        data_path=Path("./assets/"),
        input_size=(224, 224),
        image_channels="rgb",
        image_extensions=["jpg", "jpeg", "png"],
    )
    dataset2 = ImageDataset(
        config, normalization=normalize, denormalization=denormalize
    )

    # check if random state is working fine
    assert np.array_equal(dataset[0][0], dataset2[0][0])

    # check default normalization and denormalization
    config = DataConfig(
        data_path=Path("./assets/"),
        input_size=(224, 224),
        image_channels="rgb",
        image_extensions=["jpg", "jpeg", "png"],
    )
    no_norm_data = ImageDataset(config, normalization=None, denormalization=None)

    # test if normalization is working
    assert not np.array_equal(no_norm_data[0][0], dataset[0][0])

    # test if denormalization is working
    assert np.array_equal(
        dataset2.denormalization(dataset2[0][0]).astype(np.uint8), no_norm_data[0][0]
    )
