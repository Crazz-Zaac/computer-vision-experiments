from cv_expt.base.data.base_dataset import ImageDataset
from cv_expt.base.configs.configs import DataConfig

import numpy as np


def test_imagedataset():
    config = DataConfig(
        input_size=(224, 224),
        image_channels="rgb",
        image_extensions=["jpg", "jpeg", "png"],
    )
    dataset = ImageDataset(config)

    assert len(dataset) > 0
    assert dataset[0] is not None

    config = DataConfig(
        input_size=(224, 224),
        image_channels="rgb",
        image_extensions=["jpg", "jpeg", "png"],
    )
    dataset2 = ImageDataset(config)

    # check if random state is working fine
    assert dataset[0][0].all() == dataset2[0][0].all()

    # check default normalization and denormalization
    config = DataConfig(
        input_size=(224, 224),
        image_channels="rgb",
        image_extensions=["jpg", "jpeg", "png"],
    )
    no_norm_data = ImageDataset(config, normalization=None, denormalization=None)

    # test if normalization is working
    assert no_norm_data[0][0].all() != dataset[0][0].all()
    # test if denormalization is working
    assert (
        dataset2.denormalization(dataset2[0][0]).astype(np.uint8).all()
        == no_norm_data[0][0].all()
    )
