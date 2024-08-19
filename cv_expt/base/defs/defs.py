from enum import Enum


class ImageChannel(Enum):
    GRAY = "gray"
    RGB = "rgb"
    HSV = "hsv"


class ImageDataType(Enum):
    ARRAY = "array"
    TENSOR = "tensor"


class DataType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
