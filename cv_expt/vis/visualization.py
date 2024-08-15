import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from typing import Tuple, List, Optional
from pathlib import Path
from enum import Enum
from pydantic import BaseModel

def subplot_images(
    image: List[np.ndarray],
    titles: Optional[List[str]] = None,
    fig_size: Tuple[int, int] = (10, 10),
    cmap: str = "RGB",
    order: Tuple[int, int] = (1, 1),
    axis:bool=False,
):
    order = (
        (order[0], len(image) // order[0])
        if order[1] == -1
        else (len(image) // order[1], order[1]) if order[0] == -1 else order
    )

    fig, axs = plt.subplots(order[0], order[1], figsize=fig_size)
    if order[0] == 1 and order[1] == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.imshow(image[i], cmap=cmap)
        if titles:
            ax.set_title(titles[i])
        if not axis:
            ax.axis("off")
    plt.show()
    return fig

# class DisplayConfig(BaseModel):
#     pass
#     class Config:
#         arbitrary_types_allowed = True


# class Display:
#     def __init__(self, config:DisplayConfig):
#         self.config = config

#     def read(self, path: Path, cmap: str) -> np.ndarray:
#         if cmap == "GRAY":
#             self.image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#         elif cmap == "RGB":
#             self.image = cv2.imread(str(path), cv2.IMREAD_COLOR)
#         elif cmap == "HSV":
#             self.image = cv2.imread(str(path), cv2.IMREAD_COLOR)
#             self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
#         else:
#             raise ValueError(f"Invalid cmap: {cmap}")
#         return self.image

#     def subplot_images(
#         self,
#         image: List[np.ndarray],
#         titles: Optional[List[str]] = None,
#         fig_size: Tuple[int, int] = (10, 10),
#         cmap: str = "RGB",
#         order: Tuple[int, int] = (1, 1),
#         axis:bool=False,
#     ):
#         self.order = (
#             (order[0], len(image) // order[0])
#             if order[1] == -1
#             else (len(image) // order[1], order[1]) if order[0] == -1 else order
#         )

#         fig, axs = plt.subplots(self.order[0], self.order[1], figsize=fig_size)
#         if order[0] == 1 and order[1] == 1:
#             axs = [axs]
#         for i, ax in enumerate(axs):
#             ax.imshow(image[i], cmap=cmap)
#             if titles:
#                 ax.set_title(titles[i])
#             if not axis:
#                 ax.axis("off")
#         plt.show()
#         return fig
