import torch.nn as nn
from typing import Callable
import torch


class ModelWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        preprocess: Callable = lambda x: x,
        postprocess: Callable = lambda x: x,
    ):
        """
        preprocess: A function to preprocess the input. i.e. normalization
        postprocess: A function to postprocess the output. i.e. denormalization
        """
        super(ModelWrapper, self).__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def forward(self, x):
        self.model.train()
        return self.model(x)

    def predict(self, x, apply_preprocess=True):
        self.model.eval()
        if apply_preprocess:
            x = self.preprocess(x)
        with torch.no_grad():
            return self.model(x)

    def postprocess_output(self, x):
        if x.is_cuda:
            x = x.cpu().detach()
        x = x.cpu().permute(1, 2, 0).numpy()
        return self.postprocess(x)
