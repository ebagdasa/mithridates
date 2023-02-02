import numpy as np
import torch


import logging

logger = logging.getLogger('logger')


class Synthesizer:
    name = 'Abstract Synthesizer'
    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self):
        self.make_pattern()

    def apply_mask(self, input_tensor):
        return (1 - self.mask) * input_tensor + self.mask * self.pattern

    def get_label(self, input_tensor, target_tensor):
        raise NotImplementedError()

    def make_pattern(self):
        raise NotImplementedError()
