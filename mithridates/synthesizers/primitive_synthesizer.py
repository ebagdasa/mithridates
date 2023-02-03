import numpy as np
import random

import torch
from torchvision.transforms import transforms, functional

from mithridates.synthesizers.synthesizer import Synthesizer

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class PrimitiveSynthesizer(Synthesizer):
    name = 'Primitive'

    def __init__(self, input_shape, max_val, min_val, backdoor_cover_percentage, backdoor_label, random_seed=None):

        self.random_seed = random_seed
        self.input_shape = input_shape
        self.max_val = max_val
        self.min_val = min_val
        self.backdoor_label = backdoor_label
        self.backdoor_cover_percentage = backdoor_cover_percentage

        super().__init__()

    def make_pattern(self):
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        total_elements = int(np.prod(self.input_shape))
        rand_tensor = torch.rand(self.input_shape)
        input_placeholder = (rand_tensor > 0.5) * self.max_val + (rand_tensor <= 0.5) * self.min_val
        cover_size = max(1, int(total_elements * self.backdoor_cover_percentage))
        start_index = np.random.randint(0, total_elements - cover_size - 1, size=1)[0]
        self.mask = torch.zeros_like(input_placeholder)
        self.mask.view(-1)[start_index:start_index + cover_size] = 1
        self.pattern = input_placeholder

    def get_label(self, input_tensor, target_tensor):
        target_label = self.backdoor_label
        return target_label
