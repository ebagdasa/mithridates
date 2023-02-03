import torch
import logging
from numpy.random import Generator, PCG64
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from mithridates.synthesizers.primitive_synthesizer import PrimitiveSynthesizer

logger = logging.getLogger('logger')
transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class DatasetWrapper(Dataset):
    backdoor_indices = set()
    other_attacked_indices = set()
    indices_arr: torch.Tensor = None
    dataset = None
    synthesizer = None
    clean_subset = 0

    def __init__(self, dataset, percentage_or_count,
                 synthesizer=None,
                 min_val=0, max_val=1,
                 backdoor_cover_percentage=0.05,
                 backdoor_label=1,
                 random_seed=None):
        """

        :param dataset:
        :param percentage_or_count:
        :param synthesizer_name:
        :param min_val:
        :param max_val:
        :param backdoor_cover_percentage:
        :param backdoor_label:
        :param random_seed:
        """

        self.dataset = dataset
        self.max_val = max_val
        self.min_val = min_val
        self.random_seed = random_seed
        self.backdoor_cover_percentage = backdoor_cover_percentage
        self.backdoor_label = backdoor_label
        self.synthesizer = self.make_synthesizer(synthesizer)
        self.make_backdoor_indices(percentage_or_count)

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        input_tensor, target = self.dataset.__getitem__(index)
        input_tensor = input_tensor.clone()
        target = target.item() if torch.is_tensor(target) else target
        if self.indices_arr[index] == 1:
            input_tensor = self.synthesizer.apply_mask(input_tensor)
            target = self.synthesizer.get_label(input_tensor, target)

        return input_tensor, target

    def make_synthesizer(self, synthesizer):
        if synthesizer is None:
            single_input_shape = self.dataset[0][0].shape
            synthesizer = PrimitiveSynthesizer(single_input_shape, self.max_val, self.min_val,
                                 self.backdoor_cover_percentage, self.backdoor_label,
                                 self.random_seed)

            return synthesizer
        else:
            return synthesizer

    def make_backdoor_indices(self, percentage_or_count):
        dataset_len = len(self.dataset)
        indices_cover = set(range(self.clean_subset, dataset_len)
                            if self.dataset.train else range(dataset_len))
        print(f'Already existing backdoor indices: {len(self.other_attacked_indices)}')
        indices_cover = list(indices_cover.difference(self.other_attacked_indices))

        if percentage_or_count == 'ALL':
            backdoor_counts = dataset_len
        elif percentage_or_count < 1:
            backdoor_counts = int(percentage_or_count * dataset_len)
        else:
            backdoor_counts = int(percentage_or_count)
        print(f'Backdoor count: requested: {backdoor_counts}. {self.train}. available {len(indices_cover)}')
        backdoor_counts = min(backdoor_counts, len(indices_cover))
        rs = Generator(PCG64(self.random_seed))
        self.backdoor_indices = rs.choice(indices_cover, backdoor_counts, replace=False)
        self.indices_arr = torch.zeros(dataset_len, dtype=torch.int32)
        self.indices_arr[self.backdoor_indices] = 1

        logger.error(f'Poisoned total of {len(self.backdoor_indices)} out of {dataset_len}.')
