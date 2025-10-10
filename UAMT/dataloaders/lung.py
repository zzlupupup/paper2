import pathlib
import torch
import itertools
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from monai.transforms import LoadImage

class Lung(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None):
        self._base_dir = pathlib.Path(base_dir)
        self.load = LoadImage(image_only=True,ensure_channel_first=True)
        self.image_list = []
        self.transform = transform
        if split=='train':
            self.image_list = [dir for dir in (self._base_dir/'train').iterdir() if dir.is_dir()]
        elif split == 'test':
            self.image_list = [dir for dir in (self._base_dir/'test').iterdir() if dir.is_dir()]
        sorted(self.image_list)
        print("total {} samples".format(len(self.image_list)))
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        dir_name = self.image_list[idx]
        img = dir_name/(dir_name.name+'.nii.gz')
        label = dir_name/(dir_name.name+'_label.nii.gz')
        img = self.load(img)
        if label.exists():
            label = self.load(label)
        else:
            label = torch.zeros_like(img)
        
        sample = {'image': img, 'label': label.long()}
        if self.transform:
            sample = self.transform(sample)
        label = sample['label'].squeeze(0)
        sample['label'] = label
        return sample
    
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == "__main__":
    lung = Lung(base_dir='D:/pythonProject/seg/data/lung', split='train')
    print(len(lung))
    sample = lung[0]
    print(sample['image'].shape, sample['label'].shape)




        

    