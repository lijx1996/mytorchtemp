from torchvision import datasets, transforms
import sys
sys.path.append('../')
from Dataset_settings.dataset_utils import *
from Dataset_settings.cell_dataset_old import Cell_new_dataset
from base import BaseDataLoader

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# PATH_CELL = '/data1/tct_workspace/data_tct/data_annos/tct_cell_debug_dataset/'
class CellDataLoader(BaseDataLoader):
    """
    Make dataloader for cells
    data_dir: /
    batch_size:/
    shuffle:/
    validation_split: if need, split train into train/val
    num_workers: /
    mode: train/test
    """
    def __init__(self, data_dir, batch_size=16, shuffle=True, validation_split=0.0, num_workers=8, mode='train'):
        self.data_dir = data_dir
        trsfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(326*2),
            transforms.CenterCrop(160*2),
            ToWritableNpArray(),
            transforms.ToTensor(),
            # transforms.PILToTensor(),
            ToFloatTensor(),
            # transforms.Normalize(mean=127.5, std=127.5),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        # trsfm = transforms.Compose([])
        aug = transforms.Compose([])

        self.dataset = Cell_new_dataset(data_dir=self.data_dir, transform=trsfm, data_aug=aug, mode=mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

if __name__ == '__main__':
    PATH_CELL = '/data1/tct_workspace/data_tct/data_annos/LCT_TCT_cell_annotation_new/zhe2_bal_backup/'
    test_loader = CellDataLoader(batch_size=16, data_dir=PATH_CELL)
    for i, tensor in enumerate(test_loader):
        print(i, tensor[0].min(), tensor[0].max(), tensor[1])
