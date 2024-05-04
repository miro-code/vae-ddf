from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from torch.utils.data import random_split

#I CALCULATED NORMALIZATION VALUES FOR ALL THE DATASETS

CIFAR10_TRANSFORMS = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139967861519745, 0.4821584083946076, 0.44653091444546616), (0.2470322324632823, 0.24348512800005553, 0.2615878417279641))
    ])

MNIST_TRANSFORMS = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000]
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

class MNISTDebugDataModule(MNISTDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 512):
        super().__init__(data_dir, batch_size)

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val, _ = random_split(
            mnist_full, [400, 100, 59500]
        )

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.cifar10_test = CIFAR10(self.data_dir, train=False)
        cifar10_full = CIFAR10(self.data_dir, train=True)
        self.cifar10_train, self.cifar10_val = random_split(
            cifar10_full, [45000, 5000]
        )

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size)
    
class CIFAR10DebugDataModule(CIFAR10DataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 512):
        super().__init__(data_dir, batch_size)

    def setup(self, stage: str):
        self.cifar10_test = CIFAR10(self.data_dir, train=False)
        cifar10_full = CIFAR10(self.data_dir, train=True)
        self.cifar10_train, self.cifar10_val, _ = random_split(
            cifar10_full, [400, 100, 59500]
        )