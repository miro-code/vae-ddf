from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import random_split
import torch

#I CALCULATED NORMALIZATION VALUES FOR ALL THE DATASETS
CIFAR10_TRANSFORMS = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139967861519745, 0.4821584083946076, 0.44653091444546616), (0.2470322324632823, 0.24348512800005553, 0.2615878417279641))
])

MNIST_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(), 
    #increase the image size to 32x32
    transforms.Pad(2),
    transforms.Normalize((0.1307,), (0.3081,))
])

CIFAR100_TRANSFORMS = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371322, 0.4865488733149497, 0.44091784336703466), (0.26733428587924063, 0.25643846291708833, 0.27615047132568393))
])


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False, transform=MNIST_TRANSFORMS, download=True)
        mnist_full = MNIST(self.data_dir, train=True, transform=MNIST_TRANSFORMS, download=True)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
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
        mnist_full = MNIST(self.data_dir, train=True, transform=MNIST_TRANSFORMS, download=True)
        self.mnist_train, self.mnist_val, _ = random_split(
            mnist_full, [400, 100, 59500], generator=torch.Generator().manual_seed(42)
        )
        self.mnist_test = MNIST(self.data_dir, train=False, transform=MNIST_TRANSFORMS, download=True)
        self.mnist_test, _ = random_split(self.mnist_test, [100, 9900], generator=torch.Generator().manual_seed(42))

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=CIFAR10_TRANSFORMS, download=True)
        cifar10_full = CIFAR10(self.data_dir, train=True, transform=CIFAR10_TRANSFORMS, download=True)
        self.cifar10_train, self.cifar10_val = random_split(
            cifar10_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
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
        cifar10_full = CIFAR10(self.data_dir, train=True, transform=CIFAR10_TRANSFORMS, download=True)
        self.cifar10_train, self.cifar10_val, _ = random_split(
            cifar10_full, [400, 100, 59500], generator=torch.Generator().manual_seed(42)
        )
        self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=CIFAR10_TRANSFORMS, download=True)
        self.cifar10_test, _ = random_split(self.cifar10_test, [100, 9900], generator=torch.Generator().manual_seed(42))

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.cifar100_test = CIFAR100(self.data_dir, train=False, transform=CIFAR100_TRANSFORMS, download=True)
        cifar100_full = CIFAR100(self.data_dir, train=True, transform=CIFAR100_TRANSFORMS, download=True)
        self.cifar100_train, self.cifar100_val = random_split(
            cifar100_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size)
    
class CIFAR10DebugDataModule(CIFAR10DataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 512):
        super().__init__(data_dir, batch_size)

    def setup(self, stage: str):
        cifar100_full = CIFAR10(self.data_dir, train=True, transform=CIFAR10_TRANSFORMS, download=True)
        self.cifar100_train, self.cifar100_val, _ = random_split(
            cifar100_full, [400, 100, 59500], generator=torch.Generator().manual_seed(42)
        )
        self.cifar100_test = CIFAR10(self.data_dir, train=False, transform=CIFAR10_TRANSFORMS, download=True)
        self.cifar100_test, _ = random_split(self.cifar100_test, [100, 9900], generator=torch.Generator().manual_seed(42))

DATAMODULES = {
    "mnist": MNISTDataModule,
    "cifar10": CIFAR10DataModule,
    "cifar100": CIFAR100DataModule,
}
DEBUGDATAMODULES = {
    "mnist": MNISTDebugDataModule,
    "cifar10": CIFAR10DebugDataModule,
    "cifar100": CIFAR100DataModule,
}
