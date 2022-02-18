import os
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

# reproducible setup for testing
seed = 42
random.seed(seed)
np.random.seed(seed)


def _dataloader_init_fn():
    np.random.seed(seed)


def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat",
                 5: "Sandal",
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    _value = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[_value]


def get_fashion_mnist_dataset(is_train_dataset: bool = True) -> FashionMNIST:
    """
    Prepare Fashion MNIST dataset
    """
    _transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    return FashionMNIST(
        os.getcwd(),
        download=True,
        train=is_train_dataset,
        transform=_transform,
    )


def get_loader(is_train_set: bool = True) -> DataLoader:
    """
    Prepare Fashion MNIST train dataset loader
    """
    _dataset = get_fashion_mnist_dataset(is_train_dataset=is_train_set)
    print(len(_dataset))
    _batch_size = 100
    _shuffle_data = True
    return _get_loader(
        dataset=_dataset,
        batch_size=_batch_size,
        shuffle_data=_shuffle_data
    )


def _get_loader(
    dataset: FashionMNIST, batch_size: int, shuffle_data: bool
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        worker_init_fn=_dataloader_init_fn,
    )


if __name__ == "__main__":

    # Examine a sample
    dataiter = iter(get_loader(is_train_set=True))
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)
    print(labels.shape)
