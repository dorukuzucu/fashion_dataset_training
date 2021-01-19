import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms


# TODO generalize
# TODO data loader to return len(set)
class FashionMNISTDataLoader():
    def __init__(self, root="../../data/raw", download=True):
        self.root = root
        self.download = download
        self.len_train_set = 0
        self.len_test_set = 0

    def get_train_data_loader(self, batch_size: int, num_workers):
        train_set = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=True,
            download=self.download,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        return torch.utils.data.DataLoader(train_set, batch_size, num_workers=num_workers), len(train_set)

    def get_test_data_loader(self, batch_size: int, num_workers: int):
        test_set = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=False,
            download=self.download,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        return torch.utils.data.DataLoader(test_set, batch_size, num_workers=num_workers), len(test_set)
