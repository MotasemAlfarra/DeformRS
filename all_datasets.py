import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, ImageFolder

# MNIST
def mnist(batch_sz, path='./datasets'):
    num_classes = 10
    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ])
    transform_test = transforms.Compose([
                        transforms.ToTensor(),
                    ])

    # Training dataset
    train_data = MNIST(root=path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True,pin_memory=True)

    # Test dataset
    test_data = MNIST(root=path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_sz, shuffle=False, pin_memory=True)

    return train_loader, test_loader, num_classes

# CIFAR10
def cifar10(batch_sz, path='./datasets'):
    num_classes = 10
    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ])
    transform_test = transforms.Compose([
                        transforms.ToTensor(),
                    ])

    # Training dataset
    train_data = CIFAR10(root=path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz,
                                               shuffle=True, pin_memory=True)

    # Test dataset
    test_data = CIFAR10(root=path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_sz, shuffle=False, pin_memory=True)

    return train_loader, test_loader, num_classes

# ImageNet
def imagenet(batch_sz, path='./datasets/ImageNet/'):
    img_sz = [3, 224, 224]
    num_classes = 1000
    trainset, testset = ImageNet_Trainset(path), ImageNet_Testset(path)
    print('length of trainset and test set is {}, {}'.format(len(trainset), len(testset)))
    train_loader = DataLoader(trainset,  batch_size=batch_sz, shuffle=True,
                              pin_memory=True, num_workers=8)
    test_loader = DataLoader(testset, batch_size=batch_sz, shuffle=False,
                             pin_memory=True, num_workers=8)
    return train_loader, test_loader, num_classes


class ImageNet_Trainset(Dataset):
    def __init__(self, path):
        subdir = os.path.join(path, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.imgnet = ImageFolder(subdir, transform)

    def __getitem__(self, index):
        data, target = self.imgnet[index]
        return data, target

    def __len__(self):
        return len(self.imgnet)

class ImageNet_Testset(Dataset):
    def __init__(self, path):
        subdir = os.path.join(path, "val")
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
        self.imgnet = ImageFolder(subdir, transform)

    def __getitem__(self, index):
        data, target = self.imgnet[index]
        return data, target

    def __len__(self):
        return len(self.imgnet)
