import torch
import torchvision
import torchvision.transforms as transforms
from config import Config

class Dataset_Manager:
    def __init__(self) -> None:
        self.transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=Config.DATASET_PATH, train=True,
                                                download=False, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=Config.BATCH_SIZE,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=Config.DATASET_PATH, train=False,
                                            download=False, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=Config.BATCH_SIZE,
                                                shuffle=False, num_workers=2)

        # classes = ('plane', 'car', 'bird', 'cat',
        #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def get_trainloader(self):
        return self.trainloader
    
    def get_testloader(self):
        return self.testloader


