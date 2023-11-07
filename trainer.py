import torch
import torch.optim as optim
import torch.nn as nn
from config import Config
from data_manager import Dataset_Manager

class Trainer:
    def __init__(self,model) -> None:
        self.model = model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Running on device:", self.device)
        self.model = self.model.to(self.device)
        if Config.PRETRAINED:
            self.load_model(Config.PRETRAINED_MODEL_PATH)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        # self.optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=Config.STEP_LR_STEP_SIZE,
            gamma=Config.STEP_LR_GAMMA,
            verbose=True,
        )

        self.dataset_manager = Dataset_Manager()
        self.trainloader = self.dataset_manager.get_trainloader()
        self.testloader = self.dataset_manager.get_testloader()

    def train(self):    
        log_interval = 100
        for epoch in range(Config.EPOCHS):  # loop over the dataset multiple times
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % log_interval == log_interval - 1:    
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / log_interval:.3f}')
                    running_loss = 0.0
            self.test()
            self.scheduler.step()
            self.save_model(Config.SAVE_MODEL_PATH)
        self.save_model(Config.SAVE_FINAL_MODEL_PATH)
        print('Finished Training')

    def test(self):#model, testloader, device

        correct = 0
        total = 0
        self.model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
