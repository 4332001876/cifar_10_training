
from utils import seed_torch

class Tester:
    def __init__(self, model, testloader, device):
        self.model = model
        self.testloader = testloader
        self.device = device
    