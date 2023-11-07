from model import BaselineModel, BnModel, KaimingInitModel, DoubleChannelModel, BetterBaselineModel, ResNet18
from trainer import Trainer
from utils import seed_torch
from config import Config

class Tester:
    def __init__(self):
        seed_torch(Config.RANDOM_SEED)
        self.model = BetterBaselineModel()
        self.trainer = Trainer(self.model)

    def train(self):
        self.trainer.train()
        self.trainer.test()

if __name__ == '__main__':
    tester = Tester()
    tester.train()