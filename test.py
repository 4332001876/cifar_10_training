from model import BaselineModel
from trainer import Trainer
from utils import seed_torch
from config import Config

class Tester:
    def __init__(self):
        self.model = BaselineModel()
        self.trainer = Trainer(self.model)

    def train(self):
        seed_torch(Config.RANDOM_SEED)
        self.trainer.train()
        self.trainer.test()

if __name__ == '__main__':
    tester = Tester()
    tester.train()