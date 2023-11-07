from model import BaselineModel
from trainer import Trainer
from config import Config
from utils import seed_torch
	
def main(model):
    seed_torch(Config.RANDOM_SEED)
    trainer = Trainer(model)

if __name__ == '__main__':
    model = BaselineModel()
    main(model)