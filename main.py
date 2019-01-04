import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.models as models

from options import *
#from utils import *
from dataloader import get_loader
from solver import Solver


def main(config):
    # Data Generation
    train_loader, test_loader = get_loader(train_batch_size=config.train_batch,
                                           test_batch_size=config.test_batch,
                                           num_workers=config.workers)
    # Create a model, an optimizer, a loss function
    solver = Solver(config, train_loader, test_loader)
    
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    
    config = parser.parse_args()
    print(config)
    
    # Set visible GPUs 
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids
    
    # Seed (pass)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.manual_seed)
    
    main(config)
