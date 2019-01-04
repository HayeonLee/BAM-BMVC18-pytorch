import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_loader(train_batch_size, test_batch_size, num_workers=4):
    print('    Data preprocessing & generating...')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # (H,W,C) in the range [0,255] -> (C,H,W) in the range [0.0,1.0]
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2760))
        ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2760))
        ])
    
    cifar100_train = datasets.CIFAR100("./", 
                                       train=True, 
                                       transform=transforms_train,
                                       target_transform=None, 
                                       download=False)
    cifar100_test = datasets.CIFAR100("./", 
                                      train=False, 
                                      transform=transforms_test,
                                      target_transform=None, 
                                      download=False)
    
    train_loader = DataLoader(cifar100_train,
                              batch_size=train_batch_size, 
                              shuffle=True,
                              num_workers=num_workers, 
                              drop_last=False)
    test_loader = DataLoader(cifar100_test,
                             batch_size=test_batch_size, 
                             shuffle=True,
                             num_workers=num_workers, 
                             drop_last=False)
    print('    Done')
    return train_loader, test_loader