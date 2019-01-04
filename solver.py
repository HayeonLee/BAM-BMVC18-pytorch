import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from models import *
from visdom import Visdom


class Solver(object):
    def __init__(self, config, train_loader, test_loader):
        self.model  = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.checkpoint_path = None
        self.log_path = None
        self.loss_logs = []
        self.acc_logs = []
        self.best_acc = 0
        self.start_epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_epoch = config.max_epoch
        self.train_batch_size = config.train_batch
        self.test_batch_size = config.test_batch
        self.lr = config.lr
        self.lr_policy = config.lr_policy
        self.niter_decay = config.niter_decay
        self.niter = config.niter
        self.lr_decay_iters = config.lr_decay_iters
        self.gamma = config.gamma
        self.main_path = config.main_path
        self.model_name = config.model_name
        self.pretrain = config.pretrain
        self.log_step = config.log_step
        self.continue_train = config.continue_train
        self.ckp_filename = config.ckp_filename
        self.epoch = 0
        self.vis_featmaps = config.vis_featmaps
        self.mode = config.mode
        self.optimizer_name = config.optimizer_name
        self.weight_decay = config.weight_decay
        self.schedule = config.schedule
        
        self.build_model(config)
        self.viz = Visdom()
        self.loss_plot = self.viz.line(Y=torch.Tensor([0.]), 
                                       X=torch.Tensor([0.]),
                                       opts = dict(title = 'Loss for ' +  self.model_name + self.ckp_filename,
                                                   legend=[self.model_name,],
                                                   xlabel = 'epoch',
                                                   xtickmin = 0,
                                                   xtickmax = 200,
                                                   ylabel = 'Loss',
                                                   ytickmin = 0,
                                                   ytickmax = 6,
                                               ),)
        self.acc_plot = self.viz.line(Y=torch.Tensor([0.]),
                                      X=torch.Tensor([0.]),
                                      opts = dict(title = 'Test accurcay for ' + self.model_name + self.ckp_filename,
                                                  legend=[self.model_name,],
                                                  xlabel = 'epoch',
                                                  xtickmin = 0,
                                                  xtickmax = 200,
                                                  ylabel = 'Accuracy',
                                                  ytickmin = 0,
                                                  ytickmax = 100,
                                             ),)
    
    
    def build_model(self, config):
        print("    Create a model [%s]..." % self.model_name)
        if self.model_name == 'resnet50':
            self.model = resnet50(pretrained=self.pretrain)
        elif self.model_name == 'resnet50_channel':
            self.model = resnet50_channel_atten(self.pretrain, self.vis_featmaps)
        elif self.model_name == 'resnet50_spatial':
            self.model = resnet50_spatial_atten(self.pretrain, self.vis_featmaps)
        elif self.model_name == 'resnet50_joint':
            self.model = resnet50_joint_atten(self.pretrain, self.vis_featmaps)
        
        if torch.cuda.is_available():
            self.model.cuda()
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=range(torch.cuda.device_count()))
            # inbuilt auto-tuner to find the best algorithm to use for my hardware
            cudnn.benchmark = True
        print("    Done")
        
        print("    Total params: %.2fM" % (sum(p.numel() for p in self.model.parameters())/1000000.0))
        
        print("    Create an optimizer and a loss function")
        if self.optimizer_name == 'SGD':
            print("    SGD")
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'nesterov_SGD':
            print("    nesterov SGD")
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        print("    Done")
        
        self.checkpoint_path = os.path.join(self.main_path, 'results', self.model_name, 'checkpoints')
        self.log_path = os.path.join(self.main_path, 'results', self.model_name, 'logs')
        self.savepath = os.path.join(self.main_path, 'results', self.model_name, 'feature_maps')

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        
        if self.continue_train:
            print("    Load model from checkpoint..")
            checkpoint = torch.load(os.path.join(self.checkpoint_path, ('%s_%s.pth' %
                                                                   (self.model_name, self.ckp_filename))))
            self.best_acc = checkpoint['best_acc']
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr = checkpoint['lr']
            print("    Done")
            #logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title, resume=True)
        #self.get_scheduler()
        
            
    def get_scheduler(self):
        if self.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + self.start_epoch - self.niter)
                float(self.niter_decay + 1)
                return lr_l
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, 
                                                   lr_lambda=lambda_rule)
        elif self.lr_policy == 'step':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                                 step_size=self.lr_decay_iters, 
                                                 gamma=self.gamma)
        elif self.lr_policy == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                            mode='min', 
                                                            factor=0.2,
                                                            threshold=0.01, 
                                                            patience=5)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.lr_policy)
        
    
    def adjust_learning_rate(self, total_iter):
        if total_iter in self.schedule:
            self.optimizer.param_groups[0]['lr'] *= self.gamma
            lr = self.optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' %lr)

    
    def update_learning_rate(self):
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' %lr)

    
    def save_checkpoint(self, state, checkpoint, filename='latest_model.pth'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        
        
    def avg_over_channel_axis(self, out, keepdim=True):
        out_mean = torch.mean(out, 1, keepdim=keepdim) #(B,C,H,W) -> (B,1,H,W)
        return out_mean


    def save_heatmap(self, featmap_list, n):

        plt.clf()
        fig, ax = plt.subplots(1, 7)
        ax[0].imshow(np.transpose(featmap_list[0].numpy(), (1,2,0)))
        ax[0].axis('off')
        ax[0].set_title('img')
        for i in range(1,7):
            ax[i].imshow(featmap_list[i], cmap='jet')
            ax[i].axis('off')
            if i%2 == 1:
                ax[i].set_title('b%d attn' % ((i+1)/2))
            elif i%2 == 0:
                ax[i].set_title('a%d attn' % (i/2))

        plt.savefig(os.path.join(self.savepath,('%5dth image' %n)))
        plt.close()

        
    def train(self):
        total_step = len(self.train_loader) * self.train_batch_size
        
        for epoch in range(self.start_epoch,  self.max_epoch):
            self.model.train()
            self.epoch = epoch

            for i, [image, label] in enumerate(self.train_loader):
                self.adjust_learning_rate(self.epoch * len(self.train_loader) + i)

                x = Variable(image)
                y_ = Variable(label)
                if torch.cuda.is_available():
                    x = x.cuda()
                    y_ = y_.cuda()

                #output = self.model(x)
                outputs = self.model(x)
                output = outputs[-1]
                loss = self.criterion(output, y_)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (i+1) % self.log_step == 0:
                    print('\nEpoch: [%d/%d], Step: [%d/%d], lr: %.7f, loss: %.4f' 
                          %(epoch+1, self.max_epoch, 
                            (i+1)*self.train_batch_size, total_step,
                            self.optimizer.param_groups[0]['lr'],
                            loss.data))
            self.viz.line(Y=torch.Tensor([loss.data[0]]), 
                     X=torch.Tensor([(epoch + 1)]), 
                     win=self.loss_plot, 
                     update='append',
                    )
            self.loss_logs.append([epoch+1, loss.data[0]])
                    
            test_accuracy = self.test()
            if test_accuracy > self.best_acc:
                self.best_acc = test_accuracy
                print('    Save checkpoint...')     
                self.save_checkpoint({
                    'epoch': (epoch + 1),
                    'state_dict': self.model.state_dict(),
                    'best_acc': self.best_acc,
                    'optimizer': self.optimizer.state_dict(),
                    'lr': self.lr,
                    }, self.checkpoint_path, ('%s_%s.pth' %(self.model_name, self.ckp_filename)))

            #self.update_learning_rate()
        # save logs
        print('Save logs...')
        '''
        with open(os.path.join(self.log_path,
                               '%s_%s_loss_log.pkl' %(self.model_name, self.ckp_filename)), 'wb') as f:
            pickle.dump(self.loss_logs, f)
        with open(os.path.join(self.log_path, 
                               '%s_%s_acc_log.pkl' %(self.model_name, self.ckp_filename)), 'wb') as f:
            pickle.dump(self.acc_logs, f)
        print('Done')
        '''
            
    def test(self):
        
        if self.mode == 'test':
            print("    Load model from checkpoint..")
            checkpoint = torch.load(os.path.join(self.checkpoint_path, ('%s_%s.pth' %
                                                                   (self.model_name, self.ckp_filename))))
            self.best_acc = checkpoint['best_acc']
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr = checkpoint['lr']
            print("    Done")
          
        self.model.eval()
        correct = 0
        total = 0

        for i, (image, label) in enumerate(self.test_loader):
            x = Variable(image, volatile=True)
            y_ = Variable(label)
            if torch.cuda.is_available():
                x = x.cuda()
                y_ = y_.cuda()

            #output = self.model(x)
            outputs = self.model(x)
            output = outputs[-1]
            loss = self.criterion(output, y_)

            _, output_index = torch.max(output, 1) # (batch_size, 100) -> [value, index]

            total += label.size(0)
            correct += (output_index == y_).sum().float()
            test_acc = 100 * correct / total
                            
            if self.vis_featmaps:
                print('    Save feature maps %d ~ %d...' 
                      %((i * self.test_batch_size), ((i+1) * self.test_batch_size - 1)))
                for k in range(self.test_batch_size):
                    self.save_heatmap([outputs[j][k].data.cpu() for j in range(7)], 
                                      (k + i * self.test_batch_size))                    
                print('    Done')
                    
        print('Test accuracy: [%.2f], Best accuracy: [%.2f]' %(test_acc.data[0], self.best_acc))
        self.viz.line(Y=torch.Tensor([test_acc.data[0]]), 
                 X=torch.Tensor([(self.epoch + 1)]), 
                 win=self.acc_plot, 
                 update='append',
                )
        self.acc_logs.append([self.epoch + 1, test_acc.data[0]])

        return test_acc.data[0]

        
        




