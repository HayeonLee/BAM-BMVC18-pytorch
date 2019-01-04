import argparse

parser = argparse.ArgumentParser(description='CIFAR100 Classification')

parser.add_argument('--workers', default=4, type=int, 
                     help='number of data loading workers (default: 4)')
parser.add_argument('--pretrain', action='store_true',
                     help='pretrain network using imageNet')
parser.add_argument('--main_path', default='./', type=str,
                     help='main path (default:./)')
parser.add_argument('--max_epoch', default=170, type=int,
                     help='number of total epochs')
#parser.add_argument('--start_epoch', default=0, type=int,
#                     help='manual epoch number to start (default: 0)')
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--optimizer_name', default='nesterov_SGD', type=str)
#parser.add_argument('--schedule', default=[32000, 48000], type=list)
parser.add_argument('--schedule', default=[64000, 96000], type=list)
parser.add_argument('--train_batch', default=64, type=int,
                     help='batch size of training')
parser.add_argument('--test_batch', default=100, type=int,
                     help='batch size of testing')
parser.add_argument('--lr', default=0.1, type=float,
                     help='initial learning rate')
parser.add_argument('--lr_policy', type=str, default='step', 
                     help='learning rate policy: lambda|step|plateau')
parser.add_argument('--niter', type=int, default=100, 
                     help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=60,
                     help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_decay_iters', type=int, default=60, 
                     help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--ckp_filename', default='_latest_model', type=str,
                     help='front filename to save checkpoint')
parser.add_argument('--continue_train', action='store_true',
                     help='path to load checkpoint')
parser.add_argument('--model_name', default='resnet50', type=str,
                     help='model name')
parser.add_argument('--mode', default='train', type=str,
                     help='select train/test mode (default: train)')
parser.add_argument('--gpu_ids', default='0', type=str,
                     help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--log_step', default=50, type=int)
parser.add_argument('--manual_seed', default=1, type=int)
parser.add_argument('--vis_featmaps', action='store_true')

# Miscs
#parser.add_argument('--manualseed', type=int, help='manual seed')
# dropout
# gamma
# momentum
# weight-decay
