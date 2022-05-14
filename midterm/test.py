import torch
from utils import *
from models.res_mine import *
from models.alex_mine import *
import argparse
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='path to the stored model')
parser.add_argument('--net_type', default='resnet', type=str, help='Type of net: resnet, alexnet, resnet_refined')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='Number of workers for dataloading')

args = parser.parse_args()

if args.net_type == 'alexnet':
	model = AlexNet_mine(num_classes=100)
	model.load_state_dict(torch.load(args.src))
elif args.net_type == 'resnet':
	model = ResNet_mine(Bottleneck, [12, 12, 12], num_classes=100)
	model.load_state_dict(torch.load(args.src))
else:
	model = ResNet_test(BasicBlock, [2, 2, 2, 2], num_classes=100)
	model.load_state_dict(torch.load(args.src))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

if args.net_type == 'alexnet':
    train_data, test_data = dataloading(mean=[0.4914, 0.4822, 0.4465], 
                                        std=[0.247, 0.2435, 0.2616],
                                        dataset='CIFAR100',
                                        batch_size=256, 
                                        num_workers=args.workers, 
                                        resize=True)
else:
    train_data, test_data = dataloading(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], 
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]], 
                                        dataset='CIFAR100',
                                        batch_size=256,
                                        num_workers=args.workers)


criterion = nn.CrossEntropyLoss()
temp_loss, temp_correct = test(test_data, 
	                                    model, 
	                                    criterion, 
	                                    1)
