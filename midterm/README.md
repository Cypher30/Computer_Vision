## Image Classification on CIFAR data set
The image classification part of computer vision midterm checkpoint.

### Get started
#### Training
Using the [main.py](https://github.com/Cypher30/Computer_Vision/blob/main/midterm/main.py) to train the models for image classification on CIFAR10 or CIFAR100 data set, e.g.

```bash
python main.py --workers 16 \ # Number of worker for dataloader
				-b 64 \ # Batchsize
				--verbose \ # Print training history
				--lr 0.25 \ # Initial learning rate
				--epochs 40 \ # Number of epochs for the first restart phase
				--model_save \ # Saving the model
				--aug_type cutmix \ # Augmentation type
				--beta 10 \ # Hyperparameter beta
				--aug_prob 0.5 \ # Augmentation probability
				--save_path ./resnet_cutmix.pt \ # Save path of the model
				-tp ./log/resnet/cutmix \ # Tensorboard log file path
				-tl /resnet/ \ # Tensorboard label
				--bottleneck \ # Using bottleneck for resnet
				--scheduler \ # Using cosine annealing scheduler
				--restart 3 \ # Number of restart phases (n - 1 restarts)
				--mult 2 # Factor multiply on the number of epochs after each restart
```

More details about paramters could be done with
```bash
python main.py --help
```

#### Testing
Using the test.py to load the models trained for this task, e.g.

```bash
python test.py --workers 16 \ # Number of worker for dataloader
			   --net_type resnet \ # Net type (resnet, alexnet, refined_resnet)
			   --src resnet_cutmix.pt \ # .pt file that store the parameters of the net
```

Using --help for more informations.
You could download the trained model from my [BaiduNetDisk](https://pan.baidu.com/s/1d---q__eczFNsB2Mq3wx7w) with code 9pvi

You could reach the other parts with the [link](https://github.com/403forbiddennn/DATA130051-Computer-Vision/blob/main/cv-midterm.md)
