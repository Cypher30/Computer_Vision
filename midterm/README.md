## Image Classification on CIFAR data set
The image classification part of computer vision midterm checkpoint.
Using the main.py to train the models for image classification on CIFAR10 or CIFAR100 data set, e.g.

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

You could reach the other parts with the [link](https://github.com/403forbiddennn/DATA130051-Computer-Vision/blob/main/cv-midterm.md)
