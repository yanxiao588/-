

import torch
from torchvision import datasets, transforms
# 根据给定的数据集名称加载对应的训练数据集和评估数据集
def get_dataset(dir, name):

    # 加载MNIST数据集
	if name=='mnist':
		train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
		eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())

	# 加载CIFAR-10数据集
	elif name=='cifar':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		train_dataset = datasets.CIFAR10(dir, train=True, download=True,
										transform=transform_train)
		eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
		
	
	return train_dataset, eval_dataset