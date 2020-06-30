import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from clf_models import BasicBlock, build_model
from utils import save_checkpoint
from dataset_splits import (
	build_celeba_classification_dataset,
	build_multi_celeba_classification_datset,
)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('model_name', type=str, help='celeba')
	parser.add_argument('out_dir', type=str, help='where to save outputs')
	parser.add_argument('--ckpt-path', type=str, default=None, 
						help='if test=True, path to clf checkpoint')
	parser.add_argument('--batch-size', type=int, default=64,
	                    help='minibatch size [default: 64]')
	parser.add_argument('--lr', type=float, default=0.001,
	                    help='learning rate [default: 0.001]')
	parser.add_argument('--epochs', type=int, default=10,
	                    help='number of epochs [default: 10]')
	parser.add_argument('--log_interval', type=int, default=10,
	                    help='number of steps to log after during training')
	parser.add_argument('--class_idx', type=int, default=20,
						help='CelebA class label for training.')
	parser.add_argument('--multi', type=bool, default=False, 
						help='If True, runs multi-attribute classifier')
	parser.add_argument('--cuda', action='store_true', default=True,
	                    help='enables CUDA training')
	args = parser.parse_args()
	args.cuda = args.cuda and torch.cuda.is_available()

	# reproducibility
	torch.manual_seed(777)
	np.random.seed(777)

	device = torch.device('cuda' if args.cuda else 'cpu')

	if not os.path.isdir(args.out_dir):
	    os.makedirs(args.out_dir)

	# get data: idx 20 = male, idx 8 = black hair
	if not args.multi:
		train_dataset = build_celeba_classification_dataset(
			'train', args.class_idx)
		valid_dataset = build_celeba_classification_dataset(
			'val', args.class_idx)
		test_dataset = build_celeba_classification_dataset(
			'test', args.class_idx)
		n_classes = 2
	else:
		train_dataset = build_multi_celeba_classification_datset('train')
		valid_dataset = build_multi_celeba_classification_datset('val')
		test_dataset = build_multi_celeba_classification_datset('test')
		n_classes = 4


	print(len(train_dataset))
	print(len(valid_dataset))

	# train/validation split
	train_loader = torch.utils.data.DataLoader(
	    train_dataset, batch_size=args.batch_size, shuffle=True)
	valid_loader = torch.utils.data.DataLoader(
	    valid_dataset, batch_size=args.batch_size, shuffle=False)
	test_loader = torch.utils.data.DataLoader(
		test_dataset, batch_size=100, shuffle=False)

	# build model
	model_cls = build_model(args.model_name)
	model = model_cls(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=n_classes, grayscale=False)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	model.to(device)


	def train(epoch):
	    model.train()
	    for batch_idx, (data, target) in enumerate(train_loader):
	        data, target = data.to(device), target.to(device)
	        data = data.float() / 255.
	        target = target.long()
	        
	        # NOTE: here, female (y=0) and male (y=1)
	        logits, probas = model(data)
	        loss = F.cross_entropy(logits, target)	        
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	        if batch_idx % args.log_interval == 0:
	            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
	                epoch, batch_idx * len(data), len(train_loader.dataset),
	                100. * batch_idx / len(train_loader), loss.item()))


	def test(epoch, loader):
		model.eval()
		test_loss = 0
		correct = 0
		num_examples = 0
		with torch.no_grad():
			for data, target in loader:
				data, target = data.to(device), target.to(device)
				data = data.float() / 255.
				target = target.long()

				# run through model
				logits, probas = model(data)
				test_loss += F.cross_entropy(logits, target, reduction='sum').item() # sum up batch loss
				_, pred = torch.max(probas, 1)
				num_examples += target.size(0)
				correct += (pred == target).sum()

		test_loss /= num_examples

		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		    test_loss, correct, num_examples,
		    100. * correct / num_examples))
		return test_loss

	# classifier has finished training, evaluate sample diversity
	best_loss = sys.maxsize

	print('beginning training...')
	for epoch in range(1, args.epochs + 1):
		train(epoch)
		valid_loss = test(epoch, valid_loader)

		is_best = valid_loss < best_loss
		best_loss = min(valid_loss, best_loss)
		state_dict = model.state_dict()
		if is_best:
			print('saving checkpoint at epoch {}'.format(epoch))
			save_checkpoint({
				'state_dict': model.state_dict(),
				'optimizer_state_dict' : optimizer.state_dict(),
				'cmd_line_args': args,
			}, is_best, epoch, folder=args.out_dir)
			best_idx = epoch
			best_state = model.state_dict()

	# finished training, want to test on final test set
	print('finished training...testing on final test set with epoch {} ckpt'.format(best_idx))
	# reload best model
	model_cls = build_model('celeba')
	model = model_cls(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=n_classes, grayscale=False)
	model = model.to(device)
	model.load_state_dict(best_state)

	# get test
	test_loss = test(epoch, test_loader)