"""
training a classifier to approximate the density ratio of males : females
here we ask it to classify whether a given sample is from the balanced dataset (y=1) or from the unbalanced dataset (y=0)

we equalize the proportion of balanced/unbalanced within a minibatch.

NOTE: once we run this function, we automatically save all the balanced/unbalanced datasets we will be using for our generative model
"""
import os
import sys
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from clf_models import BasicBlock, build_model
from utils import save_checkpoint
from dataset_splits import (
	BagOfDatasets,
	build_90_10_unbalanced_datasets_clf_celeba,
	build_80_20_unbalanced_datasets_clf_celeba,
	build_multi_datasets_clf_celeba,
)

from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset_name', type=str, help='celeba')
	parser.add_argument('model_name', type=str, help='celeba')
	parser.add_argument('--batch-size', type=int, default=64,
	                    help='minibatch size [default: 64]')
	parser.add_argument('--ckpt-path', type=str, default=None, 
											help='if test=True, path to clf checkpoint')
	parser.add_argument('--lr', type=float, default=1e-4,
	                    help='learning rate [default: 1e-4]')
	parser.add_argument('--epochs', type=int, default=10,
	                    help='number of epochs [default: 10]')
	parser.add_argument('--log-interval', type=int, default=100,
	                    help='interval for printing [default: 100]')
	parser.add_argument('--perc', type=float, default=1.0,
	                    help='size of balanced dataset [0.1, 0.25, 0.5, 1.0]')
	parser.add_argument('--cuda', action='store_true', default=True,
	                    help='enables CUDA training')
	parser.add_argument('--test', action='store_true', default=False,
	                    help='if True, tests the performance of a given classifier')
	parser.add_argument('--bias', type=str, default='90_10',
	                    help='type of bias [90_10, 80_20, multi]')
	args = parser.parse_args()
	args.cuda = args.cuda and torch.cuda.is_available()

	# reproducibility
	torch.manual_seed(777)
	np.random.seed(777)

	device = torch.device('cuda' if args.cuda else 'cpu')

	# grab appropriate dataset splits
	assert args.perc in [0.1, 0.25, 0.5, 1.0]
	if args.bias == '90_10':
		balanced_train_dataset, unbalanced_train_dataset = build_90_10_unbalanced_datasets_clf_celeba(
			args.dataset_name, 'train', args.perc)
		balanced_valid_dataset, unbalanced_valid_dataset = build_90_10_unbalanced_datasets_clf_celeba(
			args.dataset_name, 'val', args.perc)
		bias = '90_10_perc{}'.format(args.perc)
	elif args.bias == '80_20':
		balanced_train_dataset, unbalanced_train_dataset = build_80_20_unbalanced_datasets_clf_celeba(
			args.dataset_name, 'train', args.perc)
		balanced_valid_dataset, unbalanced_valid_dataset = build_80_20_unbalanced_datasets_clf_celeba(
			args.dataset_name, 'val', args.perc)
		bias = '80_20_perc{}'.format(args.perc)
	elif args.bias == 'multi':
		balanced_train_dataset, unbalanced_train_dataset = build_multi_datasets_clf_celeba(
			args.dataset_name, 'train', args.perc)
		balanced_valid_dataset, unbalanced_valid_dataset = build_multi_datasets_clf_celeba(
			args.dataset_name, 'val', args.perc)
		bias = 'multi_perc{}'.format(args.perc)
	else:
		raise NotImplementedError

	# save outputs in correct directory
	args.out_dir = '../data/celeba_{}'.format(bias)
	print('outputs will be saved to: {}'.format(args.out_dir))
	if not os.path.isdir(args.out_dir):
	    os.makedirs(args.out_dir)

	# for training the classifier
	train_dataset = BagOfDatasets([balanced_train_dataset, unbalanced_train_dataset])
	train_loader = torch.utils.data.DataLoader(
	    train_dataset, batch_size=100, shuffle=True)

	# adjust size of unbalanced validation set to check calibration
	if args.perc != 1.0:
		print('shrinking the size of the unbalanced dataset to assess classifier calibration!')
		to_shrink = len(balanced_valid_dataset)
		# shrink validation set according to the right proportions!
		d, g, l = unbalanced_valid_dataset.dataset.tensors
		if '90' in args.bias or '80' in args.bias:
			females = torch.where(g==0)[0]
			males = torch.where(g==1)[0]
			if '90' in args.bias:
				f_idx = int(to_shrink * 0.9)
				m_idx = int(to_shrink * 0.1)
			else:
				f_idx = int(to_shrink * 0.8)
				m_idx = int(to_shrink * 0.2)
			f_idx = females[0:f_idx]
			m_idx = males[0:m_idx]
			print('females: {}'.format(len(f_idx)/to_shrink))
			print('males: {}'.format(len(m_idx)/to_shrink))
			d = torch.cat([d[f_idx], d[m_idx]])
			l = torch.cat([l[f_idx], l[m_idx]])
			g = torch.cat([g[f_idx], g[m_idx]])
		else:  # multi-attribute
			print('balancing for multi-attribute')
			a = torch.where(g==0)[0]
			b = torch.where(g==1)[0]
			c = torch.where(g==2)[0]
			e = torch.where(g==3)[0]
			
			# get indices (NOTE: these are true proportions used in paper)
			a_idx = int(to_shrink * 0.436)
			b_idx = int(to_shrink * 0.415)
			c_idx = int(to_shrink * 0.064)
			e_idx = int(to_shrink * 0.085)
			
			# get all indices
			a_idx = a[0:a_idx]
			b_idx = b[0:b_idx]
			c_idx = c[0:c_idx]
			e_idx = e[0:e_idx]
			# print stats
			print('00: {}'.format(len(a_idx)/to_shrink))
			print('01: {}'.format(len(b_idx)/to_shrink))
			print('10: {}'.format(len(c_idx)/to_shrink))
			print('11: {}'.format(len(e_idx)/to_shrink))
			# aggregate all data
			d = torch.cat([d[a_idx], d[b_idx], d[c_idx], d[e_idx]])
			l = torch.cat([l[a_idx], l[b_idx], l[c_idx], l[e_idx]])
			g = torch.cat([g[a_idx], g[b_idx], g[c_idx], g[e_idx]])

	# balance validation set size for proper calibration assessment
	d, g, l = unbalanced_valid_dataset.dataset.tensors
	adj_unbalanced_valid_dataset = torch.utils.data.TensorDataset(d,g,l)
	valid_dataset = torch.utils.data.ConcatDataset([balanced_valid_dataset, adj_unbalanced_valid_dataset])
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=100, shuffle=False)

	# build model and optimizer
	model_cls = build_model(args.model_name)
	model = model_cls(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=2, grayscale=False)  # just for attr
	model = model.to(device)
	if args.test:
		print('Skipping training; loading best classifier...')
		ckpt = torch.load(os.path.join(args.out_dir, 'model_best.pth.tar'))
		model.load_state_dict(ckpt['state_dict'])
	optimizer = optim.Adam(model.parameters(), lr=args.lr)


	def train(epoch):
		model.train()
		correct = 0.
		num_examples = 0.
		preds = []
		true_labels = []
		probs = []

		for batch_idx, data_list in enumerate(train_loader):
			# concatenate data and labels from both balanced + unbalanced, and make sure that each minibatch is balanced
			n_unbalanced = len(data_list[0][1])
			data = torch.cat(
				(data_list[0][0][0:n_unbalanced], data_list[0][1])).to(device)
			attr = torch.cat(
				(data_list[1][0][0:n_unbalanced], data_list[1][1])).to(device)
			target = torch.cat(
				(data_list[2][0][0:n_unbalanced], data_list[2][1])).to(device)

			# random permutation of data
			idx = torch.randperm(len(data))
			data = data[idx]
			target = target[idx]
			attr = attr[idx]

			# minor adjustments
			data = data.float() / 255.
			target = target.long()

			# NOTE: here, balanced (y=1) and unbalanced (y=0)
			logits, probas = model(data)
			loss = F.cross_entropy(logits, target)
			_, pred = torch.max(probas, 1)	   

			# check performance
			num_examples += target.size(0)
			correct += (pred == target).sum()
			preds.append(pred)
			probs.append(probas)
			true_labels.append(target)

			if not args.test:
				# backprop
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			# log performance
			if batch_idx % args.log_interval == 0:
			    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			        epoch, batch_idx * len(data), len(train_loader.dataset),
			        100. * batch_idx / len(train_loader), loss.item()))
		# aggregate results
		train_acc = float(correct)/num_examples
		preds = torch.cat(preds)
		true_labels = torch.cat(true_labels)
		preds = np.ravel(preds.data.cpu().numpy())
		
		return train_acc, preds


	def test(epoch, loader):
		model.eval()
		test_loss = 0.
		correct = 0.
		num_examples = 0.
		num_pos_correct = 0.
		num_neg_correct = 0.

		num_pos_examples = 0.
		num_neg_examples = 0.

		preds = []
		targets = []

		with torch.no_grad():
			for data, attr, target in loader:
				data, attr, target = data.to(device), attr.to(device), target.to(device)
				
				# i also need to modify the data a bit here
				data = data.float() / 255.
				target = target.long()

				logits, probas = model(data)
				test_loss += F.cross_entropy(logits, target, reduction='sum').item() # sum up batch loss
				_, pred = torch.max(probas, 1)
				num_examples += target.size(0)

				# split correctness by pos/neg examples
				num_pos_examples += target.sum()
				num_neg_examples += (target.size(0) - target.sum())

				# correct should also be split
				num_pos_correct += (pred[target==1] == target[target==1]).sum()
				num_neg_correct += (pred[target==0] == target[target==0]).sum()

				preds.append(pred)
				targets.append(target)
			preds = torch.cat(preds)
			targets = torch.cat(targets)
			preds = np.ravel(preds.data.cpu().numpy())
			targets = np.ravel(targets.data.cpu().numpy())

		test_loss /= num_examples
		# test_acc = float(correct) / num_examples

		# average over weighted proportions of positive/negative examples
		test_acc = ((num_pos_correct.float()/num_pos_examples) + (num_neg_correct.float()/num_neg_examples)) * 0.5

		print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))

		return test_loss, test_acc, preds

	def run_loop():
		ratios = []
		labels = []
		probs = []

		model.eval()

		with torch.no_grad():
			# iterate through entire dataset
		    for data, attr, target in valid_loader:
		        data = data.float() / 255.
		        data, target = data.to(device), target.to(device).long()
		        logits, probas = model(data)
		        probs.append(probas)
		        
		        # save data, density ratios, and labels
		        labels.append(target)
		    labels = torch.cat(labels)
		    probs = torch.cat(probs)
		return labels, probs

	# if not training, skip to calibration
	if args.test:
		# check accuracies on validation set
		valid_loss, valid_acc, valid_preds = test(0, valid_loader)

		print('valid loss: {}, valid acc: {}'.format(valid_loss, valid_acc))

		valid_labels, valid_probs = run_loop()
		y_valid = valid_labels.data.cpu().numpy()
		valid_prob_pos = valid_probs.data.cpu().numpy()

		print('running through different bins:')
		for bins in [5, 6, 7, 8, 9, 10]:
			fraction_of_positives, mean_predicted_value = calibration_curve(y_valid, valid_prob_pos[:, 1], n_bins=bins)

			# save calibration results
			np.save(os.path.join(args.out_dir, 'fraction_of_positives'), fraction_of_positives)
			np.save(os.path.join(args.out_dir, 'mean_predicted_value.npy'), mean_predicted_value)

			# obtain figure
			plt.figure(figsize=(10,5))
			plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='dset_clf')
			plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

			plt.title('Validation Set: Calibration Curve',fontsize=22)
			plt.ylabel('Fraction of positives',fontsize=22)
			plt.tick_params(axis='both', which='major', labelsize=20)
			plt.tick_params(axis='both', which='minor', labelsize=20)
			plt.legend()
			plt.savefig(os.path.join(args.out_dir, 'calibration_curve_{}bins.png'.format(bins)), dpi=300)
		print('Completed calibration eval, ending program.')
		sys.exit(0)

	# if not testing, train the model
	best_loss = -np.inf
	valid_accs_db = np.zeros(args.epochs)
	train_accs_db = np.zeros(args.epochs)
	preds_db = np.zeros((args.epochs, len(valid_dataset)))
	train_preds_db = np.zeros((args.epochs, len(train_dataset)*2))

	print('beginning training...')
	for epoch in range(1, args.epochs + 1):
		train_acc, train_preds = train(epoch)
		valid_loss, valid_acc, valid_preds = test(epoch, valid_loader)
		train_accs_db[epoch-1] = train_acc
		valid_accs_db[epoch-1] = valid_acc
		preds_db[epoch-1] = valid_preds
		train_preds_db[epoch-1] = train_preds

		# model checkpointing
		is_best = valid_acc > best_loss
		best_loss = max(valid_acc, best_loss)
		print('epoch {}: is_best: {}'.format(epoch, is_best))
		if is_best:
			best_state_dict = model.state_dict()
			best_epoch = epoch
		save_checkpoint({
		    'state_dict': model.state_dict(),
		    'optimizer_state_dict' : optimizer.state_dict(),
		    'cmd_line_args': args,
		}, is_best, epoch, folder=args.out_dir)

		# save accuracies at validation time
		# np.save(os.path.join(args.out_dir, 'valid_accs.npy'), valid_accs_db)
		# np.save(os.path.join(args.out_dir, 'train_accs.npy'), train_accs_db)
		# np.save(os.path.join(args.out_dir, 'train_preds.npy'), train_preds_db)
		# np.save(os.path.join(args.out_dir, 'valid_preds.npy'), preds_db)

	# EXTRACT BEST CLASSIFIER AND LOAD MODEL
	print('best model finished training at epoch {}: {}, loading checkpoint'.format(best_epoch, best_loss))
	model_cls = build_model(args.model_name)
	model = model_cls(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=2, grayscale=False)  # just for attr
	model = model.to(device)
	best_state_dict = torch.load(os.path.join(args.out_dir, 'model_best.pth.tar'))['state_dict']
	model.load_state_dict(best_state_dict)

	# STEP 2: assess calibration
	valid_labels, valid_probs = run_loop()
	y_valid = valid_labels.data.cpu().numpy()
	valid_prob_pos = valid_probs.data.cpu().numpy()

	print('running through different bins:')
	for bins in [3, 4, 5, 6, 7, 8, 9, 10]:
		fraction_of_positives, mean_predicted_value = calibration_curve(y_valid, valid_prob_pos[:, 1], n_bins=bins)

		# save calibration results
		np.save(os.path.join(args.out_dir, 'fraction_of_positives'), fraction_of_positives)
		np.save(os.path.join(args.out_dir, 'mean_predicted_value.npy'), mean_predicted_value)

		# obtain figure
		plt.figure(figsize=(10,5))
		plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='dset_clf')
		plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

		plt.title('Validation Set: Calibration Curve',fontsize=22)
		plt.ylabel('Fraction of positives',fontsize=22)
		plt.tick_params(axis='both', which='major', labelsize=20)
		plt.tick_params(axis='both', which='minor', labelsize=20)
		plt.legend()
		plt.savefig(os.path.join(args.out_dir, 'calibration_curve_{}bins.png'.format(bins)), dpi=300)

	# classifier has finished training, evaluate sample diversity
	# run through unbalanced dataset and save density ratios
	print('saving unbalanced dataset with density ratios...')
	# save values
	unbalanced_train_loader = torch.utils.data.DataLoader(unbalanced_train_dataset.dataset, batch_size=100, shuffle=False)
	train_ratios = []
	train_data = []
	train_labels = []
	attrs = []

	# MAKE SURE YOU TURN BATCHNORM OFF!
	model.eval()

	with torch.no_grad():
		# only iterating through unbalanced dataset!
		for data, attr, target in unbalanced_train_loader:
			data, attr, target = data.to(device), attr.to(device), target.to(device)
			data = data.float() / 255.
			attr, target = attr.long(), target.long()
			logits, probas = model(data)
			density_ratio = probas[:,1]/probas[:,0]

			# save data, density ratios, and labels
			train_data.append(data)
			train_ratios.append(density_ratio)
			train_labels.append(target)
			attrs.append(attr)
		train_ratios = torch.cat(train_ratios)
		train_data = torch.cat(train_data)
		train_labels = torch.cat(train_labels)
		attrs = torch.cat(attrs)
	train_data = (train_data * 255).to(torch.uint8)

	# save files
	torch.save(train_ratios.data.cpu(), os.path.join(args.out_dir, 'celeba_unbalanced_train_density_ratios.pt'))
	# NOTE: we are returning the true attr labels so that we can look at the density ratios across classes later
	torch.save(attrs.data.cpu(), os.path.join(args.out_dir, 'celeba_unbalanced_train_attr_labels.pt'))
	torch.save(train_data.data.cpu(), os.path.join(args.out_dir, 'celeba_unbalanced_train_data.pt'))
	torch.save(train_labels.data.cpu(), os.path.join(args.out_dir, 'celeba_unbalanced_train_labels.pt'))

	# save balanced dataset
	balanced_train_loader = torch.utils.data.DataLoader(balanced_train_dataset.dataset, batch_size=100, shuffle=False)
	train_data = []
	train_labels = []

	# MAKE SURE YOU TURN BATCHNORM OFF!
	model.eval()

	# save density ratios and labels
	with torch.no_grad():
	    # only iterating through unbalanced dataset!
	    for data,attr,target in balanced_train_loader:
	        data,target = data.to(device), target.to(device)
	        data = data.float() / 255.
	        data,target = data.to(device), target.to(device).long()
	        
	        # save data, density ratios, and labels
	        train_data.append(data)
	        train_labels.append(target)
	    train_data = torch.cat(train_data)
	    train_labels = torch.cat(train_labels)
	train_data = (train_data * 255).to(torch.uint8)
	torch.save(train_data.data.cpu(), os.path.join(args.out_dir, 'celeba_balanced_train_data.pt'))
	torch.save(train_labels.data.cpu(), os.path.join(args.out_dir,'celeba_balanced_train_labels.pt'))
