"""
multi-attribute labeling scheme:

[0. 0.] 0 (black hair = 0, male = 0)
[0. 1.] 1 (black hair = 0, male = 1)
[1. 0.] 2 (black hair = 1, male = 0)
[1. 1.] 3 (black hair = 1, male = 1)
"""
import os
import math 
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import urllib
import gzip
import pickle


# BASE_PATH = '/atlas/u/kechoi/fair_generative_modeling/data/'
BASE_PATH = './data/'


class BagOfDatasets(Dataset):
    """Wrapper class over several dataset classes. from @mhw32
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.n = len(datasets)

    def __len__(self):
        lengths = [len(dataset) for dataset in self.datasets]
        return max(lengths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image1, image2, ...)
        """
        items = []
        attrs = []
        labels = []
        for dataset in self.datasets:
            item = dataset.__getitem__(index)
            if isinstance(item, tuple):
                data = item[0]
                attr = item[1]  # true female/male label
                label = item[2]  # fake data balanced/unbalanced label
            items.append(data)
            labels.append(label)
            attrs.append(attr)

        return items, attrs, labels


class LoopingDataset(Dataset):
    """
    Dataset class to handle indices going out of bounds when training
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index >= len(self.dataset):
            index = np.random.choice(len(self.dataset))
        item, attr, label = self.dataset.__getitem__(index)
        return item, attr, label


def build_celeba_classification_dataset(split, class_idx, class_idx2=None):
	"""
	Returns a dataset used for classification for given class label <class_idx>. If class_idx2 is not None, returns both labels (this is typically used for downstream tasks)
	
	Args:
	    split (str): one of [train, val, test]
	    class_idx (int): class label for protected attribute
	    class_idx2 (None, optional): additional class for downstream tasks
	
	Returns:
	    TensorDataset for training attribute classifier
	"""
	data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
	labels = torch.load(BASE_PATH + '{}_labels_celeba_64x64.pt'.format(split))
	labels1 = labels[:, class_idx]

	# return appropriate split
	if class_idx2:
	    labels2 = labels[:, class_idx2]
	    dataset = torch.utils.data.TensorDataset(data, labels1, labels2)
	    return dataset
	else:
	    dataset = torch.utils.data.TensorDataset(data, labels1)
	    return dataset


def build_multi_celeba_classification_datset(split):
	"""
	Loads data for multi-attribute classification
	
	Args:
	    split (str): one of [train, val, test] 
	
	Returns:
	    TensorDataset for training attribute classifier
	"""
	data = torch.load(
		BASE_PATH + '{}_celeba_64x64.pt'.format(split))
	print('returning labels for (black hair, gender) multi-attribute')
	labels = torch.load(
		BASE_PATH + '{}_multi_labels_celeba_64x64.pt'.format(split))
	dataset = torch.utils.data.TensorDataset(data, labels)
	return dataset


def build_90_10_unbalanced_datasets_clf_celeba(dataset_name, split, perc=1.0):
	"""
	Builds (90-10) and (50-50) biased/unbiased dataset splits.
	
	Args:
	    dataset_name (str): celeba
	    split (str): one of [train, val, test] 
	    perc (float, optional): [0.1, 0.25, 0.5, 1.0], size of unbiased dataset relative to biased dataset
	
	Returns:
	    LoopingDataset with (data, true_gender_label, balanced/unbalanced label)
	"""
	assert dataset_name == 'celeba'

	data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
	labels = torch.load(BASE_PATH + '{}_labels_celeba_64x64.pt'.format(split))

	# get appropriate gender splits
	females = np.where(labels[:, 20]==0)[0]
	males = np.where(labels[:, 20]==1)[0]

	# this is 90-10
	if split == 'train':
	    # need 40504 males total, currently have 94509 females in train
	    total_examples = 135012
	    males = males[0:40504]
	elif split == 'val':
	    # need 4889 males total, currently have 11409 females in val
	    total_examples = 16298
	    males = males[0:4889]

	# construct unbiased dataset
	n_balanced = (total_examples // 2)
	if perc < 1.0:
	    print('cutting down balanced dataset to {} its original size'.format(perc))
	to_stop = int((n_balanced // 2) * perc)
	balanced_indices = np.hstack((males[0:to_stop], females[0:to_stop]))
	balanced_dataset = data[balanced_indices]
	balanced_labels = labels[balanced_indices][:,20]
	print('balanced dataset ratio: {}'.format(np.unique(balanced_labels.numpy(), return_counts=True)))

	# construct biased dataset
	unbalanced_indices = np.hstack((females[(n_balanced//2):], males[(n_balanced//2):]))
	unbalanced_dataset = data[unbalanced_indices]
	unbalanced_labels = labels[unbalanced_indices][:, 20]
	print('unbalanced dataset ratio: {}'.format(np.unique(unbalanced_labels.numpy(), return_counts=True)))

	print('converting labels from gender to balanced/unbalanced...')
	data_balanced_labels = torch.ones_like(balanced_labels)  # y = 1 for balanced
	data_unbalanced_labels = torch.zeros_like(unbalanced_labels)  # y = 0 for unbalanced

	# construct dataloaders
	balanced_train_dataset = torch.utils.data.TensorDataset(
		balanced_dataset, balanced_labels, data_balanced_labels)
	unbalanced_train_dataset = torch.utils.data.TensorDataset(unbalanced_dataset, unbalanced_labels, data_unbalanced_labels)
	
	# make sure things don't go out of bounds during trainin
	balanced_train_dataset = LoopingDataset(balanced_train_dataset)
	unbalanced_train_dataset = LoopingDataset(unbalanced_train_dataset)

	return balanced_train_dataset, unbalanced_train_dataset


def build_80_20_unbalanced_datasets_clf_celeba(dataset_name, split, idx=20, perc=1.0):
	"""
	Builds (80-20) and (50-50) biased/unbiased dataset splits.
	
	Args:
	    dataset_name (str): celeba
	    split (str): one of [train, val, test] 
	    perc (float, optional): [0.1, 0.25, 0.5, 1.0], size of unbiased dataset relative to biased dataset
	
	Returns:
	    LoopingDataset with (data, true_gender_label, balanced/unbalanced label)
	"""
	assert dataset_name == 'celeba'

	data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
	labels = torch.load(BASE_PATH + '{}_labels_celeba_64x64.pt'.format(split))

	females = np.where(labels[:, 20]==0)[0]
	males = np.where(labels[:, 20]==1)[0]

	# this is 90-10
	if split == 'train':
	    # need 40504 males total, currently have 94509 females in train 
	    total_examples = 135012
	    males = males[0:40504]
	elif split == 'val':
	    # need 4889 males total, currently have 11409 females in val
	    total_examples = 16298
	    males = males[0:4889]

	n_balanced = (total_examples // 2)
	if perc < 1.0:
	    print('cutting down balanced dataset to {} its original size'.format(perc))
	to_stop = int((n_balanced // 2) * perc)
	balanced_indices = np.hstack((males[0:to_stop], females[0:to_stop]))
	balanced_dataset = data[balanced_indices]
	balanced_labels = labels[balanced_indices][:, 20]
	print('balanced dataset ratio: {}'.format(
		np.unique(balanced_labels.numpy(), return_counts=True)))

	if split == 'train':
	    # adjust proportions of unbalanced_indices
	    new_females = females[(n_balanced//2):-6750]
	    # additional_males
	    additional_males = np.where(labels[:,20]==1)[0][40504:]
	    new_males = np.hstack(
	    	(males[(n_balanced//2):], additional_males[0:6750]))
	elif split == 'val':
	    # adjust proportions of unbalanced_indices
	    new_females = females[(n_balanced//2):-815]
	    additional_males = np.where(labels[:,20]==1)[0][4889:]
	    new_males = np.hstack(
	    	(males[(n_balanced//2):], additional_males[0:815]))

	unbalanced_indices = np.hstack((new_females, new_males))
	unbalanced_dataset = data[unbalanced_indices]
	unbalanced_labels = labels[unbalanced_indices][:,20]
	print('unbalanced dataset ratio: {}'.format(np.unique(unbalanced_labels.numpy(), return_counts=True)))

	print('converting labels from gender to balanced/unbalanced...')
	data_balanced_labels = torch.ones_like(balanced_labels)  # y = 1 for balanced
	data_unbalanced_labels = torch.zeros_like(unbalanced_labels)  # y = 0 for unbalanced

	# construct dataloaders
	balanced_train_dataset = torch.utils.data.TensorDataset(
		balanced_dataset, balanced_labels, data_balanced_labels)
	unbalanced_train_dataset = torch.utils.data.TensorDataset(unbalanced_dataset, unbalanced_labels, data_unbalanced_labels)

	# make sure things don't go out of bounds during training
	balanced_train_dataset = LoopingDataset(balanced_train_dataset)
	unbalanced_train_dataset = LoopingDataset(unbalanced_train_dataset)

	return balanced_train_dataset, unbalanced_train_dataset


def build_multi_datasets_clf_celeba(dataset_name, split, perc=1.0):
	"""
	Constructs a multi-attribute dataset that splits by black hair and gender
	
	Args:
	    dataset_name (str): celeba 
	    split (str): one of [train, val, test] 
	    perc (float, optional): [0.1, 0.25, 0.5, 1.0], size of unbiased dataset relative to biased dataset
	
	Returns:
	    LoopingDataset with (data, true_gender_label, balanced/unbalanced label)
	"""
	assert dataset_name == 'celeba'

	data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
	multi_labels = torch.load(BASE_PATH + '{}_multi_labels_celeba_64x64.pt'.format(split))

	# obtain proper number of examples
	if split == 'train':
	    total_examples = 120000
	elif split == 'val':
	    # need 4889 males total, currently have 11409 females in total
	    total_examples = 10520

	n_balanced = (total_examples // 2)

	if split == 'train':
	    f_other_hair = np.where(multi_labels==0)[0][0:((n_balanced // 4) + 26216)]  # (0, 0), 41216
	    m_other_hair = np.where(multi_labels==1)[0][0:((n_balanced // 4) + 24878)]  # (0, 1), 48139
	    f_black_hair = np.where(multi_labels==2)[0]  # (1, 0), 18784
	    m_black_hair = np.where(multi_labels==3)[0]  # (1, 1), 20122
	elif split == 'val':
	    f_other_hair = np.where(multi_labels==0)[0][0:((n_balanced // 2) + 983)]   # (0,0), 9752
	    m_other_hair = np.where(multi_labels==1)[0][0:((n_balanced // 2) + 866)]   # (0,1), 5971
	    f_black_hair = np.where(multi_labels==2)[0][0:((n_balanced // 4) + 331)]  # (1, 0), 1657
	    m_black_hair = np.where(multi_labels==3)[0][0:((n_balanced // 4) + 449)]  # (1,1), 2487

	# construct balanced dataset
	if perc < 1.0:
	    print('cutting down balanced dataset to {} its original size'.format(perc))
	to_stop = int((n_balanced // 4) * perc)  # 4 categories
	balanced_indices = np.hstack(
		(f_black_hair[0:to_stop], f_other_hair[0:to_stop], 
		m_black_hair[0:to_stop], m_other_hair[0:to_stop]))
	balanced_dataset = data[balanced_indices]
	balanced_labels = multi_labels[balanced_indices]
	print('balanced dataset ratio: {}'.format(np.unique(balanced_labels.numpy(), return_counts=True)))

	unbalanced_indices = np.hstack((f_black_hair[(n_balanced // 4):], f_other_hair[(n_balanced // 4):], m_black_hair[(n_balanced // 4):], m_other_hair[(n_balanced // 4):]))
	unbalanced_dataset = data[unbalanced_indices]
	unbalanced_labels = multi_labels[unbalanced_indices]
	print('unbalanced dataset ratio: {}'.format(np.unique(unbalanced_labels.numpy(), return_counts=True)))

	print('converting labels from gender to balanced/unbalanced...')
	data_balanced_labels = torch.ones_like(balanced_labels)  # y = 1 for balanced
	data_unbalanced_labels = torch.zeros_like(unbalanced_labels)  # y = 0 for unbalanced

	# construct dataloaders
	balanced_train_dataset = torch.utils.data.TensorDataset(balanced_dataset, balanced_labels, data_balanced_labels)
	unbalanced_train_dataset = torch.utils.data.TensorDataset(unbalanced_dataset, unbalanced_labels, data_unbalanced_labels)

	# make sure things don't go out of bounds during training
	balanced_train_dataset = LoopingDataset(balanced_train_dataset)
	unbalanced_train_dataset = LoopingDataset(unbalanced_train_dataset)

	return balanced_train_dataset, unbalanced_train_dataset