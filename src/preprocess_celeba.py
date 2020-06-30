# preprocessing code for CelebA dataset adapted from @ruishu and @mhw32
import os
import torch
import tqdm
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import torchvision
import torchvision.transforms as transforms


VALID_PARTITIONS = {'train': 0, 'val': 1, 'test': 2}
ATTR_TO_IX_DICT = {'Sideburns': 30, 'Black_Hair': 8, 'Wavy_Hair': 33, 
                    'Young': 39, 'Heavy_Makeup': 18,
                   'Blond_Hair': 9, 'Attractive': 2, '5_o_Clock_Shadow': 0, 
                   'Wearing_Necktie': 38,
                   'Blurry': 10, 'Double_Chin': 14, 'Brown_Hair': 11, 
                   'Mouth_Slightly_Open': 21,
                   'Goatee': 16, 'Bald': 4, 'Pointy_Nose': 27, 'Gray_Hair': 17,
                   'Pale_Skin': 26,
                   'Arched_Eyebrows': 1, 'Wearing_Hat': 35, 
                   'Receding_Hairline': 28, 'Straight_Hair': 32,
                   'Big_Nose': 7, 'Rosy_Cheeks': 29, 'Oval_Face': 25, 
                   'Bangs': 5, 'Male': 20, 'Mustache': 22,
                   'High_Cheekbones': 19, 'No_Beard': 24, 'Eyeglasses': 15, 
                   'Bags_Under_Eyes': 3,
                   'Wearing_Necklace': 37, 'Wearing_Lipstick': 36, 
                   'Big_Lips': 6, 'Narrow_Eyes': 23,
                   'Chubby': 13, 'Smiling': 31, 
                   'Bushy_Eyebrows': 12, 'Wearing_Earrings': 34}

# NOTE: we use all the attributes...
IX_TO_ATTR_DICT = {v:k for k,v in ATTR_TO_IX_DICT.items()}
N_ATTRS = len(ATTR_TO_IX_DICT)
N_IMAGES = 202599
IMG_SIZE = 64
ATTR_PATH = 'attributes.pt'


def preprocess_images(args):
    # automatically save outputs to "data" directory
    IMG_PATH = os.path.join(args.out_dir, '{1}_celeba_{0}x{0}.pt'.format(
        IMG_SIZE, args.partition))
    LABEL_PATH = os.path.join(args.out_dir, '{1}_labels_celeba_{0}x{0}.pt'.format(IMG_SIZE, args.partition))

    print('preprocessing partition {}'.format(args.partition))
    # NOTE: datasets have not yet been normalized to lie in [-1, +1]!
    transform = transforms.Compose(
        [transforms.CenterCrop(140),
        transforms.Resize(IMG_SIZE)])
    eval_data = load_eval_partition(args.partition, args.data_dir)
    attr_data = load_attributes(eval_data, args.partition, args.data_dir)

    if os.path.exists(IMG_PATH):
        print("{} already exists.".format(IMG_PATH))
        return

    N_IMAGES = len(eval_data)
    data = np.zeros((N_IMAGES, 3, IMG_SIZE, IMG_SIZE), dtype='uint8')
    labels = np.zeros((N_IMAGES, 40))

    print('starting conversion...')
    for i in tqdm.tqdm(range(N_IMAGES)):
        os.path.join(
            args.data_dir, 'img_align_celeba/', '{}'.format(eval_data[i]))
        with Image.open(os.path.join(args.data_dir, 'img_align_celeba/', 
            '{}'.format(eval_data[i]))) as img:
            if transform is not None:
                img = transform(img)
        img = np.array(img)
        data[i] = img.transpose((2, 0, 1))
        labels[i] = attr_data[i]

    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)

    print("Saving images to {}".format(IMG_PATH))
    torch.save(data, IMG_PATH)
    torch.save(labels, LABEL_PATH)


def load_eval_partition(partition, data_dir):
    eval_data = []
    with open(os.path.join(data_dir, 'list_eval_partition.txt')) as fp:
        rows = fp.readlines()
        for row in rows:
            path, label = row.strip().split(' ')
            label = int(label)
            if label == VALID_PARTITIONS[partition]:
                eval_data.append(path)
    return eval_data


def load_attributes(paths, partition, data_dir):
    if os.path.isfile(os.path.join(data_dir, 'attr_%s.npy' % partition)):
        attr_data = np.load(os.path.join(data_dir, 'attr_%s.npy' % partition))
    else:
        attr_data = []
        with open(os.path.join(data_dir, 'list_attr_celeba.txt')) as fp:
            rows = fp.readlines()
            for ix, row in enumerate(rows[2:]):
                row = row.strip().split()
                path, attrs = row[0], row[1:]
                if path in paths:
                    attrs = np.array(attrs).astype(int)
                    attrs[attrs < 0] = 0
                    attr_data.append(attrs)
        attr_data = np.vstack(attr_data).astype(np.int64)
    attr_data = torch.from_numpy(attr_data).float()
    return attr_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str, 
        help='path to downloaded celebA dataset (e.g. /path/to/celeba/')
    parser.add_argument('--out_dir', default='../data/', type=str, 
        help='destination of outputs')
    parser.add_argument('--partition', default='train', type=str, 
        help='[train,valid,test]')
    args = parser.parse_args()
    preprocess_images(args)
