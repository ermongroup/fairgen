#!/usr/bin/env python3
# from: https://github.com/bioinf-jku/TTUR/blob/master/precalc_stats_example.py
# this code is from the original tensorflow FID computation codebase!

import os
import glob
# NOTE: set GPU thing here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fast_fid as fid
# from scipy.misc.pilutil import imread
from cv2 import imread
import tensorflow as tf
import time

########
# PATHS
########

# splits = [0.1, 0.2, 0.3, 0.4, 0.5]
splits = [0.5]
set_name = 'test'
for split in splits:
	# load sample and specify output path
	# sample_path = '/atlas/u/kechoi/fair_generative_modeling/fid_stats/unbiased_all_gender_samples.npz'
	sample_path = '/atlas/u/kechoi/fair_generative_modeling/fid_stats/unbiased_all_multi_samples.npz'


	# output_path = '/atlas/u/kechoi/fair_generative_modeling/fid_stats/unbiased_all_gender_fid_stats.npz' 
	output_path = '/atlas/u/kechoi/fair_generative_modeling/fid_stats/unbiased_all_multi_fid_stats.npz'

	# load inception model
	inception_path = None
	print("check for inception model..", end=" ", flush=True)
	inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
	print("ok")

	# loads all images into memory (this might require a lot of RAM!)
	print("load images..", end=" " , flush=True)
	# image_list = glob.glob(os.path.join(sample_path, '*.jpg'))
	# images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

	# this is for the unbiased gender samples
	# images = np.load(sample_path)
	images = np.load(sample_path)['x']
	print("%d images found and loaded" % len(images))

	print("create inception graph..", end=" ", flush=True)
	fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
	print("ok")

	print("calculte FID stats..", end=" ", flush=True)
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
	    np.savez_compressed(output_path, mu=mu, sigma=sigma)
	print("finished saving pre-computed statistics to: {}".format(output_path))