import numpy as np
import gzip, pickle
import tensorflow as tf
from scipy import linalg
import pathlib
import urllib
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'
# INCEPTION_INPUT = 'Mul:0'
INCEPTION_INPUT = 'ExpandDims:0'

INCEPTION_OUTPUT = 'logits:0'
INCEPTION_FINAL_POOL = 'pool_3:0'


def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)[0]
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.GFile(pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='Inception_Net')

def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / INCEPTION_FROZEN_GRAPH
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract(INCEPTION_FROZEN_GRAPH, str(model_file.parent))
    return str(model_file)

def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'Inception_Net/' + INCEPTION_FINAL_POOL
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

    layername = 'Inception_Net/' + INCEPTION_OUTPUT
    logits = sess.graph.get_tensor_by_name(layername)
    ops = logits.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    
    return pool3, logits
#-------------------------------------------------------------------------------


def calculate_frechet_distance(real_features, 
                        gen_features, 
                        eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)

    mu2 = np.mean(gen_features, axis=0)
    sigma2 = np.cov(gen_features, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        #warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    mean_diff = diff.dot(diff)
    cov_diff = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return mean_diff+cov_diff, mean_diff, cov_diff

def calculate_frechet_distance_stats(real_stats, 
                        gen_features, 
                        eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = real_stats[0]
    sigma1 = real_stats[1]

    mu2 = np.mean(gen_features, axis=0)
    sigma2 = np.cov(gen_features, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        #warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    mean_diff = diff.dot(diff)
    cov_diff = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return mean_diff+cov_diff, mean_diff, cov_diff

def get_activations(images, 
                    sess, 
                    batch_size=50, 
                    verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer, logits_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pred_arr = np.empty((n_used_imgs,2048))
    logits_arr = np.empty((n_used_imgs,1008))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        end = start + batch_size
        batch = images[start:end]
        pred, logits = sess.run([inception_layer, logits_layer], {'Inception_Net/'+INCEPTION_INPUT: batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
        logits_arr[start:end] = logits.reshape(batch_size,-1)

    if verbose:
        print(" done")
    return pred_arr, logits_arr

def get_activation_stats(images, 
                        sess,
                        batch_size):

    features, logits = get_activations(images, 
                            sess, 
                            batch_size,
                            verbose=True)
    
    return logits, features

def compute_scores_tf(real_images, 
                    gen_images, 
                    batch_size):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    inception_path = check_or_download_inception(inception_path)
    create_inception_graph(str(inception_path))
    print('inception graph created', flush=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        real_logits, real_features = get_activation_stats(real_images,
                                                            sess,
                                                            batch_size=batch_size)
        gen_logits, gen_features = get_activation_stats(gen_images,
                                                            sess, 
                                                            batch_size=batch_size)


    fid, _, _ = calculate_frechet_distance(real_features, gen_features)

    return fid

def compute_scores_tf_activations(real_stats, 
                    gen_images, 
                    batch_size,
                    inception_path=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    inception_path = check_or_download_inception(inception_path)
    create_inception_graph(str(inception_path))
    print('inception graph created', flush=True)
    with tf.Session(config=config) as sess:
        print('sess running start')
        sess.run(tf.global_variables_initializer())
        print('sess running end')
        gen_logits, gen_features = get_activation_stats(gen_images,
                                                            sess, 
                                                            batch_size=batch_size)

        

    fid, _, _ = calculate_frechet_distance_stats(real_stats, gen_features)

    return fid

def main(args):
    print('running {}'.format(args.exp_id))

    # grab proper inception moments
    if args.multi == False:
        print('running FID calculation for single-attribute experiment')
        fname = '../../fid_stats/unbiased_all_gender_fid_stats.npz'
    else:
        # multi-attribute experiment
        print('running FID for multi-attribute experiment')
        fname = '../../fid_stats/unbiased_all_multi_fid_stats.npz'

    # load both biased and unbiased stats
    print('loading unbiased stats from {}'.format(fname))
    f = np.load(fname)
    real_mu, real_cov = f['mu'][:], f['sigma'][:]
    unbiased_stats = (real_mu, real_cov)

    # load biased stats
    fname = '../../fid_stats/fid_stats_celeba.npz'
    print('loading original celebA FID stats from: {}'.format(fname))
    f = np.load(fname)
    real_mu, real_cov = f['mu'][:], f['sigma'][:]
    biased_stats = (real_mu, real_cov)

    # start running 10 sets of FID scores
    samples_root = './samples'
    fid_unbiased_db = np.zeros(args.n_replicates)
    fid_biased_db = np.zeros(args.n_replicates)
    print('fixed evaluation to be FID samples')

    # iterate through 10 sets of 10K samples for FID computation
    for i in range(args.n_replicates):
        # load samples
        npz_filename = '%s/%s/fid_samples_%s.npz' % (
            samples_root, args.exp_id, i)
        gen_images = np.load(npz_filename)['x']
        gen_images = np.transpose(gen_images, (0,2,3,1))

        # samples shape
        print(real_mu.shape, real_cov.shape, gen_images.shape)
        start = time.time()
        unbiased_fid = compute_scores_tf_activations(
            unbiased_stats, gen_images, 100)
        biased_fid = compute_scores_tf_activations(
            biased_stats, gen_images, 100)
        end = time.time()
        print('Unbiased FID:', unbiased_fid, ' Time:', end-start)
        print('Biased FID:', biased_fid, ' Time:', end-start)

        # save values
        fid_unbiased_db[i] = unbiased_fid
        fid_biased_db[i] = biased_fid
    # save all 10 runs
    print('unbiased: {}'.format(fid_unbiased_db))
    print('biased: {}'.format(fid_biased_db))
    np.save(os.path.join(samples_root, args.exp_id, 'unbiased_fid.npy'), fid_unbiased_db)
    np.save(os.path.join(samples_root, args.exp_id, 'biased_fid.npy'), fid_biased_db)

        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', type=str, 
                        help='name of experiment ID in samples')
    parser.add_argument('--multi', type=bool,
                        help='whether the experiment is multi-attribute')
    parser.add_argument('--n_replicates', type=int, default=10,
                        help='number of sets of 10K samples for evaluation.')
    args = parser.parse_args()
    main(args)
