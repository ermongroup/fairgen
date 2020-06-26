''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange
import pickle


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
from clf_models import ResNet18, BasicBlock


CLF_PATH = '/atlas/u/kechoi/fair_generative_modeling/results/gender_clf/model_best.pth.tar'
MULTI_CLF_PATH = '/atlas/u/kechoi/fair_generative_modeling/results/multi_clf/model_best.pth.tar'


def classify_examples(model, sample_path):
    """
    classifies generated samples into appropriate classes 
    """
    import numpy as np
    model.eval()
    preds = []
    probs = []
    samples = np.load(sample_path)['x']
    n_batches = samples.shape[0] // 1000

    with torch.no_grad():
        # generate 10K samples
        for i in range(n_batches):
            x = samples[i*1000:(i+1)*1000]
            samp = x / 255.  # renormalize to feed into classifier
            samp = torch.from_numpy(samp).to('cuda').float()

            # get classifier predictions
            logits, probas = model(samp)
            _, pred = torch.max(probas, 1)
            probs.append(probas)
            preds.append(pred)
        preds = torch.cat(preds).data.cpu().numpy()
        probs = torch.cat(probs).data.cpu().numpy()

    return preds, probs


def run(config):
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num_fair': 0, 'save_best_num_fid': 0, 'best_IS': 0, 'best_FID': 999999, 'best_fair_d': 999999, 'config': config}

    # Optionally, get the configuration from the state dict. This allows for
    # recovery of the config provided only a state dict and experiment name,
    # and can be convenient for writing less verbose sample shell scripts.
    if config['config_from_name']:
        utils.load_weights(None, None, state_dict, config['weights_root'],
                           config['experiment_name'], config['load_weights'], None,
                           strict=False, load_optim=False)
        # Ignore items which we might want to overwrite from the command line
        for item in state_dict['config']:
            if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
                config[item] = state_dict['config'][item]

    # update config (see train.py for explanation)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = 1  # HACK
    if config['conditional']:
        config['n_classes'] = 2
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    device = 'cuda'
    config['sample_num_npz'] = 10000
    print(config['ema_start'])

    # Seed RNG
    # utils.seed_rng(config['seed'])  # config['seed'])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    G = model.Generator(**config).cuda()
    utils.count_parameters(G)

    # Load weights
    print('Loading weights...')
    assert config['mode'] in ['fair', 'fid']

    # find best weights for either FID or fair
    import os
    import glob
    # grab best weights according to FID/fair mode
    print('TODO: REMOVE THIS')
    weights_root = config['weights_root']
    # weights_root = '/atlas/u/kechoi/fair_generative_modeling/src/KL-BigGAN/rebuttal_scripts/weights'
    # weights_root = '/atlas/u/kechoi/fair_generative_modeling/scripts/weights'
    ckpts = glob.glob(os.path.join(weights_root, experiment_name, 'state_dict_best_{}*'.format(config['mode'])))
    best_ckpt = 'best_{}{}'.format(config['mode'],len(ckpts)-1)
    config['load_weights'] = best_ckpt

    # only make samples that don't exist
    # sample_path = '%s/%s/{}_samples_0.npz' % (config['samples_root'], experiment_name, config['mode'])
    # if os.path.exists(sample_path):
    #     print('samples already exist for {}, exiting'.format(config['mode']))
    #     quit()

    # uncomment this later
    # Here is where we deal with the ema--load ema weights or load normal weights
    import sys
    try:
        # utils.load_weights(G if not (config['use_ema']) else None, None, state_dict, config['weights_root'], experiment_name, config['load_weights'], G if config['ema'] and config['use_ema'] else None,
        #     strict=False, load_optim=False)
        utils.load_weights(G if not (config['use_ema']) else None, None, state_dict, weights_root, experiment_name, config['load_weights'], G if config['ema'] and config['use_ema'] else None,
            strict=False, load_optim=False)
        # print('weights found')
        # quit()
    except FileNotFoundError:
        # print('NOTE: does not exist, loading from other directory')
        # new_weights_root = '/atlas/u/trsingh/fair_generative_modeling/src/KL-BigGAN/weights'
        # ckpts = glob.glob(os.path.join(new_weights_root,experiment_name, 'state_dict_best_{}*'.format(config['mode'])))
        # best_ckpt = 'best_{}{}'.format(config['mode'],len(ckpts)-1)
        # config['load_weights'] = best_ckpt

        # # issue with the classes :(
        # try:
        #     utils.load_weights(G if not (config['use_ema']) else None, None, state_dict, new_weights_root, experiment_name, config['load_weights'], G if config['ema'] and config['use_ema'] else None,strict=False, load_optim=False)
        # except RuntimeError:
        #     config['n_classes'] = 4
        #     G = model.Generator(**config).cuda()
        #     utils.load_weights(G if not (config['use_ema']) else None, None, state_dict, new_weights_root, experiment_name, config['load_weights'], G if config['ema'] and config['use_ema'] else None,strict=False, load_optim=False)
        # print('weights *now* found, proceeding')
        pass
    # Update batch size setting used for G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'],
                               z_var=config['z_var'])

    if config['G_eval_mode']:
        print('Putting G in eval mode..')
        G.eval()
    else:
        print('G is in %s mode...' % ('training' if G.training else 'eval'))

    # Sample function
    sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
    if config['accumulate_stats']:
        print('Accumulating standing stats across %d accumulations...' %
              config['num_standing_accumulations'])
        utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])
    # end of uncomment this later

    # # # Sample a number of images and save them to an NPZ, for use with TF-Inception

    # # check sample status
    sample_path = '%s/%s/' % (config['samples_root'], experiment_name)
    print('looking in sample path {}'.format(sample_path))
    if not os.path.exists(sample_path):
        print('creating sample path: {}'.format(sample_path))
        os.mkdir(sample_path)

    # Lists to hold images and labels for images
    if config['mode'] == 'fid':
        print('saving FID samples')
        for k in range(10):  # need to do this for 10 repeated runs
            npz_filename = '%s/%s/fid_samples_%s.npz' % (
                config['samples_root'], experiment_name, k)
            if os.path.exists(npz_filename):
                print('samples already exist, skipping...')
                continue
            x, y = [], []
            print('Sampling %d images and saving them to npz...' %
                  config['sample_num_npz'])
            for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
                with torch.no_grad():
                    images, labels = sample()
                x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
                y += [labels.cpu().numpy()]
            x = np.concatenate(x, 0)[:config['sample_num_npz']]
            y = np.concatenate(y, 0)[:config['sample_num_npz']]
            print('checking labels: {}'.format(y.sum()))
            print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
            npz_filename = '%s/%s/fid_samples_%s.npz' % (
                config['samples_root'], experiment_name, k)
            print('Saving npz to %s...' % npz_filename)
            np.savez(npz_filename, **{'x': x, 'y': y})
    else:
        print('saving fair samples')
        # this was initially commented out
        for k in range(10):  # need to do this for 10 repeated runs
            npz_filename = '%s/%s/fair_samples_%s.npz' % (
                config['samples_root'], experiment_name, k)
            if os.path.exists(npz_filename):
                print('samples already exist, skipping...')
                continue
            x, y = [], []
            print('Sampling %d images and saving them to npz...' %
                  config['sample_num_npz'])
            for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
                with torch.no_grad():
                    images, labels = sample()
                x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
                y += [labels.cpu().numpy()]
            x = np.concatenate(x, 0)[:config['sample_num_npz']]
            y = np.concatenate(y, 0)[:config['sample_num_npz']]
            print('checking labels: {}'.format(y.sum()))
            print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
            npz_filename = '%s/%s/fair_samples_%s.npz' % (
                config['samples_root'], experiment_name, k)
            print('Saving npz to %s...' % npz_filename)
            np.savez(npz_filename, **{'x': x, 'y': y})

    # classify proportions
    metrics = {'l2': 0, 'l1': 0, 'kl': 0}
    l2_db = np.zeros(10)
    l1_db = np.zeros(10)
    kl_db = np.zeros(10)

    if config['mode'] == 'fair':
        fname = '%s/%s/fair_disc.p' % (
                config['samples_root'], experiment_name)
    else:
        fname = '%s/%s/fair_disc_fid_samples.p' % (
                config['samples_root'], experiment_name)
    # if os.path.exists(fname):
    #     print('metrics already exist, exiting')
    #     quit()

    # load classifier
    if not config['multi']:
        print('Pre-loading pre-trained single-attribute classifier...')
        clf_state_dict = torch.load(CLF_PATH)['state_dict']
        clf_classes = 2
    else:
        # multi-attribute
        print('Pre-loading pre-trained multi-attribute classifier...')
        clf_state_dict = torch.load(MULTI_CLF_PATH)['state_dict']
        clf_classes = 4
    # load attribute classifier here
    clf = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], 
                    num_classes=clf_classes, grayscale=False) 
    clf.load_state_dict(clf_state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = clf.to(device)
    clf.eval()  # turn off batch norm

    # classify examples and get probabilties
    n_classes = 2
    if config['multi']:
        n_classes = 4

    # number of classes
    probs_db = np.zeros((10, 10000, n_classes))
    for i in range(10):
        # fair or fid samples
        npz_filename = '{}/{}/{}_samples_{}.npz'.format(
            config['samples_root'], experiment_name, config['mode'], i)
        preds, probs = classify_examples(clf, npz_filename)
        l2, l1, kl = utils.fairness_discrepancy(preds, clf_classes)

        # save metrics
        l2_db[i] = l2
        l1_db[i] = l1
        kl_db[i] = kl
        probs_db[i] = probs
        print('fair_disc for iter {} is: l2:{}, l1:{}, kl:{}'.format(i, l2, l1, kl))
    metrics['l2'] = l2_db
    metrics['l1'] = l1_db
    metrics['kl'] = kl_db
    print('fairness discrepancies saved in {}'.format(fname))
    print(l2_db)
    
    # save all metrics
    with open(fname, 'wb') as fp:
        pickle.dump(metrics, fp)
    np.save(os.path.join(config['samples_root'], experiment_name, 'clf_probs.npy'), probs_db)


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
