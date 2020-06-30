''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

import utils
import losses
from clf_models import ResNet18, BasicBlock

# NOTE: this is only for the binary attribute classifier!
CLF_PATH = '../results/attr_clf/model_best.pth.tar'
MULTI_CLF_PATH = '../results/multi_clf/model_best.pth.tar'


# Dummy training function for debugging
def dummy_training_function():
    def train(x, y, ratio):
        return {}
    return train


def select_loss(config):
    if config['loss_type'] == 'hinge':
        return losses.loss_hinge_dis, losses.loss_hinge_gen
    elif config['loss_type'] == 'kl':
        return losses.loss_kl_dis, losses.loss_kl_gen
    elif config['loss_type'] == 'kl_gen':
        return losses.loss_hinge_dis, losses.loss_kl_gen
    elif config['loss_type'] == 'kl_dis':
        return losses.loss_kl_dis, losses.loss_hinge_gen
    elif config['loss_type'] == 'kl_grad':
        return losses.loss_kl_grad_dis, losses.loss_kl_grad_gen
    elif config['loss_type'] == 'f_kl':
        return losses.loss_f_kl_dis, losses.loss_f_kl_gen
    elif config['loss_type'] == 'chi2':
        return losses.loss_chi_dis, losses.loss_chi_gen
    elif config['loss_type'] == 'dv':
        return losses.loss_dv_dis, losses.loss_dv_gen
    else:
        raise ValueError('loss not defined')


def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
    discriminator_loss, generator_loss = select_loss(config)

    def train(x, y, ratio):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        ratio = torch.split(ratio, config['batch_size'])
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                # only feed in 0's for y if "unconditional"
                if not config['conditional']:
                    y_.zero_()
                    y_counter = torch.zeros_like(y[counter]).to(y_.device).long()
                else:
                    y_.sample_()
                    y_counter = y[counter]
                D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']], x[counter], y_counter, train_G=False,
                    split_D=config['split_D'])
                # reweight discriminator loss
                # modified discriminator loss to reflect flattening coefficient
                D_loss_real, D_loss_fake = discriminator_loss(
                    D_fake, D_real, ratio[counter], alpha=config['alpha'])
                D_loss = (D_loss_real + D_loss_fake) / \
                    float(config['num_D_accumulations'])

                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            # NOTE: setting all labels to 0 to train as unconditional model
            if not config['conditional']:
                y_.zero_()
            D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
            # we don't need to do anything for the generator loss
            G_loss = generator_loss(
                D_fake) / float(config['num_G_accumulations'])
            G_loss.backward()

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            # Debug print to indicate we're using ortho reg in G
            print('using modified ortho reg in G')
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item())}
        # Return G's loss and the components of D's loss.
        return out
    return train


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''


def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                    state_dict, config, experiment_name):
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, None, G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name,
                           'copy%d' % state_dict['save_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_num'] = (
            state_dict['save_num'] + 1) % config['num_save_copies']

    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    # Accumulate standing statistics?
    if config['accumulate_stats']:
    # NOTE: setting all labels to 0 to train as unconditional model
        if not config['conditional']:
            y_.zero_()
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])

    # Save a random sample sheet with fixed z and y
    with torch.no_grad():
        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(
                which_G, (fixed_z, which_G.shared(fixed_y)))
        else:
            fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    utils.sample_sheet(which_G,
                       classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                       num_classes=config['n_classes'],
                       samples_per_class=10, parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       z_=z_)
    # Also save interp sheets
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        utils.interp_sheet(which_G,
                           num_per_sheet=16,
                           num_midpoints=8,
                           num_classes=config['n_classes'],
                           parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=experiment_name,
                           folder_number=state_dict['itr'],
                           sheet_number=0,
                           fix_z=fix_z, fix_y=fix_y, device='cuda')

''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''
def update_FID(G, D, G_ema, state_dict, config, FID, experiment_name, test_log, epoch):
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

    # obtain classifier predictions for samples
    preds = classify_examples(clf, config)  # (10K,)
    fair_d, l1_fair_d, kl_fair_d = utils.fairness_discrepancy(preds, clf_classes)
    # when comparing, just look at L2 for now!
    print('Fairness discrepancy metric is: {}'.format(fair_d))

    print('Itr %d: PYTORCH UNOFFICIAL FID is %5.4f' %
          (state_dict['itr'], FID))
    # If improved over previous best metric, save appropriate copy

    # save model by both best FID (inaccurate) and fairness discrepancy
    if fair_d < state_dict['best_fair_d']:
        print('%s improved over previous best fair_d, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name, 'best_fair%d' % state_dict['save_best_num_fair'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_fair'] = (
            state_dict['save_best_num_fair'] + 1) % config['num_best_copies']


    # save model by both best FID or fairness discrepancy
    if FID < state_dict['best_FID']:
        print('%s improved over previous best FID, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name, 'best_fid%d' % state_dict['save_best_num_fid'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_fid'] = (
            state_dict['save_best_num_fid'] + 1) % config['num_best_copies']
    # update best fairness discrepancy and FID score
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    state_dict['best_fair_d'] = min(state_dict['best_fair_d'], fair_d)
    # Log results to file
    test_log.log(epoch=int(epoch), itr=int(state_dict['itr']), IS_mean=float(0), IS_std=float(0), FID=float(FID), FAIR=float(fair_d), L1_FAIR=float(l1_fair_d), KL_FAIR=float(kl_fair_d))


def classify_examples(model, config):
    """
    classifies generated samples into appropriate classes 
    """
    model.eval()
    preds = []
    samples = np.load(config['sample_path'])['x']
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
            preds.append(pred)
        preds = torch.cat(preds).data.cpu().numpy()

    return preds