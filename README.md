# Fair Generative Modeling via Weak Supervision
This repo contains a reference implementation for fairgen as described in the paper:
> Fair Generative Modeling via Weak Supervision </br>
> [Kristy Choi*](http://kristychoi.com/), [Aditya Grover*](http://aditya-grover.github.io/), [Trisha Singh](https://profiles.stanford.edu/trisha-singh), [Rui Shu](http://ruishu.io/about/), [Stefano Ermon](https://cs.stanford.edu/~ermon/) </br>
> International Conference on Machine Learning (ICML), 2020. </br>
> Paper: https://arxiv.org/abs/1910.12008 </br>


## 1) Data setup:
(a) Download the CelebA dataset here (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) into the `data/` directory (if elsewhere, note the path for step b)

(b) Preprocess the CelebA dataset for faster training:
```
python3 preprocess_celeba.py --data_dir=/path/to/downloaded/dataset/celeba/ --out_dir=../data --partition=train
```

You should run this script for `--partition=[train, val, test]` to cache all the necessary data. The preprocessed files will then be saved in `data/`.


## 2) Pre-train attribute classifier
For a single-attribute:
```
python3 train_attribute_clf.py celeba ./results/attr_clf
```

For multiple attributes, add the `--multi=True` flag.

Then, the trained attribute classifier will be saved in `./results/attr_clf` and will be used for downstream evaluation for generative model training.


## 3) Pre-train density ratio classifier
The density ratio classifier should be trained for the appropriate `bias` and `perc` setting, which can be adjusted in the script below:
```
python3 get_density_ratios.py celeba celeba --perc=[0.1, 0.25, 0.5, 1.0] --bias=[90_10, 80_20, multi]
```
Note that the best density ratio classifier will be saved in its corresponding directory under `./data/`


## 4) Pre-compute unbiased FID scores:
We have provided both (a) biased and (b) unbiased FID statistics in the `fid_stats/` directory.

(a) `fid_stats/fid_stats_celeba.npz` contains the original activations from the *entire* CelebA dataset, as in: https://github.com/ajbrock/BigGAN-PyTorch

(b) `fid_stats/unbiased_all_gender_fid_stats.npz` contains activations from the entire CelebA dataset, where the gender attribute (female, male) are balanced.

(c) `fid_stats/unbiased_all_multi_fid_stats.npz` contains activations from the entire CelebA dataset, where the 4 attribute classes (black hair, other hair, female, male) are balanced.

These pre-computed FID statistics are for model checkpointing (during GAN training) and downstream evaluation of sample quality only, and should be substituted for other statistics when using a different dataset/attribute splits.


## 5) Train generative model (BigGAN)
A sample script to train the model can be found in `scripts/`:

`bash run_celeba_90_10_perc1.0_impweight.sh`

You should add different arguments for different model configurations. For example:
(a) for the multi-attribute setting, append ` --multi 1`
(b) for the equi-weighted baseline, append ` --reweight 0`
(c) for the conditional baseline, append `--conditional 1 --y 1 --reweight 0`
(d) for the importance-weighted model, append `--reweight 1 --alpha 1.0`


## Sample from trained model
A sample script to sample from the (trained) model can be found in `scripts/`:

`bash sample_celeba_90_10_perc1.0_impweight.sh`

You can either append the argument `--load_weights name_of_weights` to load a specific set of weights, or pass in the `--name_suffix my_experiment` argument for the script to find the most recent checkpoint with the best FID.


## Compute FID scores
To compute FID scores after running the sampling script, (using the original Tensorflow implementation), run the following:
`python3 fast_fid.py my_experiment --multi=[True,False]`

This code assumes that there are 10 sets of 10K samples generated from the model (as per `sample.py`), and will evaluate the samples on both (a) the original FID scores and (b) unbiased FID scores (as per Step #4)


## References
If you find this work useful in your research, please consider citing the following paper:
```
@article{grover2019fair,
  title={Fair Generative Modeling via Weak Supervision},
  author={Grover, Aditya and Choi, Kristy and Shu, Rui and Ermon, Stefano},
  journal={arXiv preprint arXiv:1910.12008},
  year={2019}
}
```
