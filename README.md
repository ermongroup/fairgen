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
## CHANGE NAME: if multi, then append --multi 1

```
python3 src/KL-BigGAN/train.py --shuffle --batch_size 128 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset CA64 --data_root /atlas/u/kechoi/fair_generative_modeling/data --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --save_every 1000 --test_every 1000 --num_best_copies 50 --num_save_copies 1 --loss_type hinge --seed 777 --num_epochs 150 --start_eval 40 --reweight 1 --alpha 1.0 --name_suffix experiment_id --bias 90_10 --perc=1.0
```


## Sample from trained model
```
python3 src/KL-BigGAN/sample.py --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset CA64 --data_root /atlas/u/kechoi/fair_generative_modeling/data --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --sample_npz --save_every 1000 --test_every 1000 --num_best_copies 500 --num_save_copies 2 --loss_type hinge --seed 777 --name_suffix icml_multi_perc0.5_conditional --bias multi --perc 0.5 --multi 1 --conditional 1 --y 1 --reweight 0 --num_epochs 200 --start_eval 50 --load_weights best_fid35
```

## Compute FID scores


## References
If you find this work useful in your research, please consider citing the following paper:
