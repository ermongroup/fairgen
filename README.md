# Fair Generative Modeling via Weak Supervision
This repo contains a reference implementation for fairgen as described in the paper:
> Fair Generative Modeling via Weak Supervision </br>
> [Kristy Choi*](http://kristychoi.com/), [Aditya Grover*](http://aditya-grover.github.io/), [Trisha Singh](https://profiles.stanford.edu/trisha-singh), [Rui Shu](http://ruishu.io/about/), [Stefano Ermon](https://cs.stanford.edu/~ermon/) </br>
> International Conference on Machine Learning (ICML), 2020. </br>
> Paper: https://arxiv.org/abs/1910.12008 </br>


## Data Setup:
(1) Download the CelebA dataset here (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) into the `data/` directory (if elsewhere, note the path)

(2) Preprocess the CelebA dataset for faster training:
```
python3 preprocess_celeba.py --data_dir=/path/to/downloaded/dataset/celeba/ --out_dir=../data --partition=train
```

You should run this script for `--partition=[train, val, test]` to cache all the necessary data. The preprocessed files will then be saved in `data/`.


## Pre-train Attribute Classifier
### for a single-attribute:
```
python3 train_attribute_clf.py celeba ./results/gender_clf
```

### for multiple attributes:
```
python3 train_attribute_clf.py celeba ./results/multi_clf --multi=True
```


## Pre-compute unbiased FID scores:
```
python3 KL-BigGAN/calculate_unbiased_inception_moments.py --fid_type multi
python3 KL-BigGAN/calculate_unbiased_inception_moments.py --fid_type gender
# or maybe we can just attach the FID statistics here instead
```


# Pre-train density ratio classifier
```
python3 get_density_ratios.py celeba celeba --small=[0.1, 0.25, 0.5, 1.0] --balance_type=[90_10, 80_20, multi]
```


# Train generative model (BigGAN)
## Train GAN
## CHANGE NAME: if multi, then append --multi 1

```
python3 src/KL-BigGAN/train.py --shuffle --batch_size 128 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset CA64 --data_root /atlas/u/kechoi/fair_generative_modeling/data --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --save_every 1000 --test_every 1000 --num_best_copies 50 --num_save_copies 1 --loss_type hinge --seed 777 --num_epochs 150 --start_eval 40 --reweight 1 --alpha 1.0 --name_suffix experiment_id --bias 90_10 --small=1.0
```


# Sample from trained model
```
python3 src/KL-BigGAN/sample.py --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset CA64 --data_root /atlas/u/kechoi/fair_generative_modeling/data --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --sample_npz --save_every 1000 --test_every 1000 --num_best_copies 500 --num_save_copies 2 --loss_type hinge --seed 777 --name_suffix icml_multi_small0.5_conditional --bias multi --small 0.5 --multi 1 --conditional 1 --y 1 --reweight 0 --num_epochs 200 --start_eval 50 --load_weights best_fid35
```

# Compute FID scores


## References
If you find this work useful in your research, please consider citing the following paper:
