# AAE-PyTorch
Adversarial autoencoder (basic/semi-supervised/supervised)

First, 

```bash
$ python create_datasets.py
```
It takes some times....

Then, you get data/MNIST, data/subMNIST (automatically downloded in data/ directory), which are MNIST image datasets.
you also get train_labeled.p, train_unlabeled.p, validation.p, which are list of tr_l, tr_u, tt image.

Second,

```bash
$ python aae_baisc.py
```
or


```bash
$ python aae_supervised.py
```
You can get category conditional images like below,


<img src="https://user-images.githubusercontent.com/51259168/143198299-f0c87643-998e-4949-95f9-3c38be746091.png" width="300" height="300"/>

or


```bash
$ python aae_semi_supervised.py
```
