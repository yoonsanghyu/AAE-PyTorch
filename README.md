# AAE-PyTorch
Adversarial autoencoder (basic/semi-supervised/supervised)

First, 

```bash
$ python create_datasets.py
```

Then, you get data/MNIST, data/subMNIST, which are MNIST image datasets.
you also get train_labeled.p, train_unlabeled.p, validation.p, which are list of tr_l, tr_u, tt image list.

Second,

```bash
$ python aae_baisc.py
```
or


```bash
$ python aae_supervised.py
```

or


```bash
$ python aae_semi_supervised.py
```
