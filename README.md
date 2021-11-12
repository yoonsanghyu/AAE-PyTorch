# AAE-PyTorch
Adversarial autoencoder (basic/semi-supervised/supervised)

First, 

```bash
$ python create_datasets.py
```

Then, you get data/MNIST, data/subMNIST (atomatically downloded in data/ directory), which are MNIST image datasets.
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


![image](https://user-images.githubusercontent.com/51259168/141420652-db958bc1-251f-42c9-a8eb-02a2e31c2343.png)


or


```bash
$ python aae_semi_supervised.py
```
