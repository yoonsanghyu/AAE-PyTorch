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
![image](https://user-images.githubusercontent.com/51259168/143195813-82ef4186-7718-46dc-8b17-786f63d33830.png)


or


```bash
$ python aae_semi_supervised.py
```
