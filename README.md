# README

Data science datasets in python.



# installing

`vi ~/.profile`

```
export PATH=/path/to/datasets:$PATH
```

# usage

```
import datasets
```

# MNIST

`datasets.mnist.get_mnist(directory='None')`

**Examples**
```
X_train, y_train, X_test, y_test = get_mnist()
```
# CIFAR

## get CIFAR

`datasets.cifar.get_cifar100(directory='/tmp/datasets/cifar/', channel='rgb')`

**arguments**
directory : the directory contained CIFAR100 dataset python pickle.
channel: 'rgb' by default, and you can set to 'bgr'

**return values**
meta: include fine_label_names and coarse_label_names.
X_train: 10000 train datas with shape 32 X 32 X 3.
y_train_fl: 10000 train fine labels.
X_train: 10000 train datas with shape 32 X 32 X 3.
y_train_fl: 10000 train fine labels.

**example**

```
meta, X_train, y_train,  X_test, y_test = datasets.cifar.get_cifar100()
```



## get CIFAR-100 labels

`datasets.cifar.get_label(fine_label_idx, meta=None)`

if meta isn't None, it return (coarse label names, fine label names).
otherwise it only return the index of coarse label.

**examples**

```
get_label(y_predict[15])
get_label(y_predict[15], meta)
```