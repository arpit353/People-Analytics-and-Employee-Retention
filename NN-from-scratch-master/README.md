# Coding up a Neural Network classifier from scratch

<p align="center">
<img src="https://github.com/ankonzoid/NN-from-scratch/blob/master/images/NN.png" width="50%">
</p>
 
We build and train a single hidden fullly-connected layer neural network classifier written from scratch (no high-level libraries such as tensorflow, keras, pytorch, etc), and train it via back-propagation on the "seeds_dataset.csv" (https://archive.ics.uci.edu/ml/datasets/seeds) data set containing measurements of geometrical properties of kernels belonging to three different varieties of wheat. The loss function is assumed to be L2-norm, and we do not include any biases in the activation calculation. Also, a sigmoid transfer function is used on all nodes. The delta rule (gradient descent) is used as our weight update rule which assumes the L2-loss function. More about the delta rule can be found at: https://en.wikipedia.org/wiki/Delta_rule.

 If you would like to import your own dataset, then follow the format of the data being entirely of floats except for the last column which should be only integer class labels. The code does pre-processing of the data X by normalizing each feature column.

 This code was inspired by:
 https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

### Usage:

Run the command

> python NN_classifier.py

The output should look similar to:

```
Neural network model:
 n_hidden_nodes = [5]
 l_rate = 0.6
 n_epochs = 800
 n_folds = 4

Reading 'data/seeds_dataset.csv'...
 X.shape = (210, 7)
 y.shape = (210,)
 n_classes = 3

Training and cross-validating...
 Fold 1/4: train acc = 94.94%, test acc = 96.15% (n_train = 158, n_test = 52)
 Fold 2/4: train acc = 92.41%, test acc = 90.38% (n_train = 158, n_test = 52)
 Fold 3/4: train acc = 98.10%, test acc = 92.31% (n_train = 158, n_test = 52)
 Fold 4/4: train acc = 95.57%, test acc = 94.23% (n_train = 158, n_test = 52)

Avg train acc = 95.25%
Avg test acc = 93.27%
```

### Libraries required:

* numpy
