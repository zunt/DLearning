import torch
import torchvision
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("./src")
import linear_regression as dl

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=tf.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=tf.ToTensor())
# datasets，数组，每个单元下为包含feature和label

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, label)

x,y = [], []
for i in range (10):
    x.append(mnist_train[i][0])
    y.append(mnist_train[i][1])

dl.show_fashion_mnist(x, dl.get_fashion_mnist_labels(y))