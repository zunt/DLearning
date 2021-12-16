import torch
import torchvision
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import time
import sys
import linear_regression as d2l
import matplotlib


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle', 'boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().get_xaxis().set_visible(False)
        f.axes.get_yaxis().get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                    transform=tf.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=tf.ToTensor())
    # datasets，数组，每个单元下为包含feature和label

    print(type(mnist_train))
    print(len(mnist_train), len(mnist_test))

    feature, label = mnist_train[0]
    print(feature.shape, label)

    x, y = [], []
    for i in range(10):
        x.append(mnist_train[i][0])
        y.append(mnist_train[i][1])
    show_fashion_mnist(x, get_fashion_mnist_labels(y))
