#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2021/10/13 16:22
# software: PyCharm

# 使用MnIST手写数字数据集来实现一个神经网络的分类方法
# 使用Keras框架搭建一个神经网络解决手写体数字是被问题
# Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow，CNTK 或者 Theano 作为后端运行。
#
# Keras 具有如下优点：
#
# 由于用户友好，高度模块化，可扩展性，可以简单而快速的进行原型设计。
# 同时支持卷积神经网络和循环神经网络，以及两者的组合。
# 在 CPU 和 GPU 上无缝运行。

# 导入包

# 数据集 mnist
from tensorflow.keras.datasets import mnist
# 序列模型 Sequential
from tensorflow.keras.models import Sequential
# 神经网络层 Dense,Activation,Dropout
from tensorflow.keras.layers import Dense, Activation, Dropout
# 工具 np_utils
from keras.utils import np_utils

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# 查看一些图片
def plot_images(imgs):
    """绘制几个样本图片
    :param show: 是否显示绘图
    :return:
    """
    sample_num = min(9, len(imgs))
    img_figure = plt.figure(1)
    img_figure.set_figwidth(5)
    img_figure.set_figheight(5)
    for index in range(0, sample_num):
        ax = plt.subplot(3, 3, index + 1)
        ax.imshow(imgs[index].reshape(28, 28), cmap='gray')
        ax.grid(False)
    plt.margins(0, 0)
    plt.show()


def create_model():
    """
    采用 keras 搭建神经网络模型
    :return: 神经网络模型
    """
    # 选择模型，选择序贯模型（Sequential())
    model = Sequential()
    # 添加全连接层，共 512 个神经元
    model.add(Dense(512, input_shape=(784,), kernel_initializer='he_normal'))

    # 添加激活层，激活函数选择 relu
    model.add(Activation('relu'))

    # 添加全连接层，共 512 个神经元
    model.add(Dense(512, kernel_initializer='he_normal'))

    # 添加激活层，激活函数选择 relu
    model.add(Activation('relu'))

    # 添加全连接层，共 10 个神经元
    model.add(Dense(nb_classes))

    # 添加激活层，激活函数选择 softmax
    model.add(Activation('softmax'))

    return model


def fit_and_predict(model, model_path):
    """
    训练模型、模型评估、保存模型
    :param model: 搭建好的模型
    :param model_path:保存模型路径
    :return:
    """
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 模型训练
    model.fit(X_train, y_train, epochs=7, batch_size=64, verbose=1, validation_split=0.05)

    # 保存模型
    model.save(model_path)

    # 模型评估，获取测试集的损失值和准确率
    loss, accuracy = model.evaluate(X_test, y_test)

    # 打印结果
    print('Test loss:', loss)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    # 获取数据
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 将训练集数据形状从（60000,28,28）修改为（60000,784）
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    plot_images(X_train)

    # 将数据集图像像素点的数据类型从 uint8 修改为 float32
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # 把数据集图像的像素值从 0-255 放缩到[-1,1]之间
    X_train = (X_train - 127) / 127
    X_test = (X_test - 127) / 127

    # 数据集类别个数
    nb_classes = 10

    # 把 y_train 和 y_test 变成了 one-hot 的形式，即之前是 0-9 的一个数值，
    # 现在是一个大小为 10 的向量，它属于哪个数字，就在哪个位置为 1，其他位置都是 0。
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    # 实例化模型
    model = create_model()
    # 训练模型和评估模型
    fit_and_predict(model, model_path='./model.h5')
