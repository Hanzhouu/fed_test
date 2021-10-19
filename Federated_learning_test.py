#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:HanZhou
# datetime:2021/10/18 14:37
# software: PyCharm
import collections
import threading

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from pytorch_study import NeuralNetwork  # 神经网络类
from pytorch_study import train  # 训练方法
from pytorch_study import Model_test  # 模型评估方法
from threading import Thread  # 导入线程


def Device_training(epochs, models_list, num):
    # 联邦学习的设备分别进行训练，参数为每个设备的训练轮次，模型列表，线程序号

    print('%d号子线程开启! thread (%s%d) is training...' % (num, threading.current_thread().name, num))

    # Download training data from open datasets.
    # 加载各个设备的本地数据用于模型的训练
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # 训练设备类型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smodel = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(smodel.parameters(), lr=1e-3)

    for t in range(epochs):
        # print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, smodel, loss_fn, optimizer, device)
        # test/(test_dataloader, model, loss_fn)

    #  添加此线程训练的结果
    models_list.append(smodel)

    print('%d号子线程关闭! thread (%s%d) finished training!' % (num, threading.current_thread().name, num))


def Avg_model(model_list):
    #  对多个模型进行进行参数平均，存放在模型model中
    #  获取所有模型的所有参数
    worker_state_dict = [x.state_dict() for x in model_list]
    #  获取模型参数的名称
    weight_keys = list(worker_state_dict[0].keys())
    #  实现字典中对于元素的排序
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:  # 遍历模型的所有参数
        key_sum = 0
        for i in range(len(model_list)):  # 遍历所有的模型的此参数
            # print(worker_state_dict[i][key])
            key_sum = (key_sum + worker_state_dict[i][key])
            # print(key_sum)
        # fed_state_dict[key] = key_sum / len(model_list)
        fed_state_dict[key] = torch.div(key_sum, float(len(model_list)))
        # print(fed_state_dict[key])
    #  返回模型的参数
    return fed_state_dict


def main():
    threads = []  # 线程列表
    models_list = []  # 训练模型列表
    epochs = 2  # 每个设备进行训练的轮次
    device_num = 2  # 进行分别训练的设备数量

    print('主线程开启！ thread %s is running...' % threading.current_thread().name)
    for i in range(device_num):
        cur_thread = Thread(target=Device_training(epochs, models_list, i))  # 创建训练的线程
        threads.append(cur_thread)
    # 子线程开始训练
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    #  测试每一个模型的准确率
    for model in models_list:
        Model_test(1, epochs, model)

    print("进行模型整合中......")
    fl_dict = Avg_model(models_list)
    # 对各个设备训练得到的模型进行整合
    fl_model = NeuralNetwork()  # 最终的模型
    # 整合各个设备训练的神经网络模型,加载模型返回后的参数
    fl_model.load_state_dict(fl_dict)

    # 对整合后的模型进行准确度评估
    Model_test(device_num, epochs, fl_model)


if __name__ == '__main__':
    main()
