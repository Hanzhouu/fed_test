import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 一个图像分类算法，共有60000个图像
# 50000个训练集，10000个测试集
# 图像共有10个类别，随机的正确率为10%

# 展示图像的函数
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 创建神经网络
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)  # 二维卷积网络，RGB图像输入三通道，输出卷积6通道，卷积核的大小为5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b，全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))  # 使用三个全连接层进行训练
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./datasets/5f6b1577787e9d5bb70800a4-momodel', train=True,
                                            download=False, transform=transform)
    # 训练数据集，并且进行了打乱
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./datasets/5f6b1577787e9d5bb70800a4-momodel', train=False,
                                           download=False, transform=transform)
    # 测试数据集，并且进行了打乱
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 获取随机数据
    dataiter = iter(trainloader)
    images0, labels = dataiter.next()

    # 展示图像
    imshow(torchvision.utils.make_grid(images0))
    # 显示图像标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net()  # 创建神经网络

    # 交叉验证
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # 开始训练
    for epoch in range(5):  # 多批次循环

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):  # 训练数据集
            # 获取输入
            inputs, labels = data

            # 梯度置0
            optimizer.zero_grad()

            # 正向传播，反向传播，优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            running_loss += loss.item()
            if i % 3 == 2:  # 每3批次打印一次， 在 gpu 上训练请调大此参数
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # 显示图片
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # 预测的输出
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
