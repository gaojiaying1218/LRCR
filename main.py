# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)



def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
def get_images_and_labels(dir_path):

    dir_path = Path(dir_path)
    classes = []
    for category in dir_path.iterdir():
        if category.is_dir():
            classes.append(category.name)
    images_list = []
    labels_list = []
    for index, name in enumerate(classes):
        class_path = dir_path / name
        if not class_path.is_dir():
            continue
        for img_path in class_path.glob('*.jpg'):
            images_list.append(str(img_path))
            labels_list.append(int(index))
    return  images_list, labels_list

def get_images_and_labels_from_excel(dir_path):

    data = pd.read_csv(dir_path)

    labels_list = data.values[:, 1]
    images_list = data.values[:, 2]
    subPath_list = data.values[:, 0]

    return images_list, labels_list, subPath_list

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=(0.5), std=(0.5))
])

class MyDataset(Dataset):

    def __init__(self, dir_path, Size = 256, transform=transform):
        self.dir_path = dir_path
        self.Size = Size
        self.transform = transform
        self.images, self.labels , self.subPath = get_images_and_labels_from_excel(self.dir_path)

    def __len__(self):
        return len(self.images)
    def __getitem__(self, index, root = r'D:\DataSet\JiDaTop100/'):
        img_path = root + str(self.subPath[index]) + '/'+  self.images[index]
        #img_path = self.images[index] #当使用def get_images_and_labels_from_excel()函数时使用此句
        label = self.labels[index]
        img = Image.open(img_path).convert('L')
        img = img.resize((self.Size,self.Size))
        img = np.array(img)
       # img = img.float()
        sample = {'image':img, 'labels':label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample





# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    blk = Residual(3, 3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    Y.shape
    blk = Residual(3, 6, use_1x1conv=True, strides=2)
    blk(X).shape

    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 100))
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.05, 5, 128
    train_dataset = MyDataset(r'D:\DataSet\JiDaTop100\ResNet_JiDa100\Top100_jida_train.csv')
    test_dataset = MyDataset(r'D:\DataSet\JiDaTop100\ResNet_JiDa100\Top100_jida_test.csv')
    train_iter = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_iter = DataLoader(train_dataset, batch_size=128, shuffle=False)
   # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter , num_epochs, lr, d2l.try_gpu())

    torch.save(net.state_dict(),'D:\Code\python\ResNet_d2l\weights/' + 'ResNet_JiDa100_v2.pt')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
