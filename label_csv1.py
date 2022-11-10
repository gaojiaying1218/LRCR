
#coding=gbk
import os
import pandas as pd
from sklearn.utils import shuffle
import torchvision
import torch
import numpy as np
from PIL import Image
#dir_name = 'D:/代码/Python代码/ResNet/jiagu_trans/jiagu_trans_50mix/'#分类50类
#dir_name = 'D:/数据集/古文字数据集/pic_trans'#测试pic中有多少类

dir_name = 'D:/数据集/古文字数据集/pic_trans整理/jiagu_Jinwen/'


def create_Hebin_label():#author GJY

    path = 'D:\数据集\古文字数据集\JiDaTop6941_Aug/'
    label_list = []
    name_list = []
    for file in os.listdir(path):
        print('打开',file)
        sub_path = path+file+'/'
        for file_sub in os.listdir(sub_path):
            name_list.append(file_sub)
            label_list.append(file)
    csv = {'name': name_list, 'label': label_list}
    csv = pd.DataFrame(csv)
    print(csv)
    csv.to_csv('D:\代码\Python代码\ResNet_JiDa6941/Created6941_label.csv',index=False)

def labelNum6941(): #author GJY
    all_label = pd.read_csv('D:\DataSet\JiDaTop100\ResNet_JiDa100\Created_Top100_labelNum.csv')
    df = pd.DataFrame(all_label['label'].value_counts())
    df.columns = ['nums']
    newdf = df[df['nums'] < 0]
    delindexs = newdf.index
    all_label = all_label[~all_label['label'].isin(delindexs)]
    drop_duplicates = all_label.drop_duplicates(['label'])

    drop_duplicates.to_csv('D:\DataSet\JiDaTop100\ResNet_JiDa100/labelNum100.csv', index=False)

    print(drop_duplicates)



def labelNum():

    all_label = pd.read_csv('D:\代码\Python代码\ResNet_JiDa6941/Created6941_label.csv')
    df = pd.DataFrame(all_label['label'].value_counts())
    df.columns = ['nums']
    newdf = df[df['nums'] < 0]
    delindexs = newdf.index
    all_label = all_label[~all_label['label'].isin(delindexs)]
    drop_duplicates = all_label.drop_duplicates(['label'])
    drop_duplicates['label_number'] = range(len(drop_duplicates))
    print('drop_duplicates', drop_duplicates)
    new_all_data = pd.merge(drop_duplicates, all_label, how='inner', on=['label'])
    new_all_data = new_all_data.drop(['name_x'], axis=1)
    new_all_data = new_all_data.rename(columns={'name_y': 'name'})
    new_all_data.to_csv('D:\代码\Python代码\ResNet_JiDa6941/Created6941_labelNum.csv', index=False)
def splitDataset():
    all_label = pd.read_csv('D:\代码\Python代码\ResNet_JiDa6941/Created6941_labelNum.csv')
    print(all_label)
    all_label = shuffle(all_label)
    train_label = all_label[:int(len(all_label) * 4 / 5)]
    test_label = all_label[int(len(all_label) * 4 / 5):]

    train_label.to_csv('D:\代码\Python代码\ResNet_JiDa6941/Top6941_train.csv', index=False)
    test_label.to_csv('D:\代码\Python代码\ResNet_JiDa6941/Top6941_test.csv', index=False)


if __name__ == '__main__':
    create_Hebin_label()
    labelNum()
    splitDataset()
    labelNum6941()
