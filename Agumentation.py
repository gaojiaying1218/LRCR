import os
import sys
import torch
import os
from torch.utils.data import Dataset
#import cv2
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torch
import torchvision
from d2l import torch as d2l
import shutil
import random
# def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
#     Y = [aug(img) for _ in range(num_rows * num_cols)]

    #保存Y们到本地路径
    #d2l.show_images(Y, num_rows, num_cols, scale=scale)

def Create_folder(filename): #创建文件夹
    filename = filename.strip()
    filename = filename.rstrip("\\")
    isExists = os.path.exists(filename)

    if not isExists:
        os.makedirs(filename, mode=777)
        print(filename+"创建成功")
        return  True
    else:
        print(filename+"已存在")
        return False

def img2augs(path):
    img = d2l.Image.open(path)
    index = random.randint(0, 3)
    if index == 0:
        img_aug = color_aug(img)
    elif index == 1:
        img_aug = shape_aug(img)
    elif index == 2:
        img_aug = color_aug(img)
    else:
        img_aug = augs(img)
    return img_aug

def Augment(source_path,destnation_path):




    for root, dirs, files in os.walk(source_path):
        for dir in dirs:
            path_dir = os.path.join(root,dir)
            list_dir = os.listdir(path_dir)
            numImgs = len(list_dir)
            num_needImgs = 400-numImgs
            folderName = dir[:]

            if num_needImgs >= numImgs:
                num_Loop = num_needImgs//numImgs
                j = 1
                for i in range(0,num_Loop):
                    for each in os.listdir(path_dir):
                        path = path_dir+'/'+each
                        img_aug = img2augs(path)
                        Create_folder(destnation_path+folderName)
                        shutil.copy(path,destnation_path+folderName)
                        img_aug.save(destnation_path+'/'+folderName+'/'+folderName+'_'+str(j)+'.jpg')
                        print('扩充'+str(j)+'/'+str(num_needImgs))
                        numImgs = numImgs+1
                        j = j + 1

            elif num_needImgs<0:
                picknumber = abs(num_needImgs)
                #sample = random.sample(list_dir, picknumber)
                j = 1
                for each in os.listdir(path_dir):
                    path = path_dir + '/' + each
                    img_aug = img2augs(path)
                    Create_folder(destnation_path + folderName)
                    shutil.copy(path, destnation_path + folderName)
                    #img_aug.save(destnation_path + '/' + folderName + '/' + folderName + '_' + str(j) + '.jpg')
                    print('扩充' + str(j) + '/' + str(num_needImgs))
                    j = j + 1

            else:
                picknumber = num_needImgs
                sample = random.sample(list_dir,picknumber)
                j=1

                for eacha in os.listdir(path_dir):
                    path = path_dir+ '/' + eacha
                    Create_folder(destnation_path + folderName)
                    shutil.copy(path, destnation_path + folderName)
                for each in sample:
                    path = path_dir + '/' + each
                    img_aug = img2augs(path)
                    Create_folder(destnation_path + folderName)
                    #shutil.copy(path, destnation_path + folderName)
                    img_aug.save(destnation_path + '/'+ folderName+'/'+folderName+ '_' + str(j) + '.jpg')
                    print('扩充'+str(j)+'/'+str(num_needImgs))
                    j = j+1
    return


def Aug_Test(source_path,destnation_path):
    for root, dirs, files in os.walk(source_path):
        for dir in dirs:
            path_dir = os.path.join(root,dir)
            list_dir = os.listdir(path_dir)
            folderName = dir[:]
            j=1
            for each in os.listdir(path_dir):
                path = path_dir + '/' + each
                img_aug = img2augs(path)
                Create_folder(destnation_path + folderName)
                #shutil.copy(path, destnation_path + folderName)
                img_aug.save(destnation_path + '/' + folderName + '/' + folderName + '_' + str(j) + '.jpg')
                print('扩充' + str(j) + '/' + str(len(list_dir)))
                j = j + 1

if __name__ == '__main__':

    source_path = 'D:\数据集\古文字数据集\JiDa/'
    destnation_path = 'D:\数据集\古文字数据集\JiDaTop6941_Aug/'
    color_aug = torchvision.transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    shape_aug = torchvision.transforms.RandomResizedCrop(
        (200, 200), scale=(0.8, 1), ratio=(0.5, 2))
    rotate_aug = torchvision.transforms.RandomRotation(degrees=(-30,30),expand=False)
    augs = torchvision.transforms.Compose([
        color_aug, shape_aug,rotate_aug])
    Augment(source_path,destnation_path)
    #Aug_Test(source_path,destnation_path)

    # path1 = source_path
    # path2 = destnation_path
    # path3 = 'D:\数据集\古文字数据集\JiDaTop800_plus/'
    # Hebin(path1,path2,path3)

