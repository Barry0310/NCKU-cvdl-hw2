from torch.utils.data import Dataset
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
from random import random


class DogAndCat(Dataset):
    def __init__(self, img_dir, transform=None, mode='train'):
        super(DogAndCat, self).__init__()
        dog = []
        cat = []
        for i in os.listdir(img_dir+'/Dog/'):
            dog.append(img_dir+'/Dog/'+i)
        for i in os.listdir(img_dir+'/Cat/'):
            cat.append(img_dir+'/Cat/'+i)
        self.train = [{'feature': x, 'label': 0} for x in dog[:int(len(dog)*0.8)]] + \
                     [{'feature': x, 'label': 1} for x in cat[:int(len(cat)*0.8)]]
        self.valid = [{'feature': x, 'label': 0} for x in dog[int(len(dog)*0.8):int(len(dog)*0.9)]] + \
                     [{'feature': x, 'label': 1} for x in cat[int(len(cat)*0.8):int(len(dog)*0.9)]]
        self.test = [{'feature': x, 'label': 0} for x in dog[int(len(dog)*0.9):]] + \
                    [{'feature': x, 'label': 1} for x in cat[int(len(cat)*0.9):]]
        self.transform = transform
        self.mode = mode

    def randomHorizontalFlip(self, image):
        prob = 0.5
        if (random() <= prob):
            image = image.flip(-1)
        return image

    def __len__(self):
        dataset = None
        if self.mode == 'train':
            dataset = self.train
        elif self.mode == 'valid':
            dataset = self.valid
        elif self.mode == 'test':
            dataset = self.test
        return len(dataset)

    def __getitem__(self, idx):
        dataset = None
        if self.mode == 'train':
            dataset = self.train
        elif self.mode == 'valid':
            dataset = self.valid
        elif self.mode == 'test':
            dataset = self.test
        image = plt.imread(dataset[idx]['feature'])
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = np.uint8(image)
        label = dataset[idx]['label']
        if self.transform:
            image = self.transform(image)
        image = self.randomHorizontalFlip(image)
        return image, label
