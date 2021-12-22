import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from hw2_ui import Ui_MainWindow
import cv2
import torch
from torchsummary import summary
from random import random
from dataset import DogAndCat
from torchvision import transforms
from matplotlib import pyplot as plt


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_8.clicked.connect(self.showModel)
        self.ui.pushButton_7.clicked.connect(self.showTensorboard)
        self.ui.pushButton_6.clicked.connect(self.test)
        self.ui.pushButton_9.clicked.connect(self.dataAugmantation)

    def showModel(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=2)
        if torch.cuda.is_available():
            model = model.cuda()
        summary(model, (3, 224, 224))

    def showTensorboard(self):
        img = cv2.imread('origin_resnet50.png')
        cv2.namedWindow('Tensorboard', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tensorboard', 1200, 600)
        cv2.imshow('Tensorboard', img)


    def test(self):
        classes = ('Dog', 'Cat')
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=2)
        if torch.cuda.is_available():
            model = model.cuda()
        model.load_state_dict(torch.load('resnet50_aug.pth'))
        test_data = DogAndCat(img_dir='./dog_and_cat', transform=transforms.ToTensor(), mode='test')
        index = int(self.ui.lineEdit_2.text())
        pic = test_data[index][0].transpose(0, 2).transpose(0, 1).cpu().detach().numpy()
        data = torch.unsqueeze(test_data[index][0], 0)
        if torch.cuda.is_available():
            data = data.cuda()
        _, pred = model(data).max(1)
        plt.figure()
        plt.title(classes[pred])
        plt.imshow(pic)
        plt.axis('off')
        plt.show()


    def randomHorizontalFlip(image):
        prob = 0.5
        if (random() <= prob):
            image = image.flip(-1)
        return image

    def dataAugmantation(self):
        img = cv2.imread('compare.png')
        cv2.imshow('Compare', img)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
