import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from hw2_ui import Ui_MainWindow
import os
import cv2
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.findCorners)
        self.ui.pushButton_2.clicked.connect(self.findIntrinsic)
        self.ui.pushButton_5.clicked.connect(self.findExtrinsic)
        self.ui.pushButton_3.clicked.connect(self.findDistortion)
        self.ui.pushButton_4.clicked.connect(self.showResult)


    def findCorners(self):
        path = 'Dataset_OpenCvDl_Hw2/Q2_Image/'
        list = os.listdir(path)
        output = []
        for pic in list:
            img = cv2.imread(path+pic)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                cv2.drawChessboardCorners(img, (11, 8), corners, ret)
                output.append(img)
            else:
                print('error')
        cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Window', 600, 600)
        for i in output:
            cv2.imshow('Window', i)
            cv2.waitKey(500)
        cv2.destroyAllWindows()

    def findIntrinsic(self):
        path = 'Dataset_OpenCvDl_Hw2/Q2_Image/'
        list = os.listdir(path)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        for pic in list:
            img = cv2.imread(path + pic)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print('error')
        _, mtx, _, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print('Intrisic:')
        print(mtx)

    def findExtrinsic(self):
        path = 'Dataset_OpenCvDl_Hw2/Q2_Image/'
        list = os.listdir(path)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        for pic in list:
            img = cv2.imread(path + pic)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print('error')
        _, _, _, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        j = int(self.ui.lineEdit.text())-1
        if j not in range(0, 15):
            print('error')
            return
        dst, _ = cv2.Rodrigues(rvecs[j])
        dst = dst.tolist()
        for i in range(len(dst)):
            dst[i].append(tvecs[j][i][0])
        print('Extrinsic:')
        print(np.array(dst))

    def findDistortion(self):
        path = 'Dataset_OpenCvDl_Hw2/Q2_Image/'
        list = os.listdir(path)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        for pic in list:
            img = cv2.imread(path + pic)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print('error')
        _, _, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print('Distortion:')
        print(dist)

    def showResult(self):
        path = 'Dataset_OpenCvDl_Hw2/Q2_Image/'
        list = os.listdir(path)
        output = []
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        final = []
        for pic in list:
            img = cv2.imread(path + pic)
            output.append(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print('error')
        _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        for img in output:
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            img = np.concatenate((img, dst), axis=1)
            final.append(img)
        cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Window', 1200, 600)
        for i in final:
            cv2.imshow('Window', i)
            cv2.waitKey(500)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
