# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(300, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 10, 261, 571))
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox_2)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(19, 29, 221, 521))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_8 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_8.setObjectName("pushButton_8")
        self.verticalLayout.addWidget(self.pushButton_8)
        self.pushButton_7 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout.addWidget(self.pushButton_7)
        self.pushButton_6 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout.addWidget(self.pushButton_6)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout.addWidget(self.lineEdit_2)
        self.pushButton_9 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_9.setObjectName("pushButton_9")
        self.verticalLayout.addWidget(self.pushButton_9)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "cvdl hw2"))
        self.groupBox_2.setTitle(_translate("MainWindow", "5. Dogs and Cats classification"))
        self.pushButton_8.setText(_translate("MainWindow", "1. Show Model Structure"))
        self.pushButton_7.setText(_translate("MainWindow", "2. Show Tensorboard"))
        self.pushButton_6.setText(_translate("MainWindow", "3. Test"))
        self.pushButton_9.setText(_translate("MainWindow", "4. Data Augmantation"))
