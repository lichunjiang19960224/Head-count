# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1096, 653)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(10, 10, 960, 544))
        self.image.setFrameShape(QtWidgets.QFrame.Box)
        self.image.setFrameShadow(QtWidgets.QFrame.Plain)
        self.image.setLineWidth(3)
        self.image.setText("")
        self.image.setObjectName("image")
        self.single = QtWidgets.QLabel(self.centralwidget)
        self.single.setGeometry(QtCore.QRect(1010, 30, 16, 16))
        self.single.setFrameShape(QtWidgets.QFrame.Box)
        self.single.setText("")
        self.single.setObjectName("single")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1030, 30, 31, 16))
        self.label.setObjectName("label")
        self.text = QtWidgets.QTextBrowser(self.centralwidget)
        self.text.setGeometry(QtCore.QRect(980, 60, 111, 31))
        self.text.setObjectName("text")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1096, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "信号"))