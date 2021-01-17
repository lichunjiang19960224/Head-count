from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class CrowdCount(QtWidgets.QMainWindow):

    def __init__(self):
        super(CrowdCount, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 400)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.retranslateUi(MainWindow)

        self.img = QtWidgets.QLabel(self.centralWidget)
        self.img.setGeometry(QtCore.QRect(50, 50, 300, 200))
        self.img.setObjectName("img")
        self.img.setPixmap(QPixmap('forQTtest.png'))
        self.img.setScaledContents(True)

        self.startbutton = QtWidgets.QPushButton(self.centralWidget)
        self.startbutton.setCheckable(True)
        self.startbutton.move(50, 280)
        self.startbutton.setText("开始")
        self.startbutton.clicked[bool].connect(self.setText)

        self.endbutton = QtWidgets.QPushButton(self.centralWidget)
        self.endbutton.setCheckable(True)
        self.endbutton.move(276, 280)
        self.endbutton.setText("结束")
        self.endbutton.clicked[bool].connect(self.setText)

        self.lb1 = QtWidgets.QLabel(self.centralWidget)
        self.lb1.setGeometry(QtCore.QRect(50, 160, 300, 200))
        self.lb1.setObjectName("text")
        self.lb1.setText("显示文本")
        self.lb1.setAlignment(Qt.AlignCenter)

        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(276, 320, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("选取视频")
        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.openfile)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "人群计数"))

    def openfile(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'Excel files(*.xlsx , *.xls)')

    def setText(self, pressed):

        source = self.sender()

        if pressed:
            pass
        else:
            pass

        if source.text() == "开始":
            pass
        elif source.text() == "结束":
            pass


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = CrowdCount()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
