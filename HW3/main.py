import sys
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel
from UI import Ui_MainWindow
from swapAlg import *
import shutil
import os

class MyMainWindow(QMainWindow, Ui_MainWindow):
    FACE_PICTURE_PATH = ""
    HEAD_PICTURE_PATH = ""
    RESULT_PICTURE_PATH = ""

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 初始化照片方框的背景图片
        self.faceLabel.setStyleSheet("border-image:url(./icons/initial1.png);")
        self.headLabel.setStyleSheet("border-image:url(./icons/initial1.png);")
        self.resultLabel.setStyleSheet("border-image:url(./icons/initial2.png);")
	
        # 设置点击鼠标事件
        self.facePushButton.clicked.connect(self.getFaceImage)
        self.headPushButton.clicked.connect(self.getHeadImage)
        self.savePushButton.clicked.connect(self.saveImage)
        self.clearPushButton.clicked.connect(self.clearImage)
        self.swapPushButton.clicked.connect(self.swapFace)
        self.exitPushButton.clicked.connect(self.exitProgram)

    # 获取脸部图片
    def getFaceImage(self):
    	# 选择图片，默认文件夹为test文件夹，可选类型为png图片和jpg图片
        pathname, filetype = QFileDialog.getOpenFileName(self,
                  "Select Image",
                  "./test/",
                  "Image Files (*.png *.jpg)")
        # 如果没有选择图片，不执行下面的代码
        if pathname == "":
            return
		# 将图片展示在faceLabel上
        self.faceLabel.setStyleSheet("border-image:url(" + pathname + ");")
        # 记录脸部图片路径
        self.FACE_PICTURE_PATH = pathname

    # 获取头部图片
    def getHeadImage(self):
        # 选择图片，默认文件夹为test文件夹，可选类型为png图片和jpg图片
        pathname, filetype = QFileDialog.getOpenFileName(self,
                  "Select Image",
                  "./test/",
                  "Image Files (*.png *.jpg)")
        # 如果没有选择图片，不执行下面的代码
        if pathname == "":
            return
        # 将图片展示在headLabel上
        self.headLabel.setStyleSheet("border-image:url(" + pathname + ");")
        # 记录头部图片路径
        self.HEAD_PICTURE_PATH = pathname

    # 保存换脸结果
    def saveImage(self):
        # 如果没有换脸结果，那么不保存图
        if self.RESULT_PICTURE_PATH == "":
            _translate = QtCore.QCoreApplication.translate
            self.resultLabel.setStyleSheet("border-image:url(./icons/initial2.png);")
            self.resultLabel.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; font-weight:600;\">There is<br>No result!</span></p></body></html>"))
            return
        # 选择路径保存换脸图片
        image = Image.open("./tmp/tmp_result.png")
        pathname, filetype = QFileDialog.getSaveFileName(self,
                  "Save Image",
                  "./",
                  "Image Files (*.png *.jpg)")
        # 如果没有选择路径，不执行下面的代码
        if pathname == "":
            return
        image.save(pathname)

    # 进行换脸然后显示结果
    def swapFace(self):
        _translate = QtCore.QCoreApplication.translate
        # 如果没有选择脸部图片或头部图片，那么不换脸
        if self.FACE_PICTURE_PATH == "" or self.HEAD_PICTURE_PATH == "":
            self.resultLabel.setStyleSheet("border-image:url(./icons/initial2.png);")
            self.resultLabel.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; font-weight:600;\">Please choose<br>Face first!</span></p></body></html>"))
            return

        try:
            # 初始化中间的结果图片框
            self.resultLabel.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; font-weight:600;\"></span></p></body></html>"))
            # 根据图像的路径名读入图像，将图像转为3通道BGR彩色图像
            head_image = cv2.imread(self.HEAD_PICTURE_PATH, cv2.IMREAD_COLOR)
            face_image = cv2.imread(self.FACE_PICTURE_PATH, cv2.IMREAD_COLOR)
            # 获取图像的特征点
            head_landmarks = acquire_landmarks(head_image)
            face_landmarks = acquire_landmarks(face_image)
            # 获取头部图片遮罩和脸部图片遮罩
            headshade = acquire_shade(head_image, head_landmarks)
            faceshade = acquire_shade(face_image, face_landmarks)
            # 获取仿射变换矩阵
            aff_tra_matrix = acquire_aff_tra_matrix(head_landmarks, face_landmarks)
            # 利用仿射变换将脸部图像遮罩映射到头部图片中，生成符合头部图片坐标的新脸部遮罩
            warpAffine_shade = warpAffine_face(faceshade, aff_tra_matrix, head_image)
            # 利用仿射变换将脸部图像映射到头部图片中
            warpAffine_image = warpAffine_face(face_image, aff_tra_matrix, head_image)
            # 得到修正边缘之后的图像
            revise_edge_image = revise_edge(head_image, warpAffine_image, head_landmarks)
            # 将符合头部照片的新的脸部图片遮罩和头部图片遮罩结合为一个，尽可能表现出头部图片遮罩的特性
            combined_shade = numpy.max([headshade, warpAffine_shade], 0)
            # 应用遮罩，输出换脸图像
            result_image = head_image * (1.0 - combined_shade) + revise_edge_image * combined_shade
            # 在缓存文件夹中保存换脸图片
            cv2.imwrite("./tmp/tmp_result.png", result_image)
            # 设置换脸图片
            self.resultLabel.setStyleSheet("border-image:url(./tmp/tmp_result.png);")
            # 记录图片路径
            self.RESULT_PICTURE_PATH = "./tmp/tmp_result.png"
        # 捕捉多于一个人脸的异常
        except MoreThanOneFaces:
            _translate = QtCore.QCoreApplication.translate
            self.resultLabel.setStyleSheet("border-image:url(./icons/initial2.png);")
            self.resultLabel.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:24pt; font-weight:600;\">More than<br>One faces!</span></p></body></html>"))
            self.RESULT_PICTURE_PATH = ""
        # 捕捉没有人脸的异常
        except ZeroFaces:
            _translate = QtCore.QCoreApplication.translate
            self.resultLabel.setStyleSheet("border-image:url(./icons/initial2.png);")
            self.resultLabel.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:30pt; font-weight:600;\">Zero<br>Face!</span></p></body></html>"))
            self.RESULT_PICTURE_PATH = ""

    # 清空所有照片
    def clearImage(self):
        self.faceLabel.setStyleSheet("border-image:url(./icons/initial1.png);")
        self.headLabel.setStyleSheet("border-image:url(./icons/initial1.png);")
        self.resultLabel.setStyleSheet("border-image:url(./icons/initial2.png);")

        self.FACE_PICTURE_PATH = ""
        self.HEAD_PICTURE_PATH = ""
        self.RESULT_PICTURE_PATH = ""

    # 退出程序时，清空缓存
    def exitProgram(self):
        shutil.rmtree("tmp")
        os.mkdir("tmp")
        QApplication.instance().quit()

if __name__ == "__main__":
	# 解决兼容性问题
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
