import cv2
import dlib
import numpy

# 提取图像中的面部特征点，将图像转换为一个矩阵数组，矩阵中每个元素对应每一行的一个x，y坐标
def acquire_landmarks(image):
    # 首先从dlib库中获取一个人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 然后利用这个人脸检测器对图像画人脸框，返回的是一个人脸检测矩形框的4点坐标
    faces = detector(image, 1)

    # 如果检测器检测到人脸的个数为0，说明图像没有人脸，抛出一个异常
    if len(faces) == 0:
        raise ZeroFaces
    # 如果检测器检测到人脸的个数大于1，说明图像中不止一个人脸，抛出一个异常
    if len(faces) > 1:
        raise MoreThanOneFaces

    # 从dlib库中通过已经预训练好的关键点模型，返回一个人脸关键点预测器，用来标记人脸关键点
    predictor = dlib.shape_predictor("./predictor.dat")
    
    # 定位人脸关键点，faces[0]是开始内部形状预测的边界框，最后返回关键点的位置
    shape = predictor(image, faces[0])
    matrix = numpy.matrix([[point.x, point.y] for point in shape.parts()])
    return matrix

# 得到人脸遮罩
def acquire_shade(image, landmarks):
    shape = image.shape[:2]
    image = numpy.zeros(shape, numpy.float64)

    # 获取dlib库中识别眉毛和眼睛的识别点区域
    brow_and_eye_points = list(range(17, 48))
    # 获取dlib库中识别鼻子和嘴巴的识别点区域
    nose_and_mouth_points = list(range(27, 35)) + list(range(48, 61))

    # 利用OpenCV库中convexHull函数得到眉毛和眼睛的凸包，从而得到眉毛和眼睛的轮廓
    brow_and_eye_landmarks = cv2.convexHull(landmarks[brow_and_eye_points])
    # 得到鼻子和嘴巴的凸包，从而得到鼻子和嘴巴的轮廓
    nose_and_mouth_landmarks = cv2.convexHull(landmarks[nose_and_mouth_points])

    # 填充眉毛和眼睛、鼻子和嘴巴的轮廓
    cv2.fillConvexPoly(image, brow_and_eye_landmarks, 1)
    cv2.fillConvexPoly(image, nose_and_mouth_landmarks, 1)

    # 将矩阵转换为图像表示
    image = numpy.array([image, image, image]).transpose((1, 2, 0))
    
    # 使用高斯滤波对图像进行降噪处理
    plume_amount = 11
    # 画出的两个区域的轮廓向遮罩的边缘外部羽化扩展plume_amount个像素，可以隐藏不连续的区域
    image = (cv2.GaussianBlur(image, (plume_amount, plume_amount), 0) > 0) * 1.0
    return cv2.GaussianBlur(image, (plume_amount, plume_amount), 0)

# 获取仿射变换矩阵
def acquire_aff_tra_matrix(head_landmarks, face_landmarks):
    # 获取要转换的识别点区域
    head_landmarks = head_landmarks[list(range(17, 61))]
    face_landmarks = face_landmarks[list(range(17, 61))]

    # 先将获取到的头部图像特征点矩阵的数字类型转换为浮点数，以得到更精确的列均值
    head_landmarks = head_landmarks.astype(numpy.float64)
    # 对头部图像特征点矩阵的各列求均值，得到各列均值，即矩心
    col_aver1 = numpy.mean(head_landmarks, 0)
    # 按照普式分析法，各列减去矩心
    head_landmarks = head_landmarks - col_aver1
    # 对矩阵的各列求标准差，得到各列的标准差
    SD1 = numpy.std(head_landmarks)
    # 各列除以标准差，按照标准差缩放，减少了有问题的组件的缩放偏差
    head_landmarks = head_landmarks / SD1

    # 获取到的脸部特征点矩阵也按照上面的过程处理
    face_landmarks = face_landmarks.astype(numpy.float64)
    col_aver2 = numpy.mean(face_landmarks, 0)
    face_landmarks = face_landmarks - col_aver2
    SD2 = numpy.std(face_landmarks)
    face_landmarks = face_landmarks / SD2

    col_aver1 = col_aver1.T
    col_aver2 = col_aver2.T
    SD_div = SD2 / SD1 * 1.0

    # 使用奇异值分解计算旋转部分，返回三个矩阵，分别为左奇异值、奇异值、右奇异值
    head_landmarks_tra = head_landmarks.T
    u, s, vh = numpy.linalg.svd(head_landmarks_tra * face_landmarks)

    # 公式假设矩阵在右边，有行向量，解决方案要求矩阵在左边，有列向量，所以进行转置
    R = u * vh
    R = R.T

    # 返回根据普式分析法得到的仿射变换矩阵
    return numpy.vstack([numpy.hstack((SD_div * R, col_aver2 - SD_div * R * col_aver1)), numpy.matrix([0., 0., 1.])])

# 利用仿射变换映射图片
def warpAffine_face(face_image, aff_tra_matrix, head_image):
    warpAffine_image = numpy.zeros(head_image.shape, face_image.dtype)
    width = head_image.shape[1]
    height = head_image.shape[0]
    cv2.warpAffine(face_image, aff_tra_matrix[:2],
                   (width, height), warpAffine_image,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return warpAffine_image

# 修正由于图像肤色和光线的不同导致脸部覆盖区域边缘的不连续问题
def revise_edge(head_image, face_image, head_landmarks):
    # 取左眼的识别点矩阵的矩心
    left_eye_points_col_aver = numpy.mean(head_landmarks[list(range(42, 48))], 0)
    # 取右眼识别点矩阵的矩心
    right_eye_points_col_aver = numpy.mean(head_landmarks[list(range(36, 42))], 0)
    # 取两个矩心之差，取范数后乘以一个模糊量系数得到一个为奇数的高斯内核
    eye_points_col_aver_sub = left_eye_points_col_aver - right_eye_points_col_aver
    blur_fra = 0.6
    kernel_size = int(blur_fra * numpy.linalg.norm(eye_points_col_aver_sub)) + 1
    if kernel_size % 2 == 0:
        kernel_size = kernel_size - 1
    # 使用高斯滤波函数得到两个图像的高斯模糊值
    head_image_blur = cv2.GaussianBlur(head_image, (kernel_size, kernel_size), 0)
    face_image_blur = cv2.GaussianBlur(face_image, (kernel_size, kernel_size), 0)
    face_image_blur = face_image_blur + 128 * (face_image_blur <= 1.0)
    
    # 将使用到的值都转换为float64类型的
    face_image = face_image.astype(numpy.float64)
    head_image_blur = head_image_blur.astype(numpy.float64)
    face_image_blur = face_image_blur.astype(numpy.float64)

    # 使用脸部图片乘以头部图片模糊值，再除以脸部图片模糊值，得到修正边缘后的图像
    return face_image * head_image_blur / face_image_blur

# 当图像中没有人脸时，抛出这个异常
class ZeroFaces(Exception):
    pass

# 当图像中人脸个数大于1时，抛出这个异常
class MoreThanOneFaces(Exception):
    pass