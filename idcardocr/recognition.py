import cv2
import numpy as np
import pytesseract

def reverse(img):
    '''
    像素反转
    :param img:
    :return:
    '''
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            if(img[i][j]>100):
                img[i][j]=0
            else:
                img[i][j]=255
    return img

def denoise(binary,n):
    '''
    对投影后的向量进行降噪
    :param binary: 投影得到的数组
    :param n: 单行噪点小于n时，设为0
    :return:
    '''
    for i in range(len(binary)):
        if(binary[i]<n):
            binary[i] = 0
    return binary

def denoise_2d(binary):
    """
    当某个像素点值为255,但是其上下左右像素值都是0的话,认做为孤立像素点,将其值也设置为0
    :param binary:二维的二值图
    :return:binary
    """
    for i in range(1, len(binary) - 1):
        for j in range(1, len(binary[0]) - 1):
            if binary[i][j] == 255:
                # if条件成立的话,则说明当前像素点为孤立点
                if binary[i - 1][j] == binary[i + 1][j] == binary[i][j - 1] == binary[i][j + 1] == 0:
                    binary[i][j] = 0
    return binary

def hProject(binary):
    '''
    处理水平方向投影
    :param binary: 二维的二值图
    :return: 长度为h的一维数组和投影图
    '''
    h, w = binary.shape
    # 水平投影
    hprojection = np.zeros(binary.shape, dtype=np.uint8)
    # 创建h长度都为0的数组
    h_h = [0]*h
    for j in range(h):
        for i in range(w):
            if binary[j,i] == 0:
                h_h[j] += 1
    # 画出投影图
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j,i] = 255
    # cv2.imshow('hpro', hprojection)
    return h_h,hprojection

def vProject(binary):
    '''
    处理垂直方向投影
    :param binary: 二维的二值图
    :return: 长度为w的一维数组和投影图
    '''
    h, w = binary.shape
    # 垂直投影
    vprojection = np.zeros(binary.shape, dtype=np.uint8)
    # 创建 w 长度都为0的数组
    w_w = [0]*w
    for i in range(w):
        for j in range(h):
            if binary[j, i ] == 0:
                w_w[i] += 1
    for i in range(w):
        for j in range(w_w[i]):
            vprojection[j,i] = 255
    # cv2.imshow('vpro', vprojection)
    return w_w,vprojection

def getPosition_noProcess(cropImg, isDenoise = [0,0]):
    '''
    对未做水平投影的身份证号区域进行处理
    :param cropImg: 身份证号区域的二值图
    :param isDenoise: 默认【0，0】不进行降噪处理，数字是降噪力度
    :return: 有信息的文字位置
    '''
    h, w = cropImg.shape
    # 先水平投影取的文字高度位置
    h_h, _ = hProject(cropImg)
    if isDenoise[0] > 0:
        h_h = denoise(h_h, isDenoise[0])
    start = 0
    h_start, h_end = [], []
    # 根据水平投影进行分割
    for i in range(len(h_h)):
        if h_h[i] > 0 and start == 0:
            h_start.append(i)
            start = 1
        if h_h[i] ==0 and start == 1:
            h_end.append(i)
            start = 0
    # 垂直投影进行分割
    position = []
    for i in range(len(h_start)):
        cropImg_w = cropImg[h_start[i]:h_end[i], 0:w]
        w_w, _ = vProject(cropImg_w)
        if isDenoise[1] > 0:
            w_w = denoise(w_w, isDenoise[0])
        wstart , wend, w_start, w_end = 0, 0, 0, 0
        for j in range(len(w_w)):
            if w_w[j] > 0 and wstart == 0:
                w_start = j
                wstart = 1
                wend = 0
            if w_w[j] ==0 and wstart == 1:
                w_end = j
                wstart = 0
                wend = 1
            # 当确认了起点和终点之后保存坐标
            if wend == 1:
                position.append([w_start, h_start[i], w_end, h_end[i]])
                wend = 0
    return position

def getPosition(cropImg, isDenoise = [0]):
    '''
    对已经做过水平投影的身份证号区域进行处理
    :param cropImg: 身份证号区域的二值图
    :param isDenoise: 默认【0】不进行降噪处理，数字是降噪力度
    :return: 有信息的文字位置
    '''
    h, w = cropImg.shape
    position = []
    # 只做垂直投影
    w_w, _ = vProject(cropImg)
    if isDenoise[0] > 0:
        w_w = denoise(w_w, isDenoise[0])
    wstart, wend, w_start, w_end = 0, 0, 0, 0
    for j in range(len(w_w)):
        if w_w[j] > 0 and wstart == 0:
            w_start = j
            wstart = 1
            wend = 0
        if w_w[j] == 0 and wstart == 1:
            w_end = j
            wstart = 0
            wend = 1
        # 当确认了起点和终点之后保存坐标
        if wend == 1:
            position.append([w_start, 0, w_end, h])
            wend = 0
    return position

def kuangxuan(img,position):
    '''
    画图框选
    :param img: 需要框选的图片
    :param position: 位置信息
    :return: 框选好的img
    '''
    for p in position:
        cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)
    return img

def kuangxuanID_h(cropImg,img):
    h_h = hProject(cropImg)
    start = 0
    h_start, h_end = [], []
    for i in range(len(h_h)):
        if h_h[i] > 0 and start == 0:
            h_start.append(i)
            start = 1
        if h_h[i] == 0 and start == 1:
            h_end.append(i)
            start = 0

    h, w = cropImg.shape
    position = []
    # 只做垂直投影
    w_w = vProject(cropImg)

    wstart, wend, w_start, w_end = 0, 0, 0, 0
    for j in range(len(w_w)):
        if w_w[j] > 0 and wstart == 0:
            w_start = j
            wstart = 1
            wend = 0
        if w_w[j] == 0 and wstart == 1:
            w_end = j
            wstart = 0
            wend = 1
        # 当确认了起点和终点之后保存坐标
        if wend == 1:
            position.append([w_start, h_start[0], w_end, h_end[0]])
            wend = 0

    # # 确定分割位置
    # for p in position:
    #     print(p)
    #     cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)

    return position

def kuangxuanID(cropImg,img):
    h, w = cropImg.shape
    position = []
    # 只做垂直投影
    w_w = vProject(cropImg)

    wstart, wend, w_start, w_end = 0, 0, 0, 0
    for j in range(len(w_w)):
        if w_w[j] > 0 and wstart == 0:
            w_start = j
            wstart = 1
            wend = 0
        if w_w[j] == 0 and wstart == 1:
            w_end = j
            wstart = 0
            wend = 1

        # 当确认了起点和终点之后保存坐标
        if wend == 1:
            position.append([w_start, 0, w_end, h])
            wend = 0

    # 确定分割位置
    for p in position:
        cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)

    return img

def kuangxuanName(cropImg,img):
    h, w = cropImg.shape
    position = []
    # 只做垂直投影
    w_w = vProject(cropImg)
    wstart, wend, w_start, w_end = 0, 0, 0, 0
    for j in range(len(w_w)):
        if w_w[j] > 0 and wstart == 0:
            w_start = j
            wstart = 1
            wend = 0
        if w_w[j] == 0 and wstart == 1:
            w_end = j
            wstart = 0
            wend = 1
        # 当确认了起点和终点之后保存坐标
        if wend == 1:
            position.append([w_start, 0, w_end, h])
            wend = 0
    # 确定分割位置
    for p in position:
        img_name = img[p[1]:p[3],p[0]:p[2]]
        img_name = cv2.cvtColor(img_name, cv2.COLOR_BGR2GRAY)
        img_name = cv2.threshold(img_name, 140, 255, cv2.THRESH_BINARY)[1]
        cv2.resize(img_name, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        cv2.imshow("img_name",img_name)
        print(pytesseract.image_to_string(img_name, lang='chi_sim', config="--psm 7"))
        cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)

    # for i in range(3):
    #     if i>2:
    #         break
    #
    #     # print([position[0][0]+i*h, position[0][1],position[0][0]+(i+1)*h, position[0][3]])
    #     cv2.rectangle(img, (position[0][0]+i*(h)+i, position[0][1]), (position[0][0]+(i+1)*(h+3)+i, position[0][3]), (0, 0, 255), 2)


    return img

def img_preprocess_two(img,coefficients = [0, 1, 1],size=(10,5),isReverse=True):
    '''
        预处理
        :param img:要预处理的图片
        :param threshold_param: 二值化参数
        :param size: 膨胀系数
        :return: 处理好的图片
    '''
    img_denoise = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # 降噪
    # 转换成灰度图
    # coefficients = [0, 1, 1]
    m = np.array(coefficients).reshape((1, 3))
    img_gray = cv2.transform(img_denoise, m)
    # 反向二值化图像
    img_binary_inv = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)
    # 自定义的降噪代码,将孤立像素点去除
    img_binary_inv = denoise_2d(img_binary_inv)
    # 膨胀操作,将图像变成一个个矩形框，用于下一步的筛选，找到身份证号码对应的区域
    ele = cv2.getStructuringElement(cv2.MORPH_RECT, size)  # 经过测试,(10,5)为水平和垂直方向的膨胀size
    th = cv2.dilate(img_binary_inv, ele, iterations=1)
    if isReverse:
        th = reverse(th)
    # 针对不同图需要调整阈值
    # cv2.imshow('gray', th)
    return th

def img_preprocess(img,threshold_param =80,size=(10,5)):
    '''
    预处理
    :param img:要预处理的图片
    :param threshold_param: 二值化参数
    :param size: 膨胀系数
    :return: 处理好的图片
    '''
    # 灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, img_gray = cv2.threshold(gray, threshold_param, 255, cv2.THRESH_BINARY)
    # 反向二值化图像
    img_binary_inv = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)
    # 自定义的降噪代码,将孤立像素点去除
    img_binary_inv = denoise_2d(img_binary_inv)
    # 膨胀操作,将图像变成一个个矩形框，用于下一步的筛选，找到身份证号码对应的区域
    ele = cv2.getStructuringElement(cv2.MORPH_RECT, size)  # 经过测试,(10,5)为水平和垂直方向的膨胀size
    th = cv2.dilate(img_binary_inv, ele, iterations=1)
    th = reverse(th)
    # 针对不同图需要调整阈值
    # cv2.imshow('gray', th)
    return th

def cutIdcard(th,img):
    '''
    剪裁身份证号
    :param th:预处理之后的图片
    :param img: 原图
    :return:
    '''
    h, w = th.shape
    # step1水平方向投影，切分身份证号
    h_h, im = hProject(th)
    cv2.imshow("im",im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    start = 0
    h_start, h_end = [], []
    for i in range(len(h_h)):
        if h_h[i] > 0 and start == 0:
            h_start.append(i)
            start = 1
        if h_h[i] == 0 and start == 1:
            h_end.append(i)
            start = 0
    h_length = len(h_start)

    # 抛去身份证号部分
    exceptIdCardImg = th[0:h_start[h_length - 1], 0:w]
    exceptIdCardImg_origin= img[0:h_start[h_length - 1], 0:w]
    # 身份证号部分
    idCardImg = th[h_start[h_length - 1]:h_end[h_length-1], 0:w]
    idCardImg_origin = img[h_start[h_length - 1]:h_end[h_length-1], 0:w]

    return exceptIdCardImg,exceptIdCardImg_origin,idCardImg,idCardImg_origin

def cutImg(th, img):
    h, w = th.shape
    # 垂直剪裁去掉人头
    w_w, _ = vProject(th)
    w_w = denoise(w_w,10)
    wstart = 0
    w_start, w_end = [], []
    for i in range(len(w_w)):
        if w_w[i] > 0 and wstart == 0:
            w_start.append(i)
            wstart = 1
        if w_w[i] == 0 and wstart == 1:
            w_end.append(i)
            wstart = 0
    w_length = len(w_start)
    # 提取图片垂直方向二值图
    faceImg = th[0:h, w_start[w_length - 1]:w_end[w_length-1]]
    # 提取图片垂直方向彩图
    faceImg_origin = img[0:h, w_start[w_length - 1]:w_end[w_length - 1]]
    # cv2.imshow("faceImg_origin", faceImg_origin)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    expectFaceImg = th[0:h, 0:w_start[w_length - 1]]
    expectFaceImg_origin = img[0:h, 0:w_start[w_length - 1]]

    # 对人头进行水平切割
    h_h, _ = hProject(faceImg)
    start = 0
    h_start, h_end = [], []
    for i in range(len(h_h)):
        if h_h[i] > 0 and start == 0:
            h_start.append(i)
            start = 1
        if h_h[i] == 0 and start == 1:
            h_end.append(i)
            start = 0
    faceImg = faceImg[h_start[0]:h_end[len(h_start)-1], :]
    faceImg_origin = faceImg_origin[h_start[0]:h_end[len(h_start)-1], :]
    # cv2.imshow("faceImg_origin", faceImg_origin)

    return expectFaceImg_origin,faceImg_origin

def cutName(th,img):
    # 提取姓名
    h, w = th.shape
    # 根据水平投影获取垂直分割
    h_h, img1 = hProject(th)
    start = 0
    h_start, h_end = [], []
    for i in range(len(h_h)):
        if h_h[i] > 0 and start == 0:
            h_start.append(i)
            start = 1
        if h_h[i] == 0 and start == 1:
            h_end.append(i)
            start = 0
    # if(h_end[0]-h_start[0]>22):
    #     nameImg = th[h_start[0]:h_start[0]+22, 0:w]
    #     nameImg_origin = img[h_start[0]:h_start[0]+22, 0:w]
    # else:
    nameImg = th[h_start[0]:h_end[0], 0:w]
    nameImg_origin = img[h_start[0]:h_end[0], 0:w]
    return nameImg, nameImg_origin

def cut(th):

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    h, w = th.shape

    # step1水平方向投影，切分身份证号
    h_h = hProject(th)
    start = 0
    h_start, h_end = [], []
    for i in range(len(h_h)):
        if h_h[i] > 0 and start == 0:
            h_start.append(i)
            start = 1
        if h_h[i] == 0 and start == 1:
            h_end.append(i)
            start = 0
    h_length = len(h_start)
    # 抛去身份证号部分
    exceptIdCardImg = th[0:h_start[h_length - 1], 0:w]
    exceptIdCardImg_origin= img[0:h_start[h_length - 1], 0:w]

    # 身份证号部分
    idCardImg = th[h_start[h_length - 1]:h_end[h_length-1], 0:w]
    idCardImg_origin = img[h_start[h_length - 1]:h_end[h_length-1], 0:w]


    # 垂直剪裁去掉人头
    w_w = vProject(exceptIdCardImg)
    w_w = denoise(w_w)
    wstart = 0
    w_start, w_end = [], []
    for i in range(len(w_w)):
        if w_w[i] > 0 and wstart == 0:
            w_start.append(i)
            wstart = 1
        if w_w[i] == 0 and wstart == 1:
            w_end.append(i)
            wstart = 0
    w_length = len(w_start)
    # 提取图片垂直方向二值图
    faceImg = exceptIdCardImg[0:h, w_end[w_length - 1] - 170:w_end[w_length-1]]
    # 提取图片垂直方向彩图
    faceImg_origin = exceptIdCardImg_origin[0:h, w_end[w_length - 1] - 170:w_end[w_length - 1]]


    # 对人头进行水平切割
    h_h = hProject(faceImg)
    start = 0
    h_start, h_end = [], []
    for i in range(len(h_h)):
        if h_h[i] > 0 and start == 0:
            h_start.append(i)
            start = 1
        if h_h[i] == 0 and start == 1:
            h_end.append(i)
            start = 0
    faceImg = faceImg[h_start[0]:h_end[len(h_start)-1], :]
    faceImg_origin = faceImg_origin[h_start[0]:h_end[len(h_start)-1], :]
    # cv2.imshow("faceImg_origin", faceImg_origin)


    # 去掉人脸和身份证的图
    cropImg = exceptIdCardImg[0:h, 0:w_end[w_length - 1] - 170]
    # 提取姓名
    h, w = cropImg.shape
    # 根据水平投影获取垂直分割
    h_h = hProject(cropImg)
    start = 0
    h_start, h_end = [], []
    for i in range(len(h_h)):
        if h_h[i] > 0 and start == 0:
            h_start.append(i)
            start = 1
        if h_h[i] == 0 and start == 1:
            h_end.append(i)
            start = 0

    if(h_end[0]-h_start[0]>22):
        nameImg = cropImg[h_start[0]:h_start[0]+22, 0:w]
        nameImg_origin = img[h_start[0]:h_start[0]+22, 0:w]
    else:
        nameImg = cropImg[h_start[0]:h_end[0], 0:w]
        nameImg_origin = img[h_start[0]:h_end[0], 0:w]

    # 取姓名
    # cv2.imshow("name",nameImg)
    return nameImg, nameImg_origin, idCardImg, idCardImg_origin

def find_number_region(img):
    '''
    在膨胀处理后的img中,找到身份证号码所在矩形区域,
    :param img:
    :return: box
    '''
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    card_number_region = []  # 用来保存最终返回的region
    max_area = 0  # 存储最大的矩形面积,用于筛选出身份证边框所在区域
    for i in range(len(contours)):
        # 返回点集cnt的最小外接矩形，(外接矩形中心坐标(x,y),(外接矩形宽，外接矩形高)，旋转角度)
        rect = cv2.minAreaRect(contours[i])
        box = np.int0(cv2.boxPoints(rect))  # box是外接矩形四个点的坐标,np.int0()用来去除小数点之后的数字
        width, height = rect[1]
        if 0 not in box:  # 剔除一些越界的矩形,如果矩阵坐标中包含0,说明该矩阵不是我们想要找的矩阵框
            if 9 < width / height < 16 or 9 < height / width < 16:
                area = width * height
                if area > max_area:
                    max_area = area
                    card_number_region = box
    return card_number_region

def get_number_img(origin_img, region):
    '''
    根据上一步找到的边框,从原始图像中,裁剪出身份证号码区域的图像
    :param origin_img:
    :param region:
    :return: image
    '''
    # 根据四个点的左边裁剪区域
    h = abs(region[0][1] - region[2][1])
    w = abs(region[0][0] - region[2][0])
    x_s = [i[0] for i in region]
    y_s = [i[1] for i in region]
    x1 = min(x_s)
    y1 = min(y_s)
    return origin_img[y1:y1 + h, x1:x1 + w]

def rotate_image(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    rotate_matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(src, rotate_matrix, (w, h), borderValue=(255, 255, 255))
    return rotate

def degree_trans(theta):
    res = theta / np.pi * 180
    return res

def calc_degree(img):
    mid_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst_image = cv2.Canny(mid_image, 50, 200, 3)
    line_image = img.copy()
    # 通过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高
    lines = cv2.HoughLines(dst_image, 1, np.pi / 180, 15)
    # 排除lines为None的异常情况
    if lines is None:
        return 0
    # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    count = 0
    # 依次画出每条线段
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))
            # 只选角度最小的作为旋转角度
            count += theta
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

    # 对所有角度求平均，这样做旋转效果会更好
    average = count / len(lines)
    angle = degree_trans(average) - 90
    return angle

def horizontal_correct(img,img_resize,n=3):
    degree = calc_degree(img)
    # 在测试中发现,如果扭曲角度角度(大于3),则进行水平矫正,否则不进行矫正
    if abs(degree) > n:
        img_rotate = rotate_image(img, degree)
        img__resize_rotate = rotate_image(img_resize, degree)
        return img_rotate, img__resize_rotate
    return img, img_resize

def sort_contours(cnts,method="left-to-right"):
    reverse=False
    i=0

    if method=="right-to-left" or method =="bottom-to-top":
        reverse=True
    if method=="top-to-bottom" or method=="bottom-to-top":
        i=1

    '''
    cv2.boundingRect(c) 
    返回四个值，分别是x，y，w，h；
    x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    '''
    boundingBoxes=[cv2.boundingRect(c) for c in cnts] #在轮廓信息中找到一个外接矩形
    (cnts,boundingBoxes)=zip(*sorted(zip(cnts,boundingBoxes),key=lambda b:b[1][i],reverse=reverse))
    return cnts,boundingBoxes

def resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    dim=None
    (h,w)=image.shape[:2] #(200,300,3)
    if width is None and height is None:
        return image
    if width is None:
        r=height/float(h)
        dim=(int(w*r),height)
    else:
        r=width/float(w)
        dim=(width,int(h*r))

    resized=cv2.resize(image,dim,interpolation=inter)
    return resized

def nameOcr(img, position):
    res = []
    for p in position:
        img_name = img[p[1]:p[3], p[0]:p[2]]
        img_name = cv2.cvtColor(img_name, cv2.COLOR_BGR2GRAY)
        img_name = cv2.threshold(img_name, 100, 255, cv2.THRESH_BINARY)[1]
        res.append(pytesseract.image_to_string(img_name, lang='chi_sim', config="--psm 7"))
    return res

def loadTmp():
    tmp = cv2.imread(r"E:\class\back\flask\idcardocr\tmp1.png")
    # 转灰度
    ref = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    # 二值图
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

    _, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tmp, refCnts, -1, (0, 0, 255), 3)
    cv2.imshow('tmp', tmp)

    print(np.array(refCnts).shape)
    refCnts = sort_contours(refCnts, method="left-to-right")[0]  # 排序从左到右，从上到下
    digits = {}
    # 遍历每一个轮廓
    for (i, c) in enumerate(refCnts):
        # 计算外接矩形并且resize成合适大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 58))
        if i == 8:
            cv2.imshow("roi", roi)
        # 每一个数字对应一个模板
        digits[i] = roi
    return digits

def ocrId(img, position,digits):
    # 读取输入图像，预处理
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
    img_gray = denoise_2d(img_gray)
    cv2.imshow("img_gray",img_gray)
    position = getPosition_noProcess(img_gray)
    output = []
    j = 0
    for i in range(position.__len__()).__reversed__():
        j+=1
        if j>18:
            break
        p = position[i]
        img_num = img_gray[p[1]:p[3], p[0]:p[2]]
        img_num = cv2.resize(img_num, (57, 58))
        img_num = reverse(img_num)
        cv2.imshow('num', img_num)
        # if i==1:
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # 计算匹配得分
        scores = []
        # 在模板章中计算每一个得分
        for (digit, digiROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(img_num, digiROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # 得到合适的数字
        output.append(str(np.argmax(scores)))
    output.reverse()
    return output

def ocr(img):
    tmp = cv2.imread(r"E:\class\back\flask\idcardocr\tmp1.png")
    # 转灰度
    ref = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
    # 二值图
    ref = cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]

    _, refCnts, hierarchy = cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tmp, refCnts, -1, (0, 0, 255), 3)
    cv2.imshow('tmp', tmp)

    print(np.array(refCnts).shape)
    refCnts = sort_contours(refCnts, method="left-to-right")[0]  # 排序从左到右，从上到下
    digits = {}
    # 遍历每一个轮廓
    for (i, c) in enumerate(refCnts):
        # 计算外接矩形并且resize成合适大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 58))
        if i==8:
            cv2.imshow("roi", roi)
        # 每一个数字对应一个模板
        digits[i] = roi

    # 读取输入图像，预处理
    image = img.copy()

    image = resize(image, height=100)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]

    img_gray = denoise_2d(img_gray)

    position = getPosition_noProcess(img_gray)

    output = []
    for p in position:
        img_num = img_gray[p[1]:p[3],p[0]:p[2]]
        img_num = cv2.resize(img_num, (57, 58))
        img_num = reverse(img_num)
        cv2.imshow('num',img_num)
        # 计算匹配得分
        scores = []
        # 在模板章中计算每一个得分
        for (digit, digiROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(img_num, digiROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # 得到合适的数字
        output.append(str(np.argmax(scores)))

    return output

def main(filePath):
    img = cv2.imread('E:/class/back/flask/static/image/'+filePath)
    img_2 = img.copy()
    # -----------------------------是否需要旋转-------------------------------------------
    img_resize = cv2.resize(img_2, (428, 270), interpolation=cv2.INTER_CUBIC)
    image_preprocessed = img_preprocess_two(img_resize, size=(10, 5), isReverse=False)
    number_region = find_number_region(image_preprocessed)
    image_id_number = get_number_img(img_resize, number_region)
    # 判断是否需要旋转
    image_correct, img = horizontal_correct(image_id_number, img, n=3)
    # --------------------------把身份证分为三部分------------------------------------------
    # 得到身份证号
    th = img_preprocess(img, size=(5, 5))
    cv2.imshow("th", th)
    exceptIdCardImg, exceptIdCardImg_origin, idCardImg, idCardImg_origin = cutIdcard(th, img)
    # 得到头像
    th = img_preprocess(exceptIdCardImg_origin)
    expectFaceImg_origin, faceImg_origin = cutImg(th, exceptIdCardImg_origin)
    # 得到姓名
    th = img_preprocess_two(expectFaceImg_origin, size=(3, 1))
    nameImg, nameImg_origin = cutName(th, expectFaceImg_origin)
    # --------------------------得到精准的位置---------------------------------------------
    namePostion = getPosition(nameImg)
    idPostion = getPosition(idCardImg)
    # --------------------------识别开始--------------------------
    # 加载比对的数字
    digits = loadTmp()
    # 识别身份证
    id = ocrId(idCardImg_origin, idPostion, digits)
    print('id is :', id)
    # 识别姓名
    name = nameOcr(nameImg_origin, namePostion)
    print('name is :', name)
    return {"id":"".join(id),"name":"".join(name)}

if __name__ == '__main__':
    # main("Aa")
    img = cv2.imread('./img.png')
    # img = cv2.imread('./res/pic_input/2.jpg')
    img_2 = img.copy()
    # -----------------------------是否需要旋转-------------------------------------------
    img_resize = cv2.resize(img_2, (428, 270), interpolation=cv2.INTER_CUBIC)
    image_preprocessed = img_preprocess_two(img_resize, size=(10, 5), isReverse=False)
    number_region = find_number_region(image_preprocessed)
    image_id_number = get_number_img(img_resize, number_region)
    # 判断是否需要旋转
    image_correct, img = horizontal_correct(image_id_number, img, n=3)
    # --------------------------把身份证分为三部分------------------------------------------
    # 得到身份证号
    th = img_preprocess(img,size=(5, 5))
    cv2.imshow("th",th)
    exceptIdCardImg, exceptIdCardImg_origin, idCardImg, idCardImg_origin = cutIdcard(th,img)
    # 得到头像
    th = img_preprocess(exceptIdCardImg_origin)
    expectFaceImg_origin ,faceImg_origin= cutImg(th,exceptIdCardImg_origin)
    # 得到姓名
    th = img_preprocess_two(expectFaceImg_origin, size=(3, 1))
    nameImg, nameImg_origin = cutName(th, expectFaceImg_origin)
    # --------------------------得到精准的位置---------------------------------------------
    namePostion = getPosition(nameImg)
    idPostion = getPosition(idCardImg)
    # --------------------------识别开始--------------------------
    # 加载比对的数字
    digits = loadTmp()
    # 识别身份证
    id = ocrId(idCardImg_origin,idPostion,digits)
    print('id is :', id)
    # 识别姓名
    name = nameOcr(nameImg_origin,namePostion)
    print('name is :', name)

    # --------------------------框选图片进行可视化-----------------------------------------
    frameNameImg = kuangxuan(nameImg_origin, namePostion)
    frameIdCardImg = kuangxuan(idCardImg_origin, idPostion)
    # frameNameImg = kuangxuanName(nameImg, nameImg_origin)
    # frameIdCardImg = kuangxuanID(idCardImg, idCardImg_origin)
    cv2.imshow("name", frameNameImg)
    cv2.imshow("id", frameIdCardImg)
    cv2.imshow("face", faceImg_origin)
    cv2.imshow("img", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



