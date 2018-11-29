import cv2
import numpy as np

'''
    边缘检测函数
    模糊滤波器的核大小7×7
    边缘检测滤波器的核大小5×5
'''

def strokeEdges(src, blurKsize=7, edgeKsize=5):
    '''

    :param src: 需要处理的图像的路径
    :param blurKsize:模糊滤波器的核大小
    :param edgeKsize:边缘检测滤波器的核大小
    :return:返回图像边缘检测后的得到的图
    '''

    img = cv2.imread(src);

    # 第一步，先进行模糊处理，去噪
    if blurKsize >= 3:
        blurredImg = cv2.medianBlur(img, blurKsize)
        grayImg = cv2.cvtColor(blurredImg, cv2.COLOR_BGR2GRAY)
    else:
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转化色彩空间为灰度空间，使用高斯边缘检测时效果更明显

    # 第二步，检测边缘
    cv2.Laplacian(grayImg, cv2.CV_8U, grayImg, ksize=edgeKsize)

    normalizedInverseAlpha = (1.0 / 255) * (255 - grayImg) # 归一化

    channels = cv2.split(img)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha # 乘以原图想得到对应的边缘检测图
    return cv2.merge(channels)



if __name__=="__main__":


    new_pic = strokeEdges(src='myPic.png')
    # 显示边缘检测后的图像
    cv2.imshow('new_pic', new_pic)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    # 显示BGR图像中的B通道
    blue = cv2.split(new_pic)[0]
    cv2.imshow('blue', blue)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()