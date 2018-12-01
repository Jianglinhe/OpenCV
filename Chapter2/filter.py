import cv2
import numpy as np
'''
    稍微做一下测试就能找到适合的核
'''



# 边缘检测函数
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
    cv2.Laplacian(grayImg, cv2.CV_8U, grayImg, ksize=edgeKsize) # cv2.CV_8U表示每个通道为8位

    normalizedInverseAlpha = (1.0 / 255) * (255 - grayImg) # 归一化

    channels = cv2.split(img)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha # 乘以原图想得到对应的边缘检测图
    return cv2.merge(channels)


# 一般的卷积滤波器
class VConvolutionFilter(object):
    '''
    A filter that applies a convolution to V（or all od BGR）
    '''
    def __init__(self, kernel):
        self.__kernel = kernel

    def apply(self, src):
        '''Apply the filter with a BGR or Gray source/destination'''
        dst = cv2.filter2D(src, -1, self.__kernel) # src原图像，dst目标图像
        return dst

# 特定的锐化滤波器
class SharpenFilter(VConvolutionFilter):
    '''A sharpen filter with a 1-pixel raius'''

    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)

# 边缘检测核，将边缘转化为白色，非边缘区域转化为黑色
class FindEdgesFilter(VConvolutionFilter):
    '''An edge-finding filter with a 1-pixel dadius'''

    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]
                           ])
        VConvolutionFilter.__init__(self, kernel)


# 构建一个模糊滤波器（为了达到模糊效果，通常权重和应该为1，而且临近像素的权重全为正）
class BlurFilter(VConvolutionFilter):
    '''A blur filter with a 2-pixel radius'''

    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)



# 锐化、边缘检测以及模糊等滤波器都使用了高度对称的核，但有时不对称的核也会得到一些有趣的效果
# 同时具有模糊和锐化的作用，会产生一脊状或者浮雕的效果
class EmbossFilter(VConvolutionFilter):
    '''An emboss fliter with a 1-pixel radius'''

    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]])
        VConvolutionFilter.__init__(self, kernel)





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