import cv2
import numpy as np

# 读入一张图像
img = cv2.imread('hammer.png')

# cv2.threshold() 简单阈值，选取一个全局阈值，然后将整幅图像分成非黑即白的二值图像
# 输入4个参数，第一个参数为原图，第二个参数为自己设定的阈值，第三个参数是高于（低于）阈值时富裕的新值，第四个参数是一个方法选择参数
# 返回两个参数，第一个得到的阈值值，第二个就是阈值化后的图像
# cv2.THRESH_BINARY（黑白二值）
# cv2.THRESH_BINARY_INV（黑白二值反转）
# cv2.THRESH_TRUNC （得到的图像为多像素值）
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY) # 对原图进行二值化

# cv2.findContours()会改变输入图像,故上面使用img.copy()，避免原图被改变
# cv2.findContours()有三个参数，第一个输入图像，第二个是轮廓检索模式，第三个是轮廓近似方法
# 返回的第二元素是轮廓，这个轮廓是一个列表，列表的每个元素代表一个轮廓
# 参数2
# cv2.RETR_LIST 从解释的角度来看，这中应是最简单的。它只是提取所有的轮廓，而不去创建任何父子关系。
# cv2.RETR_EXTERNAL 如果你选择这种模式的话，只会返回最外边的的轮廓，所有的子轮廓都会被忽略掉。
# cv2.RETR_CCOMP 在这种模式下会返回所有的轮廓并将轮廓分为两级组织结构。
# cv2.RETR_TREE 这种模式下会返回所有轮廓，并且创建一个完整的组织结构列表。
# 参数3
# cv2.CHAIN_APPROX_NONE 表示边界上的所有的点都会被存储
# cv2.CHAIN_APPROX_SIMPLE 会压缩轮廓，将轮廓上冗余点去掉，比如说四边形就会只储存四个角点
image, countours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


print("总共有%d个轮廓"%len(countours))


# 遍历所有轮廓
for c in countours:

    # 第一种边框(矩形,没有旋转)
    # 计算出一个简单的边界框,用最小的矩形将找到的形状包起来,不带旋转
    # 将轮廓信息转换成(x, y)坐标，并加上举行的高度与宽度，画出一个矩形
    # 画矩形,第一个参数img是原图,(x,y)是矩形的左上点的坐标,(x+w, y+h)是矩阵的右下点坐标,(0, 255, 0)画线的颜色信息,1为线的宽度信息
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)  # 绿色

    # 第二种边框(矩形,有旋转)
    # 计算出包围目标最小的矩形区域,带旋转
    rect = cv2.minAreaRect(c) # 返回最小外接矩形的(中心(x, y), (宽, 高), 旋转角度)
    box = cv2.boxPoints(rect) # 获取最小外接矩形的4个顶点
    box = np.int0(box)  # opencv没有函数能直接从轮廓信息中计算最小矩形顶点的坐标,需要计算出最小矩形区域,然后计算这个矩形的顶点.计算出来的都是浮点数,但是所有像素的坐标值是整数,需要转换
    cv2.drawContours(img, [box], 0, (0, 0, 255), 1) # 红色

    # 第三种边框(圆)
    # 寻找最小包围圆形
    (x, y), radius = cv2.minEnclosingCircle(c) # 返回两个元素,第一个元素为圆心的坐标组成的二元组,第二个元素为元的半径值
    center = (int(x), int(y))   # 转换为整型
    radius = int(radius)
    img = cv2.circle(img, center, radius, (255, 0, 0), 1) # 蓝色


# cv2.drawContours()用来绘制轮廓
# 第一个参数是一张图片,可以是原图或者其他;第二个参数是轮廓,也可以说是cv2.findCountours()找出来的点集;第三个参数是对轮廓(第二个参数)的索引,当需要绘制独立轮廓时很有用,若要全部绘制可设为-1
# 第四种边框(贴着物体描边)
cv2.drawContours(img, countours, -1, (128, 255, 128), 2)


cv2.imshow("contours", img)
cv2.waitKey(10000)
cv2.destroyAllWindows()


