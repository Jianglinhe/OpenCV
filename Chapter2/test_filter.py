import filter
import cv2
'''
    测试filter.py中的4中滤波器
'''

img = cv2.imread('myPic.png')
print(img.shape)

# 特定锐化卷积滤波器
sharpenFilter = filter.SharpenFilter()
dst1 = sharpenFilter.apply(img)
cv2.imwrite('sharpen.png', dst1)

# 边缘检测核
findEdgesFilter = filter.FindEdgesFilter()
dst2 = findEdgesFilter.apply(img)
cv2.imwrite('edge.png', dst2)

# 模糊滤波器
blurFilter = filter.BlurFilter()
dst3 = blurFilter.apply(img)
cv2.imwrite('blur.png', dst3)

# 同时具有模糊和锐化的浮雕效果
embossFilter = filter.EmbossFilter()
dst4 = embossFilter.apply(img)
cv2.imwrite('emboss.png', dst4)