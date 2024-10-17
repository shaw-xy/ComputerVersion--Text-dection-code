import cv2
import pytesseract
import numpy as np
from imutils.object_detection import non_max_suppression



#加载图片
image = cv2.imread('cash.jpg')
ORI_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 图像预处理函数
# get grayscale image 灰度图
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal 降噪
def remove_noise(image):
    return cv2.medianBlur(image,5)

#thresholding 二值化
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation 膨胀
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

#erosion 侵蚀
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)




#EAST
# 定义 EAST 文本检测器的路径
east_path = r"D:\EAST\frozen_east_text_detection.pb"
net = cv2.dnn.readNet(east_path)

#确定图像最终尺寸，注意EAST需求图像长宽均为32的倍数
height, width = image.shape[:2]
new_width = 320
new_height = 320
resized_image = cv2.resize(image, (new_width, new_height))
(H, W) = resized_image.shape[:2]

#设置bolb，尺寸，均值，缩放因子，满足网络输入需求
blob = cv2.dnn.blobFromImage(resized_image, 1.0, (H, W), (123.68, 116.78, 103.94), swapRB=True, crop=False)

#输入bolb进入网络的输出层
net.setInput(blob)
(scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

#收集网络输出的信息，遍历每个元素，调整位置
(num_rows, num_cols) = scores.shape[2:4]
rects = []
confidences = []


for y in range(0, num_rows):
    scores_data = scores[0, 0, y]
    x_data_0 = geometry[0, 0, y]
    x_data_1 = geometry[0, 1, y]
    x_data_2 = geometry[0, 2, y]
    x_data_3 = geometry[0, 3, y]
    angles_data = geometry[0, 4, y]

    for x in range(0, num_cols):
        if scores_data[x] < 0.5:
            continue
    #偏移
        offset_x = x * 4.0
        offset_y = y * 4.0
        #获取角度值，然后计算得到sin和cos
        angle = angles_data[x]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        #文本高度和宽度
        h = x_data_0[x] + x_data_2[x]
        w = x_data_1[x] + x_data_3[x]
        #右下
        end_x = int(offset_x + (cos_a * x_data_1[x]) + (sin_a * x_data_2[x]))
        end_y = int(offset_y - (sin_a * x_data_1[x]) + (cos_a * x_data_2[x]))
        #左上
        start_x = int(end_x - w)
        start_y = int(end_y - h)

        rects.append((start_x, start_y, end_x, end_y))
        confidences.append(scores_data[x])


#抑制非极大值
boxes = non_max_suppression(np.array(rects), probs=confidences)



#在图像绘制矩形框
for (start_x, start_y, end_x, end_y) in boxes:
    start_x = int(start_x * (width / W))
    start_y = int(start_y * (height / H))
    end_x = int(end_x * (width / W))
    end_y = int(end_y * (height / H))
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    text_box_image = ORI_image[start_y:end_y, start_x:end_x]
    # Use Tesseract to add text around the rectangle
    custom_config = '--oem 3 --psm 6'
    text = pytesseract.image_to_string(text_box_image, config=custom_config)
    cv2.putText(image, text.strip(), (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# Tesseract方法，识别图中文本并打印
img = ORI_image
custom_config = '--oem 3 --psm 6'
text = pytesseract.image_to_string(img, config=custom_config)
print(text)

# 展示成果
cv2.imshow("Text Detection result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

