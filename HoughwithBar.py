import cv2
import numpy as np

#载入图片
img_original=cv2.imread('5.jpg')
scale_percent = 30 # percent of original size
width = int(img_original.shape[1] * scale_percent / 100)
height = int(img_original.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img_original, dim, interpolation = cv2.INTER_AREA) 

#设置窗口
cv2.namedWindow('Canny', 0)    
cv2.resizeWindow('Canny', 1024, 768)
#定义回调函数
def nothing(x):
    pass
#创建两个滑动条，分别控制threshold1，threshold2
cv2.createTrackbar('threshold1','Canny',50,400,nothing)
cv2.createTrackbar('threshold2','Canny',100,400,nothing)
cv2.createTrackbar('minLineLength','Canny',10,1000,nothing)
cv2.createTrackbar('maxLineGap','Canny',10,1000,nothing)
while(1):
    #在新图中画线
    resized_new = resized.copy()
    #返回滑动条所在位置的值
    threshold1=cv2.getTrackbarPos('threshold1','Canny')
    threshold2=cv2.getTrackbarPos('threshold2','Canny')
    threshold3=cv2.getTrackbarPos('minLineLength','Canny')
    threshold4=cv2.getTrackbarPos('maxLineGap','Canny')

    #Canny边缘检测
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #转灰度图
    dst = cv2.equalizeHist(gray) #直方图均衡化
    gaussian = cv2.GaussianBlur(dst, (5, 5), 1) #高斯模糊 高斯核 5x5
    img_edges=cv2.Canny(gaussian,threshold1,threshold2)

    lines = cv2.HoughLinesP(img_edges,1,np.pi/180,100,minLineLength=threshold3,maxLineGap=threshold4) #概率霍夫变换
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(resized_new,(x1,y1),(x2,y2),(0,255,0),2)   #画线
    #显示图片
    cv2.namedWindow('original', 0)    
    cv2.resizeWindow('original', 1024, 768)
    cv2.imshow('original',resized_new)
    cv2.imshow('Canny',img_edges)
    #cv2.waitKey(0)  
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()