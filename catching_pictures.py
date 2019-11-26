# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:49:03 2019

@author: cqiuac
"""

import cv2
import sys
 
from PIL import Image
 
def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(0)                
    
    #告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("C:\\Users\\cqiuac\\Downloads\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")
    
    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
    
    num = 0    
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:            
            break                
    
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将当前桢图像转换成灰度图像            
        

        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect                        
                

                img_name = '%s/%d.jpg'%(path_name, num)                
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)                                
                                
                num += 1                
                if num > (catch_pic_num):
                    break
                

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)                
        

        if num > (catch_pic_num): break                
                       
        #显示图像
        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(33)
       #''' if (c != 255):
          #break
        if c & 0xFF == ord('q'):
            break        
    

    cap.release()
    cv2.destroyAllWindows() 
    
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo("截取人脸", 0, 400, r'C:\Users\cqiuac\Desktop\Xiehao')

