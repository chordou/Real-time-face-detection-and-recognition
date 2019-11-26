# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:15:23 2019

@author: cqiuac
"""
 
 
import cv2
import sys
from PIL import Image
 
def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)
    

    cap = cv2.VideoCapture(0)                
    

    classfier = cv2.CascadeClassifier("C:\\Users\\cqiuac\\Downloads\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")
    
    #RGB
    color = (0, 255, 0)
        
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:            
            break  
 

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                 
        

        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect        
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                        

        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
    
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows() 
    
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo("recognizing face", 0)
        