#importing the necessary libraries for the project
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

import pandas as pd



#code to get the video from the harddrive of my pc to program
cap = cv2.VideoCapture('Synchro2.mov')

#for yellow color
yellow_lower = np.array([20,100,100],np.uint8)
yellow_upper = np.array([30,255,255],np.uint8)




while(cap.isOpened()):
    ret, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame,(5, 5), 0)
    #resizing the video
    resized = cv2.resize(blurred_frame, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    #croping the frontal view
    y=0
    x=0
    h=500
    w=450
    crop = resized[y:y+h, x:x+w]
    crop = cv2.medianBlur(crop,5)
    
    #turning the crop video file into the HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    #thresholding the HSV to only get yellow color
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask = cv2.erode (mask, None, iterations = 2)
    mask = cv2.dilate (mask, None, iterations = 2)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  
    for contour in contours:
        #finding the area of the contour
        area = cv2.contourArea(contour)
        #drawing the contours with green line
        cv2.drawContours(crop, contours, -1, (0, 255, 0), 1)
        

        print(contours)
        #print(area)
        #print(hierarchy)
        

    # Morphological Transform, Dilation
    #red = cv2.dilate(red, kernal)
    #res_red = cv2.bitwise_and(crop, crop, mask = red)
    
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(crop,crop, mask= mask)
    


            
    cv2.imshow("Original",crop)        
    cv2.imshow("Mask",mask)
    
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()   

fname = "output.csv"
with open(fname,'w',newline='') as f:
    writer = csv.writer(f)
    for row in contours:
        writer.writerow(row)
        
    style.use("fivethirtyeight")
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    
    def animate(i):
        graph_data = open("output.txt",'r').read()
        lines = graph_data.split("\n")
        xs = []
        ys = []
        for line in lines:
            if len(line)>1:
                x, y = line.split("2")
                xs.append(x)
                ys.append(y)
        ax1.clear()
        ax1.plot(xs, ys)
    ani = animation.FuncAnimation(fig, animate, interval = 1000)
    plt.show()








        

        

   
        




    
    
    
    
