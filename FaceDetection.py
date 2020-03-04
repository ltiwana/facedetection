# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:04:18 2019

@author: ltiwa
"""

from imutils.video import VideoStream
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import time
import cv2

cam = 0
capture_training_dir = "captured_for_training\\"
capture_detection_dir = "captured_for_recognition\\"
model_dir = "ModelFiles\\"

# load our serialized model from disk
print ("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(model_dir + "deploy.prototxt.txt", model_dir + "res10_300x300_ssd_iter_140000.caffemodel")

# intialize the video stream 
print ("[INFO] Starting video stream...")
vs = VideoStream(src=cam).start()
time.sleep(2.0)

# loop over the frames from video stream
while True:
    
    # grab the fram from the threaded video stream and resize it
    # to miaximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)    
    
    # grab the frame dimensions and convert it to blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, 
                                 (300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the detection
    # and prediction
    net.setInput(blob)
    detections = net.forward()
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    # loop over the detections
    for i in range (0, detections.shape[2]):
        
        # extract the confidence (i.e. probability) assocated with the 
        # prediction
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the confidence is 
        # greater than the minimum confidence
        if confidence < 0.5:
            continue
        
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # draw the bouding box of the fqqqqace along with the associated 
        # probability
        text = "{:.2f}%".format(confidence * 100)
        f, s, t = (cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        (text_width, text_height) = cv2.getTextSize(text, f, s, t)[0]
        y = startY - 10 if startY - 10 > 10 else startY + 10
        
        # Extract face shot and body shot from frame\
        # creating a copy of frame
        frame_copy = frame.copy()
        
        # calculate box width
        box_width = endX - startX
        
        # calculate box height
        box_height = endY - startY
        
        # extracting face shot
        # expanding the rectangle to make a better face crop
        # expanding the top by 50%, right by 20%
        # expanding the below chin part by only 10% and left by 20%
        bottom_left, top_left, top_right, bottom_right = (startX - int(startX * (30/100)),
                                                  startY - int(startY * (50/100)),
                                                  endX + int(endX * (20/100)),
                                                  endY + int(endY * (10/100))
                                                  )
        
        face_shot = frame_copy[top_left:bottom_right, bottom_left:top_right]
        cv2.imwrite(capture_training_dir + "my-image-face_" + timestr + ".png", face_shot)
        
        # extracting face shot for ecognition
        face_shot = frame_copy[startY:endY, startX:endX]
        cv2.imwrite(capture_detection_dir + "my-image-face_" + timestr + ".png", face_shot)
        
        # Extract body from frame
        
        # expanding the rectangle to make a better face crop
        # expanding the top by 50%, right by 20%
        # expanding the below chin part by only 10% and left by 20%
        bottom_left, top_left, top_right, bottom_right = (startX - box_width,
                                                  startY - int(startY * (50/100)),
                                                  endX + box_width,
                                                  endY + box_height
                                                  )

        body_shot = frame_copy[top_left:bottom_right, bottom_left:top_right]
        cv2.imwrite(capture_training_dir + "my-image-body_" + timestr + ".png", body_shot)
        
        
        cv2.rectangle(frame, (startX, startY), ( startX + text_width + 10, y - text_height - 5), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.putText(frame, text, (startX + 5, y),
            f, s, (0,0,0), t)
        
    # show the output frame
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the q key was pressed, break the from the loop
    if key == ord("q"):
        break
        
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
