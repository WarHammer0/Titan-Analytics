#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 18:24:40 2019

@author: aneesh
"""

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

#import SSIM for finding snap
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import csv

# import miscellaneous modules
import cv2
import numpy as np
import time

from mse_estime import testMSE

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from random import  randint
import sys

from perspective_transform import get_homography, warp_image, rectify_image

"""Iterates through play and finds two frames with greatest structural similarity.
Saves frame as snap.jpg. Also saves next two frames as snap2.jpg and snap3.jpg 
in case the Hough Transform can't find lines on snap.jpg later in the program"""
def find_snap():
    print("FINDING THE SNAP FRAME")
    global video_path, goodCounter
    counter=0
    lowestSSIM = 0
    cap = cv2.VideoCapture(video_path)
    ret,old_frame = cap.read()
    while(1):
        ret,new_frame = cap.read()
        if counter>=10 and counter%10==0:
            if type(new_frame) is np.ndarray:
                s = ssim(old_frame, new_frame, multichannel=True)
                if s>lowestSSIM:
                    snapFrame = new_frame
                    lowestSSIM = s
                    goodCounter = counter
                old_gray = new_frame.copy()
            else:
                break    
        if counter>100:
            break
        counter+=1
    cap = cv2.VideoCapture(video_path)
    
    counter=0
    while(1):
        ret,new_frame = cap.read()
        if type(new_frame) is np.ndarray:
            if counter == goodCounter:
                cv2.imwrite("snap.jpg", new_frame)
            elif counter == goodCounter+4:
                cv2.imwrite("snap2.jpg", new_frame)
            elif counter == goodCounter+8:
                cv2.imwrite("snap3.jpg", new_frame)
            elif counter > goodCounter + 10:
                break
        else:
            break    
        counter+=1


"""Gets session for CNN object detection"""
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
    
    
"""Loads in ML model and performs object detection on snap.jpg.
    The bounding boxes are saved as p0"""
def object_detection():
    print("PERFORMING OBJECT DETECTION")
    global p0, boxes, scores, labels
    p0 = np.array([[[0, 0]]], dtype=np.float32)
    
    
    keras.backend.tensorflow_backend.set_session(get_session())
    model_path = 'saved_good_files/player_detection_inference_model.h5'
    model = models.load_model(model_path, backbone_name='resnet50')
    
    # load image
    image = read_image_bgr("snap.jpg")
    
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    
    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    # correct for image scale
    boxes /= scale
    
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
            
        color = label_color(label)
        
        b = box.astype(int)
        center = [(b[0]+b[2])/2, (b[1]+b[3])/2]
        p1 = np.array([[[center[0], center[1]]]], dtype=np.float32)
        p0 = np.concatenate((p0, p1))
        draw_box(draw, b, color=color)
        

        
    """
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()
    """
    
    
"""Finds the direction of the offense, which is saved as 'direction'. 
The function drops optical flow pins on the bounding boxes from p0 and then 
sums up the horizontal direction of all the points. Based on the assumption that 
offense is moving forward, it finds which direction the offense is on"""
def optical_flow():
    print("FINDING DIRECTION OF OFFENSE")
    global time_counter, original_points, goodCounter, p0, direction, finding, type_of_play
    time_counter = 0
    original_points = p0
    
    
    cap = cv2.VideoCapture(video_path)
    
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorneßs = 100,
                           qualityLevel = 0.9,
                           minDistance = 7,
                           blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Create some random colors
    color = np.random.randint(0,255,(100,3))    
    
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(old_frame)

    while(True):
        time.sleep(.005)
        time_counter+=1
        ret,frame = cap.read()
        try: 
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            break
    
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
        p0 = good_new.reshape(-1,1,2)
        #cv2.imshow('frame',img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        
        #Only analyze the first 190 frames
        if time_counter==goodCounter + 190:
            saved_points = p1
        
    cv2.destroyAllWindows()
    
    sum=0
    print(type(saved_points))
    print(saved_points)

    #Sum up the final x displacement of each pin
    for i in range(len(saved_points)):
        sum += saved_points[i][0][0] - original_points[i][0][0]
    print(sum)
    
    if sum>0:
        if finding == "Offense":    
            direction = "Left"
        else:
            if direction == "Left":
                type_of_play = "Pass"
            else:
                type_of_play = "Run"
    else :
        if finding == "Offense":
            direction = "Right"
        else:
            if direction == "Left":
                type_of_play = "Run"
            else:
                type_of_play = "Pass"
                
                
    print(direction)


"""Returns the bounding box around the 'box' near the line of scrimmage using 
    the CNN model for LOS. Stores the bounding box in maxBox. """
def find_los():
    print("FINDING THE LINE OF SCRIMMAGE")
    global maxBox, los_boxes, los_scores, los_labels
    
    keras.backend.tensorflow_backend.set_session(get_session())
    
    model_path = 'saved_good_files/los_detection_inference_model.h5'
    
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    
    image = read_image_bgr('snap.jpg')
    
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    
    los_boxes, los_scores, los_labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    # correct for image scale
    los_boxes /= scale
    
    maxScore=0
    maxBox = None
    # Find the box with the greatest confidence
    for box, score, label in zip(los_boxes[0], los_scores[0], los_labels[0]):
        # scores are sorted so we can break
        if score >maxScore:
            maxScore = score
            maxBox = box
            

"""Return only white pixels in img"""
def only_white(img):
	lower_white = np.array([200,200,200])
	upper_white = np.array([250,250,250])
	mask = cv2.inRange(img,lower_white,upper_white)
	output = cv2.bitwise_and(img, img, mask = mask)
	return output


good_boxes =[]

"""Get the midpoint lol"""
def get_midpoint(x1, x2, y1, y2, y):
    return ((y-y1)*(x2-x1) / (y2-y1)) + x1
    

"""Returns lines of performing Hough Transform on image at image_path"""
def hough_transform(image_path):
    img = cv2.imread("snap.jpg")
    img = cv2.resize(img, (852,482))
    wimg = only_white(img)
    gray = cv2.cvtColor(wimg,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,200,400,apertureSize = 3)
    
    lines = cv2.HoughLines(edges,1,np.pi/90,100)
    return lines


"""Returns the midpoint and endpoint coordinates of the line from the Hough 
transform lines that is closest to the line of scrimmage. Closest is defined as 
smallest horizontal direction from good_x which is the midpoint of the 'box' found
from find_los()
"""
def find_closest_line(lines):
    closest = 10000000
    for lineT in lines:
        for rho,theta in lineT:            
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 10000*(-b))
            y1 = int(y0 + 10000*(a))
            x2 = int(x0 - 10000*(-b))
            y2 = int(y0 - 10000*(a))
            if abs((y2-y1)/(x2-x1)) >= 1:
                #cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
                midpoint = get_midpoint(x1, x2, y1, y2, good_y)
                if abs(good_x - midpoint)<closest:
                    closest = abs(good_x - midpoint)    
                    best_midpoint = midpoint
                    final_points = [x1, x2, y1, y2]
    return best_midpoint, final_points


"""Finds the closest Hough transform line to the LOS. Snap.jpg is used and then 
snap2.jpg and snap3.jpg can potentially be used if no Hough lines are found in 
snap.jpg. After finding the LOS, the players on the side of the offense, stored
in 'direction', are drawn on the image. The offensive players' bounding boxes 
are stored in good_boxes to be used later in KLT tracking"""
def draw_line():
    global p0, finding
    print("DRAWING THE LINE OF SCRIMMAGE")
    global los_box, good_x, good_y, boxes, scores, labels, direction
    
    lines = hough_transform("snap.jpg")
    best_midpoint, final_points = find_closest_line(lines)
    hor_dist = int(good_x - best_midpoint)
    print("FIRST IMAGE")
    if abs(hor_dist) > 50:
        lines = hough_transform("snap2.jpg")
        best_midpoint, final_points = find_closest_line(lines)
        hor_dist = int(good_x - best_midpoint)
        print("SECOND IMAGE")
        
    if abs(hor_dist) > 50:
        lines = hough_transform("snap3.jpg")
        best_midpoint, final_points = find_closest_line(lines)
        hor_dist = int(good_x - best_midpoint)
        print("THIRD IMAGE")
    
    print("HORIZONTAL DISTANC: " + str(hor_dist))
    #rectangle_img = img.copy()
    #cv2.rectangle(rectangle_img,(int(los_box[0]),int(los_box[1])),
                  #(int(los_box[2]),int(los_box[3])),(0,255,0),3)

    line_points = [int(final_points[0])+hor_dist, int(final_points[2]), int(final_points[1])+hor_dist, int(final_points[3])]
    cv2.line(img, (int(final_points[0])+hor_dist, int(final_points[2])) ,(int(final_points[1])+hor_dist, int(final_points[3])),(0,0,255),3)

    count=0

    numLeft=0
    numRight=0
    p0 = np.array([[[0, 0]]], dtype=np.float32)
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        count+=1
        # scores are sorted so we can break
        if score < 0.5:
            break

        b = box.astype(int)

        center = [(b[0]+b[2])/2, (b[1]+b[3])/2]
        

        if center[0] > get_midpoint(line_points[0],line_points[2], line_points[1],
                        line_points[3], center[1]):
            numRight+=1
        else:
            numLeft+=1
            
        mid = get_midpoint(line_points[0],line_points[2], line_points[1], line_points[3], center[1])

        if direction == "Right":
            if mid < center[0] < 200 + mid:
                good_boxes.append(b)
                cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), (255,0,0), 3)
            elif mid - 120 < center[0] < mid - 50 and los_box[1]-40<center[1]<los_box[3]+40 :
                cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), (0,0,255), 3)
                p1 = np.array([[[center[0], center[1]]]], dtype=np.float32)
                p0 = np.concatenate((p0, p1))
                
        elif direction == "Left":
            if mid - 200 <= center[0] <= mid:
                good_boxes.append(b)
                cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), (255,0,0), 3)
            elif mid + 50 < center[0] < mid + 120 and los_box[1]-40<center[1]<los_box[3]+40:
                cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), (0,0,255), 3)
                p1 = np.array([[[center[0], center[1]]]], dtype=np.float32)
                p0 = np.concatenate((p0, p1))


    
    print(p0)

    """
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 852,480)
    cv2.imshow("image",img)
    #cv2.imshow("box", rectangle_img)
    
    while True:  
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    """
    
    cv2.destroyAllWindows()
    print("GOOD BOXES")
    print(good_boxes)
    finding = "Type of play"
    
    
"""Track the boxes in good_boxes (offensive players). Performs KLT tracking 
algorithm on these boxes and stores their location in new_mid. Lines between 
their points are also drawn with mask which is outputted to the screen. Additionally, 
the perspective transform is calculated and the points are transformed and also 
outputted."""
def track(bboxes, vp1, vp2, clip_factor):
     counter=0

     cap = cv2.VideoCapture(video_path)
     csvfile = open("playerData.csv", "w")
     writer = csv.writer(csvfile, delimiter="|")

     success, frame = cap.read()
     #frame = warp_image(frame, vp1, vp2, clip_factor)
     if not success:
         sys.exit(1)
     colors = []
     old_mid = []
     temp = []
     
     for bbox in bboxes:
         old_mid.append((((bbox[0] + bbox[2])/2), ((bbox[1] + bbox[3])/2)))
         t = tuple((bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
         temp.append(t)
         
     bboxes = temp
     multiTracker = cv2.MultiTracker_create()
     #frame = warp_image(frame, vp1, vp2, clip_factor)
     for bbox in bboxes:
         colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
         multiTracker.add(cv2.TrackerCSRT_create(), frame, bbox)
    
    
     mask = np.zeros_like(frame)
     
     #Find perspective transform matrix
     transformed_mask = np.zeros_like(frame)
     homography = get_homography(frame, vp1, vp2, 4)
     while cap.isOpened():
       counter+=1
       success, frame = cap.read()

       warped_frame = warp_image(frame, vp1, vp2, clip_factor)
       if not success:
         break
    
       success, boxes = multiTracker.update(frame)
       for i, newbox in enumerate(boxes):
         print(i)
         print(newbox)
         p1 = (int(newbox[0]), int(newbox[1]))
         p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
         
         #Find midpoint of moved bounding box
         new_mid = (((p1[0] + p2[0])/2), ((p1[1] + p2[1])/2))
         
         #Compute new lines to be drawn
         mask = cv2.line(mask, (int(new_mid[0]),int(new_mid[1])),(int(old_mid[i][0]), int(old_mid[i][1])), colors[i], 2)
         
         #Transform points with transformation matrix 
         old_mid_array = np.array([[old_mid[i][0]], [old_mid[i][1]], [1]], dtype=float)
         transformed_old_mid = np.matmul(homography, old_mid_array)
         transformed_old_mid = transformed_old_mid / transformed_old_mid[2]
         
         new_mid_array = np.array([[new_mid[0]], [new_mid[1]], [1]], dtype=float)
         transformed_new_mid = np.matmul(homography, new_mid_array)
         transformed_new_mid = transformed_new_mid / transformed_new_mid[2]
         
         #Computer transformed mask 
         transformed_mask = cv2.line(transformed_mask, (int(transformed_new_mid[0]),int(transformed_new_mid[1])),
                                        (int(transformed_old_mid[0]), int(transformed_old_mid[1])), colors[i], 2)
         
         #Draw new bounding boxes
         cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
         old_mid[i] = new_mid
         new_img = cv2.add(frame,mask)

         #Resize transformed mask
         transformed_mask = cv2.resize(transformed_mask,(len(warped_frame[0]),len(warped_frame)))
         warped_frame = warped_frame.astype(float)
         transformed_mask = transformed_mask.astype(float)
         new_warped_image= cv2.add(warped_frame, transformed_mask)
         
         #Save points in csv
         r,g,b=colors[i]
         playid=str(r)+str(g)+str(b)
         x=int(new_mid[0])
         y=int(new_mid[1])
         csvin=playid+ "," + str(x) + "," + str(y) + "," + str(counter)
         writer.writerow([csvin])

        #Show original video, video with tracking, transformed video with tracking
       cv2.imshow('MultiTracker', new_img)
       cv2.imshow("aneesh2", mask)
       #cv2.imshow("transformed2", transformed_mask)
       cv2.imshow("transformed", new_warped_image)

        #Save image of tracked routes at some hard coded frame
       if counter==230:
           cv2.imwrite("routes.jpg", mask)
           cv2.imwrite("routes_on_play.jpg", new_img)

    
       if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
         break
    

    
     cv2.destroyAllWindows()
     cap.release()
    
    

    
    
    
if __name__ == "__main__":
    finding = "Offense"
    video_path = sys.argv[-1]
    find_snap()
    object_detection()
    optical_flow()
    direction="Right"
    find_los()
    img = cv2.imread("snap.jpg")
    img = cv2.resize(img, (852,480))
    los_box = maxBox
    good_x = int((los_box[0] + los_box[2])/2)
    good_y = int((los_box[1] + los_box[3])/2)
    draw_line()
    optical_flow()
    #print(type_of_play)
    vp1, vp2, clip_factor = rectify_image("snap.jpg", 4, algorithm="independent")
    track(good_boxes, vp1, vp2, clip_factor)



    
#track([[ 78, 128, 108, 176], [328, 298, 365, 357], [147, 244, 183, 318], [125, 170, 158, 217]])



        
