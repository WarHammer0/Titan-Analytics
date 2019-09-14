# Titan-Analytics
Project to automatically generate scouting reports from football video using Machine Learning and Computer Vision. 

## Introduction
This project uses object detection/classification models and optical flow to gather statistics on football player's formations, movements, and target frequency. There are 7 major steps in the pipeline. 

1. Determine the time of snap and take a screenshot of the snap frame. 
2. Run the object detection model to find the locations of the players. 
3. Find the field lines using Hough transform.
4. Find the 'box' - main line of scrimmage area - and locate the Line of Scrimmage
5. Determine the offensive side and isolate players on the offense
6. Use optical flow to track the offensive player's movements
7. Match players movements to standard football routes and determine the offensive play. 

## 1. Determine the time of snap and take a screenshot of the snap frame

The input to the pipeline is a video, so it is essential to take a screenshot at the time of the snap. This is done by running a Structual Similarity model every fifteen frames. The idea is that while the quarterback is calling the cadence, the offense will stay still and wait for the ball to be snapped. So whichever two frames have the most structural similarity are considered the time of the snap and a screenshot is taken. The following are examples of screenshots at the time of the snap. 

## 2. Run the object detection model to find the locations of the players. 

A Keras Retinanet model was trained on 300 training images. The training images were extracted from Hudl, annotated with RectLabel, and augmented with OpenCV. The model works well with a .88 mAP and stores the locations of each object of class 'player'. The following are results of running the object detection model on a snap frame screenshot. 

![Object Detection](https://github.com/Aneesh1212/Titan-Analytics/blob/master/pictures/object_detection.jpg)

## 3. Find the field lines using Hough transform.
Identifying and knowing the angle of the field lines is very essential for recognizing perspective and the Line of Scrimmage. The safe assumption is that the field background will be green and the field lines will be white. First, a Canny Edge Detector is applied to the frame and then a Hough Transform is used to find the angles of the lines. The field lines are colored red in the following examples. 

## 4. Find the 'box' - main line of scrimmage area - and locate the Line of Scrimmage
Another object detection model is applied to find the 'box'. The 'box' is an imaginary area in football that includes the lineman, linebackers, and offensive backfield. The Line of Scrimmage will always run through the center of the 'box', so it is important to get the correct location of this object. The model was trained on the same images as the earlier model. It is not gaurunteed that a white field line will go directly through the box, so the two closest field lines are found and their angles are averaged. This angle is used as the angle of the line of scrimmage and drawn through the center of the box. 

## 5. Determine the offensive side and isolate players on the offense
At this point, all the players on the field are considered in the same class. So, using the line of scrimmage from Step 4, the offense and defense are separated into two teams based on their side of the line. However, it is unknown which group is the offense or defense, so an optical flow algorithm is ran on all the players. One football assumption is that the offense will be going forward while the defense will be backing up in response, so the overall X displacement is calculated using optical flow. The side that has a positive X displacement is considered the offense, and the opposite for the defense. 

## 6. Use optical flow to track the offensive player's movements
Once the offense is isolated, KLT Tracking is used only on the offensive players. KLT tracking is very helpful because it tracks a block of pixels rather than a single pixel, so if players cross field lines, their box is not dropped. However, when players overlap, there has been some trouble in maintainig the box on the right player. The lines drawn by each player is considered their 'route' and is drawn on the video and also a black screen. 
![Routes](https://github.com/Aneesh1212/Titan-Analytics/blob/master/pictures/routes1.jpg)

## 7. Match players movements to standard football routes and determine the offensive play. 
There are about 9 standard football routes in the 'route tree'. Using Mean Squared Error, the route drawn by Step 6 is matched to each standard route to determine which one it is. By analyzing all the players' routes, the total play can be determined. 
